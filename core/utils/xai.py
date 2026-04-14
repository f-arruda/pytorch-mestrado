import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients, FeatureAblation

class SolarXAIEngine:
    def __init__(self, model, device):
        """Engine de XAI escalável para Séries Temporais Solares."""
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)
        
        # O FeatureAblation agora aponta pro _model_wrapper para consistência dimensional
        self.ig = IntegratedGradients(self._model_wrapper)
        self.ablator = FeatureAblation(self._model_wrapper)
        self._attention_buffer = []

    def _model_wrapper(self, inputs, target_step_idx=0, feature_idx=0):
        output = self.model(inputs)
        # Retorna a previsão do horizonte futuro (target_step) para a variável específica (kt ou power)
        return output[:, target_step_idx, feature_idx]

    def compute_global_feature_importance(self, loader, feature_names=None, save_path=None):
        print("      🔍 Calculando Importância Global (Ablation)...")
        importances_list = []
        with torch.backends.cudnn.flags(enabled=False):
            for batch in loader:
                # O loader retorna uma tupla (x, y, mask, aux), pegamos apenas o x
                x = batch[0].to(self.device).float()
                seq_len, n_feats = x.shape[1], x.shape[2]
                mask = torch.arange(n_feats).view(1, 1, -1).repeat(1, seq_len, 1).to(self.device)
                
                attr = self.ablator.attribute(x, feature_mask=mask)
                batch_imp = attr[:, 0, :].cpu().detach().numpy()
                importances_list.append(batch_imp)
                
        all_imps = np.concatenate(importances_list, axis=0)
        global_imp = np.mean(np.abs(all_imps), axis=0)
        global_imp_norm = (global_imp / global_imp.sum()) * 100
        
        if feature_names and save_path:
            self.plot_global_importance(global_imp_norm, feature_names, save_path)
        return global_imp_norm

    def compute_temporal_importance(self, dataloader, feature_names=None, target_step_idx=0, max_samples=300, save_path=None):
        print(f"      ⏳ Calculando Padrão Temporal (IG) para horizonte h={target_step_idx+1}...")
        accumulated_saliency = None
        count = 0
        prev_cudnn_state = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            for batch in dataloader:
                x = batch[0]
                if count >= max_samples: break
                x = x.to(self.device).float().requires_grad_()
                
                current_wrapper = lambda inp: self._model_wrapper(inp, target_step_idx=target_step_idx)
                local_ig = IntegratedGradients(current_wrapper)
                
                attr, _ = local_ig.attribute(x, baselines=torch.zeros_like(x), n_steps=15, return_convergence_delta=True)
                batch_saliency = attr.abs().cpu().detach().numpy()
                sum_saliency = np.sum(batch_saliency, axis=0) 
                
                if accumulated_saliency is None: 
                    accumulated_saliency = sum_saliency
                else: 
                    accumulated_saliency += sum_saliency
                count += x.shape[0]
        finally:
            torch.backends.cudnn.enabled = prev_cudnn_state
            
        mean_saliency = accumulated_saliency / count
        
        if feature_names and save_path:
            self.plot_temporal_profile(mean_saliency, feature_names, save_path)
        return mean_saliency

    def get_local_explanation(self, x, target_idx=0, feature_names=None, save_path=None):
        prev_cudnn_state = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            x = x.to(self.device).float().requires_grad_()
            current_wrapper = lambda inp: self._model_wrapper(inp, target_step_idx=0, feature_idx=target_idx)
            local_ig = IntegratedGradients(current_wrapper)
            
            attr, _ = local_ig.attribute(x, baselines=torch.zeros_like(x), n_steps=50, return_convergence_delta=True)
            attr_np = attr.squeeze(0).cpu().detach().numpy()
            
            if feature_names and save_path:
                self.plot_heatmap(attr_np, feature_names, title=f"Explicação Local (Alvo: {target_idx})", save_path=save_path)
            return attr_np
        finally:
            torch.backends.cudnn.enabled = prev_cudnn_state

    # --- NOVO: Extração de Atenção ---

    def _attention_hook(self, module, input, output):
        """
        Hook executado toda vez que a camada de atenção funciona.
        O output da Attention Layer geralmente é (context, attn_weights).
        Pegamos o attn_weights (índice 1).
        """
        # Atenção: Output pode variar dependendo da implementação exata da classe Attention
        # Geralmente: return context, attn_weights
        if isinstance(output, tuple):
            weights = output[1] # [Batch, Output_Len(1), Input_Len]
            self._attention_buffer.append(weights.detach().cpu())
        else:
            # Fallback se retornar só os pesos
            self._attention_buffer.append(output.detach().cpu())

    def collect_attention_maps(self, input_tensor):
        """
        Roda uma inferência e coleta os mapas de atenção para uma amostra.
        """
        self._attention_buffer = [] # Limpa buffer
        
        # 1. Tenta registrar o hook na camada de atenção
        # Precisamos achar model.decoder.attention
        handle = None
        found = False
        
        # Busca recursiva pela camada 'attention'
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() and 'decoder' in name.lower():
                # Encontrou a camada de atenção do decoder
                handle = module.register_forward_hook(self._attention_hook)
                found = True
                break
        
        if not found:
            print("⚠️ Camada de atenção não encontrada automaticamente pelo nome.")
            return None

        # 2. Faz inferência
        with torch.no_grad():
            self.model(input_tensor.to(self.device))
        
        # 3. Remove o hook (limpeza)
        handle.remove()
        
        # 4. Processa o buffer
        # O buffer terá uma lista de tensores, um para cada passo do decoder (se for loop)
        # ou um tensor único se for paralelizado.
        # Assumindo loop: [Step1_Weights, Step2_Weights, ...]
        
        if len(self._attention_buffer) == 0:
            return None
            
        # Concatena os passos de tempo do decoder
        # Cada item é [Batch=1, 1, Input_Seq] -> Queremos [Input_Seq, Output_Seq] para plotar
        
        try:
            # Empilha ao longo da dimensão de saída
            # Buffer: List of [1, 1, In_Seq] (se batch=1)
            attn_stack = torch.cat(self._attention_buffer, dim=1) # [1, Out_Seq, In_Seq]
            return attn_stack.squeeze(0).numpy().T # Transpõe para [In_Seq (Y), Out_Seq (X)] ou vice-versa
        except:
            return None

    # --- Métodos de Plotagem ---

    def plot_attention_map(self, attn_matrix, feature_names=None, title="Mapa de Atenção", save_path=None):
        """
        Plota Input (Passado) vs Output (Futuro).
        attn_matrix shape esperado: [Input_Seq, Output_Seq]
        """
        plt.figure(figsize=(10, 8))
        
        # Eixo X: Futuro (Horizonte de Previsão)
        # Eixo Y: Passado (Histórico)
        sns.heatmap(attn_matrix, cmap='viridis', cbar_kws={'label': 'Peso de Atenção $\\alpha$'})
        
        plt.xlabel('Horizonte de Previsão (Futuro)')
        plt.ylabel('Histórico (Passado - Lags)')
        plt.title(title)
        
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    # ... [MANTENHA OS PLOTS ANTERIORES: plot_global_importance, plot_temporal_profile, plot_heatmap] ...
    def plot_global_importance(self, importances, feature_names, save_path=None):
        plt.figure(figsize=(10, 6))
        indices = np.argsort(importances)
        plt.barh(range(len(indices)), importances[indices], color='#4c72b0')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importância Relativa (%)')
        plt.title('Importância Global das Variáveis (Feature Ablation)')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_temporal_profile(self, saliency_matrix, feature_names, save_path=None):
        plt.figure(figsize=(12, 6))
        seq_len = saliency_matrix.shape[0]
        lags = np.arange(seq_len, 0, -1) 
        for i, feat in enumerate(feature_names):
            plt.plot(lags, saliency_matrix[:, i], label=feat, marker='o', markersize=4)
        plt.gca().invert_xaxis() 
        plt.xlabel('Horas no Passado (Lags)')
        plt.ylabel('Influência Média (IG)')
        plt.title('Decaimento Temporal da Importância')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def plot_heatmap(self, saliency_matrix, feature_names, title, save_path=None):
        plt.figure(figsize=(12, 6))
        sns.heatmap(saliency_matrix.T, yticklabels=feature_names, cmap='inferno', cbar_kws={'label': 'Importância'})
        plt.xlabel('Tempo (Passos na Janela)')
        plt.title(title)
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # --- NOVO: Extração de Feature Attention ---
    def _feature_hook(self, module, input, output):
        # FeatureAttention retorna (weighted_input, weights)
        # Queremos os weights (índice 1)
        if isinstance(output, tuple) and len(output) > 1:
            weights = output[1] # Shape: [Batch, Seq_Len, Features]
            self._feature_buffer.append(weights.detach().cpu())

    def collect_feature_weights(self, input_tensor):
        """
        Coleta os pesos da Feature Attention (Se existir).
        Retorna matriz: [Features, Seq_Len (Tempo)]
        """
        self._feature_buffer = []
        handle = None
        
        # 1. Busca a camada feat_att no encoder
        if hasattr(self.model.encoder, 'feat_att') and self.model.encoder.feat_att is not None:
            handle = self.model.encoder.feat_att.register_forward_hook(self._feature_hook)
        else:
            return None # Modelo não tem feature attention

        # 2. Inferência
        with torch.no_grad():
            self.model(input_tensor.to(self.device))
        
        # 3. Limpeza
        handle.remove()
        
        if len(self._feature_buffer) == 0: return None
        
        # 4. Processamento
        # Pega o primeiro item do buffer (Batch=1) -> [1, Seq, Feat]
        weights = self._feature_buffer[0] 
        # Remove batch e transpõe para plotagem (Features no Y, Tempo no X)
        return weights.squeeze(0).numpy().T # Shape final: [Features, Seq_Len]

    def plot_feature_weights(self, weight_matrix, feature_names, title="Feature Attention Weights", save_path=None):
        """
        Plota o mapa de calor da Feature Attention.
        Eixo Y: Variáveis
        Eixo X: Tempo (Passado)
        """
        plt.figure(figsize=(12, 6))
        sns.heatmap(weight_matrix, yticklabels=feature_names, cmap='viridis', cbar_kws={'label': 'Importância $\\beta$'})
        plt.xlabel('Histórico (Lags)')
        plt.title(title)
        if save_path: plt.savefig(save_path, bbox_inches='tight')
        plt.close()