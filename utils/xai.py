import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, FeatureAblation

class SolarXAIEngine:
    def __init__(self, model, device):
        """
        Engine de XAI escalável para modelos de Séries Temporais.
        Args:
            model: Qualquer modelo PyTorch que siga input (B, T, F) -> output (B, T_out, 1).
            device: 'cpu' ou 'cuda'.
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def _model_wrapper(self, inputs, target_step_idx):
        """
        Função interna que transforma o output de sequência em escalar.
        O Captum precisa de um único valor para calcular o gradiente.
        
        Args:
            inputs: Tensor de entrada.
            target_step_idx: Qual passo do futuro queremos explicar (0 a N_futuro).
        """
        # O modelo retorna [Batch, Seq_Len_Out, 1]
        output = self.model(inputs)
        
        # Retorna apenas o valor previsto no passo específico desejado
        # Ex: Queremos explicar apenas a previsão da 1ª hora (idx=0) ou da 12ª (idx=11)
        return output[:, target_step_idx, 0]

    def get_local_saliency(self, input_tensor, target_step_idx=0):
        """
        Usa Integrated Gradients para gerar um mapa de calor (Tempo x Features).
        Responde: "Para prever a hora X, quais pontos do passado foram vitais?"
        """
        input_tensor = input_tensor.to(self.device).requires_grad_()
        
        # Instancia o IG com a função wrapper focada no passo desejado
        ig = IntegratedGradients(lambda x: self._model_wrapper(x, target_step_idx))
        
        # Calcula atribuições (comparando com baseline de zeros)
        attributions, delta = ig.attribute(
            input_tensor,
            baselines=torch.zeros_like(input_tensor),
            n_steps=50,
            return_convergence_delta=True
        )
        
        # Retorna array numpy [Seq_Len_In, Features]
        # (Removemos a dimensão de batch assumindo batch=1 para análise)
        return attributions.squeeze(0).cpu().detach().numpy()

    def get_feature_importance(self, input_tensor, feature_names):
        """
        Usa Feature Ablation para calcular a importância GLOBAL de cada variável.
        Responde: "Se eu remover a Temperatura inteira da série, quanto o erro muda?"
        """
        input_tensor = input_tensor.to(self.device)
        
        # --- MÁGICA DA MÁSCARA ---
        # Feature Ablation por padrão remove ponto a ponto.
        # Nós queremos remover a feature INTEIRA (todas as horas de uma vez).
        # Precisamos criar uma feature_mask onde cada feature tem um ID único.
        
        # input shape: [1, Seq_Len, N_Features]
        seq_len = input_tensor.shape[1]
        n_features = input_tensor.shape[2]
        
        # Cria mascara: [1, Seq_Len, N_Features]
        # Ex: Feature 0 tem valor 0 em todo o tempo. Feature 1 tem valor 1...
        feature_mask = torch.arange(n_features).unsqueeze(0).repeat(seq_len, 1).unsqueeze(0).to(self.device)
        
        # Ablation para explicar a SOMA de todas as previsões futuras (Impacto Geral)
        # Wrapper soma todo o output [Batch, Output_Len, 1] -> Escalar
        def agg_wrapper(inputs):
            return torch.sum(self.model(inputs), dim=1)

        ablator = FeatureAblation(agg_wrapper)
        
        # Calcula impacto
        attributions = ablator.attribute(
            input_tensor,
            feature_mask=feature_mask # Agrupa por feature
        )
        
        # O resultado vem no shape do input. Como usamos máscara por feature,
        # o valor é igual para todos os tempos daquela feature. Pegamos a média/primeira linha.
        # Shape final: [N_Features]
        importance = attributions[0, 0, :].cpu().detach().numpy()
        
        # Normaliza para porcentagem
        importance = np.abs(importance)
        importance = importance / importance.sum()
        
        return importance

    # --- Métodos de Plotagem Profissional ---
    
    def plot_saliency_map(self, saliency_matrix, feature_names, target_hour):
        """Plota o mapa de calor do Integrated Gradients"""
        plt.figure(figsize=(10, 6))
        # Transpõe para ficar Features (Y) x Tempo (X)
        plt.imshow(saliency_matrix.T, aspect='auto', cmap='inferno', origin='lower')
        plt.colorbar(label='Importância (Gradiente Integrado)')
        plt.yticks(ticks=range(len(feature_names)), labels=feature_names)
        plt.xlabel('Horas do Passado (Janela de Entrada)')
        plt.title(f'Mapa de Saliência: O que influenciou a previsão da {target_hour}ª hora futura?')
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, importance_scores, feature_names):
        """Plota o gráfico de barras do Feature Ablation"""
        indices = np.argsort(importance_scores)
        
        plt.figure(figsize=(8, 5))
        plt.title('Importância Global das Features (Feature Ablation)')
        plt.barh(range(len(indices)), importance_scores[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importância Relativa (%)')
        plt.tight_layout()
        plt.show()