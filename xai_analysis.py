import torch
import pandas as pd
import numpy as np
import os
import json
import ast
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Importações do projeto
from models.encdec_model import EncDecModel 
from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from utils.xai import SolarXAIEngine

# ================= CONFIGURAÇÃO =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lista de experimentos para analisar
EXPERIMENTS_DIRS = [
    # Adicione seus modelos aqui
    "trained_models/2025-12-29_18-50-50_EDLSTM_K_Factor", 
]

OUTPUT_DIR = "analysis_outputs/Analise_XAI_Profunda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= FUNÇÕES AUXILIARES =================

def safe_parse_config(config):
    """Garante que strings de lista no JSON sejam convertidas para listas Python."""
    parsed = config.copy()
    for key in ['hidden_sizes', 'feature_cols']:
        if key in parsed and isinstance(parsed[key], str):
            try: parsed[key] = ast.literal_eval(parsed[key])
            except: pass 
    for key in ['dropout', 'learning_rate', 'dropout_prob']:
        if key in parsed: parsed[key] = float(parsed[key])
    for key in ['input_seq_len', 'output_seq_len', 'input_size', 'epochs']:
        if key in parsed: parsed[key] = int(parsed[key])
    return parsed

def load_experiment_config(exp_dir):
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json não encontrado em {exp_dir}")
    with open(config_path, 'r') as f:
        return safe_parse_config(json.load(f))

def get_preprocessing_params(config):
    """Extrai parâmetros físicos e mapeamentos do config, com fallback."""
    pp_conf = config.get('preprocessing', {})
    
    return {
        'latitude': pp_conf.get('latitude', config.get('latitude', -15.60)),
        'longitude': pp_conf.get('longitude', config.get('longitude', -47.70)),
        'timezone': pp_conf.get('timezone', config.get('timezone', 'Etc/GMT+3')),
        'nominal_power': pp_conf.get('nominal_power', config.get('nominal_power', 156.0)),
        'target_col_csv': pp_conf.get('pv_power_col_csv', pp_conf.get('target_col_csv', 'Pot_BT')),
        'column_mapping': pp_conf.get('column_mapping', config.get('column_map', None)),
        'csv_path': config.get('csv_path', 'data/pv0.csv')
    }

def get_safe_features(matrix, feature_names):
    if matrix is None: return feature_names
    n_cols = matrix.shape[0] if matrix.ndim == 1 else matrix.shape[1]
    if len(feature_names) > n_cols: return feature_names[:n_cols]
    elif len(feature_names) < n_cols: return feature_names + [f"F_{i}" for i in range(len(feature_names), n_cols)]
    return feature_names

# ================= MAIN XAI PIPELINE =================
def main():
    if not EXPERIMENTS_DIRS:
        print("❌ Lista de experimentos vazia.")
        return

    print(f"\n🚀 Iniciando Análise XAI em {len(EXPERIMENTS_DIRS)} modelos...")

    # 1. DETECÇÃO DE AMBIENTE
    print(f"🔍 Detectando parâmetros físicos de: {EXPERIMENTS_DIRS[0]}")
    try:
        base_config = load_experiment_config(EXPERIMENTS_DIRS[0])
        base_params = get_preprocessing_params(base_config)
        
        preprocessor = SolarPreprocessor(
            latitude=base_params['latitude'], 
            longitude=base_params['longitude'], 
            timezone=base_params['timezone'], 
            nominal_power=base_params['nominal_power'], 
            target_col=base_params['target_col_csv'],
            column_mapping=base_params['column_mapping']
        )
    except Exception as e:
        print(f"❌ Erro crítico ao ler config base: {e}")
        return

    # 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS
    csv_path = base_params['csv_path']
    if not os.path.exists(csv_path): csv_path = 'data/pv0.csv'
    
    print(f"⏳ Carregando dados: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    if 'Date_Time' in df_raw.columns:
        df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time'])
        df_raw = df_raw.drop_duplicates(subset=['Date_Time'], keep='first').set_index('Date_Time').sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
    
    SCALER_DIR = "artifacts"
    try:
        preprocessor.load_scalers(SCALER_DIR)
        print("✅ Scalers carregados de artifacts.")
    except:
        print("⚠️ Scalers não encontrados. Fitando localmente.")
        preprocessor.fit(df_raw)
        
    df_processed = preprocessor.transform(df_raw)

    # 3. SPLIT DE TESTE
    last_year = df_processed.index.year.max()
    print(f"📅 Definindo conjunto de XAI: Ano {last_year}")
    df_test = df_processed[df_processed.index.year == last_year].copy()
    
    if df_test.empty:
        print("❌ Erro: Dataset de teste vazio.")
        return

    # 4. LOOP DE ANÁLISE
    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir): continue
        
        model_name = os.path.basename(exp_dir)
        save_folder = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(save_folder, exist_ok=True)
        
        try:
            config = load_experiment_config(exp_dir)
            feature_cols = config.get('feature_cols', [])
            target_col = config.get('target_col', 'target')
            
            print(f"\n🔬 Analisando: {model_name}")
            
            df_xai_input = df_test.copy()
            if target_col not in df_xai_input.columns:
                print(f"   ⚠️ Coluna alvo '{target_col}' não encontrada. Pulando.")
                continue

            dataset = SolarEfficientDataset(
                df=df_xai_input, 
                feature_cols=feature_cols, 
                target_col=target_col,
                n_past=config['input_seq_len'], 
                n_future=config['output_seq_len']
            )
            
            full_loader = DataLoader(dataset, batch_size=32, shuffle=False)

            model = EncDecModel(
                input_size=len(feature_cols), 
                hidden_sizes=config['hidden_sizes'],
                output_seq_len=config['output_seq_len'],
                output_dim=1,
                cell_type=config.get('cell_type', 'lstm'),
                bidirectional=config.get('bidirectional', False),
                use_attention=config.get('use_attention', False),
                use_feature_attention=config.get('use_feature_attention', False)
            ).to(DEVICE)
            
            w_path = os.path.join(exp_dir, 'best_model.pt')
            if not os.path.exists(w_path): w_path = os.path.join(exp_dir, 'best_model.pth')
            model.load_state_dict(torch.load(w_path, map_location=DEVICE), strict=False)
            model.eval()

            xai = SolarXAIEngine(model, DEVICE)
            
            # --- ANÁLISE 1: Importância Global ---
            print("   📊 [1/6] Importância Global...")
            global_imp = xai.compute_global_feature_importance(full_loader)
            safe_feats = get_safe_features(global_imp, feature_cols)
            xai.plot_global_importance(global_imp, safe_feats, save_path=os.path.join(save_folder, '1_Global.png'))
            
            # --- ANÁLISE 2: Perfil Temporal ---
            print("   ⏳ [2/6] Perfil Temporal...")
            t_map = xai.compute_temporal_importance(full_loader, target_step_idx=0)
            safe_feats = get_safe_features(t_map, feature_cols)
            xai.plot_temporal_profile(t_map, safe_feats, save_path=os.path.join(save_folder, '2_Temporal.png'))

            # --- ANÁLISE 3: Física ---
            print("   ☁️ [3/6] Análise Física...")
            indicator = 'k' if 'k' in df_xai_input.columns else 'irr_clearsky_ratio'
            if indicator in df_xai_input.columns:
                k_values = df_xai_input[indicator].values
                dataset_k = k_values[dataset.valid_indices]
                
                idx_clear = np.where(dataset_k > 0.6)[0]
                idx_cloudy = np.where(dataset_k < 0.3)[0]
                
                def create_subloader(indices, max_n=200):
                    if len(indices) == 0: return None
                    if len(indices) > max_n: indices = np.random.choice(indices, max_n, replace=False)
                    return DataLoader(Subset(dataset, indices), batch_size=32)

                ldr_c = create_subloader(idx_clear)
                ldr_n = create_subloader(idx_cloudy)
                
                if ldr_c:
                    map_c = xai.compute_temporal_importance(ldr_c, target_step_idx=0)
                    xai.plot_heatmap(map_c, safe_feats, "Importância (Céu Claro)", os.path.join(save_folder, '3a_ClearSky.png'))
                if ldr_n:
                    map_n = xai.compute_temporal_importance(ldr_n, target_step_idx=0)
                    xai.plot_heatmap(map_n, safe_feats, "Importância (Nublado)", os.path.join(save_folder, '3b_Cloudy.png'))

            # --- ANÁLISE 4: Casos Extremos (Com Proteção de Erro) ---
            print("   🎯 [4/6] Estudos de Caso de Erro...")
            try:
                errors_list = []
                with torch.no_grad():
                    for x_batch, y_batch in full_loader:
                        x_batch = x_batch.to(DEVICE)
                        y_batch = y_batch.to(DEVICE)
                        preds = model(x_batch)
                        
                        # Garante shape 1D para erro
                        p = preds[:, 0, 0].view(-1)
                        y = y_batch[:, 0].view(-1)
                        
                        diff = torch.abs(p - y).cpu().numpy()
                        errors_list.append(diff)
                
                if errors_list:
                    all_err = np.concatenate(errors_list)
                    
                    # CORREÇÃO CRÍTICA: Validação de Tamanho
                    n_dataset = len(dataset)
                    n_err = len(all_err)
                    
                    if n_err != n_dataset:
                        print(f"      ⚠️ Aviso: Mismatch de tamanho (Dataset: {n_dataset}, Erros: {n_err}). Ajustando...")
                        # Corta excesso ou ignora se for menor
                        min_len = min(n_dataset, n_err)
                        all_err = all_err[:min_len]

                    if len(all_err) > 0:
                        best_idx = int(np.argmin(all_err))
                        worst_idx = int(np.argmax(all_err))
                        
                        # Explicações locais
                        x_best = dataset[best_idx][0].unsqueeze(0)
                        x_worst = dataset[worst_idx][0].unsqueeze(0)
                        
                        map_best = xai.get_local_explanation(x_best)
                        xai.plot_heatmap(map_best, safe_feats, f"Melhor Caso (Erro: {all_err[best_idx]:.4f})", os.path.join(save_folder, '4a_Best.png'))
                        
                        map_worst = xai.get_local_explanation(x_worst)
                        xai.plot_heatmap(map_worst, safe_feats, f"Pior Caso (Erro: {all_err[worst_idx]:.4f})", os.path.join(save_folder, '4b_Worst.png'))

                        # 5 & 6: Atenção (Reusa x_worst)
                        if config.get('use_attention', False):
                            print("   🧠 [5/6] Atenção Temporal...")
                            attn = xai.collect_attention_maps(x_worst)
                            if attn is not None:
                                xai.plot_attention_map(attn, "Atenção Temporal (Pior)", os.path.join(save_folder, '5_Temporal_Attn.png'))

                        if config.get('use_feature_attention', False):
                            print("   🧬 [6/6] Atenção de Features...")
                            feat_map = xai.collect_feature_weights(x_worst)
                            if feat_map is not None:
                                sf_att = get_safe_features(feat_map.reshape(1, -1), feature_cols)
                                xai.plot_feature_weights(feat_map, sf_att, "Feature Attention (Pior)", os.path.join(save_folder, '6_Feature_Attn.png'))
            
            except Exception as e:
                print(f"      ❌ Erro na análise de casos extremos: {e}")
                # Não para o script, apenas pula essa etapa

        except Exception as e:
            print(f"❌ Erro geral em {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Análise Completa. Resultados em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()