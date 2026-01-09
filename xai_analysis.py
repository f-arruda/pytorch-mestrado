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
    "trained_models/2026-01-07_18-40-03_Teste_2", 
]

OUTPUT_DIR = "analysis_outputs/Analise_XAI_Profunda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= FUNÇÕES AUXILIARES =================

def safe_parse_config(config):
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
    pp_conf = config.get('preprocessing', {})
    return {
        'latitude': pp_conf.get('latitude', config.get('latitude', -15.60)),
        'longitude': pp_conf.get('longitude', config.get('longitude', -47.70)),
        'altitude':pp_conf.get('altitude', config.get('altitude', 0)),
        'timezone': pp_conf.get('timezone', config.get('timezone', 'Etc/GMT+3')),
        'nominal_power': pp_conf.get('nominal_power', config.get('nominal_power', 156.0)),
        'target_col': config.get('target_col', 'target'),
        'column_mapping': pp_conf.get('column_mapping', config.get('column_map', None)),
        'start_year': pp_conf.get('start_year', config.get('start_year', 2018)),
        'features_to_scale': pp_conf.get('features_to_scale', config.get('features_to_scale', []))
    }

def get_safe_features(matrix, feature_names):
    if matrix is None: return feature_names
    n_cols = matrix.shape[0] if matrix.ndim == 1 else matrix.shape[1]
    if len(feature_names) > n_cols: return feature_names[:n_cols]
    elif len(feature_names) < n_cols: return feature_names + [f"F_{i}" for i in range(len(feature_names), n_cols)]
    return feature_names

# ================= MAIN XAI PIPELINE =================
def main():
    if not EXPERIMENTS_DIRS: return

    print(f"\n🚀 Iniciando Análise XAI Sincronizada...")

    # 1. SETUP DE DADOS (Igual ao main_analysis)
    print(f"🔍 Carregando dados base...")
    try:
        base_config = load_experiment_config(EXPERIMENTS_DIRS[0])
        pp_params = get_preprocessing_params(base_config)
        
        preprocessor = SolarPreprocessor(
            latitude=pp_params['latitude'], longitude=pp_params['longitude'], 
            altitude=pp_params['altitude'], timezone=pp_params['timezone'], 
            nominal_power=pp_params['nominal_power'], start_year=pp_params['start_year'],
            features_to_scale=pp_params['features_to_scale'], target_col=pp_params['target_col'], 
            column_mapping=pp_params['column_mapping']
        )
    except Exception as e:
        print(f"❌ Erro setup: {e}")
        return

    # Carrega CSV
    df_raw = pd.read_csv('data/pv0.csv')
    if 'Date_Time' in df_raw.columns:
        df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time'])
        df_raw = df_raw.drop_duplicates(subset=['Date_Time'], keep='first').set_index('Date_Time').sort_index()
    
    # Fit/Load Scalers
    try: preprocessor.load_scalers(EXPERIMENTS_DIRS[0])
    except: preprocessor.fit(df_raw)
    
    df_processed = preprocessor.transform(df_raw)

    # Filtro de Teste (Igual ao main_analysis)
    test_year = base_config.get('test_year', df_processed.index.year.max())
    print(f"📅 Ano de Análise XAI: {test_year}")
    df_test = df_processed[df_processed.index.year == test_year].copy()
    
    if df_test.empty:
        print("❌ Erro: Dataset vazio.")
        return

    # 4. LOOP DE MODELOS
    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir): continue
        
        model_name = os.path.basename(exp_dir)
        save_folder = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(save_folder, exist_ok=True)
        
        try:
            config = load_experiment_config(exp_dir)
            feature_cols = config.get('feature_cols', [])
            target_col = config.get('target_col', 'target')
            
            print(f"\n🔬 Modelo: {model_name}")

            # Prepara Dataset
            dataset = SolarEfficientDataset(
                df=df_test.copy(), 
                feature_cols=feature_cols, 
                target_col=target_col,
                n_past=config['input_seq_len'], 
                n_future=config['output_seq_len']
            )
            full_loader = DataLoader(dataset, batch_size=32, shuffle=False)

            # --- CRIAÇÃO DO DATAFRAME GABARITO (ALINHAMENTO) ---
            # Remove os primeiros N pontos (input) para alinhar com o output (previsão)
            # O dataset gera len(dataset) amostras. 
            # O df_aligned deve ter esse mesmo tamanho.
            n_past = config['input_seq_len']
            df_aligned = df_test.iloc[n_past:].copy()
            
            # Ajuste fino de tamanho (caso o dataset descarte últimas amostras por batch)
            if len(df_aligned) > len(dataset):
                df_aligned = df_aligned.iloc[:len(dataset)]
            
            print(f"   📏 Alinhamento: Dataset ({len(dataset)}) vs DataFrame ({len(df_aligned)})")
            # ----------------------------------------------------

            # Carrega Modelo
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
            
            # 1. Importância Global
            print("   📊 [1/4] Importância Global...")
            global_imp = xai.compute_global_feature_importance(full_loader)
            safe_feats = get_safe_features(global_imp, feature_cols)
            xai.plot_global_importance(global_imp, safe_feats, save_path=os.path.join(save_folder, '1_Global.png'))
            
            # 2. Perfil Temporal
            print("   ⏳ [2/4] Perfil Temporal...")
            t_map = xai.compute_temporal_importance(full_loader, target_step_idx=0)
            xai.plot_temporal_profile(t_map, safe_feats, save_path=os.path.join(save_folder, '2_Temporal.png'))

            # 3. Análise Física (Baseada no DataFrame Alinhado)
            print("   ☁️ [3/4] Análise Física (Céu Claro vs Nublado)...")
            
            # Identifica indices no DF alinhado
            indicator = 'k' if 'k' in df_aligned.columns else 'irr_clearsky_ratio'
            
            # Pega os ÍNDICES INTEIROS (0, 1, 2...) onde a condição é verdadeira
            # Como df_aligned está sincronizado com dataset, o índice 0 do df é o dataset[0]
            indices_clear = np.where(df_aligned[indicator].values > 0.6)[0]
            indices_cloudy = np.where(df_aligned[indicator].values < 0.3)[0]
            
            def analyze_subset(indices, label, fname):
                if len(indices) == 0: return
                # Subsample se tiver muitos
                if len(indices) > 200: indices = np.random.choice(indices, 200, replace=False)
                
                # Cria DataLoader apenas com esses índices
                sub_loader = DataLoader(Subset(dataset, indices), batch_size=32)
                
                # Roda XAI
                imp_map = xai.compute_temporal_importance(sub_loader, target_step_idx=0)
                xai.plot_heatmap(imp_map, safe_feats, f"Importância ({label})", os.path.join(save_folder, fname))

            analyze_subset(indices_clear, "Céu Claro (K > 0.6)", '3a_ClearSky.png')
            analyze_subset(indices_cloudy, "Nublado (K < 0.3)", '3b_Cloudy.png')

            # 4. Casos de Erro (Com Timestamp no título)
            print("   🎯 [4/4] Casos Extremos...")
            errors_list = []
            with torch.no_grad():
                for x_b, y_b in full_loader:
                    p = model(x_b.to(DEVICE))
                    # Pega apenas horizonte 1
                    err = torch.abs(p[:, 0, 0] - y_b[:, 0, 0].to(DEVICE)).cpu().numpy()
                    errors_list.append(err)
            
            if errors_list:
                all_err = np.concatenate(errors_list)
                
                # Garante tamanho igual
                min_len = min(len(all_err), len(df_aligned))
                all_err = all_err[:min_len]
                
                if len(all_err) > 0:
                    best_idx = int(np.argmin(all_err))
                    worst_idx = int(np.argmax(all_err))
                    
                    # Pega Timestamps para o título
                    ts_best = df_aligned.index[best_idx]
                    ts_worst = df_aligned.index[worst_idx]
                    
                    # Explicações
                    x_best = dataset[best_idx][0].unsqueeze(0)
                    map_best = xai.get_local_explanation(x_best)
                    xai.plot_heatmap(map_best, safe_feats, f"Melhor ({ts_best} | Erro: {all_err[best_idx]:.4f})", os.path.join(save_folder, '4a_Best.png'))
                    
                    x_worst = dataset[worst_idx][0].unsqueeze(0)
                    map_worst = xai.get_local_explanation(x_worst)
                    xai.plot_heatmap(map_worst, safe_feats, f"Pior ({ts_worst} | Erro: {all_err[worst_idx]:.4f})", os.path.join(save_folder, '4b_Worst.png'))

                    # Atenção Temporal (Se houver)
                    if config.get('use_attention', False):
                        attn = xai.collect_attention_maps(x_worst)
                        if attn is not None:
                            xai.plot_attention_map(attn, f"Atenção Temporal - Pior Caso ({ts_worst})", os.path.join(save_folder, '5_Attention.png'))

        except Exception as e:
            print(f"❌ Erro em {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Análise XAI Concluída.")

if __name__ == "__main__":
    main()