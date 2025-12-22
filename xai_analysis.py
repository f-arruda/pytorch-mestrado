import torch
import pandas as pd
import numpy as np
import os
import json
import ast
from torch.utils.data import DataLoader, Subset

from models.encdec_model import EncDecModel 

from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from utils.xai import SolarXAIEngine

# ================= CONFIGURAÇÃO =================
CSV_PATH = 'data/pv0.csv' # Certifique-se que este é o arquivo limpo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPERIMENTS_DIRS = [
    "experiments/2025-12-19_19-09-00_EDLSTM_PAZTUVD"
    # Adicione outros experimentos aqui
]

OUTPUT_DIR = "experiments/Analise_XAI_Profunda"
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
    for key in ['bidirectional', 'use_attention', 'use_feature_attention']:
        if key in parsed:
            val = parsed[key]
            parsed[key] = val.lower() == 'true' if isinstance(val, str) else bool(val)
    return parsed

def load_config_and_model(exp_dir):
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json não encontrado em {exp_dir}")

    with open(config_path, 'r') as f:
        raw_config = json.load(f)
    
    config = safe_parse_config(raw_config)
    
    feature_cols = config.get('feature_cols', [])
    
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
    
    weights_path = os.path.join(exp_dir, 'best_model.pth') 
    if not os.path.exists(weights_path):
        weights_path = os.path.join(exp_dir, 'best_model.pt') 
        
    # Carrega pesos com flexibilidade para evitar erros bobos de dimensão
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    except RuntimeError as e:
        print(f"⚠️ Aviso de shape nos pesos: {e}")
        print("Tentando carregar com strict=False...")
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)

    model.eval()
    return model, config

def load_data(csv_path, feature_cols, n_past, n_future):
    print("⏳ Carregando e limpando dados...")
    df = pd.read_csv(csv_path)
    
    # --- LIMPEZA AGRESSIVA DE DADOS (Garante índice único) ---
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df = df.groupby('Date_Time').first().reset_index()
        df = df.set_index('Date_Time').sort_index()
    
    df = df[~df.index.duplicated(keep='first')]
    # ----------------------------------
    
    # Se quiser filtrar por ano, descomente abaixo:
    # if 'Year' not in df.columns: df['Year'] = df.index.year
    # df = df[df['Year'] == 2022].copy()
    
    df_test = df.copy() 
    
    preprocessor = SolarPreprocessor(
        latitude=-15.60, 
        longitude=-47.70,
        timezone='Etc/GMT+3',
        nominal_power=156.0,
        target_col='Pot_BT',
        start_year=2018,
        column_mapping={
            'Pot_BT': 'target',
            'Irradiação Global horária(horizontal) kWh/m2': 'ghi',
            'Irradiação Difusa horária kWh/m2': 'dhi',
            'Irradiação Global horária(Inclinada 27°) kWh/m2': 'irrad_poa',
            'Temperatura ambiente °C': 'temp_amb',
            'Umidade Relativa %': 'humidity',
            'Velocidade média do vento m/s': 'wind_speed'
        }
    )
    
    # Carrega scalers da pasta raiz (onde você disse que eles estão)
    SCALER_DIR = "." 
    try:
        print(f"♻️ Carregando scalers de: {os.path.abspath(SCALER_DIR)}")
        preprocessor.load_scalers(SCALER_DIR)
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: Não encontrei scalers em {SCALER_DIR}.")
        raise RuntimeError("Impossível rodar XAI sem os scalers originais (scaler_X.pkl, scaler_Y.pkl).")
        
    df_proc = preprocessor.transform(df_test)
    
    valid_cols = [c for c in feature_cols if c in df_proc.columns]
    
    dataset = SolarEfficientDataset(
        df_proc, 
        input_tag=valid_cols, # Usa o nome correto do argumento
        n_past=n_past, 
        n_future=n_future,
    )
    return dataset, df_proc

def get_safe_features(matrix, feature_names):
    """
    Função Auxiliar CRÍTICA:
    Garante que a lista de nomes tenha o mesmo tamanho das colunas da matriz XAI.
    Evita o erro 'IndexError: index 7 is out of bounds'.
    """
    if matrix is None: return feature_names
    
    # Se a matriz for 1D (ex: importância global), pega shape[0], se 2D (temporal) pega shape[1]
    n_cols = matrix.shape[0] if matrix.ndim == 1 else matrix.shape[1]
    
    if len(feature_names) > n_cols:
        # Corta nomes extras (geralmente o target que sobrou na lista)
        return feature_names[:n_cols]
    elif len(feature_names) < n_cols:
        # Adiciona nomes genéricos se faltar
        return feature_names + [f"Feat_{i}" for i in range(len(feature_names), n_cols)]
    return feature_names

# ================= MAIN XAI PIPELINE =================
def main():
    print(f"\n🚀 Iniciando Análise XAI em {len(EXPERIMENTS_DIRS)} modelos...")
    
    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir): continue

        model_name = os.path.basename(exp_dir)
        print(f"\n🔬 Analisando: {model_name}")
        
        save_folder = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(save_folder, exist_ok=True)
        
        try:
            model, config = load_config_and_model(exp_dir)
            feature_cols = config.get('feature_cols')
            
            # Carrega dados
            dataset, df_proc = load_data(
                CSV_PATH, 
                feature_cols, 
                config['input_seq_len'], 
                config['output_seq_len']
            )
            
            full_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            xai = SolarXAIEngine(model, DEVICE)
            
            # ANÁLISE 1: Global
            print("   📊 [1/6] Análise Global de Variáveis...")
            global_imp = xai.compute_global_feature_importance(full_loader)
            
            # USO DA FUNÇÃO SEGURA
            safe_feats = get_safe_features(global_imp, dataset.data_input)
            xai.plot_global_importance(global_imp, safe_feats, save_path=os.path.join(save_folder, '1_Global_Importance.png'))
            
            # ANÁLISE 2: Temporal
            print("   ⏳ [2/6] Análise Temporal (Memória do Modelo)...")
            temporal_map = xai.compute_temporal_importance(full_loader, target_step_idx=0, max_samples=500)
            
            # USO DA FUNÇÃO SEGURA
            safe_feats = get_safe_features(temporal_map, dataset.data_input)
            xai.plot_temporal_profile(temporal_map, safe_feats, save_path=os.path.join(save_folder, '2_Temporal_Profile.png'))

            # ANÁLISE 3: Física
            print("   ☁️ [3/6] Análise por Condição de Céu...")
            if 'k' in df_proc.columns:
                k_values = df_proc['k'].values
                indices_clear = np.where(k_values > 0.6)[0]
                indices_cloudy = np.where(k_values < 0.3)[0]
                valid_indices_arr = np.array(dataset.valid_indices)
                
                def get_subset_loader(indices_condition, max_s=200):
                    common_indices = np.intersect1d(indices_condition, valid_indices_arr)
                    if len(common_indices) == 0: return None
                    if len(common_indices) > max_s: 
                        common_indices = np.random.choice(common_indices, max_s, replace=False)
                    dataset_indices = []
                    for idx in common_indices:
                        pos = np.where(valid_indices_arr == idx)[0]
                        if len(pos) > 0: dataset_indices.append(pos[0])
                    if not dataset_indices: return None
                    return DataLoader(Subset(dataset, dataset_indices), batch_size=32)

                loader_clear = get_subset_loader(indices_clear)
                loader_cloudy = get_subset_loader(indices_cloudy)
                
                if loader_clear:
                    map_clear = xai.compute_temporal_importance(loader_clear, target_step_idx=0)
                    safe_feats = get_safe_features(map_clear, dataset.data_input)
                    xai.plot_heatmap(map_clear, safe_feats, "Importância Média - Céu Claro", save_path=os.path.join(save_folder, '3a_Heatmap_Clear.png'))
                
                if loader_cloudy:
                    map_cloudy = xai.compute_temporal_importance(loader_cloudy, target_step_idx=0)
                    safe_feats = get_safe_features(map_cloudy, dataset.data_input)
                    xai.plot_heatmap(map_cloudy, safe_feats, "Importância Média - Céu Nublado", save_path=os.path.join(save_folder, '3b_Heatmap_Cloudy.png'))
            else:
                print("      ⚠️ Variável 'k' não encontrada. Pulando análise física.")

            # ANÁLISE 4: Casos Extremos
            print("   🎯 [4/6] Estudo de Caso (Melhor vs Pior Erro)...")
            errors_list = []
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(full_loader):
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    preds = model(x_batch)
                    
                    p = preds[:, 0, 0].view(-1)
                    y = y_batch[:, 0].view(-1)
                    
                    batch_err = torch.abs(p - y).cpu().numpy()
                    errors_list.append(batch_err)
            
            if errors_list:
                errors = np.concatenate(errors_list)
                if len(errors) > 0:
                    idx_best_rel = np.argmin(errors)
                    idx_worst_rel = np.argmax(errors)
                    
                    x_best = dataset[idx_best_rel][0].unsqueeze(0)
                    
                    # Explicação Local
                    map_best = xai.get_local_explanation(x_best)
                    safe_feats = get_safe_features(map_best, dataset.data_input)
                    xai.plot_heatmap(map_best, safe_feats, f"Melhor Caso (Erro: {errors[idx_best_rel]:.4f})", save_path=os.path.join(save_folder, '4a_Case_Best.png'))

                    x_worst = dataset[idx_worst_rel][0].unsqueeze(0)
                    map_worst = xai.get_local_explanation(x_worst)
                    safe_feats = get_safe_features(map_worst, dataset.data_input)
                    xai.plot_heatmap(map_worst, safe_feats, f"Pior Caso (Erro: {errors[idx_worst_rel]:.4f})", save_path=os.path.join(save_folder, '4b_Case_Worst.png'))
                    
                    # ANÁLISE 5 e 6
                    if config.get('use_attention', False):
                        print("   🧠 [5/6] Atenção Temporal...")
                        attn_map = xai.collect_attention_maps(x_worst)
                        if attn_map is not None:
                            xai.plot_attention_map(attn_map, title="Temporal Attention - Pior Caso", save_path=os.path.join(save_folder, '5_Temporal_Attn_Worst.png'))

                    if config.get('use_feature_attention', False):
                        print("   🧬 [6/6] Atenção de Variáveis...")
                        feat_map = xai.collect_feature_weights(x_worst)
                        if feat_map is not None:
                            # Feature map geralmente vem como [Features, Tempo]
                            safe_feats = get_safe_features(feat_map.T, dataset.data_input) # Transpõe para checar dimensão feature
                            xai.plot_feature_weights(feat_map, feature_names=safe_feats, title="Feature Attention - Pior Caso", save_path=os.path.join(save_folder, '6_Feature_Attn_Worst.png'))

        except Exception as e:
            print(f"❌ Erro ao processar {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ Análise XAI Completa salva em: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()