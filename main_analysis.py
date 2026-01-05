import torch
import pandas as pd
import numpy as np
import os
import json
import ast
import joblib
from torch.utils.data import DataLoader

# Importações do projeto
from models.encdec_model import EncDecModel
from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from src.statistical_metrics import SolarStatisticalAnalyzer
from src.postprocessing import get_strategy # <--- Novo arquivo

# ================= CONFIGURAÇÃO =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_COMPARISON_DIR = "analysis_outputs/Analise_Comparativa_Final"

# Lista de experimentos para comparar
EXPERIMENTS_DIRS = [
    "trained_models/2025-12-29_18-50-50_EDLSTM_K_Factor",
    # Adicione outros caminhos aqui...
]

# ================= FUNÇÕES AUXILIARES =================

def safe_parse_config(config):
    """Garante tipos corretos ao ler do JSON."""
    parsed = config.copy()
    # Converte strings de lista de volta para listas
    for key in ['hidden_sizes', 'feature_cols']:
        if key in parsed and isinstance(parsed[key], str):
            try: parsed[key] = ast.literal_eval(parsed[key])
            except: pass
    return parsed

def load_experiment_config(exp_dir):
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json não encontrado em {exp_dir}")
    with open(config_path, 'r') as f:
        return safe_parse_config(json.load(f))

def get_preprocessing_params(exp_dir):
    """
    Tenta extrair parâmetros físicos do config.json.
    Se não encontrar (modelos antigos), usa valores padrão (Fallback).
    """
    config = load_experiment_config(exp_dir)
    
    # Procura um bloco 'preprocessing' (novo padrão)
    pp_conf = config.get('preprocessing', {})
    
    # Se não tiver bloco, tenta pegar da raiz ou usa defaults
    return {
        'latitude': pp_conf.get('latitude', config.get('latitude', -15.60)),
        'longitude': pp_conf.get('longitude', config.get('longitude', -47.70)),
        'timezone': pp_conf.get('timezone', config.get('timezone', 'Etc/GMT+3')),
        'nominal_power': pp_conf.get('nominal_power', config.get('nominal_power', 156.0)),
        'target_col_csv': pp_conf.get('pv_power_col_csv', config.get('target_col_csv', 'Pot_BT')),
        'column_mapping': pp_conf.get('column_mapping', config.get('column_map', None))
    }

def reconstruct_dataframe(y_pred, y_true, dataset, df_raw, df_processed, output_seq_len, scaler_y, config, nominal_power):
    """Reconstrói dados usando Strategy Pattern."""
    valid_indices = dataset.valid_indices
    records = []
    
    # Determina a estratégia (direct, clearsky_ratio, etc)
    mode = config.get('prediction_mode', config.get('model_type', 'direct'))
    # Pequeno ajuste para compatibilidade com nomes antigos
    if 'sem_target' in mode or 'k_factor' in mode: mode = 'clearsky_ratio'
    if 'direct' not in mode and 'clearsky' not in mode: mode = 'direct'
        
    strategy = get_strategy(mode)
    
    if not df_raw.index.is_unique:
        df_raw = df_raw[~df_raw.index.duplicated(keep='first')]

    for i, idx in enumerate(valid_indices):
        pred_seq = y_pred[i].flatten() if y_pred[i].ndim > 1 else y_pred[i]
        true_seq = y_true[i].flatten() if y_true[i].ndim > 1 else y_true[i]
        
        # Metadados do futuro (pot_cs, etc)
        future_processed = df_processed.iloc[idx : idx + output_seq_len]
        
        for h in range(output_seq_len):
            if h < len(future_processed):
                timestamp = future_processed.index[h]
                row_proc = future_processed.iloc[h]
                
                # --- DELEGAÇÃO PARA O POSTPROCESSING ---
                pred_kw, obs_kw = strategy.reconstruct(
                    pred_val=pred_seq[h],
                    true_val=true_seq[h],
                    metadata_row=row_proc,
                    scaler_y=scaler_y,
                    nominal_power=nominal_power
                )
                
                # Persistência (P1 vem do preprocessor)
                # P1 geralmente está normalizado (0-1), então multiplica por P_nom
                pers_val = row_proc.get('P1', 0.0)
                pers_kw = pers_val * nominal_power

                records.append({
                    'Timestamp': timestamp,
                    'Horizonte': h + 1,
                    'Observado': obs_kw,
                    'Previsto': pred_kw,
                    'Persistencia': pers_kw, 
                    'Hour': timestamp.hour,
                    'Modelo': config.get('model_type', 'Unknown'),
                    'Modo': mode
                })
    return pd.DataFrame(records)

# ================= MAIN =================
def main():
    if not EXPERIMENTS_DIRS:
        print("❌ Lista de experimentos vazia.")
        return

    # 1. SETUP DO PREPROCESSOR (Baseado no primeiro modelo)
    # Assume-se que todos os modelos comparados usam a mesma usina/local.
    print(f"🔍 Lendo configurações base de: {EXPERIMENTS_DIRS[0]}")
    try:
        base_params = get_preprocessing_params(EXPERIMENTS_DIRS[0])
        csv_path = 'data/pv0.csv' # Caminho fixo ou ler do config se disponível
        
        print(f"🌍 Parâmetros detectados: Lat={base_params['latitude']}, Lon={base_params['longitude']}, P_Nom={base_params['nominal_power']}")
        
        preprocessor = SolarPreprocessor(
            latitude=base_params['latitude'], 
            longitude=base_params['longitude'], 
            timezone=base_params['timezone'], 
            nominal_power=base_params['nominal_power'], 
            target_col=base_params['target_col_csv'],
            column_mapping=base_params['column_mapping']
        )
    except Exception as e:
        print(f"⚠️ Erro ao configurar preprocessor automático: {e}")
        return

    # 2. CARREGAMENTO DE DADOS
    print(f"⏳ Carregando dados: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    if 'Date_Time' in df_raw.columns:
        df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time'])
        df_raw = df_raw.drop_duplicates(subset=['Date_Time'], keep='first').set_index('Date_Time').sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]

    # Tenta carregar scalers globais (artifacts) ou fita localmente
    SCALER_DIR = "artifacts"
    try:
        preprocessor.load_scalers(SCALER_DIR)
        print("✅ Scalers carregados de artifacts.")
    except:
        print("⚠️ Scalers não encontrados. Ajustando preprocessor localmente (Fit).")
        preprocessor.fit(df_raw)
        
    df_processed = preprocessor.transform(df_raw)
    
    # 3. LOOP DE AVALIAÇÃO
    all_results = []
    print(f"\n🚀 Avaliando {len(EXPERIMENTS_DIRS)} experimentos...")

    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir): continue
            
        try:
            config = load_experiment_config(exp_dir)
            model_name = config.get('model_type', os.path.basename(exp_dir))
            
            # Recupera features usadas
            feature_cols = config.get('feature_cols', [])
            target_col = config.get('target_col', 'target')
            
            print(f"🔹 {model_name} | Target: {target_col}")

            # Adapta dataset de teste
            # Se o modelo pede 'k' e o dataset só tem 'target', o preprocessor já deve ter gerado 'k'
            # Só precisamos garantir que as colunas existam
            cols_needed = feature_cols + ([target_col] if target_col != 'target' else [])
            missing = [c for c in cols_needed if c not in df_processed.columns]
            if missing:
                print(f"   ❌ Colunas faltando: {missing}. Pulando.")
                continue

            test_dataset = SolarEfficientDataset(
                df=df_processed,
                # Usa 'input_tag' ou 'feature_cols' dependendo da versão do seu dataset_module
                feature_cols=feature_cols, 
                target_col=config['target_col'],
                n_past=config['input_seq_len'],
                n_future=config['output_seq_len']
            )
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            # Modelo
            model = EncDecModel(
                input_size=len(feature_cols),
                hidden_sizes=config['hidden_sizes'],
                output_seq_len=config['output_seq_len'],
                output_dim=1,
                cell_type=config.get('cell_type', 'lstm'),
                bidirectional=config.get('bidirectional', False),
                use_attention=config.get('use_attention', False),
                use_feature_attention=config.get('use_feature_attention', False),
                dropout_prob=0.0
            ).to(DEVICE)
            
            # Pesos
            w_path = os.path.join(exp_dir, 'best_model.pt')
            if not os.path.exists(w_path): w_path = os.path.join(exp_dir, 'best_model.pth')
            model.load_state_dict(torch.load(w_path, map_location=DEVICE), strict=False)
            model.eval()
            
            # Inferência
            preds, targets = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    out = model(x.to(DEVICE))
                    preds.append(out.cpu().numpy())
                    targets.append(y.numpy())
            
            if not preds: continue
            
            # Tenta carregar scaler_y específico do modelo, se existir (para compatibilidade antiga)
            # Se não existir, passamos None e a estratégia Direct assume apenas multiplicação
            model_scaler_y = None
            path_sy = os.path.join(exp_dir, 'scaler_Y.pkl')
            if os.path.exists(path_sy):
                model_scaler_y = joblib.load(path_sy)

            # Reconstrução
            df_model = reconstruct_dataframe(
                np.concatenate(preds, axis=0), 
                np.concatenate(targets, axis=0), 
                test_dataset, df_raw, df_processed, 
                config['output_seq_len'], 
                model_scaler_y, # Passa o scaler específico se existir
                config,
                base_params['nominal_power']
            )
            
            if not df_model.empty:
                df_model['Modelo'] = model_name
                all_results.append(df_model)
                print("   ✅ OK")

        except Exception as e:
            print(f"   ❌ Erro em {exp_dir}: {e}")
            import traceback
            traceback.print_exc()

    # 4. GRÁFICOS
    if all_results:
        print("\n📊 Gerando relatórios...")
        df_final = pd.concat(all_results, ignore_index=True)
        analyzer = SolarStatisticalAnalyzer(df_final, output_dir=OUTPUT_COMPARISON_DIR)
        
        analyzer.save_global_metrics()      
        analyzer.plot_boxplots_hourly()     
        analyzer.plot_error_by_hour_of_day() 
        analyzer.plot_scenario_days()       
        analyzer.plot_taylor_diagram()      
        analyzer.plot_scatter_hist()        
        print(f"🏁 Resultados salvos em: {OUTPUT_COMPARISON_DIR}")
    else:
        print("Nenhum resultado gerado.")

if __name__ == "__main__":
    main()