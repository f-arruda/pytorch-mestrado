import torch
import pandas as pd
import numpy as np
import joblib
import os
import json
import ast
from torch.utils.data import DataLoader

from models.encdec_model import EncDecModel

from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from src.statistical_metrics import SolarStatisticalAnalyzer

# ==========================================
# CONFIGURAÇÃO
# ==========================================
CSV_PATH = 'data/pv0.csv'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Liste aqui os experimentos que deseja comparar
EXPERIMENTS_DIRS = [
    "experiments/2025-12-19_19-09-00_EDLSTM_PAZTUVD",
    # Adicione outros aqui...
]

OUTPUT_COMPARISON_DIR = "experiments/Analise_Comparativa_Final"
DEFAULT_FEATURE_COLS = ['temp_amb', 'wind_speed', 'humidity', 'target', 'cos_zenith', 'sin_azimuth']

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
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

def load_experiment_config(exp_dir):
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ config.json não encontrado em {exp_dir}")
    with open(config_path, 'r') as f:
        return safe_parse_config(json.load(f))

def load_test_data(csv_path, preprocessor):
    print(f"⏳ Carregando dados de teste: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    
    # --- BLOCO DE LIMPEZA OBRIGATÓRIO ---
    if 'Date_Time' in df_raw.columns:
        df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time'])
        # Remove duplicatas antes de virar índice
        df_raw = df_raw.drop_duplicates(subset=['Date_Time'], keep='first')
        df_raw = df_raw.set_index('Date_Time').sort_index()
    
    # Trava final de segurança
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]
    # ------------------------------------

    # O preprocessor já deve estar carregado (fitted) ao entrar aqui
    df_processed = preprocessor.transform(df_raw)
    
    return df_raw, df_processed
    
def reconstruct_dataframe(y_pred, y_true, dataset, df_raw, df_processed, output_seq_len, scaler_y):
    """
    Reconstrói DataFrame e converte TUDO para kW.
    """
    valid_indices = dataset.valid_indices
    records = []
    
    for i, idx in enumerate(valid_indices):
        pred_seq = y_pred[i]
        true_seq = y_true[i]

        # --- CORREÇÃO DO ERRO DIM 3 ---
        # Garante que as sequências são 1D (vetores chatos) e não (24, 1)
        if pred_seq.ndim > 1: pred_seq = pred_seq.flatten()
        if true_seq.ndim > 1: true_seq = true_seq.flatten()
        # ------------------------------
        
        # Metadata futuro
        future_metadata = df_raw.iloc[idx : idx + output_seq_len]
        future_processed = df_processed.iloc[idx : idx + output_seq_len]
        
        for h in range(output_seq_len):
            if h < len(future_metadata):
                timestamp = future_metadata.index[h]
                row_raw = future_metadata.iloc[h]
                row_proc = future_processed.iloc[h]
                
                # Pega a Persistência (Normalizada)
                p_col = f"P{h+1}"
                val_persistencia_norm = row_proc[p_col] if p_col in row_proc else 0.0
                
                # --- DESNORMALIZAÇÃO ---
                # Agora true_seq[h] é um escalar, então [[scalar]] cria uma matriz (1, 1) correta (dim=2)
                obs_rel = scaler_y.inverse_transform([[true_seq[h]]])[0][0]
                pred_rel = scaler_y.inverse_transform([[pred_seq[h]]])[0][0]
                pers_rel = scaler_y.inverse_transform([[val_persistencia_norm]])[0][0]
                
                # 2. Multiplica pela Potência Nominal (Escala Relativa -> kW)
                PNOM = 156.0 
                
                obs_kw = obs_rel * PNOM
                pred_kw = pred_rel * PNOM
                pers_kw = pers_rel * PNOM

                records.append({
                    'Timestamp': timestamp,
                    'Horizonte': h + 1,
                    'Observado': obs_kw,
                    'Previsto': pred_kw,
                    'Persistencia': pers_kw, 
                    'Hour': timestamp.hour,
                    'zenith': row_raw.get('zenith', 0),
                    'Condição de céu': row_raw.get('Condição de céu', 'Desconhecido')
                })
    return pd.DataFrame(records)

# ==========================================
# MAIN
# ==========================================
def main():
    # 1. Instancia o Preprocessor
    preprocessor = SolarPreprocessor(
        latitude=-15.60,
        longitude=-47.70,
        timezone='Etc/GMT+3',
        nominal_power=156.0,
        target_col='Pot_BT',
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

    # 2. Carrega os Scalers (Essencial para não dar erro de Fit)
    # Tenta carregar do diretório atual
    SCALER_DIR = "." 
    
    try:
        print(f"♻️ Carregando scalers de: {os.path.abspath(SCALER_DIR)}")
        preprocessor.load_scalers(SCALER_DIR)
    except FileNotFoundError:
        print(f"❌ ERRO: Não encontrei 'scaler_X.pkl' e 'scaler_Y.pkl' na pasta raiz: {os.path.abspath(SCALER_DIR)}")
        print("Certifique-se que os arquivos .pkl gerados no treino estão na mesma pasta do script.")
        return

    # 3. Carrega Dados
    df_raw, df_processed = load_test_data(CSV_PATH, preprocessor)
    
    all_results = []
    print(f"\n📂 Avaliando {len(EXPERIMENTS_DIRS)} experimentos...\n")

    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir): continue
            
        try:
            config = load_experiment_config(exp_dir)
            model_name = config.get('model_type', os.path.basename(exp_dir))
            feature_cols = config.get('feature_cols', DEFAULT_FEATURE_COLS)
            
            # Filtra colunas que realmente existem no df_processed
            valid_cols = [c for c in feature_cols if c in df_processed.columns]
            if len(valid_cols) != len(feature_cols):
                print(f"⚠️ Aviso: Ajustando feature_cols. Faltando: {set(feature_cols) - set(valid_cols)}")
            
            print(f"🔹 {model_name}")

            # Dataset
            test_dataset = SolarEfficientDataset(
                df_processed, 
                input_tag=config['feature_cols'],  # MUDANÇA: Usando 'feature_cols' que é o nome correto no init do Dataset
                n_past=config['input_seq_len'], 
                n_future=config['output_seq_len'],
            )
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            # Modelo
            model = EncDecModel(
                input_size=len(valid_cols),
                hidden_sizes=config['hidden_sizes'],
                output_seq_len=config['output_seq_len'],
                output_dim=1,
                cell_type=config.get('cell_type', 'lstm'),
                bidirectional=config.get('bidirectional', False),
                use_attention=config.get('use_attention', False),
                dropout_prob=config.get('dropout', 0.0) ,
                use_feature_attention=config.get('use_feature_attention', False)
            ).to(DEVICE)
            
            # Carrega Pesos
            weights_path = os.path.join(exp_dir, 'best_model.pth')
            if not os.path.exists(weights_path):
                 weights_path = os.path.join(exp_dir, 'best_model.pt') # Tenta extensão alternativa
            
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            
            # Inferência
            preds, targets = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    out = model(x.to(DEVICE))
                    preds.append(out.cpu().numpy())
                    targets.append(y.numpy())
            
            if not preds: continue

            y_pred = np.concatenate(preds, axis=0)
            y_true = np.concatenate(targets, axis=0)
            
            # Reconstrução (Passando scaler_y do preprocessor)
            df_model = reconstruct_dataframe(
                y_pred, y_true, test_dataset, df_raw, df_processed, 
                config['output_seq_len'], preprocessor.scaler_y
            )
            
            if not df_model.empty:
                df_model['Modelo'] = model_name
                all_results.append(df_model)
                print("   ✅ OK")

        except Exception as e:
            print(f"   ❌ Erro ao processar {exp_dir}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("Nenhum resultado gerado.")
        return

    print("\n📊 Consolidando dados...")
    df_final = pd.concat(all_results, ignore_index=True)
    
    analyzer = SolarStatisticalAnalyzer(df_final, output_dir=OUTPUT_COMPARISON_DIR)
    
    analyzer.save_global_metrics()      
    analyzer.plot_metrics_by_horizon()  
    analyzer.plot_boxplots_hourly()     
    analyzer.plot_taylor_diagram()      
    analyzer.plot_scatter_hist()        
    
    print(f"\n🏁 Análise completa salva em: {OUTPUT_COMPARISON_DIR}")

if __name__ == "__main__":
    main()