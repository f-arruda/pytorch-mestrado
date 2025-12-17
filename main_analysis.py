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
CSV_PATH = 'data/base_teste.csv'
SCALER_PATH = 'scaler_Y.pkl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXPERIMENTS_DIRS = [
    "experiments/2025-12-17_15-04-58_GRU_Bi_Attention",
    "experiments/2025-12-17_15-54-04_LSTM_Bi_Attention",
    "experiments/2025-12-17_16-19-56_LSTM_Bi_Attention_PAZ"
]
OUTPUT_COMPARISON_DIR = "experiments/Analise_Comparativa_Final"
DEFAULT_FEATURE_COLS = ['Velocidade média do vento m/s', 'Temperatura ambiente °C', 'Umidade Relativa %', 'Pot_BT', 'cos_zenith', 'sin_azimuth']

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
    for key in ['bidirectional', 'use_attention']:
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

def load_test_data(csv_path):
    print("⏳ Carregando dados de teste...")
    df = pd.read_csv(csv_path, index_col=0)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    if 'Year' not in df.columns: df['Year'] = df.index.year
    df_test_raw = df[df['Year'] == 2022].copy()
    
    preprocessor = SolarPreprocessor(nominal_power=156.0)
    try:
        preprocessor.load_scalers(input_dir='.')
    except FileNotFoundError:
        print("⚠️ Scalers não encontrados na raiz.")
        raise
    df_test_processed = preprocessor.transform(df_test_raw)
    return df_test_raw, df_test_processed

def reconstruct_dataframe(y_pred, y_true, dataset, df_raw, df_processed, output_seq_len):
    """
    Reconstrói DataFrame e converte TUDO para kW para o Skill Score funcionar.
    """
    valid_indices = dataset.valid_indices
    records = []
    
    for i, idx in enumerate(valid_indices):
        pred_seq = y_pred[i]
        true_seq = y_true[i]
        
        future_metadata = df_raw.iloc[idx : idx + output_seq_len]
        future_processed = df_processed.iloc[idx : idx + output_seq_len]
        
        for h in range(output_seq_len):
            if h < len(future_metadata):
                timestamp = future_metadata.index[h]
                row_raw = future_metadata.iloc[h]
                row_proc = future_processed.iloc[h]
                
                # Pega a Persistência P1 (ou P2, P3 se o horizonte fosse maior)
                p_col = f"P{h+1}"
                val_persistencia = row_proc[p_col] if p_col in row_proc else np.nan
                
                # --- CORREÇÃO DE UNIDADES (Tudo para kW) ---
                # 1. scaler_y inverte para a escala 0-1 (Potencia Normalizada)
                # 2. Multiplicamos por 156.0 para voltar para kW
                
                obs_kw = true_seq[h] * 156.0
                pred_kw = pred_seq[h] * 156.0
                pers_kw = val_persistencia * 156.0

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
    df_raw, df_processed = load_test_data(CSV_PATH)
    scaler_y = joblib.load(SCALER_PATH)
    
    all_results = []
    print(f"\n📂 Avaliando {len(EXPERIMENTS_DIRS)} experimentos...\n")

    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir): continue
            
        try:
            config = load_experiment_config(exp_dir)
            model_name = config.get('model_type', os.path.basename(exp_dir))
            feature_cols = config.get('feature_cols', DEFAULT_FEATURE_COLS)
            
            print(f"🔹 {model_name}")

            # Dataset
            test_dataset = SolarEfficientDataset(
                df_processed, 
                input_tag=feature_cols, 
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
                dropout_prob=config.get('dropout', 0.0) 
            ).to(DEVICE)
            
            # Pesos
            model.load_state_dict(torch.load(os.path.join(exp_dir, 'best_model.pt'), map_location=DEVICE))
            model.eval()
            
            # Inferência
            preds, targets = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    out = model(x.to(DEVICE))
                    preds.append(out.cpu().numpy())
                    targets.append(y.numpy())
            
            # Reconstrução
            y_pred = np.concatenate(preds, axis=0)
            y_true = np.concatenate(targets, axis=0)
            
            N, T, _ = y_pred.shape
            # Inversão do Scaler (retorna 0 a 1)
            y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(N, T)
            y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(N, T)
            
            # Reconstroi e converte para kW
            df_model = reconstruct_dataframe(y_pred_inv, y_true_inv, test_dataset, df_raw, df_processed, config['output_seq_len'])
            
            if not df_model.empty:
                df_model['Modelo'] = model_name
                all_results.append(df_model)
                print("   ✅ OK")

        except Exception as e:
            print(f"   ❌ Erro: {e}")

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