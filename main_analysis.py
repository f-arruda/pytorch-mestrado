import torch
import pandas as pd
import numpy as np
import joblib
import os
import json
from torch.utils.data import DataLoader

# Imports Locais
from models.encdec_model import EncDecModel
from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from src.statistical_metrics import SolarStatisticalAnalyzer

# ==========================================
# 1. CONFIGURAÇÃO DA ANÁLISE
# ==========================================
CSV_PATH = 'data/base_teste.csv'  # Caminho da base original
SCALER_PATH = 'scaler_Y.pkl'      # Scaler do target
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LISTA DE EXPERIMENTOS A COMPARAR ---
# Agora você coloca APENAS o caminho da pasta. O script descobre o resto.
EXPERIMENTS_DIRS = [
    "experiments/2025-12-17_13-49-49_LSTM_Bi_Attention", 
    "experiments/2025-12-17_13-52-59_GRU_Bi_Attention"
]

# Nome da pasta onde salvar os gráficos comparativos finais

# ==========================================
# FUNÇÕES AUXILIARES DE CARREGAMENTO
# ==========================================
def load_experiment_config(exp_dir):
    """Lê o JSON de configuração dentro da pasta do experimento."""
    config_path = os.path.join(exp_dir, 'config.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ config.json não encontrado em {exp_dir}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Tratamento de string para listas (caso o JSON tenha salvo listas como strings)
    # O json.load geralmente resolve, mas por segurança em alguns sistemas:
    if isinstance(config['hidden_sizes'], str):
        config['hidden_sizes'] = eval(config['hidden_sizes'])
        
    return config

def load_test_data(csv_path):
    """Carrega e prepara dados de teste (Ano 2022)."""
    print("⏳ Carregando dados de teste...")
    df = pd.read_csv(csv_path, index_col=0)
    
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    if 'Year' not in df.columns:
        df['Year'] = df.index.year
        
    df_test_raw = df[df['Year'] == 2022].copy()
    
    # Processamento (Sem fit, apenas transform)
    preprocessor = SolarPreprocessor(nominal_power=156.0)
    df_test_processed = preprocessor.transform(df_test_raw)
    
    return df_test_raw, df_test_processed

def reconstruct_dataframe(y_pred, y_true, dataset, df_raw, output_seq_len):
    """Reconstrói o DataFrame com metadados (Zenith, Hour)."""
    valid_indices = dataset.valid_indices
    records = []
    
    for i, idx in enumerate(valid_indices):
        pred_seq = y_pred[i]
        true_seq = y_true[i]
        
        # Metadados do futuro
        future_metadata = df_raw.iloc[idx : idx + output_seq_len]
        
        for h in range(output_seq_len):
            if h < len(future_metadata):
                timestamp = future_metadata.index[h]
                row = future_metadata.iloc[h]
                
                records.append({
                    'Timestamp': timestamp,
                    'Horizonte': h + 1,
                    'Observado': true_seq[h],
                    'Previsto': pred_seq[h],
                    'Hour': timestamp.hour,
                    'zenith': row.get('zenith', 0),
                    'Condição de céu': row.get('Condição de céu', 'Desconhecido')
                })
    return pd.DataFrame(records)

# ==========================================
# LOOP PRINCIPAL INTELIGENTE
# ==========================================
def main():
    # 1. Carregar Dados de Teste
    df_raw, df_processed = load_test_data(CSV_PATH)
    feature_cols = [c for c in df_processed.columns]
    
    # Carrega Scaler
    scaler_y = joblib.load(SCALER_PATH)
    
    all_results = []
    
    print(f"\n📂 Iniciando avaliação de {len(EXPERIMENTS_DIRS)} experimentos...\n")

    # 2. Iterar sobre as pastas
    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir):
            print(f"⚠️ Pasta não encontrada: {exp_dir}. Pulando...")
            continue
            
        try:
            # A. CARREGA CONFIGURAÇÃO AUTOMÁTICA
            config = load_experiment_config(exp_dir)
            
            # Nome do modelo para a legenda (usa o model_type do config ou o nome da pasta)
            model_legend_name = config.get('model_type', os.path.basename(exp_dir))
            print(f"🔹 Processando: {model_legend_name}")
            print(f"   path: {exp_dir}")

            # B. CONFIGURA DATASET (Baseado nos parametros salvos no JSON)
            # Garante que usamos o mesmo tamanho de janela que foi treinado
            input_seq_len = int(config['input_seq_len'])
            output_seq_len = int(config['output_seq_len'])
            
            test_dataset = SolarEfficientDataset(
                df_processed, 
                input_tag=feature_cols, 
                n_past=input_seq_len, 
                n_future=output_seq_len
            )
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

            # C. INSTANCIA O MODELO (Usando parâmetros do JSON)
            model = EncDecModel(
                input_size=len(feature_cols),
                hidden_sizes=config['hidden_sizes'], # Vem do JSON!
                output_seq_len=output_seq_len,
                output_dim=1,
                cell_type=config.get('cell_type', 'lstm'),       # Vem do JSON!
                bidirectional=config.get('bidirectional', False), # Vem do JSON!
                use_attention=config.get('use_attention', False), # Vem do JSON!
                dropout_prob=config.get('dropout', 0.0)           # Vem do JSON!
            ).to(DEVICE)
            
            # D. CARREGA PESOS
            weights_path = os.path.join(exp_dir, 'best_model.pt')
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            
            # E. INFERÊNCIA
            preds_list, targets_list = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(DEVICE)
                    out = model(x)
                    preds_list.append(out.cpu().numpy())
                    targets_list.append(y.numpy())
            
            # F. RECONSTRUÇÃO
            y_pred = np.concatenate(preds_list, axis=0)
            y_true = np.concatenate(targets_list, axis=0)
            
            # Desnormaliza
            N, T, _ = y_pred.shape
            y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(N, T)
            y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(N, T)
            
            df_model = reconstruct_dataframe(y_pred_inv, y_true_inv, test_dataset, df_raw, output_seq_len)
            df_model['Modelo'] = model_legend_name # Adiciona coluna identificadora
            
            all_results.append(df_model)
            print("   ✅ Sucesso.")

        except Exception as e:
            print(f"   ❌ Erro ao processar {exp_dir}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Análise Estatística Consolidada
    if not all_results:
        print("Nenhum resultado gerado.")
        return

    print("\n📊 Consolidando resultados e gerando gráficos...")
    df_final = pd.concat(all_results, ignore_index=True)
    
    analyzer = SolarStatisticalAnalyzer(df_final, output_dir=OUTPUT_COMPARISON_DIR)
    
    analyzer.save_global_metrics()
    analyzer.plot_metrics_by_horizon()
    analyzer.plot_boxplots_hourly()
    analyzer.plot_taylor_diagram()
    analyzer.plot_scatter_reg()
    
    print(f"\n🏁 Análise Comparativa salva em: {OUTPUT_COMPARISON_DIR}")

if __name__ == "__main__":
    main()