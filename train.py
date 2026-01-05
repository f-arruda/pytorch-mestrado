import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import json
import datetime
import time
import matplotlib.pyplot as plt

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

from models.encdec_model import EncDecModel
from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from utils.early_stopping import EarlyStopping

# ================= CONFIGURAÇÃO CENTRALIZADA =================
CONFIG = {
    # --- 1. DADOS E DIVISÃO ---
    'csv_path': 'data/pv0.csv',
    'split_ratios': {'train': 0.8, 'val': 0.2}, 
    
    # --- 2. PRÉ-PROCESSAMENTO (Física & Mapeamento) ---
    'preprocessing': {
        'latitude': -23.56,
        'longitude': -46.73,
        'timezone': 'Etc/GMT+3',
        'nominal_power': 156.0,
        'pv_power_col_csv': 'Pot_BT',
        
        # DOCUMENTAÇÃO VIVA: Mapeamento "De -> Para"
        # O Preprocessor usará isso para renomear as colunas internamente.
        # Chave (Esquerda): Nome como está no CSV bruto.
        # Valor (Direita): Nome padronizado usado no código.
        'column_mapping': {
            'Pot_BT': 'target',
            'Irradiação Global horária(horizontal) kWh/m2': 'ghi',
            'Irradiação Difusa horária kWh/m2': 'dhi',
            'Irradiação Global horária(Inclinada 27°) kWh/m2': 'irrad_poa',
            'Temperatura ambiente °C': 'temp_amb',
            'Umidade Relativa %': 'humidity',
            'Velocidade média do vento m/s': 'wind_speed'
        }
    },

    # --- 3. ESTRATÉGIA DE MODELAGEM ---
    # mode: 'clearsky_ratio' (prevê k) ou 'direct' (prevê kW normalizado)
    'prediction_mode': 'clearsky_ratio',
    
    # Qual variável o modelo vai prever? ('k' ou 'target')
    'target_col': 'k', 
    
    # Features de entrada
    'feature_cols': [
        'temp_amb', 'wind_speed', 'humidity',  
        'fracao_difusa', 'irr_clearsky_ratio',
        'k'
    ],

    # --- 4. ARQUITETURA E TREINO ---
    'model_type': 'EDLSTM_K_Factor',
    'cell_type': 'lstm',
    'input_seq_len': 24,
    'output_seq_len': 1,
    'hidden_sizes': [128, 64],
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'dropout': 0.1,
    'bidirectional': False,
    'use_attention': False,
    'use_feature_attention': False,
    'patience': 20,
}

OUTPUT_ROOT = 'trained_models'
ARTIFACTS_DIR = 'artifacts'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_config(config, path):
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def plot_learning_curve(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Curva de Aprendizado')
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def main():
    # 1. Setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{timestamp}_{CONFIG['model_type']}"
    exp_dir = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    tracker = EmissionsTracker(output_dir=exp_dir, output_file="emissions.csv") if HAS_CODECARBON else None
    if tracker: tracker.start()

    print(f"🚀 Iniciando: {exp_name}")
    print(f"🎯 Modo: {CONFIG['prediction_mode']} | Alvo: {CONFIG['target_col']}")
    save_config(CONFIG, exp_dir)

    # 2. Leitura
    csv_path = CONFIG['csv_path']
    print(f"⏳ Lendo: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df = df.drop_duplicates(subset=['Date_Time'], keep='first').set_index('Date_Time').sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # 3. Pré-processamento
    pp_conf = CONFIG['preprocessing']
    
    # Instancia passando o mapa explícito. 
    # Isso garante que a padronização aconteça conforme o CONFIG acima.
    preprocessor = SolarPreprocessor(
        latitude=pp_conf['latitude'], 
        longitude=pp_conf['longitude'], 
        timezone=pp_conf['timezone'], 
        nominal_power=pp_conf['nominal_power'], 
        target_col=pp_conf['pv_power_col_csv'],
        column_mapping=pp_conf['column_mapping'] # <--- AQUI ACONTECE A MÁGICA DOCUMENTADA
    )
    
    preprocessor.fit(df)
    preprocessor.save_scalers(exp_dir)
    preprocessor.save_scalers(ARTIFACTS_DIR)
    
    # O método transform usa o column_mapping para renomear as colunas
    df_processed = preprocessor.transform(df)

    # 4. Validação de Colunas
    target_col = CONFIG['target_col']
    if target_col not in df_processed.columns:
        raise ValueError(f"❌ Coluna alvo '{target_col}' não encontrada! Verifique o column_mapping.")

    available_cols = [c for c in CONFIG['feature_cols'] if c in df_processed.columns]
    if len(available_cols) != len(CONFIG['feature_cols']):
        print(f"⚠️ Features ajustadas: {available_cols}")
        CONFIG['feature_cols'] = available_cols
        save_config(CONFIG, exp_dir)

    # 5. Split Temporal (Último Ano = Teste)
    last_year = df_processed.index.year.max()
    print(f"📅 Separando ano {last_year} para TESTE.")
    
    test_df = df_processed[df_processed.index.year == last_year].copy()
    dev_df = df_processed[df_processed.index.year < last_year].copy()
    
    if dev_df.empty:
        raise ValueError("❌ Erro no Split: Dados insuficientes antes do último ano.")

    # Split Treino/Validação
    n_dev = len(dev_df)
    train_end = int(n_dev * CONFIG['split_ratios']['train'])
    
    train_df = dev_df.iloc[:train_end].copy()
    val_df = dev_df.iloc[train_end:].copy()
    
    print(f"📊 Divisão: Treino={len(train_df)} | Val={len(val_df)} | Teste={len(test_df)}")

    # Adaptação Target (Caso o alvo não seja 'target')
    if target_col != 'target':
        print(f"🔄 Adaptando dataset: '{target_col}' -> 'target'")
        train_df['target'] = train_df[target_col]
        val_df['target'] = val_df[target_col]
    
    # DataLoaders
    train_dataset = SolarEfficientDataset(
        df=train_df, 
        feature_cols=CONFIG['feature_cols'], 
        target_col=CONFIG['target_col'],
        n_past=CONFIG['input_seq_len'], 
        n_future=CONFIG['output_seq_len']
    )
    val_dataset = SolarEfficientDataset(
        df=val_df, 
        feature_cols=CONFIG['feature_cols'], 
        target_col=CONFIG['target_col'],
        n_past=CONFIG['input_seq_len'], 
        n_future=CONFIG['output_seq_len']
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # 6. Modelo
    model = EncDecModel(
        input_size=len(CONFIG['feature_cols']),
        hidden_sizes=CONFIG['hidden_sizes'],
        output_seq_len=CONFIG['output_seq_len'],
        output_dim=1,
        cell_type=CONFIG['cell_type'],
        bidirectional=CONFIG['bidirectional'],
        use_attention=CONFIG['use_attention'],
        use_feature_attention=CONFIG['use_feature_attention'],
        dropout_prob=CONFIG['dropout']
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=True, path=os.path.join(exp_dir, 'best_model.pt'))

    # 7. Treino
    print("🔥 Iniciando épocas...")
    train_losses, val_losses = [], []
    start_time = time.time()
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        avg_train = np.mean(batch_losses)
        train_losses.append(avg_train)

        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_batch_losses.append(loss.item())
        
        avg_val = np.mean(val_batch_losses)
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
        
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping.")
            break

    # 8. Logs Finais
    if tracker: tracker.stop()
    plot_learning_curve(train_losses, val_losses, os.path.join(exp_dir, 'learning_curve.png'))
    
    pd.DataFrame({'epoch': range(1, len(train_losses)+1), 'train': train_losses, 'val': val_losses})\
        .to_csv(os.path.join(exp_dir, 'training_log.csv'), index=False)

    final_metrics = {
        "best_val_loss": min(val_losses),
        "epochs": len(train_losses),
        "time_sec": time.time() - start_time,
        "model_path": os.path.join(exp_dir, 'best_model.pt'),
        "config_snapshot": CONFIG
    }
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"✅ Fim. Modelo salvo em: {exp_dir}")

if __name__ == "__main__":
    main()