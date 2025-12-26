import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import json
import datetime
import shutil
import matplotlib.pyplot as plt
import time

# Tenta importar CodeCarbon (caso não esteja instalado, avisa)
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("⚠️ CodeCarbon não encontrado. Instale com 'pip install codecarbon' para rastrear emissões.")

# Importações do projeto
from models.encdec_model import EncDecModel
from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from utils.early_stopping import EarlyStopping

# ==========================================
# CONFIGURAÇÃO
# ==========================================
CSV_PATH = 'data/pv0.csv'
OUTPUT_ROOT = 'trained_models'
ARTIFACTS_DIR = 'artifacts'

CONFIG = {
    'input_seq_len': 24,
    'output_seq_len': 1,
    'hidden_sizes': [128, 64],
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 100,
    'dropout': 0.1,
    'bidirectional': False,
    'use_attention': True,
    'use_feature_attention': True,
    'patience': 15,
    'model_type': '2att_EDLSTM_Attention',
    'feature_cols': ['temp_amb', 'wind_speed', 'humidity', 'target', 'cos_zenith', 'sin_azimuth']
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
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
    # 1. Setup de Diretórios e Tracker
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{timestamp}_{CONFIG['model_type']}"
    exp_dir = os.path.join(OUTPUT_ROOT, exp_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    tracker = None
    if HAS_CODECARBON:
        tracker = EmissionsTracker(output_dir=exp_dir, output_file="emissions.csv")
        tracker.start()

    print(f"🚀 Iniciando treinamento: {exp_name}")
    save_config(CONFIG, exp_dir)

    # 2. Carregamento e Limpeza de Dados
    print(f"⏳ Carregando dados: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df = df.drop_duplicates(subset=['Date_Time'], keep='first')
        df = df.set_index('Date_Time').sort_index()
    
    df = df[~df.index.duplicated(keep='first')]

    # 3. Preprocessamento
    preprocessor = SolarPreprocessor(
        latitude=-15.60, longitude=-47.70, timezone='Etc/GMT+3', 
        nominal_power=156.0, target_col='Pot_BT',
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

    print("⚙️ Ajustando preprocessor...")
    preprocessor.fit(df)
    
    # Salva Scalers (Local e Artifacts)
    preprocessor.save_scalers(exp_dir)
    preprocessor.save_scalers(ARTIFACTS_DIR)
    
    df_processed = preprocessor.transform(df)
    
    # Valida Features
    available_cols = [c for c in CONFIG['feature_cols'] if c in df_processed.columns]
    if len(available_cols) != len(CONFIG['feature_cols']):
        print(f"⚠️ Ajustando features. Disponíveis: {available_cols}")
        CONFIG['feature_cols'] = available_cols
        save_config(CONFIG, exp_dir)

    # 4. Split e Dataloaders
    n = len(df_processed)
    train_df = df_processed.iloc[:int(n*0.7)]
    val_df = df_processed.iloc[int(n*0.7):int(n*0.9)]
    
    train_dataset = SolarEfficientDataset(train_df, input_tag=CONFIG['feature_cols'], n_past=CONFIG['input_seq_len'], n_future=CONFIG['output_seq_len'])
    val_dataset = SolarEfficientDataset(val_df, input_tag=CONFIG['feature_cols'], n_past=CONFIG['input_seq_len'], n_future=CONFIG['output_seq_len'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # 5. Modelo e Otimização
    model = EncDecModel(
        input_size=len(CONFIG['feature_cols']),
        hidden_sizes=CONFIG['hidden_sizes'],
        output_seq_len=CONFIG['output_seq_len'],
        output_dim=1,
        cell_type='lstm',
        bidirectional=CONFIG['bidirectional'],
        use_attention=CONFIG['use_attention'],
        use_feature_attention=CONFIG['use_feature_attention'],
        dropout_prob=CONFIG['dropout']
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=True, path=os.path.join(exp_dir, 'best_model.pt'))

    # 6. Loop de Treino
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

        # Validação
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
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
        
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("🛑 Early stopping.")
            break

    # 7. Finalização e Logs
    training_time = time.time() - start_time
    if tracker: tracker.stop()

    # Salva Curva de Aprendizado
    plot_learning_curve(train_losses, val_losses, os.path.join(exp_dir, 'learning_curve.png'))
    
    # Salva CSV de histórico
    log_df = pd.DataFrame({'epoch': range(1, len(train_losses)+1), 'train_loss': train_losses, 'val_loss': val_losses})
    log_df.to_csv(os.path.join(exp_dir, 'training_log.csv'), index=False)

    # Salva Metrics.json (Melhor Validação)
    best_val_loss = min(val_losses)
    final_metrics = {
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "training_time_sec": training_time,
        "model_path": os.path.join(exp_dir, 'best_model.pt')
    }
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"✅ Treinamento finalizado! Resultados em: {exp_dir}")

if __name__ == "__main__":
    main()