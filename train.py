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
# Loss Functions
from loss_function.cpiloss import CPILoss
from loss_function.mseloss import MaskedMSELoss
from loss_function.pyloss import PhysicsGuidedLoss

# ================= CONFIGURAÇÃO CENTRALIZADA =================
CONFIG = {
    # --- 1. PATH DA BASE DE DADOS ---
    'csv_path': 'data/pv0.csv',
    
    # --- 2. DIVISÃO DA BASE DE TREINO, TESTE E VALIDAÇÃO ---
    'split_ratios': {'train': 0.8, 'val': 0.2}, 
    'test_year':2022,

    # --- 3. PRÉ-PROCESSAMENTO (Física & Mapeamento) ---
    'preprocessing': {
        'latitude': -23.56,
        'longitude': -46.73,
        'altitude': 0,
        'timezone': 'Etc/GMT+3',
        'nominal_power': 156.0,
        'start_year': 2018,
        'features_to_scale':['temp_amb','wind_speed'],
        #'pv_power_col_csv': 'Pot_BT', # <--- AVALIAR PARA RETIRAR
        
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

    # --- 4. ESTRATÉGIA DE MODELAGEM ---
    # mode: 'sky' (prevê k) ou 'power' (prevê kW normalizado)
    'prediction_mode': 'power',
    
    # Qual variável o modelo vai prever? ('kt', 'fracao_difusa' ou 'target')
    'target_col': ['target'], 

    # mascara de dados noturnos
    'use_mask':False,   # True = Ignora a noite (ideal para K)
                        # False = Aprende a noite (ideal para Potência)
    
    # Features de entrada
    'feature_cols': [
        'cos_zenith', 'sin_azimuth', 'target',
        'wind_speed', 'temp_amb', 'humidity'
        #'elevation', 'delta_kt', 
        #'delta_fracao_difusa', 'QS', 'wind_speed',
        #'temp_amb', 'humidity'
        #'kt', 'fracao_difusa'
        
    ],

    'aux_col':[
        'ghi_cs', 'cos_zenith', 
        'elevation', 'ghi_extra'
    ],

    # --- 5. ARQUITETURA E TREINO ---
    'model_type': 'LSTM_PAZVTU',
    'cell_type': 'lstm',
    'input_seq_len': 24,
    'output_seq_len': 1,
    'hidden_sizes': [300],
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10000,
    'dropout': 0.2,
    'bidirectional': False,
    'use_attention': False,
    'use_feature_attention': False,
    'patience': 100,
    'loss_function':'mse',      # "cpi_loss", "mse", physics_loss
    'physics_base_loss': 'cpi',
    'lambda_hard':10,
    'lambda_soft':10,
}

OUTPUT_ROOT = 'trained_models'
ARTIFACTS_DIR = 'artifacts'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SALVA A CONFIGURAÇÃO DE TREINAMENTO
def save_config(config, path):
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

def save_training_log(log_data, path="outputs/training_log.csv"):
    df_log = pd.DataFrame(log_data)
    # Se o arquivo não existir, cria com cabeçalho, senão anexa
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    df_log.to_csv(path, index=False)

# SALVA A CURVA DE TREINAMENTO
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
        altitude=pp_conf['altitude'],
        timezone=pp_conf['timezone'], 
        nominal_power=pp_conf['nominal_power'], 
        start_year=pp_conf['start_year'],
        features_to_scale=pp_conf['features_to_scale'],
        target_col=CONFIG['prediction_mode'], # <--- unica variavel que não vem do preprocessing
        column_mapping=pp_conf['column_mapping'],
        cs_model = 'esra',
        kasten_corr=True
    )
    
    preprocessor.fit(df)
    preprocessor.save_scalers(exp_dir)
    #preprocessor.save_scalers(ARTIFACTS_DIR)
    
    # O método transform usa o column_mapping para renomear as colunas
    df_processed = preprocessor.transform(df)

    # 4. Validação de Colunas
    target_col = CONFIG['target_col']

    for col in target_col:
        if col not in df_processed.columns:
            raise ValueError(f"❌ Coluna alvo '{target_col}' não encontrada! Verifique o column_mapping.")

    available_cols = [c for c in CONFIG['feature_cols'] if c in df_processed.columns]
    if len(available_cols) != len(CONFIG['feature_cols']):
        print(f"⚠️ Features ajustadas: {available_cols}")
        CONFIG['feature_cols'] = available_cols

    # 5. Split Temporal (Último Ano = Teste)
    #last_year = df_processed.index.year.max() <---- REMOVER FUTURAMENTE
    last_year = CONFIG['test_year']
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

    train_dataset = SolarEfficientDataset(
        df=train_df, 
        feature_cols=CONFIG['feature_cols'], 
        target_col=CONFIG['target_col'],
        aux_col=CONFIG['aux_col'],
        n_past=CONFIG['input_seq_len'], 
        n_future=CONFIG['output_seq_len']
    )
    val_dataset = SolarEfficientDataset(
        df=val_df, 
        feature_cols=CONFIG['feature_cols'], 
        target_col=CONFIG['target_col'],
        aux_col=CONFIG['aux_col'],
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
        output_dim=len(CONFIG['target_col']),
        cell_type=CONFIG['cell_type'],
        bidirectional=CONFIG['bidirectional'],
        use_attention=CONFIG['use_attention'],
        use_feature_attention=CONFIG['use_feature_attention'],
        dropout_prob=CONFIG['dropout']
    ).to(DEVICE)

    # modificar o criterion muda a função de erro implementado a CPILoss
    if CONFIG['loss_function'] == 'mse':
        criterion = MaskedMSELoss()
    elif CONFIG['loss_function'] == 'cpi_loss':
        criterion = CPILoss()
    elif CONFIG['loss_function'] == 'physics_loss':
        criterion = PhysicsGuidedLoss(
            lambda_hard=CONFIG['lambda_hard'], 
            lambda_soft=CONFIG['lambda_soft'],
            data_loss_type=CONFIG['physics_base_loss']
        )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    early_stopping = EarlyStopping(patience=CONFIG['patience'], verbose=True, path=os.path.join(exp_dir, 'best_model.pt'))

    # 7. Treino
    print("🔥 Iniciando épocas...")
    train_losses, val_losses = [], []
    start_time = time.time()
    training_history = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_metrics = {
                    'epoch': epoch + 1,
                    'mse_stat': 0, 
                    'check_ghi':0,
                    'check_dhi':0,
                    'check_dni':0,
                    'check_diffuse_fraction':0,
                    'check_consistence':0,
                    'check_overcast_condition':0,
                    'check_maximum_direct_fraction':0,
                    'check_tracker_off':0,
                    'check_loss_physics':0,
                    'total_loss':0,
                }
        
        val_epoch_metrics = {
            'epoch': epoch + 1,
            'mse_stat': 0, 
            'check_ghi':0,
            'check_dhi':0,
            'check_dni':0,
            'check_diffuse_fraction':0,
            'check_consistence':0,
            'check_overcast_condition':0,
            'check_maximum_direct_fraction':0,
            'check_tracker_off':0,
            'check_loss_physics':0,
            'total_loss':0,
        }

        #batch_losses = []
        for x, y, mask, aux in train_loader:
            x, y, mask, aux = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE), aux.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            active_mask = mask if CONFIG['use_mask'] else None
            if CONFIG['prediction_mode'] == 'sky':
                loss, loss_dict = criterion(out, y, aux, mask=active_mask)
                loss.backward()
                for key in loss_dict:
                    epoch_metrics[key] += loss_dict[key] / len(train_loader)

            elif CONFIG['prediction_mode'] == 'power':
                loss = criterion(out, y, mask=active_mask)
                loss.backward()

            optimizer.step()
            # Acumula as métricas para o log (média ponderada pelo tamanho do batch)
            epoch_metrics['total_loss'] += loss.item() / len(train_loader)

            #batch_losses.append(loss.item())
        
        #avg_train = np.mean(batch_losses)
        train_losses.append(epoch_metrics)

        model.eval()
        #val_batch_losses = []
        with torch.no_grad():
            for x, y, mask, aux in val_loader:
                x, y, mask, aux = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE), aux.to(DEVICE)
                out = model(x)
                active_mask = mask if CONFIG['use_mask'] else None

                if CONFIG['prediction_mode'] == 'sky':
                    loss, loss_dict = criterion(out, y, aux, mask=active_mask)
                    # Acumula as métricas para o log (média ponderada pelo tamanho do batch)
                    for key in loss_dict:
                        val_epoch_metrics[key] += loss_dict[key] / len(val_loader)

                if CONFIG['prediction_mode'] == 'power':
                    loss = criterion(out, y, mask=active_mask)

                val_epoch_metrics['total_loss'] += loss.item() / len(val_loader)
        #avg_val = np.mean(val_batch_losses)
        val_losses.append(val_epoch_metrics)
        save_training_log(train_losses, os.path.join(exp_dir, 'log_xai.csv'))
        
        print(f"Epoch {epoch+1} | Train: {epoch_metrics['total_loss']:.6f} | Val: {val_epoch_metrics['total_loss']:.6f}")
        
        early_stopping(val_epoch_metrics['total_loss'], model)
        if early_stopping.early_stop:
            print("🛑 Early stopping.")
            break

    # 8. Logs Finais
    if tracker: tracker.stop()

    # Extrai apenas as perdas totais para gerar o gráfico
    train_total_only = [d['total_loss'] for d in train_losses]
    val_total_only = [d['total_loss'] for d in val_losses]
    plot_learning_curve(train_total_only, val_total_only, os.path.join(exp_dir, 'learning_curve.png'))
    
    """    
    pd.DataFrame({'epoch': range(1, len(train_losses)+1), 'train': train_losses, 'val': val_losses})\
        .to_csv(os.path.join(exp_dir, 'training_log.csv'), index=False)
    """

    final_metrics = {
        "best_val_loss": min(val_total_only),
        "epochs": len(train_losses),
        "time_sec": time.time() - start_time,
        "model_path": os.path.join(exp_dir, 'best_model.pt'),
    }
    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)

    print(f"✅ Fim. Modelo salvo em: {exp_dir}")

if __name__ == "__main__":
    main()