import torch
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

from core.models.encdec_model import EncDecModel
from core.utils.early_stopping import EarlyStopping
from core.loss_function.cpiloss import CPILoss
from core.loss_function.mseloss import MaskedMSELoss

import optuna # <--- Importante


# Mantenha o CONFIG padrão como "fallback"
DEFAULT_CONFIG = {
    'csv_path': 'data/pv0.csv',
    'split_ratios': {'train': 0.8, 'val': 0.2},
    'test_year': 2022,
    'preprocessing': {
        'latitude': -23.56, 'longitude': -46.73, 'altitude': 0,
        'timezone': 'Etc/GMT+3', 'nominal_power': 156.0, 'start_year': 2018,
        'features_to_scale':['temp_amb','wind_speed'],
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
    'prediction_mode': 'clearsky_ratio',
    'target_col': 'k',
    'use_mask': True,
    'feature_cols': ['temp_amb', 'wind_speed', 'humidity', 'fracao_difusa', 'irr_clearsky_ratio', 'kt'],
    'model_type': 'Otimizacao',
    'cell_type': 'gru',
    'input_seq_len': 24,
    'output_seq_len': 1,
    'hidden_sizes': [128, 64],
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 20, # <--- Reduzido para Otimização (Pruning resolve o resto)
    'dropout': 0.1,
    'bidirectional': False,
    'use_attention': False,
    'use_feature_attention': False,
    'patience': 5,
    'loss_function':'cpi_loss'
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(preprocessor_class, dataset_class, override_params=None, trial=None):
    """
    Função principal que o Optuna vai chamar.
    override_params: Dicionário com os hiperparâmetros que queremos testar nesta rodada.
    trial: Objeto do Optuna para fazer o Pruning.
    """
    # 1. Mesclar Configuração Padrão com a Sugestão do Optuna
    config = DEFAULT_CONFIG.copy()
    if override_params:
        config.update(override_params)
    
    # Setup de Diretórios (opcional para não encher o disco durante testes)
    # Se for trial do optuna, não salvamos artefatos pesados, apenas logs
    
    # 2. Leitura e Pré-processamento (Idêntico ao original)
    df = pd.read_csv(config['csv_path'])
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df = df.drop_duplicates(subset=['Date_Time'], keep='first').set_index('Date_Time').sort_index()
    df = df[~df.index.duplicated(keep='first')]

    pp_conf = config['preprocessing']
    preprocessor = preprocessor_class(
        latitude=pp_conf['latitude'], longitude=pp_conf['longitude'], 
        altitude=pp_conf['altitude'], timezone=pp_conf['timezone'], 
        nominal_power=pp_conf['nominal_power'], start_year=pp_conf['start_year'],
        features_to_scale=pp_conf['features_to_scale'],
        target_col=config['target_col'], 
        column_mapping=pp_conf['column_mapping'] 
    )
    preprocessor.fit(df)
    df_processed = preprocessor.transform(df)
    
    # ... (Código de Split Treino/Val/Teste igual ao original) ...
    last_year = config['test_year']
    test_df = df_processed[df_processed.index.year == last_year].copy()
    dev_df = df_processed[df_processed.index.year < last_year].copy()
    
    n_dev = len(dev_df)
    train_end = int(n_dev * config['split_ratios']['train'])
    train_df = dev_df.iloc[:train_end].copy()
    val_df = dev_df.iloc[train_end:].copy()
    
    # Adaptação Target
    if config['target_col'] != 'target':
        train_df['target'] = train_df[config['target_col']]
        val_df['target'] = val_df[config['target_col']]

    # Datasets e Loaders
    train_dataset = dataset_class(
        df=train_df, feature_cols=config['feature_cols'], target_col=config['target_col'],
        n_past=config['input_seq_len'], n_future=config['output_seq_len']
    )
    val_dataset = dataset_class(
        df=val_df, feature_cols=config['feature_cols'], target_col=config['target_col'],
        n_past=config['input_seq_len'], n_future=config['output_seq_len']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Modelo
    model = EncDecModel(
        input_size=len(config['feature_cols']),
        hidden_sizes=config['hidden_sizes'],
        output_seq_len=config['output_seq_len'],
        output_dim=1,
        cell_type=config['cell_type'],
        dropout_prob=config['dropout']
    ).to(DEVICE)

    if config['loss_function'] == 'cpi_loss':
        criterion = CPILoss()
    else:
        criterion = MaskedMSELoss()
        
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Loop de Treino com Integração Optuna (Pruning)
    min_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        for x, y, mask in train_loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y) # Simplificado para brevidade
            loss.backward()
            optimizer.step()
        
        # Validação
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Guardar o melhor para retorno
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss

        # --- INTEGRAÇÃO COM OPTUNA: PRUNING ---
        if trial:
            trial.report(avg_val_loss, epoch)
            # Se o erro for muito alto comparado aos outros, corta o treino aqui.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
    return min_val_loss

if __name__ == "__main__":
    pass