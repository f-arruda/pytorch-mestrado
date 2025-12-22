from models.encdec_model import EncDecModel
from models.stacked_rnn import Stacked_RNN
from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from utils.experiment_manager import ExperimentManager

import torch
from torchinfo import summary
import pandas as pd
from utils.early_stopping import EarlyStopping

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from codecarbon import EmissionsTracker

# --- Imports dos Seus Módulos ---
# Certifique-se que o Python enxerga a pasta raiz
import sys
sys.path.append(os.getcwd()) 

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False
    print("⚠️ CodeCarbon não encontrado. Instale com 'pip install codecarbon' para medir energia.")

# --- 1. Configurações e Hiperparâmetros ---
CONFIG = {
    'csv_path': 'data/pv0.csv',
    'target_col': 'target',
    'feature_cols': ['cos_zenith', 'sin_azimuth', 'fracao_difusa',
                     'target', 'temp_amb', 'humidity', 'wind_speed'],   # ['Pot_BT', 'cos_zenith', 'sin_azimuth']  
                                                               # ['Velocidade média do vento m/s', 'Temperatura ambiente °C', 'Umidade Relativa %']
    'input_seq_len': 24,    # n_past
    'output_seq_len': 1,   # n_fut
    'hidden_sizes': [300],
    'dropout': 0.2,
    'batch_size': 32,
    'epochs': 10000,        # Limite alto, controlado pelo Early Stopping
    'learning_rate': 0.001,
    'patience': 50,        
    'cell_type':'lstm', 
    'bidirectional':False,
    'use_attention':False,
    'use_feature_attention':False,
    'model_type':'EDLSTM_PAZTUVD',
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

def main():
    # --- 0. INICIA O GERENCIADOR (Antes de tudo) ---
    # Isso cria a pasta "experiments/DATA_HORA_GRU_Bi_Attention/" e salva o config.json
    manager = ExperimentManager(
        base_dir='experiments', 
        model_name=CONFIG['model_type'], 
        config=CONFIG
    )

    print(f"🚀 Iniciando treino no dispositivo: {CONFIG['device']}")
    
    # --- 2. Preparação dos Dados (Lógica de Divisão Customizada) ---
    print("⏳ Carregando dados brutos...")
    
    # 1. Carrega os dados brutos
    df_raw = pd.read_csv(CONFIG['csv_path'])

    # 2. LIMPEZA EXPLÍCITA (A Solução Simples)
    print("🧹 Realizando limpeza prévia de duplicatas...")

    # Garante que a coluna de data é datetime
    df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time'])

    # Remove linhas onde a DATA é duplicada (mantém a primeira aparição)
    df_raw = df_raw.drop_duplicates(subset=['Date_Time'], keep='first')

    # Define como índice e ordena
    df_raw = df_raw.set_index('Date_Time').sort_index()

    # Verifica se sobrou alguma duplicata teimosa no índice
    df_raw = df_raw[~df_raw.index.duplicated(keep='first')]

    print(f"✅ Dados limpos! Total de linhas únicas: {len(df_raw)}")
    
    # Garante que temos a coluna Year e o índice DateTime
    # (Se o CSV já tiver 'Year', ótimo. Se não, derivamos do índice)
    if not pd.api.types.is_datetime64_any_dtype(df_raw.index):
        df_raw.index = pd.to_datetime(df_raw.index)
    if 'Year' not in df_raw.columns:
        df_raw['Year'] = df_raw.index.year

    # 2. Separar o Ano de Teste (Holdout - 2022)
    # df_teste: Será usado APENAS no final de tudo (não entra no loop de epochs)
    df_teste_raw = df_raw.loc[df_raw['Year'] == 2022].copy()

    # 3. Separar o Período de Desenvolvimento (< 2022)
    df_periodo_dev = df_raw.loc[df_raw['Year'] < 2022].copy()
    
    # 4. Divisão Treino/Validação (80/20) dentro do período < 2022
    # Isso substitui seu reset_index() + loc[:round] de forma mais limpa
    split_idx = int(len(df_periodo_dev) * 0.8)
    
    df_train_raw = df_periodo_dev.iloc[:split_idx]
    df_val_raw = df_periodo_dev.iloc[split_idx:]
    
    print(f"📊 Divisão realizada:")
    print(f"   - Treino (<2022, 80%): {len(df_train_raw)} amostras")
    print(f"   - Validação (<2022, 20%): {len(df_val_raw)} amostras")
    print(f"   - Teste (==2022): {len(df_teste_raw)} amostras")

    # --- 3. Processamento e Normalização ---
    print("⚙️ Aplicando Preprocessor...")
    
    # Instancia o Preprocessor
    preprocessor = SolarPreprocessor(
        latitude=-23.33, 
        longitude=-46.44,
        timezone='Etc/GMT+3',
        nominal_power=156.0,
        target_col='target', # Nome interno que queremos usar
        start_year = 2018,
        # AQUI ESTÁ O SEGREDO DO TESTE:
        column_mapping={
            'Date_Time': 'date_time',    # Mapeia sua coluna de tempo
            'Irradiação Global horária(horizontal) kWh/m2': 'ghi',           # Mapeia irradiação
            'Temperatura ambiente °C': 'temp_amb',     # Mapeia temperatura
            'Umidade Relativa %': 'humidity',
            'Velocidade média do vento m/s': 'wind_speed',  # Mapeia vento
            'Pot_BT': 'target',    # Mapeia o alvo
            'Irradiação Difusa horária kWh/m2': 'dhi',
        }
    )
    
    # A. FIT + TRANSFORM no Treino (Aprende a escala aqui!)
    df_train = preprocessor.fit(df_train_raw).transform(df_train_raw)
    preprocessor.save_scalers(output_dir='.')

    # B. TRANSFORM na Validação e Teste (Usa a escala do Treino)
    # Importante: Não damos fit aqui para evitar vazamento de dados (Data Leakage)
    df_val = preprocessor.transform(df_val_raw)
    df_test = preprocessor.transform(df_teste_raw)
    
    # Define features       [c for c in df_train.columns]
    feature_cols = CONFIG['feature_cols']
    input_dim = len(feature_cols)

    # --- 4. Datasets e DataLoaders (Lazy Loading) ---
    # Dataset de Treino
    train_dataset = SolarEfficientDataset(
        df_train, 
        input_tag=feature_cols, 
        n_past=CONFIG['input_seq_len'], 
        n_future=CONFIG['output_seq_len']
    )
    
    # Dataset de Validação
    val_dataset = SolarEfficientDataset(
        df_val, 
        input_tag=feature_cols, 
        n_past=CONFIG['input_seq_len'], 
        n_future=CONFIG['output_seq_len']
    )
    
    # Dataset de Teste (Para inferência final)
    test_dataset = SolarEfficientDataset(
        df_test, 
        input_tag=feature_cols, 
        n_past=CONFIG['input_seq_len'], 
        n_future=CONFIG['output_seq_len']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    # Dica: O test_loader pode ter batch_size maior pois não faz backpropagation (gasta menos memória)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size']*2, shuffle=False)

    # --- 4. Instanciação do Modelo ---
    model = EncDecModel(
        input_size=input_dim,
        hidden_sizes=CONFIG['hidden_sizes'],
        output_seq_len=CONFIG['output_seq_len'],
        output_dim=1,           # Prevendo 1 variável (Potência)
        cell_type=CONFIG['cell_type'],        # Teste com GRU
        bidirectional=CONFIG['bidirectional'],     # Seu teste Bidirecional
        use_attention=CONFIG['use_attention'],
        dropout_prob=CONFIG['dropout'],      # Seu teste com Atenção
        use_feature_attention=CONFIG['use_feature_attention']
    ).to(CONFIG['device'])
    
    # Função de Perda e Otimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Callback Customizado (Early Stopping)
    early_stopping = EarlyStopping(
        patience=CONFIG['patience'], 
        verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'loss': [], 'val_loss': []}
    
    # --- CONFIGURAÇÃO DO RASTREADOR DE ENERGIA ---
    tracker = None
    if HAS_CODECARBON:
        # output_dir manda o relatório 'emissions.csv' direto para a pastinha do experimento!
        tracker = EmissionsTracker(
            project_name=CONFIG['model_type'],
            output_dir=manager.exp_dir, 
            measure_power_secs=15, # Mede a cada 15 segundos
            log_level='warning'    # Evita poluir o terminal
        )
        tracker.start()
    # --- INÍCIO DO CRONÔMETRO ---
    start_time = time.time()

    print("🔥 Iniciando Treino...")
    
    for epoch in range(CONFIG['epochs']):
        # --- TREINO ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(CONFIG['device'])
            batch_y = batch_y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0) # Acumula loss ponderada
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(CONFIG['device'])
                batch_y = batch_y.to(CONFIG['device'])
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Salva histórico
        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} \t Loss: {avg_train_loss:.6f} \t Val Loss: {avg_val_loss:.6f}")
        
        # --- SALVAMENTO AUTOMÁTICO NA PASTA CERTA ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Usa o manager para salvar na pasta criada
            manager.save_model(model, filename='best_model.pt')
            print(f"   (Novo melhor modelo salvo!)")
        else:
            patience_counter += 1
        
        # --- CHECK EARLY STOPPING ---
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print("🛑 Early stopping atingido!")
            break

    # --- PÓS TREINO ---
    


    # 1. Salvar o gráfico final na pasta
    manager.save_plot(history)
    
    # 2. Salvar métrica final
    final_rmse = np.sqrt(best_val_loss)
    # --- FIM DO CRONÔMETRO ---
    end_time = time.time()
    total_time = end_time - start_time
    
    # --- FIM DO RASTREADOR DE ENERGIA ---
    emissions_data = {}
    if tracker:
        emissions = tracker.stop() # Retorna kg de CO2eq
        # O tracker gera atributos internos úteis:
        emissions_data = {
            'CO2_Emissions_kg': emissions,
            'Energy_Consumed_kWh': tracker.final_emissions_data.energy_consumed,
            'Country_ISO': tracker.final_emissions_data.country_iso_code
        }
        print(f"🌍 Emissões: {emissions:.4f} kg CO2 | Energia: {emissions_data['Energy_Consumed_kWh']:.4f} kWh")

    # Formata o tempo para ficar legível (ex: "01:30:15")
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    
    print(f"⏱️ Tempo Total de Treino: {time_str}")

    # --- SALVANDO TUDO NO EXPERIMENT MANAGER ---
    # Adicionamos essas novas métricas ao dicionário final
    
    final_metrics = {
        'Best Val RMSE': final_rmse,
        'Epochs Trained': epoch + 1,
        'Total Training Time': time_str,
        'Total Seconds': f"{total_time:.2f}",
    }
    final_metrics.update(emissions_data)

    manager.save_metrics(final_metrics)
    
    print("✅ Experimento finalizado e salvo com sucesso!")

if __name__ == "__main__":
    main()