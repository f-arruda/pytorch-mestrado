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
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger, CSVLogger
from core.lightning_wrapper import SolarLightningModule

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

from core.models.encdec_model import EncDecModel
# Loss Functions
from core.loss_function.cpiloss import CPILoss
from core.loss_function.mseloss import MaskedMSELoss
from core.loss_function.pyloss import PhysicsGuidedLoss
from core.loss_function.pyloss_pers import PhysicsGuidedLossPers

# ================= CONFIGURAÇÃO CENTRALIZADA =================
CONFIG = {
    # --- 1. PATH DA BASE DE DADOS ---
    'csv_path': 'data/pv0.csv',
    
    # --- 2. DIVISÃO DA BASE DE TREINO, TESTE E VALIDAÇÃO ---
    'split_ratios': {'train': 0.8, 'val': 0.2}, 
    'test_year':2019,

    # --- 3. PRÉ-PROCESSAMENTO (Física & Mapeamento) ---
    'preprocessing': {
        'latitude': -23.56,
        'longitude': -46.73,
        'altitude': 0,
        'timezone': 'Etc/GMT+3',
        'nominal_power': 156.0,
        'start_year': 2015,
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
    'prediction_mode': 'sky',
    
    # Qual variável o modelo vai prever? ('kt', 'fracao_difusa' ou 'target')
    'target_col': ['kt', 'fracao_difusa'], 

    # mascara de dados noturnos
    'use_mask':True,   # True = Ignora a noite (ideal para K)
                        # False = Aprende a noite (ideal para Potência)
    
    # Features de entrada
    'feature_cols': [
        'kt', 'fracao_difusa','elevation'
        #'cos_zenith', 'sin_azimuth', 'target',
        #'wind_speed', 'temp_amb', 'humidity'
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
    'model_type': 'teste_acesso_remoto',
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
    'loss_function':'physics_loss',      # "cpi_loss", "mse", physics_loss
    'physics_base_loss': 'cpi',
    'lambda_hard':100,
    'lambda_soft':100,
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

def main(config, preprocessor_class, dataset_class):
    # 1. Setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{timestamp}_{config['model_type']}"
    exp_dir = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    tracker = EmissionsTracker(output_dir=exp_dir, output_file="emissions.csv") if HAS_CODECARBON else None
    if tracker: tracker.start()

    print(f"🚀 Iniciando: {exp_name}")
    print(f"🎯 Modo: {config['prediction_mode']} | Alvo: {config['target_col']}")
    save_config(config, exp_dir)

    # Configuração Inicial do PL para MLflow
    mlf_logger = MLFlowLogger(
        experiment_name=f"SolarForecasting_{config['prediction_mode']}",
        tracking_uri="sqlite:///mlflow.db",
        run_name=exp_name
    )
    # Logando todos os hyperparameters na interface de comparações
    clean_config = {k: str(v) if isinstance(v, (list, dict)) else v for k, v in config.items() if not isinstance(v, dict)}
    mlf_logger.log_hyperparams(clean_config)
    
    # Criando o logger em disco para exportar a curva de aprendizado local
    csv_logger = CSVLogger(save_dir=os.path.dirname(exp_dir), name=os.path.basename(exp_dir))

    # 2. Leitura
    csv_path = config['csv_path']
    print(f"⏳ Lendo: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'Date_Time' in df.columns:
        df['Date_Time'] = pd.to_datetime(df['Date_Time'])
        df = df.drop_duplicates(subset=['Date_Time'], keep='first').set_index('Date_Time').sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # 3. Pré-processamento
    pp_conf = config['preprocessing']
    
    # Instancia passando o mapa explícito. 
    # Isso garante que a padronização aconteça conforme o CONFIG acima.
    preprocessor = preprocessor_class(
        latitude=pp_conf['latitude'], 
        longitude=pp_conf['longitude'], 
        altitude=pp_conf['altitude'],
        timezone=pp_conf['timezone'], 
        nominal_power=pp_conf['nominal_power'], 
        start_year=pp_conf['start_year'],
        features_to_scale=pp_conf['features_to_scale'],
        target_col=config['prediction_mode'], # <--- unica variavel que não vem do preprocessing
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
    target_col = config['target_col']

    for col in target_col:
        if col not in df_processed.columns:
            raise ValueError(f"❌ Coluna alvo '{target_col}' não encontrada! Verifique o column_mapping.")

    available_cols = [c for c in config['feature_cols'] if c in df_processed.columns]
    if len(available_cols) != len(config['feature_cols']):
        print(f"⚠️ Features ajustadas: {available_cols}")
        config['feature_cols'] = available_cols

    # 5. Split Temporal (Último Ano = Teste)
    #last_year = df_processed.index.year.max() <---- REMOVER FUTURAMENTE
    last_year = config['test_year']
    print(f"📅 Separando ano {last_year} para TESTE.")

    test_df = df_processed[df_processed.index.year == last_year].copy()
    dev_df = df_processed[df_processed.index.year < last_year].copy()

    if dev_df.empty:
        raise ValueError("❌ Erro no Split: Dados insuficientes antes do último ano.")

    # Split Treino/Validação
    n_dev = len(dev_df)
    train_end = int(n_dev * config['split_ratios']['train'])

    train_df = dev_df.iloc[:train_end].copy()
    val_df = dev_df.iloc[train_end:].copy()

    print(f"📊 Divisão: Treino={len(train_df)} | Val={len(val_df)} | Teste={len(test_df)}")

    train_dataset = dataset_class(
        df=train_df, 
        feature_cols=config['feature_cols'], 
        target_col=config['target_col'],
        aux_col=config['aux_col'],
        n_past=config['input_seq_len'], 
        n_future=config['output_seq_len']
    )
    val_dataset = dataset_class(
        df=val_df, 
        feature_cols=config['feature_cols'], 
        target_col=config['target_col'],
        aux_col=config['aux_col'],
        n_past=config['input_seq_len'], 
        n_future=config['output_seq_len']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 6. Modelo
    model = EncDecModel(
        input_size=len(config['feature_cols']),
        hidden_sizes=config['hidden_sizes'],
        output_seq_len=config['output_seq_len'],
        output_dim=len(config['target_col']),
        cell_type=config['cell_type'],
        bidirectional=config['bidirectional'],
        use_attention=config['use_attention'],
        use_feature_attention=config['use_feature_attention'],
        dropout_prob=config['dropout']
    ).to(DEVICE)

    # modificar o criterion muda a função de erro implementado a CPILoss
    if config['loss_function'] == 'mse':
        criterion = MaskedMSELoss()
    elif config['loss_function'] == 'cpi_loss':
        criterion = CPILoss()
    elif config['loss_function'] == 'physics_loss':
        criterion = PhysicsGuidedLoss(
            lambda_hard=config['lambda_hard'], 
            lambda_soft=config['lambda_soft'],
            data_loss_type=config['physics_base_loss']
        )
    elif config['loss_function'] == 'physics_loss_pers':
        criterion = PhysicsGuidedLossPers(
            lambda_hard=config['lambda_hard'], 
            lambda_soft=config['lambda_soft'],
            lambda_pers=config.get('lambda_pers', 0.1),
            data_loss_type=config['physics_base_loss']
        )

    # 7. Treino com PyTorch Lightning
    print("⚡ Orquestrando com PyTorch Lightning...")
    start_time = time.time()
    
    # O EarlyStopping agora roda pelo framework nativo do PyTorch Lightning
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-5,
        patience=config['patience'],
        verbose=True,
        mode="min"
    )

    # O Checkpoint rastreia onde guardar o estado de forma nativa
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir,
        filename="lightning_best",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    # 1. Empacotamos seu modelo neural e as funções focadas de erro físico no Invólucro
    lightning_model = SolarLightningModule(model, criterion, config)
    
    # 2. Delegamos o poder computacional e o rastreamento para o PL!
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='auto', # Procura sua CUDA ou TPU automaticamente
        logger=[mlf_logger, csv_logger],
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    trainer.fit(lightning_model, train_loader, val_loader)

    # ==========================================
    # 8. Compatibilidade Retroativa (Extrator)
    # ==========================================
    # Os seus frameworks passados de Avaliação procuravam o State_Dict puro do PyTorch!
    # Não vamos modificar a camada de Analises: vamos extrair do PL e salvar para eles puramente:
    
    print(f"🔄 Convertendo Lightning Checkpoint para Torch padrão (.pt)")
    best_model_path = checkpoint_callback.best_model_path
    
    if os.path.exists(best_model_path):
        # Carragamos o state_dict isolado de dentro da fôrma do Lightning baseada na última validação
        best_lightning_model = SolarLightningModule.load_from_checkpoint(best_model_path, model=model, criterion=criterion, config=config)
        
        torch.save(best_lightning_model.model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))
        print(f"✅ Compatibilidade assegurada. 'best_model.pt' salvo.")

    # ==========================================
    # 9. Reconstrução dos Relatórios Clássicos
    # ==========================================
    # PL já gerou o metrics.csv internamente. Vamos parsear e devolver os relatorios fisicos que as bibliotecas e Jupyters antigos procuram:
    metrics_csv = os.path.join(csv_logger.log_dir, 'metrics.csv')
    if os.path.exists(metrics_csv):
        df_metrics = pd.read_csv(metrics_csv)
        # PL agrupa loggings por steps misturados (uns em train outros em val). O ffill alinha todos pela Época
        df_metrics = df_metrics.groupby('epoch').last().reset_index()
        
        # 1. Recria o saudoso log_xai.csv pro usuário não sentir falta (Aviso: colunas de erro carregarão val_... ou train_...)
        save_training_log(df_metrics.to_dict(orient='records'), os.path.join(exp_dir, 'log_xai.csv'))
        
        # 2. Recria a Plotagem estática das curvas de Aprendizagem
        if 'train_loss' in df_metrics.columns and 'val_loss' in df_metrics.columns:
            plot_learning_curve(df_metrics['train_loss'].tolist(), df_metrics['val_loss'].tolist(), os.path.join(exp_dir, 'learning_curve.png'))
            # Sobemos a foto final para a nuvem MLFlow também:
            mlf_logger.experiment.log_artifact(mlf_logger.run_id, os.path.join(exp_dir, 'learning_curve.png'))
            print("📈 Curva de Aprendizado e `log_xai.csv` reconstruídos a partir do core do Lightning!")

    # Registra e salva os arquivos CSV extras brutos gerados pela base PL Logger
    metrics_path = os.path.join(exp_dir, 'metrics.json')
    final_metrics = {
        "best_val_loss": float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
        "epochs": trainer.current_epoch,
        "time_sec": time.time() - start_time,
        "model_path": os.path.join(exp_dir, 'best_model.pt')
    }
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)

    # Fechamentos de memória Tracker Emisões
    if tracker: tracker.stop()

    print(f"✅ Fim. Modelo e Dados salvos em: {exp_dir}")

if __name__ == "__main__":
    pass
