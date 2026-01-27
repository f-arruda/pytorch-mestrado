import torch
import pandas as pd
import numpy as np
import os
import json
import ast
import warnings
from torch.utils.data import DataLoader

# ================= CONFIGURAÇÃO =================
EXPERIMENTS_DIRS = [
    "trained_models/2026-01-27_15-42-44_Teste_k",
    "trained_models/2026-01-27_16-19-48_Teste_k_lambda_high",
]

OUTPUT_FILE = "analysis_outputs/TABELA_COMPARATIVA_FINAL.csv"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs("analysis_outputs", exist_ok=True)
warnings.filterwarnings('ignore')

# Importações do projeto
from models.encdec_model import EncDecModel
from src.dataset_module import SolarEfficientDataset
from src.preprocessing import SolarPreprocessor
from src.postprocessing import get_strategy

# ================= FUNÇÕES AUXILIARES =================

def get_config(exp_dir):
    path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config não achado em {exp_dir}")
    with open(path, 'r') as f:
        return json.load(f)

def get_preprocessing_params(config):
    pp = config.get('preprocessing', {})
    return {
        'latitude': pp.get('latitude', config.get('latitude', -15.60)),
        'longitude': pp.get('longitude', config.get('longitude', -47.70)),
        'altitude': pp.get('altitude', config.get('altitude', 0)),
        'timezone': pp.get('timezone', config.get('timezone', 'Etc/GMT+3')),
        'nominal_power': pp.get('nominal_power', config.get('nominal_power', 156.0)),
        'target_col': config.get('target_col', 'target'),
        'column_mapping': pp.get('column_mapping', config.get('column_map', None)),
        'start_year': pp.get('start_year', config.get('start_year', 2018)),
        'features_to_scale': pp.get('features_to_scale', config.get('features_to_scale', []))
    }

def process_model(exp_dir):
    print(f"\n📂 Processando: {os.path.basename(exp_dir)}")
    
    # 1. Configs e Preprocessor (Igual ao Notebook)
    config = get_config(exp_dir)
    pp_params = get_preprocessing_params(config)
    
    preprocessor = SolarPreprocessor(
        latitude=pp_params['latitude'], longitude=pp_params['longitude'], 
        altitude=pp_params['altitude'], timezone=pp_params['timezone'], 
        nominal_power=pp_params['nominal_power'], start_year=pp_params['start_year'],
        features_to_scale=pp_params['features_to_scale'], target_col=pp_params['target_col'], 
        column_mapping=pp_params['column_mapping']
    )

    # 2. Carga de Dados
    print("   ⏳ Carregando dados...")
    df_raw = pd.read_csv('data/pv0.csv')
    if 'Date_Time' in df_raw.columns:
        df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time'])
        df_raw = df_raw.drop_duplicates(subset=['Date_Time'], keep='first').set_index('Date_Time').sort_index()

    try: preprocessor.load_scalers(exp_dir)
    except: preprocessor.fit(df_raw)
    
    df_processed = preprocessor.transform(df_raw)

    # 3. Filtro de Teste
    test_year = config.get('test_year', df_processed.index.year.max())
    df_test = df_processed[df_processed.index.year == test_year].copy()
    
    if df_test.empty:
        print(f"   ❌ Erro: Sem dados para {test_year}")
        return None, None

    # 4. Dataset
    n_past = config['input_seq_len']
    n_future = config['output_seq_len']
    
    dataset = SolarEfficientDataset(
        df=df_test.copy(),
        feature_cols=config['feature_cols'], 
        target_col=config['target_col'], 
        n_past=n_past,
        n_future=n_future
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    # 5. Modelo
    model = EncDecModel(
        input_size=len(config['feature_cols']),
        hidden_sizes=config['hidden_sizes'],
        output_seq_len=n_future,
        output_dim=1,
        cell_type=config.get('cell_type', 'lstm'),
        bidirectional=config.get('bidirectional', False),
        use_attention=config.get('use_attention', False),
        use_feature_attention=config.get('use_feature_attention', False)
    ).to(DEVICE)
    
    w_path = os.path.join(exp_dir, 'best_model.pt')
    if not os.path.exists(w_path): w_path = os.path.join(exp_dir, 'best_model.pth')
    model.load_state_dict(torch.load(w_path, map_location=DEVICE), strict=False)
    model.eval()

    # 6. Inferência
    print("   🔮 Gerando previsões...")
    preds_list = []
    valid_mask_list = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_mask in loader:
            out = model(batch_x.to(DEVICE))
            preds_list.append(out.cpu().numpy())
            valid_mask_list.append(batch_mask.cpu().numpy())
            
    # Concatena tudo: Shape (N_samples, Seq_Len, 1)
    y_pred_all = np.concatenate(preds_list, axis=0)
    mask_all = np.concatenate(valid_mask_list, axis=0)

    # 7. Conversão para Lista de Escalares (Lógica do Notebook)
    # Extrai apenas o primeiro passo da previsão (Horizonte 1) para alinhar com o index seguinte
    pred_scalars = []
    for i in range(len(y_pred_all)):
        # Pega o valor do tensor. Se for sequencia, pega o primeiro [0]
        val = y_pred_all[i]
        if isinstance(val, (np.ndarray, list)) and len(val) > 0:
            val = val[0] # Pega Horizonte 1
        if hasattr(val, 'item'):
            val = val.item()
        pred_scalars.append(float(val))

    # ==============================================================================
    # 8. Alinhamento Exato (CORREÇÃO PARA FILTRO NOTURNO)
    # ==============================================================================
    # Como o dataset pulou a noite, não podemos apenas fazer slice no df_test.
    # Precisamos recuperar os timestamps exatos que o dataset selecionou.
    
    # Recupera os índices reais que o dataset considerou "Dia" e usou para previsão
    valid_positions = dataset.valid_indices
    
    # Trava de segurança para garantir tamanhos iguais
    n_preds = len(pred_scalars)
    n_inds = len(valid_positions)
    
    if n_preds != n_inds:
        print(f"⚠️ Sincronizando: Previsões ({n_preds}) vs Índices ({n_inds})")
        min_len = min(n_preds, n_inds)
        pred_scalars = pred_scalars[:min_len]
        valid_positions = valid_positions[:min_len]

    # 1. Mapeia as previsões para os Timestamps corretos (pula a noite no index)
    aligned_dates = df_test.index[valid_positions]
    
    # 2. Cria dataframe auxiliar indexado pela data correta
    df_preds_only = pd.DataFrame({'pred_norm': pred_scalars}, index=aligned_dates)
    
    # 3. Faz o Join mantendo apenas os horários que existem na previsão
    df_aligned = df_test.join(df_preds_only, how='inner')

    # ==============================================================================
    # 9. Desnormalização e Montagem da Tabela
    # ==============================================================================
    nominal_power = pp_params['nominal_power']
    target_col = config.get('target_col', 'target')
    
    records = []
    
    # Iteramos sobre o df_aligned que agora tem APENAS os momentos válidos (dia)
    for idx, row in df_aligned.iterrows():
        # Recupera valores normalizados
        obs_norm = row.get(target_col, 0.0)
        pred_norm = row['pred_norm']
        p1_norm = row.get('P1', 0.0)
        
        # Lógica de reconstrução (se target for 'k', multiplica pelo clear sky)
        if target_col == 'k':
            pot_cs = row.get('pot_cs', 0.0)
            # Se for k, a previsão é k * pot_cs
            obs_kw = obs_norm * pot_cs
            pred_kw = pred_norm * pot_cs
            pers_kw = p1_norm * pot_cs # P1 aqui já deve ser k_lag1
        else:
            # Se for target direto (kW normalizado), multiplica pela potência nominal
            obs_kw = obs_norm * nominal_power
            pred_kw = pred_norm * nominal_power
            pers_kw = p1_norm * nominal_power
            
        records.append({
            'Timestamp': idx,
            'Observado_kW': obs_kw,
            'Persistencia_kW': pers_kw,
            'Modelo_Pred_kW': pred_kw
        })

    # ==============================================================================
    # 10. Criação do DataFrame e Filtro Final
    # ==============================================================================
    is_day_vector = mask_all.flatten().astype(bool)

    # 1. Cria o DataFrame completo (24h) com os resultados brutos
    df_full = pd.DataFrame(records).set_index('Timestamp')

    # 2. Garante que o vetor de máscara tenha o mesmo tamanho do DataFrame
    # (Trava de segurança caso tenha havido algum corte no passo 8)
    if len(is_day_vector) > len(df_full):
        is_day_vector = is_day_vector[:len(df_full)]
    elif len(is_day_vector) < len(df_full):
        # Caso raro, mas se acontecer, cortamos o DF para evitar erro
        df_full = df_full.iloc[:len(is_day_vector)]

    # 3. Aplica o filtro: Mantém apenas as linhas onde is_day_vector == True (Dia)
    # Como is_day_vector é um array booleano alinhado, podemos usar diretamente
    df_day_only = df_full[is_day_vector]

    print(f"   ✂️  Filtro Aplicado: {len(df_full)} linhas (24h) -> {len(df_day_only)} linhas (Dia)")

    return df_day_only, os.path.basename(exp_dir)

# ================= MAIN =================
def main():
    if not EXPERIMENTS_DIRS: return
    
    print("🚀 Iniciando Consolidação...")
    df_master = None
    
    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir): continue
        
        try:
            df_res, model_name = process_model(exp_dir)
            if df_res is None: continue 
            
            if df_master is None:
                df_master = df_res[['Observado_kW', 'Persistencia_kW']].copy()
            
            col_name = f"Pred_{model_name}"
            df_master[col_name] = df_res['Modelo_Pred_kW']
            print(f"   ✅ Adicionado: {col_name}")

        except Exception as e:
            print(f"   ❌ Erro: {e}")
            import traceback
            traceback.print_exc()

    if df_master is not None:
        print(f"\n💾 Salvando: {OUTPUT_FILE}")
        df_master.to_csv(OUTPUT_FILE)
        print("✅ Tabela gerada! Confira os valores no CSV.")
        print(df_master.head())
    else:
        print("⚠️ Nenhum dado gerado.")

if __name__ == "__main__":
    main()