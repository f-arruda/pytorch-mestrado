from xml.parsers.expat import model

import torch
import pandas as pd
import numpy as np
import os
import json
import ast
import warnings
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.analysis_power import PowerAnalyzer
from core.utils.xai import SolarXAIEngine

# ================= CONFIGURAÇÃO =================
EXPERIMENTS_DIRS = [
    "trained_models/2026-03-05_17-31-24_LSTM_PAZ",
    "trained_models/2026-03-05_17-38-28_LSTM_PAZVTU"
]

OUTPUT_FILE = "analysis_outputs/Potencia/"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs("analysis_outputs", exist_ok=True)
warnings.filterwarnings('ignore')

# Importações do projeto
from core.models.encdec_model import EncDecModel
from domains.previsao_potencia.dataset import SolarEfficientDataset
from domains.previsao_potencia.preprocessing import SolarPreprocessor
from domains.previsao_potencia.postprocessing import get_strategy

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
    
    # 1. Configurações e Preprocessor
    config = get_config(exp_dir)
    pp_params = get_preprocessing_params(config)
    
    preprocessor = SolarPreprocessor(
        latitude=pp_params['latitude'], longitude=pp_params['longitude'], 
        altitude=pp_params['altitude'], timezone=pp_params['timezone'], 
        nominal_power=pp_params['nominal_power'], start_year=pp_params['start_year'],
        features_to_scale=pp_params['features_to_scale'], target_col=config['prediction_mode'], 
        column_mapping=pp_params['column_mapping'],
        cs_model = 'esra',
        kasten_corr=True
    )

    # 2. Carga de Dados
    print("   ⏳ Carregando dados...")
    df_raw = pd.read_csv('data/pv0.csv')
    if 'Date_Time' in df_raw.columns:
        df_raw['Date_Time'] = pd.to_datetime(df_raw['Date_Time'])
        df_raw = df_raw.drop_duplicates(subset=['Date_Time'], keep='first').set_index('Date_Time').sort_index()

    try: 
        preprocessor.load_scalers(exp_dir)
    except: 
        preprocessor.fit(df_raw)
    
    df_processed = preprocessor.transform(df_raw)

    # 3. Filtro de Teste
    test_year = config.get('test_year', 2022)
    df_test = df_processed[df_processed.index.year == test_year].copy()
    
    if df_test.empty:
        print(f"   ❌ Erro: Sem dados para {test_year}")
        return None, None

    # 4. Dataset (Configurado para retornar x, y, mask e aux)
    n_past = config['input_seq_len']
    n_future = config['output_seq_len']
    
    dataset = SolarEfficientDataset(
        df=df_test.copy(),
        feature_cols=config['feature_cols'], 
        target_col=config['target_col'],
        aux_col=config['aux_col'], 
        n_past=n_past,
        n_future=n_future
    )
    # Batch size alto para acelerar a inferência no teste
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    # 5. Inicialização do Modelo (output_dim dinâmico)
    model = EncDecModel(
        input_size=len(config['feature_cols']),
        hidden_sizes=config['hidden_sizes'],
        output_seq_len=n_future,
        output_dim=len(config['target_col']), # Agora aceita multi-output
        cell_type=config.get('cell_type', 'gru'),
        bidirectional=config.get('bidirectional', False),
        use_attention=config.get('use_attention', False),
        use_feature_attention=config.get('use_feature_attention', False)
    ).to(DEVICE)
    
    # Carrega os pesos
    w_path = os.path.join(exp_dir, 'best_model.pt')
    if not os.path.exists(w_path): w_path = os.path.join(exp_dir, 'best_model.pth')
    model.load_state_dict(torch.load(w_path, map_location=DEVICE), strict=False)
    model.eval()

    # 6. Inferência Multi-Output
    print("   🔮 Gerando previsões...")
    preds_list, reals_list, aux_list = [], [], []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_mask, batch_aux in loader:
            out = model(batch_x.to(DEVICE))
            
            # Guardamos as predições, os reais e os metadados físicos
            preds_list.append(out.cpu().numpy())
            reals_list.append(batch_y.cpu().numpy())
            aux_list.append(batch_aux.cpu().numpy())

    # Concatenamos os resultados: Shape esperado [N_amostras, Seq_Futura, Variaveis]
    y_pred_all = np.concatenate(preds_list, axis=0)
    y_true_all = np.concatenate(reals_list, axis=0)
    aux_all = np.concatenate(aux_list, axis=0)

    # 7. Alinhamento e Montagem do DataFrame de Análise
    # Extraímos apenas o Horizonte 1 (índice 0 da sequência futura)
    valid_positions = dataset.valid_indices
    aligned_dates = df_test.index[valid_positions]

    # Criamos o DataFrame com kt e kd (Reais e Preditos)
    # Assumindo a ordem do target_col no CONFIG: [kt, fracao_difusa]
    df_analysis = pd.DataFrame({
        'power_real': y_true_all[:, 0, 0],
        'power_pred': y_pred_all[:, 0, 0],
        # Metadados físicos fundamentais capturados pelo aux_future
        #'ghi_cs': aux_all[:, 0, 0],
        #'cos_zenith': aux_all[:, 0, 1],
        #'elevation': aux_all[:, 0, 2],
        #'ghi_extra': aux_all[:, 0, 3]
    }, index=aligned_dates)

    # 8. Cálculos Físicos para Validação (W/m2)
    # GHI = kt * GHI_extra
    #df_analysis['ghi_real'] = df_analysis['kt_real'] * df_analysis['ghi_extra']
    #df_analysis['ghi_pred'] = df_analysis['kt_pred'] * df_analysis['ghi_extra']
    
    # DHI = kd * GHI
    #df_analysis['dhi_real'] = df_analysis['kd_real'] * df_analysis['ghi_real']
    #df_analysis['dhi_pred'] = df_analysis['kd_pred'] * df_analysis['ghi_pred']

    print(f"   ✅ DataFrame de análise gerado com {len(df_analysis)} timesteps diurnos.")

    df_analysis = pd.merge(df_analysis, df_processed[['P1']], left_index=True, right_index=True, how='inner')

    return df_analysis, df_test, os.path.basename(exp_dir), model, loader, dataset

# ================= MAIN =================
def main():
    if not EXPERIMENTS_DIRS: return
    
    print("🚀 Iniciando Consolidação...")
    
    
    for exp_dir in EXPERIMENTS_DIRS:
        if not os.path.exists(exp_dir): continue
        
        try:
            # 1. Gera o DataFrame consolidado (kt, kd, real, pred, aux)
            df_res, df_test, model_name, model, loader, dataset = process_model(exp_dir)
            if df_res is None: continue 

            
            #==================================
            #   --- Analise das Previsões ---
            #==================================
            
            # 2. INSTÂNCIA DO NOVO ANALYZER
            # Passamos o nome do modelo para criar a pasta organizada
            k_analyzer = PowerAnalyzer(model_name=model_name, output_dir="analysis_outputs")
            
            # 3. CHAMADA DA FIGURA DE 3 PLOTS
            # Esta função vai gerar o arquivo .png com os dados de teste
            k_analyzer.plot_clear_sky_day_analysis(df_res)
            k_analyzer.plot_high_variability_day_analysis(df_res)
            k_analyzer.plot_overcast_day_analysis(df_res)
            k_analyzer.calculate_statistical_metrics(df_res)
            k_analyzer.plot_scatter_validation(df_res)
            k_analyzer.plot_transient_day_analysis(df_res)

            #=================================
            #   --- Explainable AI (XAI) ---
            #=================================
            
            xai_engine = SolarXAIEngine(model, DEVICE)
            xai_dir = os.path.join("analysis_outputs", "XAI", model_name)
            os.makedirs(xai_dir, exist_ok=True)

            config = get_config(exp_dir)
            feature_names = config['feature_cols']

            # --- NÍVEL 1: Importância Global (Feature Ablation) ---
            xai_engine.compute_global_feature_importance(
                loader=loader,
                feature_names=feature_names,
                save_path=os.path.join(xai_dir, "global_feature_importance.png")
            )

            # --- NÍVEL 2: Importância Temporal (Integrated Gradients) ---
            xai_engine.compute_temporal_importance(
                dataloader=loader, 
                feature_names=feature_names,
                save_path=os.path.join(xai_dir, "temporal_importance_heatmap.png")
            )

            # --- NÍVEL 4: XAI de Confiabilidade Física (Local IG) ---
            print("      ⚖️ Analisando falhas de conformidade física...")
        
            df_res['error_physics'] = (df_res['power_real'] - df_res['power_pred']).abs()
            worst_moments = df_res.nlargest(5, 'error_physics').index
            
            # Cria a lista de datas na exata ordem em que o dataset as armazena logicamente
            dataset_dates = df_test.index[dataset.valid_indices]
            
            for i, timestamp in enumerate(worst_moments):
                # Encontra o índice LÓGICO (0 a N) onde essa data se encontra no dataset
                logical_idx = np.where(dataset_dates == timestamp)[0][0]
                
                # Passa o índice lógico. O próprio dataset vai usar o valid_indices internamente
                x_sample = dataset[logical_idx][0] 
                
                xai_engine.get_local_explanation(
                    x=torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0).to(DEVICE),
                    target_idx=0, 
                    feature_names=feature_names,
                    save_path=os.path.join(xai_dir, f"local_explanation_fail_{i}_{timestamp.strftime('%Y%m%d_%H%M')}.png")
                )

            print(f"   ✅ Diagnóstico de XAI concluído. Arquivos em: {xai_dir}")

        except Exception as e:
            print(f"   ❌ Erro: {e}")
            import traceback
            traceback.print_exc()

        if df_res is not None:
            path = OUTPUT_FILE + f'{exp_dir.split('/')[1]}/tabela.csv'
            print(f"\n💾 Salvando: {path}")
            df_res.to_csv(path)
            print("✅ Tabela gerada! Confira os valores no CSV.")
            print(df_res.head())
        else:
            print("⚠️ Nenhum dado gerado.")

if __name__ == "__main__":
    main()