import pandas as pd
import os
import sys

# Adiciona o diretório atual ao path para importar o módulo src
sys.path.append(os.getcwd())

# Importa a classe atualizada
from src.statistical_metrics import SolarStatisticalAnalyzer

# Configurações
INPUT_FILE = "analysis_outputs/TABELA_COMPARATIVA_FINAL.csv"
OUTPUT_DIR = "analysis_outputs/relatorios_finais"

def load_and_transform_data(file_path):
    """
    Transforma o CSV 'Largo' (Wide) em 'Longo' (Long) para análise.
    Entrada: [Timestamp, Observado_kW, Persistencia_kW, Pred_ModeloA, Pred_ModeloB...]
    Saída: [Timestamp, Observado, Persistencia, Previsto, Modelo]
    """
    if not os.path.exists(file_path):
        print(f"❌ Arquivo não encontrado: {file_path}")
        return None

    df_wide = pd.read_csv(file_path)
    
    # Conversão de Data
    if 'Timestamp' in df_wide.columns:
        df_wide['Timestamp'] = pd.to_datetime(df_wide['Timestamp'])
    else:
        print("❌ Erro: Coluna 'Timestamp' não encontrada.")
        return None

    # Identificar colunas de modelos (começam com Pred_)
    model_cols = [c for c in df_wide.columns if c.startswith('Pred_')]
    
    if not model_cols:
        print("⚠️ Nenhuma coluna de modelo encontrada.")
        return None

    print(f"📚 Modelos encontrados: {[m.replace('Pred_', '') for m in model_cols]}")

    # Transformação (Melt)
    # Vamos criar uma lista de DataFrames e concatenar (mais eficiente que melt complexo as vezes)
    dfs_long = []
    
    for m_col in model_cols:
        model_name = m_col.replace('Pred_', '')
        
        # Cria subset
        sub = df_wide[['Timestamp', 'Observado_kW', 'Persistencia_kW', m_col]].copy()
        
        # Renomeia para padrão da classe
        sub.rename(columns={
            'Observado_kW': 'Observado',
            'Persistencia_kW': 'Persistencia',
            m_col: 'Previsto'
        }, inplace=True)
        
        sub['Modelo'] = model_name
        dfs_long.append(sub)
    
    # Junta tudo
    df_long = pd.concat(dfs_long, ignore_index=True)
    
    return df_long

def main():
    print("🚀 Iniciando Geração de Relatórios Estatísticos...")
    
    # 1. Prepara Dados
    df_long = load_and_transform_data(INPUT_FILE)
    
    if df_long is None or df_long.empty:
        print("❌ Falha ao processar dados.")
        return

    # 2. Inicializa Analisador
    analyzer = SolarStatisticalAnalyzer(df_long, OUTPUT_DIR)
    
    # 3. Gera Relatórios
    print("\n📊 Calculando Métricas Globais...")
    analyzer.save_global_metrics()
    
    print("\n📈 Gerando Gráficos...")
    analyzer.plot_error_by_hour_of_day()
    analyzer.plot_boxplots_hourly()
    analyzer.plot_scatter_hist()
    analyzer.plot_scenario_days()
    
    # Opcional (se tiver a lib instalada)
    analyzer.plot_taylor_diagram()
    
    print(f"\n✅ Tudo pronto! Verifique a pasta: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()