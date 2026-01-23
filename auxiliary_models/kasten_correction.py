import numpy as np
import pandas as pd

def Kasten_Correction(df):
    """
    Aplica o fator de correção de Kasten et al. (KA) aos dados de DHI.
    """
    # 1. Cria uma cópia para não alterar o original durante os cálculos intermédios
    df_calc = df.copy()
    
    # Constantes do modelo (da Imagem)
    A = 1.161
    B = -0.112
    C = 0.0009
    D = -0.0246
    
    # 2. Pré-cálculo de variáveis auxiliares
    # Garantir que não haja divisão por zero no G0h (ghi_extra)
    # Substituímos 0 por NaN temporariamente para evitar Infinitos, 
    # ou filtramos apenas dados diurnos. Aqui, aplicamos máscara onde ghi_extra > 0.
    
    mask_day = (df_calc['ghi_extra'] > 5) & (df_calc['dhi'] > 0) # Filtro básico dia
    
    # Inicializa colunas com NaN
    df_calc['k_du'] = np.nan
    df_calc['tau_bu'] = np.nan
    df_calc['f_correction'] = np.nan
    df_calc['dhi_corrected'] = df_calc['dhi'] # Padrão é o original
    
    # --- Cálculos apenas para o período diurno ---
    
    # k_du = G_du / G_0h
    df_calc.loc[mask_day, 'k_du'] = df_calc.loc[mask_day, 'dhi'] / df_calc.loc[mask_day, 'ghi_extra']
    
    # tau_bu = k_t - k_du
    # Nota: A imagem diz para restringir tau_bu >= 0.
    # Vamos calcular e depois aplicar o clip.
    tau_bu_temp = df_calc.loc[mask_day, 'kt'] - df_calc.loc[mask_day, 'k_du']
    df_calc.loc[mask_day, 'tau_bu'] = tau_bu_temp.clip(lower=0.0001, upper=0.9999) 
    # O clip(0.0001, 0.9999) é vital porque:
    # se tau_bu <= 0 -> ln(inverso) explode ou é inválido.
    # se tau_bu = 1 -> ln(1) = 0 -> Divisão por zero no termo D.

    # 3. Cálculo dos termos da equação (24)
    # Termo 1: B * (k_du / k_t)^3
    # Tratamento para k_t próximo de 0
    kt_safe = df_calc.loc[mask_day, 'kt'].replace(0, np.nan)
    term_cubic = B * (df_calc.loc[mask_day, 'k_du'] / kt_safe)**3
    
    # Termo 2: C * delta (Assumindo delta = elevation em graus)
    term_linear = C * df_calc.loc[mask_day, 'elevation']
    
    # Termo 3: D / ln(1 / tau_bu)
    # ln(1/x) é o mesmo que -ln(x)
    term_log = D / np.log(1 / df_calc.loc[mask_day, 'tau_bu'])
    
    # 4. Cálculo do Fator f
    f = A + term_cubic + term_linear + term_log
    
    df_calc.loc[mask_day, 'f_correction'] = f
    
    # 5. Aplicação da correção
    # DHI_corrigido = DHI_medido * f
    # Nota: O fator f geralmente é > 1 (pois a banda faz sombra e "esconde" parte do difuso).
    df_calc.loc[mask_day, 'dhi_corrected'] = df_calc.loc[mask_day, 'dhi'] * f
    
    return df_calc