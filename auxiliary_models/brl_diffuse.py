import numpy as np
import pandas as pd
import pvlib

def BRL(df, betas, longitude, timezone, limite_zenith=87.0):
    dff = df.copy()
    
    # 1. Ajuste de Tempo
    if dff.index.tz is None:
        times = dff.index.tz_localize(timezone)
    else:
        times = dff.index.tz_convert(timezone)

    # ---------------------------------------------------------
    # CORREÇÃO AQUI: Extrair o .dayofyear antes de passar pro pvlib
    # ---------------------------------------------------------
    # O erro acontecia porque 'times' é uma Data completa.
    # A função quer apenas o número do dia (ex: 1, 50, 365).
    doy = times.dayofyear
    
    # Agora passamos apenas os números inteiros para a função do pvlib
    eot = pvlib.solarposition.equation_of_time_spencer71(doy)
    
    # 2. Cálculo do AST (Apparent Solar Time)
    # Precisamos converter para UTC para somar a longitude corretamente
    utc_times = times.tz_convert('UTC')
    utc_dec = utc_times.hour + utc_times.minute/60.0 + utc_times.second/3600.0
    
    # AST = UTC + Lon/15 + EoT/60
    # Nota: Se eot vier como Series, o cálculo flui normal
    ast_calc = utc_dec + (longitude / 15.0) + (eot / 60.0)
    dff['AST_calc'] = ast_calc % 24

    # 3. Cálculo do KT Diário (Ponderado pela ghi_extra)
    ghi_estimada = dff['kt'] * dff['ghi_extra']
    
    # Agrupamento pela data local
    datas_locais = pd.Series(times.date, index=dff.index) 
    
    soma_ghi_dia = ghi_estimada.groupby(datas_locais).transform('sum')
    soma_extra_dia = dff['ghi_extra'].groupby(datas_locais).transform('sum')
    
    # Evita divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        kt_diario = soma_ghi_dia / soma_extra_dia
    
    dff['KT_diario'] = np.where(soma_extra_dia > 1.0, kt_diario, np.nan)

    # 4. Cálculo do PSI (Persistência)
    is_day = dff['apparent_zenith'] < limite_zenith
    is_day_prev = is_day.shift(1).fillna(False)
    is_day_next = is_day.shift(-1).fillna(False)
    
    kt_prev = dff['kt'].shift(1)
    kt_next = dff['kt'].shift(-1)
    
    psi = pd.Series(np.nan, index=dff.index)
    mask_mid = is_day & is_day_prev & is_day_next
    mask_sunrise = is_day & (~is_day_prev)
    mask_sunset = is_day & (~is_day_next)
    
    psi.loc[mask_mid] = 0.5 * (kt_prev + kt_next)
    psi.loc[mask_sunrise] = kt_next
    psi.loc[mask_sunset] = kt_prev
    dff['psi_calc'] = psi

    # 5. Modelo BRL (Regressão Logística)
    b = betas
    Z = b[0] + \
        (b[1] * dff['kt']) + \
        (b[2] * dff['AST_calc']) + \
        (b[3] * dff['elevation']) + \
        (b[4] * dff['KT_diario']) + \
        (b[5] * dff['psi_calc'])
    
    dff['fracao_difusa_brl'] = 1 / (1 + np.exp(Z))
    dff.loc[~is_day, ['fracao_difusa_brl', 'psi_calc', 'KT_diario']] = np.nan
    
    return dff[['AST_calc', 'KT_diario', 'psi_calc', 'fracao_difusa_brl']]