import pandas as pd
import numpy as np
import os
import joblib
import warnings
from typing import Optional, Dict, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

from scipy.optimize import curve_fit
from scipy.optimize import minimize_scalar

from cs_model.esra import ESRA

from auxiliary_models.kasten_correction import Kasten_Correction

# Tenta importar pvlib
try:
    import pvlib
except ImportError:
    raise ImportError("A biblioteca 'pvlib' é obrigatória. Instale: pip install pvlib")

warnings.filterwarnings('ignore')

# --- CONSTANTES GLOBAIS ---
CONSTANTS = {
    'G_STC': 1000.0,
    'T_STC': 25.0,
    'GAMMA_SI': 0.0045,
    'U0_BOUNDS': (23.5, 26.5), 
    'U1_BOUNDS': (6.25, 7.68),
    'DEFAULT_U0': 25.0,
    'DEFAULT_U1': 6.84
}

# Mapeamento Padrão
DEFAULT_MAPPING = {
    'Temperatura ambiente °C': 'temp_amb',
    'Velocidade média do vento m/s': 'wind_speed',
    'ghi': 'ghi',
    'Pot_BT': 'target',
    'Irradiação Global horária(Inclinada 27°) kWh/m2': 'irrad_poa',
    'Umidade Relativa %': 'humidity',
    'Year': 'year',
    'Date_Time': 'date_time',
    'dhi': 'dhi',
    'dni': 'dni',
    'zenith': 'zenith',
    'azimuth': 'azimuth'
}

class SolarPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 latitude: float, 
                 longitude: float, 
                 altitude: float = 0, 
                 timezone: str = 'UTC', 
                 nominal_power: float = 156.0, 
                 start_year: int = 2018,
                 cs_model: str = 'esra',
                 degradation_rate: float = 0.05,
                 column_mapping: Optional[Dict[str, str]] = None,
                 features_to_scale: Optional[List[str]] = None,
                 target_col: str = 'power',
                 kasten_corr: bool = False,
                 auto_identify_thermal_params: bool = True):
        
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        self.location = pvlib.location.Location(latitude, longitude, timezone, altitude)
        
        self.nominal_power = nominal_power
        self.start_year = start_year
        self.degradation_rate = degradation_rate
        self.cs_model = cs_model
        
        self.column_mapping = DEFAULT_MAPPING.copy()
        if column_mapping:
            self.column_mapping.update(column_mapping)
            
        self.target_col_internal = target_col
        self.features_to_scale = features_to_scale 
        
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.u0 = CONSTANTS['DEFAULT_U0']
        self.u1 = CONSTANTS['DEFAULT_U1']
        self.auto_identify = auto_identify_thermal_params
        self._is_fitted = False
        self.kasten = kasten_corr

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renomeia colunas usando o mapeamento."""
        rename_dict = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        return df.rename(columns=rename_dict)

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Garante índice temporal, trata fuso e remove duplicatas."""
        df = df.copy()
        
        # 1. Identificar coluna de data
        col_data = None
        possible_names = ['date_time', 'Date_Time', 'Date', 'data', 'datetime']
        inv_map = {v: k for k, v in self.column_mapping.items()}
        if 'date_time' in inv_map: possible_names.insert(0, inv_map['date_time'])

        for col in possible_names:
            if col in df.columns:
                col_data = col
                break
        
        # Se achou coluna, define como index
        if col_data and not pd.api.types.is_datetime64_any_dtype(df.index):
            df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
            df.set_index(col_data, inplace=True)
        
        # Converte índice se não for datetime ainda
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
            except:
                pass

        # 2. Remover NaT antes do fuso
        if df.index.isna().any():
            df = df[df.index.notna()]

        # 3. Tratamento de Fuso Horário
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize(self.timezone, ambiguous='NaT', nonexistent='NaT')
            else:
                df.index = df.index.tz_convert(self.timezone)
        except Exception as e:
            print(f"⚠️ Erro Fuso: {e}")

        # 4. Limpeza Final (Remove Duplicatas geradas por fuso ou originais)
        # Remove NaTs gerados pelo fuso
        if df.index.isna().any():
            df = df[df.index.notna()]

        # Remove duplicatas explicitamente
        if df.index.duplicated().any():
            print(f"⚠️ Removendo {df.index.duplicated().sum()} duplicatas no índice.")
            df = df[~df.index.duplicated(keep='first')]
            
        return df.sort_index()

    def _fit_thermal_parameters(self, df: pd.DataFrame):
        """Otimiza U0 e U1. Inclui trava de segurança contra duplicatas."""
        required = [self.target_col_internal, 'ghi', 'temp_amb', 'wind_speed']
        if not all(col in df.columns for col in required):
            return

        # --- TRAVA DE SEGURANÇA CRÍTICA ---
        # Garante que não existem duplicatas antes da operação matemática
        df = df.loc[~df.index.duplicated(keep='first')].copy()
        df = df.sort_index()
        # ----------------------------------

        # Filtra dados de sol pleno
        try:
            mask = (df['ghi'] > 300) & (df['target'] > 0) & (df['wind_speed'] >= 0)
            df_fit = df.loc[mask].dropna()
        except ValueError as e:
            print(f"⚠️ Erro de índice duplicado no fit: {e}. Pulando otimização.")
            return

        if len(df_fit) < 50: return

        def physical_power_model(X, u0, u1):
            ghi, temp, wind = X
            term_vento = u0 + u1 * wind
            t_cell = temp + (ghi / term_vento)
            efficiency_loss = 1 - CONSTANTS['GAMMA_SI'] * (t_cell - CONSTANTS['T_STC'])
            return self.nominal_power * (ghi / CONSTANTS['G_STC']) * efficiency_loss

        X_data = (df_fit['ghi'].values, df_fit['temp_amb'].values, df_fit['wind_speed'].values)
        Y_data = df_fit['target'].values

        try:
            bounds = ([CONSTANTS['U0_BOUNDS'][0], CONSTANTS['U1_BOUNDS'][0]], 
                      [CONSTANTS['U0_BOUNDS'][1], CONSTANTS['U1_BOUNDS'][1]])
            
            popt, _ = curve_fit(physical_power_model, X_data, Y_data, 
                                p0=[self.u0, self.u1], bounds=bounds, method='trf')
            self.u0, self.u1 = popt
            print(f"🌡️  U0={self.u0:.2f}, U1={self.u1:.2f}")
        except:
            pass

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df = self._rename_columns(df)
        df = self._ensure_datetime_index(df) # Limpa duplicatas aqui

        if self.auto_identify & (self.target_col_internal == 'power'):
            self._fit_thermal_parameters(df) # E limpa de novo dentro, por segurança

        cols_x = [c for c in self.features_to_scale if c in df.columns]
        if cols_x: self.scaler_x.fit(df[cols_x])
            

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted: raise RuntimeError("Fit necessário antes do Transform.")
        
        df = X.copy()
        df = self._rename_columns(df)
        df = self._ensure_datetime_index(df)

        df = self._clean_index(df)
        df = self._calculate_solar_position(df)
       
        if self.cs_model == 'perez':
            df = self._is_clear(df)     # verifica se céu é claro ou não
            df = self._otimization_LT(df)   # otimiza o valor da turbidez 

        df = self._calculate_physics_features(df)
        
        if self.target_col_internal == 'sky':
            
            # Correção de irradiação difusa com anel de sombreamento usando o modelo de Kasten
            if self.kasten == True:
                dados_corrigidos = Kasten_Correction(df)
                df = df.drop(columns=['dhi', 'dni','fracao_difusa']).copy()

                df['dhi'] = dados_corrigidos['dhi_corrected']
                df['fracao_difusa'] = df['dhi'] / df['ghi']
                df['dni'] = (df['ghi'] - df['dhi'])/df['cos_zenith']

            # adiciona controle de qualidade
            df = self._quality_control(df)  

            # persistencia
            df = self._persistence(df)

        
        df = self._create_lag_features(df)
        
        
        df = self._apply_normalizations(df)

        if self.target_col_internal == "power":
            # Define o limiar (ex: 85 graus). 1 = Dia (Considerar), 0 = Noite (Ignorar)
            df['mask'] = np.where(df['zenith'] < 85, 1.0, 0.0)
        
        cols_x = [c for c in self.features_to_scale if c in df.columns]
        if cols_x: df[cols_x] = self.scaler_x.transform(df[cols_x])
            

        return df.fillna(0)

    def _clean_index(self, df):
        """Filtra ano usando coluna auxiliar (Robustez)."""
        df['__ano_temp'] = df.index.year
        df = df.loc[df['__ano_temp'] >= self.start_year].copy()
        df.drop(columns=['__ano_temp'], inplace=True)
        return df.sort_index()
    
    def _persistence(self, df):
        df['P_fracao_difusa'] = df['fracao_difusa'].shift(periods=1)
        df['P_kt'] = df['kt'].shift(periods=1)
        return df

    def _calculate_solar_position(self, df):
        solpos = self.location.get_solarposition(df.index)
        df['zenith'] = solpos['zenith']
        df['apparent_zenith'] = solpos['apparent_zenith']
        df['azimuth'] = solpos['azimuth']
        df['elevation'] = solpos['elevation']

        df['extra_rad'] = pvlib.irradiance.get_extra_radiation(df.index)
        
        df['elevation'] = solpos['elevation']
        
        df['dni_extra'] = pvlib.irradiance.get_extra_radiation(df.index)

        rel_airmass = pvlib.atmosphere.get_relative_airmass(df['apparent_zenith'])
        df['am_abs'] = pvlib.atmosphere.get_absolute_airmass(rel_airmass, 
                                                             pressure=pvlib.atmosphere.alt2pres(self.altitude))
        
        if 'zenith' in df.columns: df['cos_zenith'] = np.cos(np.deg2rad(df['zenith']))
        if 'azimuth' in df.columns: df['sin_azimuth'] = np.sin(np.deg2rad(df['azimuth']))

        df['ghi_extra'] = df['extra_rad'] * df['cos_zenith']

        #==========================
        #   Modelos de Céu Claro
        #==========================
        if self.cs_model == 'perez':
            cs = self.location.get_clearsky(df.index, 
                                            model='ineichen',
                                            linke_turbidity=3)
            df['ghi_cs_theo'] = cs['ghi']
            df['dhi_cs_theo'] = cs['dhi']

        elif self.cs_model == 'esra':

            cs = ESRA(apparent_zenith=df['apparent_zenith'],
                      dni_extra=df['dni_extra'],
                      linke_turbidity=3.0,  # Valor inicial irrelevante, será substituído
                      altitude=self.altitude)
            
            lista_dias_claros = cs.get_clear_days_pearson(df, 
                                                            threshold=0.95, 
                                                            min_ratio=0.8)
            
            df['is_clear_moment'] = np.isin(df.index.date, lista_dias_claros)

            daytime = df['elevation'] > 15
            daily_stats = df[daytime].groupby(df[daytime].index.date)['is_clear_moment'].mean()
            clear_days_dates = daily_stats[daily_stats > 0.8].index     #   <---------------
            df['is_clear_day'] = np.isin(df.index.date, clear_days_dates)
            

            tl_otimizado = cs.optimize_linke_turbidity(df, inplace=True, use_pearson=False)

            df['linke_turbidity_calc'] = tl_otimizado

            resultado = cs.get_esra_clearsky()

            df['dhi_cs'] = resultado['dhi']
            df['ghi_cs'] = resultado['ghi']
            df['dni_cs'] = resultado['dni']

        return df

    def _calculate_physics_features(self, df):
        # calculo do clearness index (Kt)
        df['kt'] = df['ghi']/df['ghi_extra'].replace(np.nan,0)
        df['kt'] = df['kt'].fillna(0)
        df['kt'] = df['kt'].clip(0, 1.5)
        
        # Calculo do DNI
        if 'dhi' in df.columns and 'ghi' in df.columns and 'dni' in df.columns:
            pass
        elif 'dhi' in df.columns and 'ghi' in df.columns:
            df['dni'] = (df['ghi'] - df['dhi'])/df['cos_zenith']
        # Calculo do DHI
        elif 'dni' in df.columns and 'ghi' in df.columns:
            df['dhi'] = df['ghi'] - (df['dni']*df['cos_zenith'])

        # 1. Fração Difusa (Prioridade: Medido > Teórico)
        if 'dhi' in df.columns and 'ghi' in df.columns:
            #df['fracao_difusa'] = df['dhi']/df['ghi'].clip(0, 1.2)
            df['fracao_difusa'] = np.where(df['ghi'] > 0.1, df['dhi']/df['ghi'], 0.0).clip(0, 1.2)
        elif 'dhi_cs_theo' in df.columns and 'ghi_cs' in df.columns:
            df['fracao_difusa'] = np.where(df['ghi_cs'] > 10, df['dhi_cs']/df['ghi_cs'], 0.0).clip(0, 1.2)

        if 'dhi' in df.columns and 'ghi' in df.columns:
            df['direct_fraction'] = df['dni']* df['cos_zenith']/df['ghi']

        df['k'] = df['ghi']/df['ghi_cs']
        df['k'] = df['k'].replace(np.nan, 0).fillna(0)
        df['k'] = df['k'].clip(0, 1)  

        #===================================
        #   --- Calculo da variancia ---
        # Castillejo-Cuberos et al. (2024)
        #===================================
        sin_alpha = np.sin(np.deg2rad(df['elevation']))

        qs = 1 - ((np.sqrt((1 - df['kt'])**2 + df['fracao_difusa']**2))/np.sqrt(2))
        df['QS'] = qs 

        x1=0.8505
        x2=0.3985
        x3=1.2972
        x4=0.9084
        x5=0.3066

        first_part = (x1 - ((df['kt'] - 0.5) ** 2 + (df['fracao_difusa'] - 0.6) ** 2)/x2) * (sin_alpha) ** x3
        second_part = (1 - (np.abs(qs-x4)/x4)) ** x5

        vs_cdfn = first_part * second_part
        df['VS_cdfn'] = vs_cdfn

            
        # calculo das variáveis necessárias para fazer o a conversão de irrad em pot
        term_vento = self.u0 + self.u1 * df['wind_speed']
        term_vento = term_vento.replace(0, 0.1) 
        df['temp_cell'] = df['temp_amb'] + (df['ghi'] / term_vento)
        
        efficiency_factor = 1 - CONSTANTS['GAMMA_SI'] * (df['temp_cell'] - CONSTANTS['T_STC'])
        df['pot_cs'] = self.nominal_power * (df['ghi_cs'] / CONSTANTS['G_STC']) * efficiency_factor
        
        return df

    def _create_lag_features(self, df):
        # Persistencia para potencia
        if self.target_col_internal == 'power':
            if 'k' in df.columns:
                for lag in [1, 2, 3]:
                    col = f'k_lag{lag}'
                    df[col] = (df['target']/df['pot_cs']).clip(0,1).shift(periods=lag)
                    if 'pot_cs' in df.columns:
                        df[f'P{lag}'] = df['pot_cs'] * df[col]
                        df[f'P{lag}'] = df[f'P{lag}'].replace(np.inf, 0)
                df.drop([f'k_lag{i}' for i in [1,2,3]], axis=1, inplace=True, errors='ignore')
        
        # Persistencia para K
        elif self.target_col_internal == 'sky':
            for lag in [1, 2, 3]:
                    df[f'P{lag}'] = (df['k']).shift(lag) 
        
        return df

    def _apply_normalizations(self, df):
        """        
        # 1. Fração Difusa (Prioridade: Medido > Teórico)
        if 'dhi' in df.columns and 'ghi' in df.columns:
            #df['fracao_difusa'] = df['dhi']/df['ghi'].clip(0, 1.2)
            df['fracao_difusa'] = np.where(df['ghi'] > 0.1, df['dhi']/df['ghi'], 0.0).clip(0, 1.2)
        elif 'dhi_cs_theo' in df.columns and 'ghi_cs' in df.columns:
            df['fracao_difusa'] = np.where(df['ghi_cs'] > 10, df['dhi_cs']/df['ghi_cs'], 0.0).clip(0, 1.2)

        if 'dhi' in df.columns and 'ghi' in df.columns:
            df['direct_fraction'] = df['dni']* df['cos_zenith']/df['ghi']"""

        if 'irrad_poa' in df.columns and 'ghi' in df.columns:
            df['irr_clearsky_ratio'] = df['irrad_poa'] / df['ghi'].replace(0, np.nan)
            df['irr_clearsky_ratio'] = df['irr_clearsky_ratio'].fillna(0)

        if 'humidity' in df.columns: df['humidity'] = df['humidity'] / 100.0

        cols_pot = ['target', 'P1', 'P2', 'P3']
        for col in cols_pot:
            if col in df.columns: df[col] = df[col] / self.nominal_power

        if 'year' in df.columns:
            years_passed = (df['year'] - self.start_year).clip(lower=0)
            df['degradacao'] = 1 - (self.degradation_rate * years_passed)

        if self.target_col_internal == 'sky':
            df['sin_elevation'] = np.sin(np.deg2rad(df['elevation']))

            for i in ['kt', 'fracao_difusa']:
                # 1. Calcula a diferença simples dos valores
                df['delta_valor'] = df[i].diff()
                
                # 2. Calcula a diferença de tempo entre as linhas
                df['diff_tempo'] = df.index.to_series().diff()
                
                # 3. Filtra: Mantém o delta APENAS se o intervalo for de 1 hora
                # (Altere '1h' para a frequência desejada dos seus dados)
                df[f'delta_{i}'] = df['delta_valor'].where(df['diff_tempo'] == pd.Timedelta('1h'))
                
                df = df.drop(columns=["delta_valor", "diff_tempo"])

        return df
    
    def _is_clear(self, df):
        # 1. Detecção de frequência mais robusta
        # Pega a frequência estimada (ex: 'T', 'min', 'H', '15T')
        freq = pd.infer_freq(df.index)
        
        # Se não conseguir inferir, tenta calcular a diferença média
        if freq is None:
            timedeltas = df.index.to_series().diff().dropna()
            median_diff = timedeltas.median()
            if median_diff <= pd.Timedelta(minutes=15):
                is_high_freq = True
            else:
                is_high_freq = False
        else:
            # Consideramos alta frequência se for menor ou igual a 15 min
            is_high_freq = 'min' in freq or 'T' in freq or freq in ['S', 'L', 'U']

        # ---------------------------------------------------------
        # CENÁRIO A: Alta Resolução (Minutos) -> Usa Reno-Hansen (detect_clearsky)
        # ---------------------------------------------------------
        if is_high_freq:
            # Preenche NaNs apenas para o algoritmo rodar, mas idealmente interpolar
            ghi_series = df['ghi'].fillna(0)
            cs_series = df['ghi_cs_theo'].fillna(0)
            
            is_clear_window = pvlib.clearsky.detect_clearsky(
                ghi_series, 
                cs_series, 
                df.index, 
                window_length=10, # 10 minutos
                mean_diff=75,
                max_diff=75,
                lower_line_length=-5,
                upper_line_length=10,
                slope_dev=8
            )
            df['is_clear_moment'] = is_clear_window

        # ---------------------------------------------------------
        # CENÁRIO B: Baixa Resolução (Horário) -> Usa Threshold Simples
        # ---------------------------------------------------------
        else:
            # Garante frequência horária sem zerar dados existentes desnecessariamente
            if freq != 'H' and freq != '1H':
                # Cuidado com asfreq fill_value=0, pode estragar médias
                # Melhor reamostrar ou apenas garantir indice
                pass 

            # Lógica para Horário: 
            # O momento é claro se o erro entre Medido e Teórico for baixo.
            # Aceitamos uma margem de erro (ex: 20%)
            
            # 1. Filtro de Noite (Elevação < 10 graus) ou GHI muito baixo
            # Se você não tem elevação no df, use o cs_theo como proxy
            mask_day = df['ghi_cs_theo'] > 10
            
            # 2. Razão de Céu Claro (Clear Sky Index - Kc)
            # Kc = GHI_medido / GHI_teorico
            # Evita divisão por zero
            kc = np.zeros(len(df))
            kc[mask_day] = df.loc[mask_day, 'ghi'] / df.loc[mask_day, 'ghi_cs_theo']
            
            # 3. Classificação
            # Um dia claro geralmente tem 0.9 <= Kc <= 1.1 
            # (pode ser 0.8 a 1.2 dependendo da calibração do seu modelo teórico)
            is_clear_moment = (kc >= 0.85) & (kc <= 1.15)
            
            # Opcional: Filtro de estabilidade temporal (opcional para horário)
            # Se a hora anterior e a próxima também forem "boas", reforça a classificação
            # Mas para otimização de TL, o filtro de magnitude acima já ajuda muito.
            
            df['is_clear_moment'] = is_clear_moment
            
            # Limpa ruído noturno
            df.loc[~mask_day, 'is_clear_moment'] = False
        

        return df

    def _otimization_LT (self, df):
        daytime = df['elevation'] > 15
        daily_stats = df[daytime].groupby(df[daytime].index.date)['is_clear_moment'].mean()
        
        clear_days_dates = daily_stats[daily_stats > 0.8].index     #   <---------------

        df['is_clear_day'] = np.isin(df.index.date, clear_days_dates)

        daily_tl = {}
        last_valid_tl = 3.0 # Valor inicial padrão (urbano) caso o primeiro dia seja nublado

        unique_days = np.unique(df.index.date)

        print("Iniciando otimização do Linke Turbidity...")

        # ⚡ Bolt Optimization: Calculate valid mask and group by day OUTSIDE the loop
        # This replaces an O(N) boolean mask scan inside the loop with an O(1) group lookup,
        # dramatically speeding up Linke Turbidity processing on large datasets.
        df_valid = df[(df['elevation'] > 10) & (df['ghi'] > 10)]
        grouped_valid = df_valid.groupby(df_valid.index.date)

        for day in unique_days:
            # Selecionar dados do dia, APENAS período diurno e válido
            try:
                sub_df = grouped_valid.get_group(day)
            except KeyError:
                sub_df = df_valid.iloc[:0].copy()
            
            # Verifica se o dia foi classificado como claro
            is_clear = day in clear_days_dates
            
            if is_clear and len(sub_df) > 10: 
                # Cenario A: Dia Claro -> Calculamos um NOVO valor
                res = minimize_scalar(
                        self._objective_function, 
                        bounds=(1.0, 10.0), 
                        args=(sub_df,), 
                        method='bounded'
                    )
                if res.success:
                    optimal_tl = res.x
                    daily_tl[day] = optimal_tl
                    
                    # ATENÇÃO AQUI:
                    last_valid_tl = optimal_tl # Atualizamos a memória com o novo valor
                else:
                    daily_tl[day] = last_valid_tl
                    
            else:
                # Cenario B: Dia Nublado -> Usamos o valor da memória
                # Como não atualizamos o 'last_valid_tl', ele continua sendo o do último dia claro
                daily_tl[day] = last_valid_tl

        # Criar uma série com os TLs diários
        tl_series = pd.Series(daily_tl, name='Daily_TL')
        tl_series.index = pd.to_datetime(tl_series.index)

        # Mapear de volta para o DataFrame original (resample/merge)
        # Maneira rápida: criar coluna 'date' e mapear
        df['temp_date'] = df.index.date
        df['Linke_Turbidity'] = df['temp_date'].map(daily_tl)
        df.drop(columns=['temp_date','ghi_cs_theo','dhi_cs_theo'], inplace=True)

        cs_calibrated = pvlib.clearsky.ineichen(
            apparent_zenith=df['apparent_zenith'],
            airmass_absolute=df['am_abs'],
            linke_turbidity=df['Linke_Turbidity'], # Aqui entra sua nova coluna!
            dni_extra=df['dni_extra'],
            perez_enhancement=True
        )

        # Salvar no DataFrame (substituindo o antigo ou criando novo)
        df['ghi_cs'] = cs_calibrated['ghi']
        df['dhi_cs'] = cs_calibrated['dhi']
        df['dni_cs'] = cs_calibrated['dni']

        return df
    
    def _objective_function(self, tl, sub_df):
        """
        Função de custo para minimizar.
        Calcula o RMSE entre o GHI Medido e o GHI Ineichen com o TL dado.
        """
        # Recalcula Ineichen com o TL candidato para este slice de dados
        cs = pvlib.clearsky.ineichen(
            apparent_zenith=sub_df['apparent_zenith'],
            airmass_absolute=sub_df['am_abs'],
            linke_turbidity=tl,
            dni_extra=sub_df['dni_extra'],
            perez_enhancement=True
        )
        
        # Calcula RMSE apenas para os dados válidos passados
        # Proteção contra NaN
        if sub_df['ghi'].isnull().all() or cs['ghi'].isnull().all():
            return 999999 
        
        return root_mean_squared_error(sub_df['ghi'], cs['ghi'])
    
    def _quality_control (self, df):
        # 0. Preparação: Converter graus para radianos (necessário para np.sin)
        # O numpy usa radianos, a imagem usa graus.
        sin_alpha = np.sin(np.deg2rad(df['elevation']))

        G =  df['ghi_extra']/df['cos_zenith']

        elevation = df['elevation']

        ghi = df['ghi']
        ghi_cs = df['ghi_cs']
        dhi = df['dhi']
        dni = df['dni']
        fracao_difusa = dhi/ghi

        # --- CRIAÇÃO DAS MÁSCARAS (FILTROS) ---

        # 1. Elevação Solar (alpha > 10)
        cond_alpha = elevation > 10

        # 2. Limites Físicos GHI
        # 0 < GHI < (100 + 1.5 * G * sin(alpha)^1.2)
        limit_ghi = 100 + 1.5 * G * (sin_alpha ** 1.2)
        cond_ghi_limits = (ghi > 0) & (ghi < limit_ghi)

        # 3. Limites Físicos DHI
        # 0 < DHI < (50 + 0.95 * G * sin(alpha)^1.2)
        limit_dhi = 50 + 0.95 * G * (sin_alpha ** 1.2)
        cond_dhi_limits = (dhi > 0) & (dhi < limit_dhi)

        # 4 e 5. Consistência de Fechamento (Closure check)
        # A imagem divide em alpha > 15 e alpha < 15, mas ambos exigem DHI + DNI*sin > 50
        sum_components = dhi + (dni * sin_alpha)
        closure_ratio = np.abs((ghi - sum_components) / ghi)

        # Parte comum: soma > 50
        cond_sum_min = sum_components > 50

        # Parte condicional do erro (0.08 para > 15 graus, 0.15 para < 15 graus)
        """
        cond_closure_accuracy = (
            ((elevation > 15) & (closure_ratio < 0.08)) | 
            ((elevation < 15) & (closure_ratio < 0.15))
        )

        cond_closure = (sum_components <= 50) | (cond_sum_min & cond_closure_accuracy)
        """
        cond_closure = ((elevation > 15) & (closure_ratio < 0.08) & (sum_components > 50)) | ((elevation < 15) & (closure_ratio < 0.15) & (sum_components > 50))
        #df['cond'] = np.where(sum_components <= 50, 1, np.where(np.logical_and(sum_components > 50, cond_closure_accuracy), 1, 0))

        # 6. Limites DNI
        cond_dni_limits = (dni >= 0) & (dni < G)

        # 7. Critério de Rejeição (Invertido para pegar os válidos)
        # Reject if GHI/GHIcs > 0.85 AND DHI/GHI > 0.85
        # Para passar, a condição deve ser FALSA, por isso o "~" (NOT)
        cond_reject_ratio = ~((ghi / ghi_cs > 0.85) & (dhi / ghi > 0.85))

        # 8. Teste de GHI mínimo relativo à elevação
        # GHI / (G * sin(alpha)) >= (alpha - 10) / 10000 (apenas se alpha > 10)
        lhs_8 = ghi / (G * sin_alpha)
        rhs_8 = (elevation - 10) / 10000
        # A condição só se aplica se alpha > 10. Se alpha <= 10, essa regra não reprova (True).
        cond_ghi_min_slope = (elevation > 10) & (lhs_8 >= rhs_8)

        # 9. Clearness Index (K) (Assumindo que você tem a coluna K ou calcula K = GHI / G_ext)
        # 0 < K < 1.1
        cond_k = (fracao_difusa > 0) & (fracao_difusa < 1.1)

        # 10. Consistência de Difusa
        # DNI sin a / GHI <= GHI/GHIcs + (-1 + 1.05/0.95)
        term_const = -1 + (1.05 / 0.95)
        cond_diffuse_consistency = (
            ((dni * sin_alpha) / ghi) <= 
            ((ghi / ghi_cs) + term_const)
        )

        # --- APLICAÇÃO DO FILTRO NO DATAFRAME (.loc) ---

        # Combinando todas as máscaras linha-a-linha
        final_mask = (
            cond_alpha & 
            cond_ghi_limits & 
            cond_dhi_limits &
            cond_closure &
            cond_dni_limits &
            cond_reject_ratio &
            cond_ghi_min_slope &
            cond_k &
            cond_diffuse_consistency
        )

        df['mask'] = np.where(final_mask, 1.0, 0.0)

        # Criação da coluna informando as reprovações
        qc_reprovals = pd.Series("", index=df.index)
        qc_reprovals += np.where(~cond_alpha, "alpha;", "")
        qc_reprovals += np.where(~cond_ghi_limits, "ghi_limits;", "")
        qc_reprovals += np.where(~cond_dhi_limits, "dhi_limits;", "")
        qc_reprovals += np.where(~cond_closure, "closure;", "")
        qc_reprovals += np.where(~cond_dni_limits, "dni_limits;", "")
        qc_reprovals += np.where(~cond_reject_ratio, "reject_ratio;", "")
        qc_reprovals += np.where(~cond_ghi_min_slope, "ghi_min_slope;", "")
        qc_reprovals += np.where(~cond_k, "k_limits;", "")
        qc_reprovals += np.where(~cond_diffuse_consistency, "diffuse_consistency;", "")
        
        df['qc_reprovals'] = qc_reprovals.str.rstrip(";")

        horas_validas_dia = df.groupby(df.index.date)['mask'].transform('sum')
        # Se o dia tiver menos de 5h válidas, zera a máscara de todos os pontos daquele dia
        mask_5h = horas_validas_dia < 5.0
        df.loc[mask_5h, 'mask'] = 0.0
        
        # Adiciona a falha de 5h na lista
        df['qc_reprovals'] = np.where(
            mask_5h,
            np.where(df['qc_reprovals'] == "", "min_5_hours", df['qc_reprovals'] + ";min_5_hours"),
            df['qc_reprovals']
        )
        
        # Marca os aprovados
        df['qc_reprovals'] = df['qc_reprovals'].replace("", "approved")

        return df

    def save_scalers(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler_x, os.path.join(output_dir, 'scaler_X.pkl'))
        #joblib.dump(self.scaler_y, os.path.join(output_dir, 'scaler_Y.pkl'))
        print(f"💾 Scalers salvos em: {output_dir}")

    def load_scalers(self, input_dir: str):
        path_x = os.path.join(input_dir, 'scaler_X.pkl')
        #path_y = os.path.join(input_dir, 'scaler_Y.pkl')
        #if not (os.path.exists(path_x) and os.path.exists(path_y)):
        #    raise FileNotFoundError(f"Scalers não encontrados em {input_dir}")
        self.scaler_x = joblib.load(path_x)
        #self.scaler_y = joblib.load(path_y)
        self._is_fitted = True
        print(f"♻️  Scalers carregados de: {input_dir}")