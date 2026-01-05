import pandas as pd
import numpy as np
import os
import joblib
import warnings
from typing import Optional, Dict, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

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
                 degradation_rate: float = 0.05,
                 column_mapping: Optional[Dict[str, str]] = None,
                 features_to_scale: Optional[List[str]] = None,
                 target_col: str = 'target',
                 auto_identify_thermal_params: bool = True):
        
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        self.location = pvlib.location.Location(latitude, longitude, timezone, altitude)
        
        self.nominal_power = nominal_power
        self.start_year = start_year
        self.degradation_rate = degradation_rate
        
        self.column_mapping = DEFAULT_MAPPING.copy()
        if column_mapping:
            self.column_mapping.update(column_mapping)
            
        self.target_col_internal = self.column_mapping.get(target_col, 'target')
        self.features_to_scale = features_to_scale or ['temp_amb', 'wind_speed', 'ghi']
        
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.u0 = CONSTANTS['DEFAULT_U0']
        self.u1 = CONSTANTS['DEFAULT_U1']
        self.auto_identify = auto_identify_thermal_params
        self._is_fitted = False

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
            mask = (df['ghi'] > 300) & (df[self.target_col_internal] > 0) & (df['wind_speed'] >= 0)
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
        Y_data = df_fit[self.target_col_internal].values

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

        if self.auto_identify:
            self._fit_thermal_parameters(df) # E limpa de novo dentro, por segurança

        cols_x = [c for c in self.features_to_scale if c in df.columns]
        if cols_x: self.scaler_x.fit(df[cols_x])
            
        #if self.target_col_internal in df.columns:
        #    target_norm = df[[self.target_col_internal]] / self.nominal_power
        #    self.scaler_y.fit(target_norm)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted: raise RuntimeError("Fit necessário antes do Transform.")
        
        df = X.copy()
        df = self._rename_columns(df)
        df = self._ensure_datetime_index(df)

        df = self._clean_index(df)
        df = self._calculate_solar_position(df)
        df = self._calculate_physics_features(df)
        df = self._create_lag_features(df)
        df = self._create_angular_features(df)
        df = self._apply_normalizations(df)
        
        cols_x = [c for c in self.features_to_scale if c in df.columns]
        if cols_x: df[cols_x] = self.scaler_x.transform(df[cols_x])
            
        #if self.target_col_internal in df.columns:
        #    df[[self.target_col_internal]] = self.scaler_y.transform(df[[self.target_col_internal]])

        return df.fillna(0)

    def _clean_index(self, df):
        """Filtra ano usando coluna auxiliar (Robustez)."""
        df['__ano_temp'] = df.index.year
        df = df.loc[df['__ano_temp'] >= self.start_year].copy()
        df.drop(columns=['__ano_temp'], inplace=True)
        return df.sort_index()

    def _calculate_solar_position(self, df):
        solpos = self.location.get_solarposition(df.index)
        df['zenith'] = solpos['zenith']
        df['azimuth'] = solpos['azimuth']
        df['extra_rad'] = pvlib.irradiance.get_extra_radiation(df.index)
        cs = self.location.get_clearsky(df.index, model='ineichen')
        df['ghi_cs_theo'] = cs['ghi']
        df['dhi_cs_theo'] = cs['dhi']
        return df

    def _calculate_physics_features(self, df):
        required = ['temp_amb', 'wind_speed', 'ghi']
        if not all(col in df.columns for col in required):
            if self.target_col_internal in df.columns:
                df['k'] = df[self.target_col_internal] / self.nominal_power
            return df

        term_vento = self.u0 + self.u1 * df['wind_speed']
        term_vento = term_vento.replace(0, 0.1) 
        df['temp_cell'] = df['temp_amb'] + (df['ghi'] / term_vento)
        
        efficiency_factor = 1 - CONSTANTS['GAMMA_SI'] * (df['temp_cell'] - CONSTANTS['T_STC'])
        df['pot_cs'] = self.nominal_power * (df['ghi_cs_theo'] / CONSTANTS['G_STC']) * efficiency_factor
        
        if self.target_col_internal in df.columns:
            denom = df['pot_cs'].replace(0, np.nan)
            df['k'] = df[self.target_col_internal] / denom
            df['k'] = df['k'].fillna(0).clip(upper=1.2)
        return df

    def _create_lag_features(self, df):
        if 'k' in df.columns:
            for lag in [1, 2, 3]:
                col = f'k_lag{lag}'
                df[col] = df['k'].shift(lag)
                if 'pot_cs' in df.columns:
                    df[f'P{lag}'] = df['pot_cs'] * df[col]
            df.drop([f'k_lag{i}' for i in [1,2,3]], axis=1, inplace=True, errors='ignore')
        return df

    def _create_angular_features(self, df):
        if 'zenith' in df.columns: df['cos_zenith'] = np.cos(np.deg2rad(df['zenith']))
        if 'azimuth' in df.columns: df['sin_azimuth'] = np.sin(np.deg2rad(df['azimuth']))
        return df

    def _apply_normalizations(self, df):
        # 1. Fração Difusa (Prioridade: Medido > Teórico)
        if 'dhi' in df.columns and 'ghi' in df.columns:
            df['fracao_difusa'] = np.where(df['ghi'] > 10, df['dhi']/df['ghi'], 0.0).clip(0, 1)
        elif 'dhi_cs_theo' in df.columns and 'ghi_cs_theo' in df.columns:
            df['fracao_difusa'] = np.where(df['ghi_cs_theo'] > 10, df['dhi_cs_theo']/df['ghi_cs_theo'], 0.0).clip(0, 1)

        if 'irrad_poa' in df.columns and 'ghi' in df.columns:
            df['irr_clearsky_ratio'] = df['irrad_poa'] / df['ghi'].replace(0, np.nan)
            df['irr_clearsky_ratio'] = df['irr_clearsky_ratio'].fillna(0)

        if 'humidity' in df.columns: df['humidity'] = df['humidity'] / 100.0

        cols_pot = [self.target_col_internal, 'P1', 'P2', 'P3']
        for col in cols_pot:
            if col in df.columns: df[col] = df[col] / self.nominal_power

        if 'year' in df.columns:
            years_passed = (df['year'] - self.start_year).clip(lower=0)
            df['degradacao'] = 1 - (self.degradation_rate * years_passed)
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