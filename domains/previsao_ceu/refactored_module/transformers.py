import pandas as pd
import numpy as np
import os
import joblib
import warnings
from typing import Optional, Dict, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import curve_fit, minimize_scalar

# Conditional imports like the original
try:
    import pvlib
except ImportError:
    raise ImportError("A biblioteca 'pvlib' é obrigatória. Instale: pip install pvlib")

from cs_model.esra import ESRA
from auxiliary_models.kasten_correction import Kasten_Correction

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

class DataSanitizer(BaseEstimator, TransformerMixin):
    def __init__(self, timezone: str = 'UTC', start_year: int = 2018, column_mapping: Optional[Dict[str, str]] = None):
        self.timezone = timezone
        self.start_year = start_year
        self.column_mapping = DEFAULT_MAPPING.copy()
        if column_mapping:
            self.column_mapping.update(column_mapping)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        # Rename columns
        rename_dict = {k: v for k, v in self.column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Ensure datetime index
        col_data = None
        possible_names = ['date_time', 'Date_Time', 'Date', 'data', 'datetime']
        inv_map = {v: k for k, v in self.column_mapping.items()}
        if 'date_time' in inv_map: possible_names.insert(0, inv_map['date_time'])

        for col in possible_names:
            if col in df.columns:
                col_data = col
                break
        
        if col_data and not pd.api.types.is_datetime64_any_dtype(df.index):
            df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
            df.set_index(col_data, inplace=True)
            
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
            except:
                pass

        if df.index.isna().any():
            df = df[df.index.notna()]

        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize(self.timezone, ambiguous='NaT', nonexistent='NaT')
            else:
                df.index = df.index.tz_convert(self.timezone)
        except Exception as e:
            print(f"⚠️ Erro Fuso: {e}")

        if df.index.isna().any():
            df = df[df.index.notna()]

        if df.index.duplicated().any():
            print(f"⚠️ Removendo {df.index.duplicated().sum()} duplicatas no índice.")
            df = df[~df.index.duplicated(keep='first')]
            
        df = df.sort_index()

        # Clean index
        df['__ano_temp'] = df.index.year
        df = df.loc[df['__ano_temp'] >= self.start_year].copy()
        df.drop(columns=['__ano_temp'], inplace=True)
        
        return df.sort_index()


class SolarPositionCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, latitude: float, longitude: float, altitude: float = 0, timezone: str = 'UTC'):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        self.location = pvlib.location.Location(latitude, longitude, timezone, altitude)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        solpos = self.location.get_solarposition(df.index)
        df['zenith'] = solpos['zenith']
        df['apparent_zenith'] = solpos['apparent_zenith']
        df['azimuth'] = solpos['azimuth']
        df['elevation'] = solpos['elevation']

        df['extra_rad'] = pvlib.irradiance.get_extra_radiation(df.index)
        df['dni_extra'] = pvlib.irradiance.get_extra_radiation(df.index)

        rel_airmass = pvlib.atmosphere.get_relative_airmass(df['apparent_zenith'])
        df['am_abs'] = pvlib.atmosphere.get_absolute_airmass(rel_airmass, 
                                                             pressure=pvlib.atmosphere.alt2pres(self.altitude))
        
        if 'zenith' in df.columns: df['cos_zenith'] = np.cos(np.deg2rad(df['zenith']))
        if 'azimuth' in df.columns: df['sin_azimuth'] = np.sin(np.deg2rad(df['azimuth']))

        df['ghi_extra'] = df['extra_rad'] * df['cos_zenith']
        return df


class ClearSkyEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, latitude: float, longitude: float, altitude: float = 0, timezone: str = 'UTC', cs_model: str = 'esra'):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        self.location = pvlib.location.Location(latitude, longitude, timezone, altitude)
        self.cs_model = cs_model

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        if self.cs_model == 'perez':
            cs = self.location.get_clearsky(df.index, 
                                            model='ineichen',
                                            linke_turbidity=3)
            df['ghi_cs_theo'] = cs['ghi']
            df['dhi_cs_theo'] = cs['dhi']
            
            df = self._is_clear(df)
            df = self._otimization_LT(df)

        elif self.cs_model == 'esra':
            cs = ESRA(apparent_zenith=df['apparent_zenith'],
                      dni_extra=df['dni_extra'],
                      linke_turbidity=3.0,
                      altitude=self.altitude)
            
            lista_dias_claros = cs.get_clear_days_pearson(df, 
                                                            threshold=0.95, 
                                                            min_ratio=0.8)
            
            df['is_clear_moment'] = np.isin(df.index.date, lista_dias_claros)

            daytime = df['elevation'] > 15
            daily_stats = df[daytime].groupby(df[daytime].index.date)['is_clear_moment'].mean()
            clear_days_dates = daily_stats[daily_stats > 0.8].index
            df['is_clear_day'] = np.isin(df.index.date, clear_days_dates)
            
            tl_otimizado = cs.optimize_linke_turbidity(df, inplace=True, use_pearson=False)
            df['linke_turbidity_calc'] = tl_otimizado

            resultado = cs.get_esra_clearsky()
            df['dhi_cs'] = resultado['dhi']
            df['ghi_cs'] = resultado['ghi']
            df['dni_cs'] = resultado['dni']

        return df

    def _is_clear(self, df):
        freq = pd.infer_freq(df.index)
        
        if freq is None:
            timedeltas = df.index.to_series().diff().dropna()
            median_diff = timedeltas.median()
            if median_diff <= pd.Timedelta(minutes=15):
                is_high_freq = True
            else:
                is_high_freq = False
        else:
            is_high_freq = 'min' in freq or 'T' in freq or freq in ['S', 'L', 'U']

        if is_high_freq:
            ghi_series = df['ghi'].fillna(0)
            cs_series = df['ghi_cs_theo'].fillna(0)
            
            is_clear_window = pvlib.clearsky.detect_clearsky(
                ghi_series, 
                cs_series, 
                df.index, 
                window_length=10, 
                mean_diff=75,
                max_diff=75,
                lower_line_length=-5,
                upper_line_length=10,
                slope_dev=8
            )
            df['is_clear_moment'] = is_clear_window
        else:
            mask_day = df['ghi_cs_theo'] > 10
            kc = np.zeros(len(df))
            kc[mask_day] = df.loc[mask_day, 'ghi'] / df.loc[mask_day, 'ghi_cs_theo']
            is_clear_moment = (kc >= 0.85) & (kc <= 1.15)
            df['is_clear_moment'] = is_clear_moment
            df.loc[~mask_day, 'is_clear_moment'] = False
        return df

    def _otimization_LT(self, df):
        daytime = df['elevation'] > 15
        daily_stats = df[daytime].groupby(df[daytime].index.date)['is_clear_moment'].mean()
        
        clear_days_dates = daily_stats[daily_stats > 0.8].index
        df['is_clear_day'] = np.isin(df.index.date, clear_days_dates)

        daily_tl = {}
        last_valid_tl = 3.0

        unique_days = np.unique(df.index.date)

        print("Iniciando otimização do Linke Turbidity...")

        for day in unique_days:
            mask_day = (df.index.date == day) & (df['elevation'] > 10) & (df['ghi'] > 10)
            sub_df = df.loc[mask_day]
            
            is_clear = day in clear_days_dates
            
            if is_clear and len(sub_df) > 10: 
                res = minimize_scalar(
                        self._objective_function, 
                        bounds=(1.0, 10.0), 
                        args=(sub_df,), 
                        method='bounded'
                    )
                if res.success:
                    optimal_tl = res.x
                    daily_tl[day] = optimal_tl
                    last_valid_tl = optimal_tl
                else:
                    daily_tl[day] = last_valid_tl
            else:
                daily_tl[day] = last_valid_tl

        tl_series = pd.Series(daily_tl, name='Daily_TL')
        tl_series.index = pd.to_datetime(tl_series.index)

        df['temp_date'] = df.index.date
        df['Linke_Turbidity'] = df['temp_date'].map(daily_tl)
        df.drop(columns=['temp_date','ghi_cs_theo','dhi_cs_theo'], inplace=True)

        cs_calibrated = pvlib.clearsky.ineichen(
            apparent_zenith=df['apparent_zenith'],
            airmass_absolute=df['am_abs'],
            linke_turbidity=df['Linke_Turbidity'],
            dni_extra=df['dni_extra'],
            perez_enhancement=True
        )

        df['ghi_cs'] = cs_calibrated['ghi']
        df['dhi_cs'] = cs_calibrated['dhi']
        df['dni_cs'] = cs_calibrated['dni']

        return df
    
    def _objective_function(self, tl, sub_df):
        cs = pvlib.clearsky.ineichen(
            apparent_zenith=sub_df['apparent_zenith'],
            airmass_absolute=sub_df['am_abs'],
            linke_turbidity=tl,
            dni_extra=sub_df['dni_extra'],
            perez_enhancement=True
        )
        
        if sub_df['ghi'].isnull().all() or cs['ghi'].isnull().all():
            return 999999 
        
        return root_mean_squared_error(sub_df['ghi'], cs['ghi'])


class PhysicalFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, nominal_power: float = 156.0, auto_identify_thermal_params: bool = True, target_col: str = 'power'):
        self.nominal_power = nominal_power
        self.auto_identify = auto_identify_thermal_params
        self.target_col_internal = target_col
        self.u0 = CONSTANTS['DEFAULT_U0']
        self.u1 = CONSTANTS['DEFAULT_U1']

    def fit(self, X: pd.DataFrame, y=None):
        if self.auto_identify and self.target_col_internal == 'power':
            self._fit_thermal_parameters(X)
        return self

    def _fit_thermal_parameters(self, df: pd.DataFrame):
        df_copy = df.loc[~df.index.duplicated(keep='first')].copy()
        df_copy = df_copy.sort_index()

        required = ['target', 'ghi', 'temp_amb', 'wind_speed']
        if not all(col in df_copy.columns for col in required):
            return

        try:
            mask = (df_copy['ghi'] > 300) & (df_copy['target'] > 0) & (df_copy['wind_speed'] >= 0)
            df_fit = df_copy.loc[mask].dropna()
        except ValueError as e:
            print(f"⚠️ Erro de índice duplicado no fit: {e}. Pulando otimização.")
            return

        if len(df_fit) < 50: return

        def physical_power_model(X_vals, u0, u1):
            ghi, temp, wind = X_vals
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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        df['kt'] = df['ghi']/df['ghi_extra'].replace(np.nan,0)
        df['kt'] = df['kt'].fillna(0)
        df['kt'] = df['kt'].clip(0, 1.5)
        
        if 'dhi' in df.columns and 'ghi' in df.columns and 'dni' in df.columns:
            pass
        elif 'dhi' in df.columns and 'ghi' in df.columns:
            df['dni'] = (df['ghi'] - df['dhi'])/df['cos_zenith']
        elif 'dni' in df.columns and 'ghi' in df.columns:
            df['dhi'] = df['ghi'] - (df['dni']*df['cos_zenith'])

        if 'dhi' in df.columns and 'ghi' in df.columns:
            df['fracao_difusa'] = np.where(df['ghi'] > 0.1, df['dhi']/df['ghi'], 0.0).clip(0, 1.2)
        elif 'dhi_cs_theo' in df.columns and 'ghi_cs' in df.columns:
            df['fracao_difusa'] = np.where(df['ghi_cs'] > 10, df['dhi_cs']/df['ghi_cs'], 0.0).clip(0, 1.2)

        if 'dhi' in df.columns and 'ghi' in df.columns:
            df['direct_fraction'] = df['dni']* df['cos_zenith']/df['ghi']

        if 'ghi_cs' in df.columns:
            df['k'] = df['ghi']/df['ghi_cs']
            df['k'] = df['k'].replace(np.nan, 0).fillna(0)
            df['k'] = df['k'].clip(0, 1)  

        sin_alpha = np.sin(np.deg2rad(df['elevation']))
        qs = 1 - ((np.sqrt((1 - df['kt'])**2 + df.get('fracao_difusa', 0)**2))/np.sqrt(2))
        df['QS'] = qs 

        x1=0.8505
        x2=0.3985
        x3=1.2972
        x4=0.9084
        x5=0.3066

        first_part = (x1 - ((df['kt'] - 0.5) ** 2 + (df.get('fracao_difusa', 0) - 0.6) ** 2)/x2) * (sin_alpha) ** x3
        second_part = (1 - (np.abs(qs-x4)/x4)) ** x5

        vs_cdfn = first_part * second_part
        df['VS_cdfn'] = vs_cdfn

        if 'wind_speed' in df.columns and 'temp_amb' in df.columns:
            term_vento = self.u0 + self.u1 * df['wind_speed']
            term_vento = term_vento.replace(0, 0.1) 
            df['temp_cell'] = df['temp_amb'] + (df['ghi'] / term_vento)
            
            efficiency_factor = 1 - CONSTANTS['GAMMA_SI'] * (df['temp_cell'] - CONSTANTS['T_STC'])
            if 'ghi_cs' in df.columns:
                df['pot_cs'] = self.nominal_power * (df['ghi_cs'] / CONSTANTS['G_STC']) * efficiency_factor
        
        return df


class KastenCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, kasten_corr: bool = False, target_col: str = 'sky'):
        self.kasten_corr = kasten_corr
        self.target_col_internal = target_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if self.target_col_internal == 'sky' and self.kasten_corr:
            dados_corrigidos = Kasten_Correction(df)
            df = df.drop(columns=['dhi', 'dni', 'fracao_difusa'], errors='ignore').copy()

            df['dhi'] = dados_corrigidos['dhi_corrected']
            df['fracao_difusa'] = df['dhi'] / df['ghi']
            df['dni'] = (df['ghi'] - df['dhi']) / df['cos_zenith']
            
        return df


class QualityControlFilter(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str = 'sky'):
        self.target_col_internal = target_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if self.target_col_internal == 'sky':
            df = self._quality_control(df)
            
        if self.target_col_internal == 'power':
            if 'zenith' in df.columns:
                df['mask'] = np.where(df['zenith'] < 85, 1.0, 0.0)
            else:
                df['mask'] = 1.0
            
        return df

    def _quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        sin_alpha = np.sin(np.deg2rad(df['elevation']))
        G =  df['ghi_extra']/df['cos_zenith']
        elevation = df['elevation']

        ghi = df['ghi']
        ghi_cs = df['ghi_cs']
        dhi = df['dhi']
        dni = df['dni']
        fracao_difusa = df.get('fracao_difusa', dhi/ghi)

        cond_alpha = elevation > 10

        limit_ghi = 100 + 1.5 * G * (sin_alpha ** 1.2)
        cond_ghi_limits = (ghi > 0) & (ghi < limit_ghi)

        limit_dhi = 50 + 0.95 * G * (sin_alpha ** 1.2)
        cond_dhi_limits = (dhi > 0) & (dhi < limit_dhi)

        sum_components = dhi + (dni * sin_alpha)
        closure_ratio = np.abs((ghi - sum_components) / ghi)

        cond_closure = ((elevation > 15) & (closure_ratio < 0.08) & (sum_components > 50)) | \
                       ((elevation < 15) & (closure_ratio < 0.15) & (sum_components > 50))

        cond_dni_limits = (dni >= 0) & (dni < G)
        cond_reject_ratio = ~((ghi / ghi_cs > 0.85) & (dhi / ghi > 0.85))

        lhs_8 = ghi / (G * sin_alpha)
        rhs_8 = (elevation - 10) / 10000
        cond_ghi_min_slope = (elevation > 10) & (lhs_8 >= rhs_8)

        cond_k = (fracao_difusa > 0) & (fracao_difusa < 1.1)

        term_const = -1 + (1.05 / 0.95)
        cond_diffuse_consistency = (((dni * sin_alpha) / ghi) <= ((ghi / ghi_cs) + term_const))

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
        mask_5h = horas_validas_dia < 5.0
        df.loc[mask_5h, 'mask'] = 0.0
        
        df['qc_reprovals'] = np.where(
            mask_5h,
            np.where(df['qc_reprovals'] == "", "min_5_hours", df['qc_reprovals'] + ";min_5_hours"),
            df['qc_reprovals']
        )
        
        df['qc_reprovals'] = df['qc_reprovals'].replace("", "approved")

        return df


class TemporalFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str = 'power'):
        self.target_col_internal = target_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        if self.target_col_internal == 'sky':
            if 'fracao_difusa' in df.columns:
                df['P_fracao_difusa'] = df['fracao_difusa'].shift(periods=1)
            if 'kt' in df.columns:
                df['P_kt'] = df['kt'].shift(periods=1)
            
            if 'k' in df.columns:
                for lag in [1, 2, 3]:
                    df[f'P{lag}'] = (df['k']).shift(lag) 
                    
        elif self.target_col_internal == 'power':
            if 'k' in df.columns and 'target' in df.columns and 'pot_cs' in df.columns:
                for lag in [1, 2, 3]:
                    col = f'k_lag{lag}'
                    df[col] = (df['target']/df['pot_cs']).clip(0,1).shift(periods=lag)
                    df[f'P{lag}'] = df['pot_cs'] * df[col]
                    df[f'P{lag}'] = df[f'P{lag}'].replace(np.inf, 0)
                df.drop([f'k_lag{i}' for i in [1,2,3]], axis=1, inplace=True, errors='ignore')
                
        return df


class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str = 'power', nominal_power: float = 156.0, start_year: int = 2018, degradation_rate: float = 0.05, features_to_scale: Optional[List[str]] = None):
        self.target_col_internal = target_col
        self.nominal_power = nominal_power
        self.start_year = start_year
        self.degradation_rate = degradation_rate
        self.features_to_scale = features_to_scale or []
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df = self._apply_normalizations(df)
        cols_x = [c for c in self.features_to_scale if c in df.columns]
        if cols_x:
            self.scaler_x.fit(df[cols_x])
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = self._apply_normalizations(df)
        
        cols_x = [c for c in self.features_to_scale if c in df.columns]
        if cols_x and self._is_fitted:
            df[cols_x] = self.scaler_x.transform(df[cols_x])
            
        return df.fillna(0)

    def _apply_normalizations(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'irrad_poa' in df.columns and 'ghi' in df.columns:
            df['irr_clearsky_ratio'] = df['irrad_poa'] / df['ghi'].replace(0, np.nan)
            df['irr_clearsky_ratio'] = df['irr_clearsky_ratio'].fillna(0)

        if 'humidity' in df.columns: 
            df['humidity'] = df['humidity'] / 100.0

        cols_pot = ['target', 'P1', 'P2', 'P3']
        for col in cols_pot:
            if col in df.columns: 
                df[col] = df[col] / self.nominal_power

        if 'year' in df.columns:
            years_passed = (df['year'] - self.start_year).clip(lower=0)
            df['degradacao'] = 1 - (self.degradation_rate * years_passed)

        if self.target_col_internal == 'sky':
            if 'elevation' in df.columns:
                df['sin_elevation'] = np.sin(np.deg2rad(df['elevation']))

            for i in ['kt', 'fracao_difusa']:
                if i in df.columns:
                    df['delta_valor'] = df[i].diff()
                    df['diff_tempo'] = df.index.to_series().diff()
                    df[f'delta_{i}'] = df['delta_valor'].where(df['diff_tempo'] == pd.Timedelta('1h'))
                    df = df.drop(columns=["delta_valor", "diff_tempo"])

        return df

    def save_scalers(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler_x, os.path.join(output_dir, 'scaler_X.pkl'))
        print(f"💾 Scalers salvos em: {output_dir}")

    def load_scalers(self, input_dir: str):
        path_x = os.path.join(input_dir, 'scaler_X.pkl')
        self.scaler_x = joblib.load(path_x)
        self._is_fitted = True
        print(f"♻️  Scalers carregados de: {input_dir}")
