import pandas as pd
import numpy as np
import os
import joblib  # <--- Importante para salvar
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import warnings

# Ignorar avisos de chained assignment
warnings.filterwarnings('ignore')

class SolarPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 nominal_power=156.0, 
                 start_year=2018,
                 degradation_rate=0.05,
                 features_to_scale=None,
                 target_col='Pot_BT'): # <--- Definimos explicitamente o alvo
        """
        Classe para limpeza e engenharia de features de dados solares.
        """
        self.nominal_power = nominal_power
        self.start_year = start_year
        self.degradation_rate = degradation_rate
        self.target_col = target_col
        
        # Constantes Físicas
        self.U0 = 25.0
        self.U1 = 6.84
        self.Pstc = 156000.0
        self.Gstc = 1000.0
        self.Tstc = 25.0
        self.gamma = 0.0045
        
        self.features_to_scale = features_to_scale or [
            'Temperatura ambiente °C', 
            'Velocidade média do vento m/s', 
            'Temp'
        ]
        
        # --- MUDANÇA: Dois Scalers Separados ---
        # scaler_x: Para as variáveis de entrada (multivariado)
        # scaler_y: EXCLUSIVO para o target (univariado) -> Crucial para inverse_transform depois
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        self._is_fitted = False

    def fit(self, X, y=None):
        """Aprende os parâmetros de normalização."""
        # 1. Faz uma cópia e aplica engenharia básica para ter as colunas calculadas (se necessário)
        # Nota: O fit deve aprender com os dados JÁ processados ou brutos? 
        # Geralmente aprendemos com os dados brutos ou parcialmente processados.
        # Aqui vamos simplificar: O fit aprende nas colunas que existirem no X passado.
        
        # Fit nas Features (X)
        if self.features_to_scale:
            cols_exist = [c for c in self.features_to_scale if c in X.columns]
            if cols_exist:
                self.scaler_x.fit(X[cols_exist])
        
        # Fit no Target (Y)
        # Atenção: Precisamos fitar o scaler_y nos dados JÁ normalizados pela potência nominal?
        # A sua lógica divide por 156. Se dividirmos por 156, o valor fica entre 0 e 1.
        # O MinMaxScaler vai aprender min=0 e max=~1. Isso funciona bem.
        # Se a coluna target existir, fitamos nela.
        if self.target_col in X.columns:
            # Precisamos simular a normalização de potência nominal antes de fitar o scaler
            # ou fitar nos dados brutos e deixar o scaler fazer todo o trabalho.
            # PARA MANTER SUA LÓGICA DE DIVISÃO POR 156: 
            # Vamos fitar o scaler nos valores brutos divididos por 156.
            
            target_values = X[[self.target_col]].copy()
            target_values = target_values / self.nominal_power # Aplica sua lógica base
            
            # Fita o scaler nesses valores já pré-processados
            self.scaler_y.fit(target_values)

        self._is_fitted = True
        return self

    def transform(self, X):
        """Aplica as transformações."""
        df = X.copy()
        
        # 1 a 5: Engenharia de Features (Física, Lags, etc)
        df = self._clean_index(df)
        df = self._calculate_physics_features(df)
        df = self._create_lag_features(df)
        df = self._create_angular_features(df)
        df = self._apply_custom_normalization(df) # Aqui divide Pot_BT por 156
        
        # 6. Normalização Sklearn (MinMax) nas Features
        cols_exist = [c for c in self.features_to_scale if c in df.columns]
        if cols_exist:
            df[cols_exist] = self.scaler_x.transform(df[cols_exist])
            
        # 7. Normalização Sklearn no Target (Opcional, mas bom para garantir 0-1)
        # Como já dividimos por 156, o valor já está quase em 0-1.
        # O scaler_y aqui vai apenas garantir o range exato se necessário,
        # mas sua principal função será ser salvo para o inverse_transform depois.
        if self.target_col in df.columns:
            df[[self.target_col]] = self.scaler_y.transform(df[[self.target_col]])

        df = self._handle_missing_values(df)
        return df
    
    def load_scalers(self, input_dir):
        """Carrega scalers salvos para reutilizar a normalização do treino."""
        # Procura pelos arquivos na pasta indicada
        scaler_x_path = os.path.join(input_dir, 'scaler_X.pkl')
        scaler_y_path = os.path.join(input_dir, 'scaler_Y.pkl')
        
        if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
            raise FileNotFoundError(f"Scalers não encontrados em {input_dir}. Certifique-se de que scaler_X.pkl e scaler_Y.pkl estão lá.")
            
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        
        # O segredo: marca como 'fitado' para o sklearn permitir o transform
        self._is_fitted = True 
        print(f"♻️ Scalers carregados com sucesso de: {input_dir}")

    def save_scalers(self, output_dir):
        """Salva os scalers em disco para uso posterior (análise)."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Salva o Scaler X
        joblib.dump(self.scaler_x, os.path.join(output_dir, 'scaler_X.pkl'))
        
        # Salva o Scaler Y (CRUCIAL PARA O SCALLER_Y QUE VOCÊ PERGUNTOU)
        joblib.dump(self.scaler_y, os.path.join(output_dir, 'scaler_Y.pkl'))
        
        print(f"💾 Scalers salvos em: {output_dir}")

    # --- MÉTODOS PRIVADOS (Mantidos do seu código) ---
    def _clean_index(self, df):
        df = df.loc[(df['Year'] >= self.start_year)].copy()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'Date_Time' in df.columns:
                df['Date_Time'] = pd.to_datetime(df['Date_Time'])
                df.set_index('Date_Time', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep='first')]
        return df.sort_index()

    def _calculate_physics_features(self, df):
        term_vento = self.U0 + self.U1 * df['Velocidade média do vento m/s']
        df['Temperatura celula'] = df['Temperatura ambiente °C'] + (df['ghi'] / term_vento)
        fator_temp = 1 - self.gamma * (df['Temperatura celula'] - self.Tstc)
        df['Pot_cs'] = self.Pstc * (df['ghi'] / self.Gstc) * fator_temp / 1000.0
        
        df['k'] = df['Pot_BT'] / df['Pot_cs'].replace(0, np.nan)
        df['k'] = df['k'].fillna(0)
        df['k'] = df['k'].clip(upper=1.0)
        return df

    def _create_lag_features(self, df):
        df['k_lag1'] = df['k'].shift(1)
        df['k_lag2'] = df['k'].shift(2)
        df['k_lag3'] = df['k'].shift(3)
        
        df['P1'] = df['Pot_cs'] * df['k_lag1']
        df['P2'] = df['Pot_cs'] * df['k_lag2']
        df['P3'] = df['Pot_cs'] * df['k_lag3']
        
        df.drop(['k_lag1', 'k_lag2', 'k_lag3'], axis=1, inplace=True)
        return df

    def _create_angular_features(self, df):
        df['cos_zenith']  = np.cos(np.deg2rad(df['zenith']))
        df['sin_azimuth'] = np.sin(np.deg2rad(df['azimuth']))
        return df

    def _apply_custom_normalization(self, df):
        # Normalizações gerais
        df['Irrad/CeuClaro'] = df['Irradiação Global horária(Inclinada 27°) kWh/m2'] / df['ghi'].replace(0, np.nan)
        df['Umidade Relativa %'] = df['Umidade Relativa %'] / 100.0
        extra_rad = df['Extra Radiation'].replace(0, np.nan)
        df['Irradiação Global horária(Inclinada 27°) kWh/m2'] = df['Irradiação Global horária(Inclinada 27°) kWh/m2'] / extra_rad
        df['Irrad'] = df['Irrad'] / extra_rad

        # Normalização pela potência nominal (Isso acontece ANTES do scaler)
        cols_potencia = ['Pot_BT', 'P1', 'P2', 'P3', 'P']
        for col in cols_potencia:
            if col in df.columns:
                df[col] = df[col] / self.nominal_power
                
        # Degradação
        years_passed = (df['Year'] - self.start_year).clip(lower=0)
        df['Degradacao'] = 1 - (self.degradation_rate * years_passed)
        
        return df

    def _handle_missing_values(self, df):
        fill_neg_one = ['Pot_BT', 'Irrad', 'Temp']
        for col in fill_neg_one:
            if col in df.columns:
                df[col] = df[col].replace([np.nan, np.inf, -np.inf], -1)
        df.fillna(0, inplace=True)
        return df