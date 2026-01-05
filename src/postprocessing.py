import numpy as np
from abc import ABC, abstractmethod

class PredictionStrategy(ABC):
    """
    Define como converter a saída do modelo (y_pred) para a grandeza real (kW).
    """
    @abstractmethod
    def reconstruct(self, pred_val, true_val, metadata_row, scaler_y=None, nominal_power=156.0):
        pass

class DirectStrategy(PredictionStrategy):
    """
    Modo 'direct': O modelo previu a Potência (normalizada).
    """
    def reconstruct(self, pred_val, true_val, metadata_row, scaler_y=None, nominal_power=156.0):
        # Cenário A: Treino antigo com MinMaxScaler (scaler_y existe)
        if scaler_y is not None:
            # O inverse_transform espera array 2D
            pred_kw = scaler_y.inverse_transform([[pred_val]])[0][0]
            obs_kw = scaler_y.inverse_transform([[true_val]])[0][0]
            # Se o scaler foi fitado na potência normalizada (0-1), 
            # ainda precisamos multiplicar pela nominal dependendo de como foi feito o fit.
            # Mas geralmente o inverse_transform já volta para a escala original do fit.
            # Assumindo que o fit foi feito em (kW / P_nom), o inverse volta para (kW / P_nom).
            # Então multiplicamos por nominal_power.
            # (Ajuste conforme seu preprocessing antigo). 
            # Se o fit foi direto no kW, não multiplica. 
            # Dado o histórico, assumimos fit no fator de capacidade:
            pred_kw = pred_kw * nominal_power
            obs_kw = obs_kw * nominal_power

        # Cenário B: Treino novo (apenas divisão por 156, sem scaler_y)
        else:
            pred_kw = pred_val * nominal_power
            obs_kw = true_val * nominal_power
            
        return max(0, pred_kw), max(0, obs_kw)

class ClearSkyRatioStrategy(PredictionStrategy):
    """
    Modo 'clearsky_ratio' (ou k-factor): O modelo previu o Índice de Claridade (k).
    Fórmula: kW = k * Potencia_Ceu_Limpo(t)
    """
    def reconstruct(self, pred_val, true_val, metadata_row, scaler_y=None, nominal_power=156.0):
        # Aqui assumimos que 'k' não passa por scaler_y (já é 0-1.2)
        # Se passar, você deve adicionar a lógica de inverse_transform aqui.
        
        k_pred = pred_val
        k_true = true_val
        
        # Trava de segurança física (k não pode ser negativo nem infinito)
        k_pred = max(0.0, min(k_pred, 2.0))
        
        # Resgata Potência Teórica calculada pelo Preprocessor
        pot_cs = metadata_row.get('pot_cs', 0.0)
        
        pred_kw = k_pred * pot_cs
        obs_kw = k_true * pot_cs
        
        return max(0, pred_kw), max(0, obs_kw)

def get_strategy(mode: str) -> PredictionStrategy:
    strategies = {
        'direct': DirectStrategy(),
        'clearsky_ratio': ClearSkyRatioStrategy(),
        # Adicione novos modos aqui (ex: 'diff')
    }
    # Fallback para direct se o modo não for reconhecido
    return strategies.get(mode, DirectStrategy())