import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class KFactorMetricsCalculator:
    """
    Classe responsável estritamente por cálculos matemáticos e estatísticos.
    Princípio de Responsabilidade Única (SRP): Não lida com plotagem, apenas processamento de dados.
    """
    def __init__(self, model_name, output_dir="analysis_outputs"):
        self.model_name = model_name
        self.save_dir = os.path.join(output_dir, "Fisica_Atmosferica", model_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def get_model_suffixes(self, df):
        """Função auxiliar para encontrar todos os sufixos de modelos nas colunas kt_pred."""
        kt_cols = [col for col in df.columns if col.startswith('kt_pred')]
        suffixes = [col[len('kt_pred'):] for col in kt_cols]
        return suffixes

    def calculate_statistical_metrics(self, df):
        print(f"   📊 Calculando métricas estatísticas para: {self.model_name}")
        suffixes = self.get_model_suffixes(df)
        summary = {'Modelo': self.model_name}

        for suff in suffixes:
            model_tag = suff.strip('_') if suff.strip('_') else 'Base'
            
            # Análise para kt
            y_true_kt = df['kt_real']
            y_pred_kt = df[f'kt_pred{suff}']
            self._compute_metrics(y_true_kt, y_pred_kt, f"kt_{model_tag}", summary)
            
            # Análise para kd
            if f'kd_pred{suff}' in df.columns:
                y_true_kd = df['kd_real']
                y_pred_kd = df[f'kd_pred{suff}']
                self._compute_metrics(y_true_kd, y_pred_kd, f"kd_{model_tag}", summary)

        df_metrics = pd.DataFrame([summary])
        save_path = os.path.join(self.save_dir, "metricas_estatisticas_k.csv")
        df_metrics.to_csv(save_path, index=False)
        print(f"   ✅ Métricas salvas em: {save_path}")
        return summary

    def _compute_metrics(self, y_true, y_pred, name_prefix, summary_dict):
        """Função auxiliar para organizar o cálculo repetitivo de métricas."""
        mean_obs = np.mean(y_true)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mbe = np.mean(y_pred - y_true)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        nrmse = (rmse / mean_obs) * 100 if mean_obs != 0 else np.nan
        nmbe = (mbe / mean_obs) * 100 if mean_obs != 0 else np.nan

        summary_dict[f'{name_prefix}_Média_Obs'] = round(mean_obs, 4)
        summary_dict[f'{name_prefix}_RMSE'] = round(rmse, 4)
        summary_dict[f'{name_prefix}_nRMSE (%)'] = round(nrmse, 2)
        summary_dict[f'{name_prefix}_MBE'] = round(mbe, 4)
        summary_dict[f'{name_prefix}_nMBE (%)'] = round(nmbe, 2)
        summary_dict[f'{name_prefix}_MAE'] = round(mae, 4)
        summary_dict[f'{name_prefix}_R2'] = round(r2, 4)
