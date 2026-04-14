import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class PowerAnalyzer:
    def __init__(self, model_name, output_dir="analysis_outputs"):
        self.model_name = model_name
        self.save_dir = os.path.join(output_dir, "Potencia", model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set_style("whitegrid")
        
        # Paleta de cores e marcadores para diferenciar múltiplos modelos
        self.colors = ['b', 'g', 'r', 'c', 'orange', 'purple']
        self.markers = ['o', 's', '^', 'D', 'v', 'p']

    def _get_pred_cols(self, df):
        """Função auxiliar para encontrar todas as colunas de predição."""
        return [col for col in df.columns if 'power_pred' in col]

    def plot_clear_sky_day_analysis(self, df, min_samples=10):
        daily_stats = df.groupby(df.index.date)['power_real'].agg(['mean', 'count'])
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        
        if valid_days.empty:
            print(f"⚠️ Nenhum dia com pelo menos {min_samples} amostras foi encontrado.")
            return

        target_date = valid_days['mean'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        print(f"   ☀️ Dia selecionado ({len(df_day)} amostras): {target_date}")

        pred_cols = self._get_pred_cols(df)

        fig = plt.figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        
        # Plot Real e Persistência
        ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day['power_real'], 'k-o', label='Power Real', markersize=5, linewidth=2)
        if 'P1' in df_day.columns:
            ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day['P1'], 'm--*', label='Power Persis', markersize=4, alpha=0.6)
        
        # Plot Dinâmico das Predições
        for i, col in enumerate(pred_cols):
            c = self.colors[i % len(self.colors)]
            m = self.markers[i % len(self.markers)]
            model_label = col.replace('power_pred', 'Pred').replace('_', ' ')
            ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day[col], f'{c}--{m}', label=model_label, markersize=4, alpha=0.8)

        ax1.set_title(f"Dados Originais - {target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Potência")
        ax1.set_ylim(0, 1); ax1.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_ceu_claro_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_high_variability_day_analysis(self, df, min_samples=10):
        daily_stats = df.groupby(df.index.date)['power_real'].agg(['std', 'count'])
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        
        if valid_days.empty:
            print(f"⚠️ Nenhum dia com variabilidade válida foi encontrado.")
            return

        target_date = valid_days['std'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        print(f"   ⛈️ Dia selecionado para alta variabilidade ({len(df_day)} amostras): {target_date}")

        pred_cols = self._get_pred_cols(df)
        time_hours = df_day.index.hour + df_day.index.minute/60

        fig = plt.figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        
        ax1.plot(time_hours, df_day['power_real'], 'k-o', label='Power Real', markersize=5, alpha=0.9, linewidth=2)
        if 'P1' in df_day.columns:
            ax1.plot(time_hours, df_day['P1'], 'm--*', label='Power Persis', markersize=4, alpha=0.6)
        
        for i, col in enumerate(pred_cols):
            c = self.colors[i % len(self.colors)]
            m = self.markers[i % len(self.markers)]
            model_label = col.replace('power_pred', 'Pred').replace('_', ' ')
            ax1.plot(time_hours, df_day[col], f'{c}--{m}', label=model_label, markersize=4, alpha=0.8)

        ax1.set_title(f"Dados Originais (Variabilidade)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Potência")
        ax1.set_ylim(0, 1); ax1.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_variabilidade_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_overcast_day_analysis(self, df, min_samples=10):
        daily_stats = df.groupby(df.index.date)['power_real'].agg(['mean', 'count'])
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        
        if valid_days.empty:
            print(f"⚠️ Nenhum dia com dados suficientes para análise overcast.")
            return

        target_date = valid_days['mean'].loc[valid_days['mean']!=0].idxmin()
        df_day = df[df.index.date == target_date].copy()
        print(f"   ☁️ Dia selecionado como Encoberto ({len(df_day)} amostras): {target_date}")

        pred_cols = self._get_pred_cols(df)
        time_hours = df_day.index.hour + df_day.index.minute/60

        fig = plt.figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        
        ax1.plot(time_hours, df_day['power_real'], 'k-o', label='Power Real', markersize=5, linewidth=2)
        if 'P1' in df_day.columns:
            ax1.plot(time_hours, df_day['P1'], 'm--*', label='Power Persis', markersize=4, alpha=0.6)
        
        for i, col in enumerate(pred_cols):
            c = self.colors[i % len(self.colors)]
            m = self.markers[i % len(self.markers)]
            model_label = col.replace('power_pred', 'Pred').replace('_', ' ')
            ax1.plot(time_hours, df_day[col], f'{c}--{m}', label=model_label, markersize=4, alpha=0.8)

        ax1.set_title(f"Dados Originais (Encoberto)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Potência")
        ax1.set_ylim(0, 1); ax1.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_overcast_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_statistical_metrics(self, df):
        print(f"   📊 Calculando métricas estatísticas para: {self.model_name}")
        pred_cols = self._get_pred_cols(df)

        #pred_cols.append('P1') if 'P1' in df.columns else None # Inclui Persistência se existir

        y_true = df['power_real']
        
        summary = {'Modelo': self.model_name}

        for col in pred_cols:
            y_pred = df[col]
            # Extrair um identificador (ex: de 'power_pred_LSTM' tira o '_LSTM')
            name = col.replace('power_pred', 'power').strip('_')
            
            mean_obs = np.mean(y_true)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mbe = np.mean(y_pred - y_true)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            nrmse = (rmse / mean_obs) * 100 if mean_obs != 0 else np.nan
            nmbe = (mbe / mean_obs) * 100 if mean_obs != 0 else np.nan

            summary[f'{name}_Média_Obs'] = round(mean_obs, 4)
            summary[f'{name}_RMSE'] = round(rmse, 4)
            summary[f'{name}_nRMSE (%)'] = round(nrmse, 2)
            summary[f'{name}_MBE'] = round(mbe, 4)
            summary[f'{name}_nMBE (%)'] = round(nmbe, 2)
            summary[f'{name}_MAE'] = round(mae, 4)
            summary[f'{name}_R2'] = round(r2, 4)

        df_metrics = pd.DataFrame([summary])
        save_path = os.path.join(self.save_dir, "metricas_estatisticas.csv")
        df_metrics.to_csv(save_path, index=False, sep =';')
        print(f"   ✅ Métricas salvas em: {save_path}")
        return summary
    
    def plot_scatter_validation(self, df):
        print(f"   📈 Gerando scatter de validação estatística...")
        pred_cols = self._get_pred_cols(df)
        n_preds = len(pred_cols)
        
        if n_preds == 0:
            print("⚠️ Nenhuma coluna de predição encontrada para o scatter.")
            return

        # Ajusta o tamanho da figura dependendo da quantidade de predições
        fig, axes = plt.subplots(1, n_preds, figsize=(8 * n_preds, 7))
        # Garante que axes seja iterável mesmo com apenas 1 predição
        if n_preds == 1: axes = [axes] 

        y_true = df['power_real']
        mean_obs = np.mean(y_true)

        for i, p_col in enumerate(pred_cols):
            y_pred = df[p_col]
            ax = axes[i]
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = (rmse / mean_obs) * 100 if mean_obs != 0 else 0
            r2 = r2_score(y_true, y_pred)
            mbe = np.mean(y_pred - y_true)

            ax.scatter(y_true, y_pred, color=self.colors[i % len(self.colors)], alpha=0.3, s=10, label='Predições')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Ideal (1:1)')

            model_label = p_col.replace('power_pred', 'Pred').replace('_', ' ')
            title_str = (f"Validação {model_label}\n"
                         f"RMSE: {rmse:.4f} | nRMSE: {nrmse:.2f}%\n"
                         f"R²: {r2:.4f} | MBE: {mbe:.4f}")
            ax.set_title(title_str, fontsize=12)
            ax.set_xlabel("Power Observado")
            ax.set_ylabel(f"Power {model_label}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "scatter_validacao_estatistica.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_transient_day_analysis(self, df, min_samples=10):
        def successive_variability(x):
            return np.abs(x.diff()).sum()

        daily_stats = df.groupby(df.index.date)['power_real'].agg([successive_variability, 'mean', 'count'])
        valid_days = daily_stats[(daily_stats['count'] >= min_samples) & (daily_stats['mean'] > 0.2)]
        
        if valid_days.empty:
            print("⚠️ Nenhum dia com perfil transiente válido foi encontrado.")
            return

        target_date = valid_days['successive_variability'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        print(f"   🌦️ Dia Transiente selecionado (Alta Variabilidade Sucessiva): {target_date}")

        pred_cols = self._get_pred_cols(df)
        time_hours = df_day.index.hour + df_day.index.minute/60

        fig = plt.figure(figsize=(18, 5))
        ax1 = fig.add_subplot(1, 3, 1)
        
        ax1.plot(time_hours, df_day['power_real'], 'k-o', label='Power Real', markersize=5, alpha=0.9, linewidth=2)
        if 'P1' in df_day.columns:
            ax1.plot(time_hours, df_day['P1'], 'm--*', label='Power Persis', markersize=4, alpha=0.6)
        
        for i, col in enumerate(pred_cols):
            c = self.colors[i % len(self.colors)]
            m = self.markers[i % len(self.markers)]
            model_label = col.replace('power_pred', 'Pred').replace('_', ' ')
            ax1.plot(time_hours, df_day[col], f'{c}--{m}', label=model_label, markersize=4, alpha=0.8)

        ax1.set_title(f"Dados Originais (Transiente)\n{target_date}")
        ax1.set_xlabel("Hora do Dia")
        ax1.set_ylabel("Potência")
        ax1.set_ylim(0, 1)
        ax1.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_transiente_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()