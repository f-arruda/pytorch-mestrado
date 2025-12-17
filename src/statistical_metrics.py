import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import skill_metrics as sm
import os

class SolarStatisticalAnalyzer:
    def __init__(self, df_combined, output_dir):
        """
        Args:
            df_combined: DataFrame com colunas ['Observado', 'Previsto', 'Modelo', 'Horizonte', 'Hour', 'zenith'].
            output_dir: Pasta onde salvar os gráficos comparativos.
        """
        self.df = df_combined
        self.save_path = output_dir
        os.makedirs(self.save_path, exist_ok=True)
        
        # Filtro Diurno (Zenith < 70)
        if 'zenith' in self.df.columns:
            self.df_day = self.df[self.df['zenith'] < 70].copy()
            print(f"☀️ Filtro Diurno aplicado: {len(self.df_day)} amostras restantes.")
        else:
            self.df_day = self.df.copy()

    def _save_fig(self, fig, name):
        path = os.path.join(self.save_path, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close(fig)
        print(f"📊 Gráfico salvo: {path}")

    def save_global_metrics(self):
        """Gera CSV comparativo de métricas globais."""
        metrics_list = []
        
        for model_name in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == model_name]
            
            rmse = np.sqrt(mean_squared_error(sub['Observado'], sub['Previsto']))
            mae = mean_absolute_error(sub['Observado'], sub['Previsto'])
            r2 = r2_score(sub['Observado'], sub['Previsto'])
            
            metrics_list.append({
                'Modelo': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            })
            
        df_metrics = pd.DataFrame(metrics_list)
        csv_path = os.path.join(self.save_path, 'global_metrics_comparison.csv')
        df_metrics.to_csv(csv_path, index=False)
        print("📝 Tabela de métricas salva.")
        return df_metrics

    def plot_metrics_by_horizon(self):
        """Lineplot comparando pRMSE por horizonte para cada modelo."""
        metrics_list = []
        mean_obs = self.df_day['Observado'].mean()
        
        # Calcula métricas agrupadas
        for model in self.df_day['Modelo'].unique():
            for h in sorted(self.df_day['Horizonte'].unique()):
                sub = self.df_day[(self.df_day['Modelo'] == model) & (self.df_day['Horizonte'] == h)]
                if len(sub) == 0: continue
                
                rmse = np.sqrt(mean_squared_error(sub['Observado'], sub['Previsto']))
                prmse = (rmse / mean_obs) * 100
                metrics_list.append({'Modelo': model, 'Horizonte': h, 'pRMSE': prmse})
        
        df_plot = pd.DataFrame(metrics_list)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_plot, x='Horizonte', y='pRMSE', hue='Modelo', style='Modelo', markers=True, dashes=False, ax=ax)
        
        ax.set_title("Comparativo: pRMSE por Horizonte")
        ax.set_ylabel("pRMSE (%)")
        ax.grid(True, linestyle='--', alpha=0.5)
        self._save_fig(fig, "comparativo_prmse_horizonte")

    def plot_boxplots_hourly(self):
        """Boxplot comparativo lado a lado por hora."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Para o gráfico não ficar poluido, plotamos Observado + Modelos
        # O hue='Modelo' vai separar as cores
        sns.boxplot(data=self.df_day, x='Hour', y='Previsto', hue='Modelo', ax=ax, showfliers=False)
        
        # Adiciona a média do Observado como linha ou referência se desejar (opcional)
        # Por enquanto, foca na comparação entre modelos
        
        ax.set_title("Distribuição das Previsões por Hora")
        ax.set_ylabel("Potência")
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, "comparativo_boxplot_hora")

    def plot_taylor_diagram(self):
        """Diagrama de Taylor com múltiplos modelos."""
        # Configuração inicial
        fig = plt.figure(figsize=(9, 9))
        
        # Referência (Observado) - Usa o primeiro modelo para pegar o observado (é igual pra todos)
        first_model = self.df_day['Modelo'].unique()[0]
        ref_data = self.df_day[self.df_day['Modelo'] == first_model]['Observado']
        ref_std = np.std(ref_data)
        
        # Listas para o skill_metrics
        sdevs = [ref_std]
        crmsds = [0]
        ccoefs = [1]
        labels = ['Observado']
        
        # Coleta estatísticas de cada modelo
        for model in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == model]
            
            # Estatísticas
            std = np.std(sub['Previsto'])
            rmse = np.sqrt(mean_squared_error(sub['Observado'], sub['Previsto']))
            corr = np.corrcoef(sub['Observado'], sub['Previsto'])[0, 1]
            
            sdevs.append(std)
            crmsds.append(rmse)
            ccoefs.append(corr)
            labels.append(model)
            
        # Plotagem
        sdevs = np.array(sdevs)
        crmsds = np.array(crmsds)
        ccoefs = np.array(ccoefs)
        
        sm.taylor_diagram(sdevs, crmsds, ccoefs,
                          markerLabel=labels,
                          markerLegend='on',
                          styleOBS='-', colOBS='black', markerobs='h',
                          titleOBS='Ref')
        
        plt.title("Comparativo: Diagrama de Taylor")
        self._save_fig(fig, "comparativo_taylor")

    def plot_scatter_reg(self):
        """Scatter plots para cada modelo (subplot)."""
        models = self.df_day['Modelo'].unique()
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6), sharey=True)
        if n_models == 1: axes = [axes]
        
        for i, model in enumerate(models):
            sub = self.df_day[self.df_day['Modelo'] == model]
            
            # Regressão
            sns.regplot(data=sub, x='Observado', y='Previsto', 
                        scatter_kws={'alpha': 0.1, 's': 5}, line_kws={'color': 'red'}, ax=axes[i])
            
            # Linha 1:1
            max_val = max(sub['Observado'].max(), sub['Previsto'].max())
            axes[i].plot([0, max_val], [0, max_val], 'k--', linewidth=1)
            
            r2 = r2_score(sub['Observado'], sub['Previsto'])
            axes[i].set_title(f"{model} (R²: {r2:.3f})")
            
        plt.tight_layout()
        self._save_fig(fig, "comparativo_scatter")