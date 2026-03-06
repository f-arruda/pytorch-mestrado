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

    def plot_clear_sky_day_analysis(self, df, min_samples=10):
        """
        Identifica um dia de céu claro VÁLIDO (com amostras suficientes)
        e gera a análise de 3 gráficos.
        """
        # 1. Agrupamos para calcular a média e a contagem de amostras por dia
        daily_stats = df.groupby(df.index.date)['power_real'].agg(['mean', 'count'])
        
        # 2. FILTRO: Consideramos apenas dias com no mínimo 'min_samples'
        # Isso evita pegar dias com apenas 1 ou 2 pontos de dados
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        
        if valid_days.empty:
            print(f"⚠️ Nenhum dia com pelo menos {min_samples} amostras foi encontrado.")
            return

        # 3. Selecionamos o dia com maior power_real médio dentro dos dias válidos
        target_date = valid_days['mean'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        
        print(f"   ☀️ Dia selecionado ({len(df_day)} amostras): {target_date}")

        # --- Início da Plotagem (Lógica de 3 imagens mantida) ---
        fig = plt.figure(figsize=(18, 5))
        
        # Imagem 1: Dados Originais
        ax1 = fig.add_subplot(1, 3, 1)
        # Usamos scatter + line para garantir visibilidade se os pontos forem espaçados
        ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day['power_real'], 'k-o', label='Power Real', markersize=4)
        ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day['power_pred'], 'b-o', label='Power Pred', markersize=4)
        ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day['P1'], 'm-o', label='Power Persis', markersize=4)
        ax1.set_title(f"Dados Originais - {target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.1); ax1.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_ceu_claro_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_high_variability_day_analysis(self, df, min_samples=10):
        """
        Identifica o dia com maior variabilidade (volatilidade) de kt
        e gera a análise de 3 gráficos comparativos.
        """
        # 1. Identificação do dia de alta variabilidade
        # Calculamos o desvio padrão (std) do power_real por dia
        daily_stats = df.groupby(df.index.date)['power_real'].agg(['std', 'count'])
        
        # Filtro de amostras mínimas para garantir um dia completo
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        
        if valid_days.empty:
            print(f"⚠️ Nenhum dia com variabilidade válida foi encontrado.")
            return

        # Selecionamos o dia com MAIOR desvio padrão (mais "nervoso")
        target_date = valid_days['std'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        
        print(f"   ⛈️ Dia selecionado para alta variabilidade ({len(df_day)} amostras): {target_date}")

        fig = plt.figure(figsize=(18, 5))
        time_hours = df_day.index.hour + df_day.index.minute/60
        
        # --- Imagem 1: Dados Originais (Lineplot com alta oscilação) ---
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(time_hours, df_day['power_real'], 'k-o', label='Power Real', markersize=4, alpha=0.8)
        ax1.plot(time_hours, df_day['power_pred'], 'b-o', label='Power Pred', markersize=4)
        ax1.plot(time_hours, df_day['P1'], 'm-o', label='Power Persis', markersize=4)
        ax1.set_title(f"Dados Originais (Variabilidade)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2); ax1.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_variabilidade_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_overcast_day_analysis(self, df, min_samples=10):
        """
        Identifica o dia com menor índice de claridade médio (overcast)
        e gera a análise de 3 gráficos comparativos.
        """
        # 1. Identificação do dia encoberto
        # Agrupamos por data e calculamos a média de power_real
        daily_stats = df.groupby(df.index.date)['power_real'].agg(['mean', 'count'])
        
        # Filtro de amostras mínimas
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        
        if valid_days.empty:
            print(f"⚠️ Nenhum dia com dados suficientes para análise overcast.")
            return

        # Selecionamos o dia com a MENOR média de kt (dia mais escuro/fechado)
        target_date = valid_days['mean'].idxmin()
        df_day = df[df.index.date == target_date].copy()
        
        print(f"   ☁️ Dia selecionado como Encoberto ({len(df_day)} amostras): {target_date}")

        fig = plt.figure(figsize=(18, 5))
        time_hours = df_day.index.hour + df_day.index.minute/60
        
        # --- Imagem 1: Dados Originais (Overcast - Linhas baixas e retas) ---
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(time_hours, df_day['power_real'], 'k-o', label='Power Real', markersize=4)
        ax1.plot(time_hours, df_day['power_pred'], 'b-o', label='Power Pred', markersize=4)
        ax1.plot(time_hours, df_day['P1'], 'm-o', label='Power Persis', markersize=4)
        ax1.set_title(f"Dados Originais (Encoberto)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2); ax1.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_overcast_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def calculate_statistical_metrics(self, df):
        """
        Calcula métricas estatísticas (RMSE, nRMSE, MBE, nMBE, R2, MAE, Média)
        separadamente para kt e kd.
        """
        print(f"   📊 Calculando métricas estatísticas para: {self.model_name}")
        
        metrics_results = []

        # Lista de pares (Real, Predito, Nome da Variável)
        targets = [
            (df['power_real'], df['power_pred'], 'power')
        ]

        summary = {'Modelo': self.model_name}

        for y_true, y_pred, name in targets:
            # Cálculos base
            mean_obs = np.mean(y_true)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mbe = np.mean(y_pred - y_true)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            # Normalização (pela média das observações)
            nrmse = (rmse / mean_obs) * 100 if mean_obs != 0 else np.nan
            nmbe = (mbe / mean_obs) * 100 if mean_obs != 0 else np.nan

            # Armazenamento no dicionário de resumo
            summary[f'{name}_Média_Obs'] = round(mean_obs, 4)
            summary[f'{name}_RMSE'] = round(rmse, 4)
            summary[f'{name}_nRMSE (%)'] = round(nrmse, 2)
            summary[f'{name}_MBE'] = round(mbe, 4)
            summary[f'{name}_nMBE (%)'] = round(nmbe, 2)
            summary[f'{name}_MAE'] = round(mae, 4)
            summary[f'{name}_R2'] = round(r2, 4)

        # 2. Salvamento em CSV individual
        df_metrics = pd.DataFrame([summary])
        save_path = os.path.join(self.save_dir, "metricas_estatisticas_k.csv")
        df_metrics.to_csv(save_path, index=False)
        
        print(f"   ✅ Métricas salvas em: {save_path}")
        return summary
    
    def plot_scatter_validation(self, df):
        """
        Gera uma figura com dois gráficos scatter (kt e kd).
        Inclui métricas (RMSE, nRMSE, R2, MBE) no título e a linha de identidade.
        """
        print(f"   📈 Gerando scatter de validação estatística para: {self.model_name}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Configurações para kt e kd
        plot_configs = [
            {'real': 'power_real', 'pred': 'power_pred', 'name': 'Power', 'ax': axes[0]},
        ]

        for cfg in plot_configs:
            y_true = df[cfg['real']]
            y_pred = df[cfg['pred']]
            ax = cfg['ax']

            # 1. Cálculo das métricas para o título
            mean_obs = np.mean(y_true)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = (rmse / mean_obs) * 100 if mean_obs != 0 else 0
            r2 = r2_score(y_true, y_pred)
            mbe = np.mean(y_pred - y_true)

            # 2. Plotagem dos dados
            ax.scatter(y_true, y_pred, color='blue', alpha=0.3, s=10, label='Predições')
            
            # 3. Linha de Identidade (Modelo Perfeito)
            ax.plot([0, 1.1], [0, 1.1], 'r--', linewidth=2, label='Ideal (1:1)')

            # 4. Formatação do Título com Métricas
            title_str = (f"Validação {cfg['name']}\n"
                         f"RMSE: {rmse:.4f} | nRMSE: {nrmse:.2f}%\n"
                         f"R²: {r2:.4f} | MBE: {mbe:.4f}")
            ax.set_title(title_str, fontsize=12)
            ax.set_xlabel(f"{cfg['name']} Observado")
            ax.set_ylabel(f"{cfg['name']} Previsto")
            ax.set_xlim(0, 1.1)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        
        # Salvamento
        save_path = os.path.join(self.save_dir, "scatter_validacao_estatistica.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Scatter de validação salvo em: {save_path}")
        plt.show()

    def plot_transient_day_analysis(self, df, min_samples=10):
        """
        Identifica o dia com maior variabilidade sucessiva (padrão claro-nublado-claro)
        e gera a análise comparativa de 3 gráficos.
        """
        # 1. Função para calcular a variabilidade sucessiva (soma dos saltos)
        def successive_variability(x):
            return np.abs(x.diff()).sum()

        # 2. Agrupamos por data para encontrar o dia com mais "saltos" no kt real
        daily_stats = df.groupby(df.index.date)['power_real'].agg([successive_variability, 'mean', 'count'])
        
        # Filtro de amostras mínimas e foco em dias que não são puramente encobertos
        # (kt médio > 0.3 garante que houve momentos de céu claro)
        valid_days = daily_stats[(daily_stats['count'] >= min_samples) & (daily_stats['mean'] > 0.2)]
        
        if valid_days.empty:
            print("⚠️ Nenhum dia com perfil transiente válido foi encontrado.")
            return

        # Selecionamos o dia com a maior variabilidade sucessiva
        target_date = valid_days['successive_variability'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        
        print(f"   🌦️ Dia Transiente selecionado (Alta Variabilidade Sucessiva): {target_date}")

        fig = plt.figure(figsize=(18, 5))
        time_hours = df_day.index.hour + df_day.index.minute/60
        
        # --- Imagem 1: Dados Originais (Lineplot com padrão ioiô) ---
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(time_hours, df_day['power_real'], 'k-o', label='Power Real', markersize=4, alpha=0.8)
        ax1.plot(time_hours, df_day['power_pred'], 'b-o', label='Power Pred', markersize=4)
        ax1.plot(time_hours, df_day['P1'], 'm-o', label='Power Persis', markersize=4)
        ax1.set_title(f"Dados Originais (Transiente)\n{target_date}")
        ax1.set_xlabel("Hora do Dia")
        ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2)
        ax1.legend()


        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_transiente_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()