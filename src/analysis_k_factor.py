import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class KFactorAnalyzer:
    def __init__(self, model_name, output_dir="analysis_outputs"):
        self.model_name = model_name
        self.save_dir = os.path.join(output_dir, "Fisica_Atmosferica", model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        sns.set_style("whitegrid")

    def plot_kt_kd_relationship(self, df):
        """
        Gera 3 plots comparativos: Observado, Previsto e Sobreposição.
        Eixo X: kt | Eixo Y: fração difusa (kd)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # Parâmetros estéticos
        scatter_kwargs = {'s': 10, 'alpha': 0.4, 'edgecolor': 'none'}
        xlims = (0, 1.1)
        ylims = (0, 1.1)

        # 1. Gráfico Dados Observados
        axes[0].scatter(df['kt_real'], df['kd_real'], color='black', **scatter_kwargs)
        axes[0].set_title(f"Dados Observados\n({self.model_name})")
        axes[0].set_xlabel("$k_t$ (Real)")
        axes[0].set_ylabel("$k_d$ (Real)")
        axes[0].set_xlim(xlims); axes[0].set_ylim(ylims)

        # 2. Gráfico Dados Previstos
        axes[1].scatter(df['kt_pred'], df['kd_pred'], color='blue', **scatter_kwargs)
        axes[1].set_title(f"Dados Previstos\n(PINN Output)")
        axes[1].set_xlabel("$k_t$ (Previsto)")
        axes[1].set_xlim(xlims)

        # 3. Sobreposição (Overlay)
        axes[2].scatter(df['kt_real'], df['kd_real'], color='gray', label='Real', **scatter_kwargs)
        axes[2].scatter(df['kt_pred'], df['kd_pred'], color='red', label='Previsto', s=8, alpha=0.3)
        #axes[2].scatter(df['P_kt'], df['P_fracao_difusa'], color='m', label='Previsto', s=8, alpha=0.4)
        axes[2].set_title("Sobreposição\n(Aderência Física)")
        axes[2].set_xlabel("$k_t$")
        axes[2].legend(loc='upper right', markerscale=2)
        axes[2].set_xlim(xlims)

        plt.tight_layout()
        
        # Salvamento
        save_path = os.path.join(self.save_dir, "relacao_kt_kd_3_plots.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Figura salva em: {save_path}")
        plt.show()

    def plot_clear_sky_day_analysis(self, df, min_samples=10):
        """
        Identifica um dia de céu claro VÁLIDO (com amostras suficientes)
        e gera a análise de 3 gráficos.
        """
        # 1. Agrupamos para calcular a média e a contagem de amostras por dia
        daily_stats = df.groupby(df.index.date)['kt_real'].agg(['mean', 'count'])
        
        # 2. FILTRO: Consideramos apenas dias com no mínimo 'min_samples'
        # Isso evita pegar dias com apenas 1 ou 2 pontos de dados
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        
        if valid_days.empty:
            print(f"⚠️ Nenhum dia com pelo menos {min_samples} amostras foi encontrado.")
            return

        # 3. Selecionamos o dia com maior kt_real médio dentro dos dias válidos
        target_date = valid_days['mean'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        
        print(f"   ☀️ Dia selecionado ({len(df_day)} amostras): {target_date}")

        # --- Início da Plotagem (Lógica de 3 imagens mantida) ---
        fig = plt.figure(figsize=(18, 5))
        
        # Imagem 1: Dados Originais
        ax1 = fig.add_subplot(1, 3, 1)
        # Usamos scatter + line para garantir visibilidade se os pontos forem espaçados
        ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day['kt_real'], 'k-o', label='$k_t$ Real', markersize=4)
        ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day['kt_pred'], 'b-o', label='$k_t$ Pred', markersize=4)
        ax1.plot(df_day.index.hour + df_day.index.minute/60, df_day['P_kt'], 'm-o', label='$k_t$ Persis', markersize=4)
        ax1.set_title(f"Dados Originais - {target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.1); ax1.legend()

        # Imagem 2: Dados Previstos
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.plot(df_day.index.hour + df_day.index.minute/60, df_day['kd_real'], 'r--o', label='$k_d$ Real', markersize=3)
        ax2.plot(df_day.index.hour + df_day.index.minute/60, df_day['kd_pred'], 'g--o', label='$k_d$ Pred', markersize=3)
        ax2.plot(df_day.index.hour + df_day.index.minute/60, df_day['P_fracao_difusa'], 'm--o', label='$k_d$ Persis', markersize=3)
        ax2.set_title(f"Dados Previstos - {self.model_name}")
        ax2.set_xlabel("Hora do Dia")
        ax2.legend()

        # Imagem 3: Correlação Física do Dia
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(df_day['kt_real'], df_day['kd_real'], color='gray', alpha=0.7, label='Real', s=40)
        ax3.scatter(df_day['kt_pred'], df_day['kd_pred'], color='red', alpha=0.7, label='Pred', s=25)
        #ax3.scatter(df_day['P_kt'], df_day['P_fracao_difusa'], color='m', alpha=0.5, label='Pred', s=25)
        ax3.set_title("Espalhamento kt vs kd (Dia Selecionado)")
        ax3.set_xlabel("$k_t$"); ax3.set_ylabel("$k_d$")
        ax3.set_xlim(0, 1.1); ax3.set_ylim(0, 1.1); ax3.legend()

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
        # Calculamos o desvio padrão (std) do kt_real por dia
        daily_stats = df.groupby(df.index.date)['kt_real'].agg(['std', 'count'])
        
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
        ax1.plot(time_hours, df_day['kt_real'], 'k-o', label='$k_t$ Real', markersize=4, alpha=0.8)
        ax1.plot(time_hours, df_day['kt_pred'], 'b-o', label='$k_t$ Pred', markersize=4)
        ax1.plot(time_hours, df_day['P_kt'], 'm-o', label='$k_t$ Persis', markersize=4)
        ax1.set_title(f"Dados Originais (Variabilidade)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2); ax1.legend()

        # --- Imagem 2: Dados Previstos (Como a PINN lidou com os saltos?) ---
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.plot(time_hours, df_day['kd_real'], 'r--o', label='$k_d$ Real', markersize=3, alpha=0.6)
        ax2.plot(time_hours, df_day['kd_pred'], 'g--o', label='$k_d$ Pred', markersize=3)
        ax2.plot(time_hours, df_day['P_fracao_difusa'], 'm--o', label='$k_d$ Persis', markersize=3)
        ax2.set_title(f"Dados Previstos - {self.model_name}")
        ax2.set_xlabel("Hora do Dia")
        ax2.legend()

        # --- Imagem 3: Correlação Física (Dispersão no Caos) ---
        ax3 = fig.add_subplot(1, 3, 3)
        # Pontos reais em cinza para o fundo
        ax3.scatter(df_day['kt_real'], df_day['kd_real'], color='gray', alpha=0.5, label='Real', s=40)
        # Pontos preditos em vermelho para destaque
        ax3.scatter(df_day['kt_pred'], df_day['kd_pred'], color='red', alpha=0.7, label='Pred', s=25)
        #ax3.scatter(df_day['P_kt'], df_day['P_fracao_difusa'], color='m', alpha=0.5, label='Pred', s=25)
        ax3.set_title("Espalhamento kt vs kd (Dia Transiente)")
        ax3.set_xlabel("$k_t$"); ax3.set_ylabel("$k_d$")
        ax3.set_xlim(0, 1.1); ax3.set_ylim(0, 1.1); ax3.legend()

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
        # Agrupamos por data e calculamos a média de kt_real
        daily_stats = df.groupby(df.index.date)['kt_real'].agg(['mean', 'count'])
        
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
        ax1.plot(time_hours, df_day['kt_real'], 'k-o', label='$k_t$ Real', markersize=4)
        ax1.plot(time_hours, df_day['kt_pred'], 'b-o', label='$k_t$ Pred', markersize=4)
        ax1.plot(time_hours, df_day['P_kt'], 'm-o', label='$k_t$ Persis', markersize=4)
        ax1.set_title(f"Dados Originais (Encoberto)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2); ax1.legend()

        # --- Imagem 2: Dados Previstos (A PINN entendeu que o céu está fechado?) ---
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.plot(time_hours, df_day['kd_pred'], 'g--o', label='$k_d$ Pred', markersize=3)
        ax2.plot(time_hours, df_day['kd_real'], 'r--o', label='$k_d$ Real', markersize=3)
        ax2.plot(time_hours, df_day['P_fracao_difusa'], 'm--o', label='$k_d$ Persis', markersize=3)
        ax2.set_title(f"Dados Previstos - {self.model_name}")
        ax2.set_xlabel("Hora do Dia")
        ax2.legend()

        # --- Imagem 3: Correlação Física (A "Cabeça" da curva kt vs kd) ---
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(df_day['kt_real'], df_day['kd_real'], color='gray', alpha=0.7, label='Real', s=40)
        ax3.scatter(df_day['kt_pred'], df_day['kd_pred'], color='red', alpha=0.7, label='Pred', s=25)
        #ax3.scatter(df_day['P_kt'], df_day['P_fracao_difusa'], color='m', alpha=0.5, label='Pred', s=25)
        ax3.set_title("Espalhamento kt vs kd (Céu Encoberto)")
        ax3.set_xlabel("$k_t$")
        ax3.set_ylabel("$k_d$")
        ax3.set_xlim(0, 1.1); ax3.set_ylim(0, 1.1); ax3.legend()

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
            (df['kt_real'], df['kt_pred'], 'kt'),
            (df['kd_real'], df['kd_pred'], 'kd')
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
            {'real': 'kt_real', 'pred': 'kt_pred', 'name': '$k_t$', 'ax': axes[0]},
            {'real': 'kd_real', 'pred': 'kd_pred', 'name': '$k_d$', 'ax': axes[1]}
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
        daily_stats = df.groupby(df.index.date)['kt_real'].agg([successive_variability, 'mean', 'count'])
        
        # Filtro de amostras mínimas e foco em dias que não são puramente encobertos
        # (kt médio > 0.3 garante que houve momentos de céu claro)
        valid_days = daily_stats[(daily_stats['count'] >= min_samples) & (daily_stats['mean'] > 0.3)]
        
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
        ax1.plot(time_hours, df_day['kt_real'], 'k-o', label='$k_t$ Real', markersize=4, alpha=0.8)
        ax1.plot(time_hours, df_day['kt_pred'], 'b-o', label='$k_t$ Pred', markersize=4)
        ax1.plot(time_hours, df_day['P_kt'], 'm-o', label='$k_t$ Persis', markersize=4)
        ax1.set_title(f"Dados Originais (Transiente)\n{target_date}")
        ax1.set_xlabel("Hora do Dia")
        ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2)
        ax1.legend()

        # --- Imagem 2: Dados Previstos (Lineplot - Resposta do Modelo) ---
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.plot(time_hours, df_day['kd_pred'], 'g--o', label='$k_d$ Pred', markersize=3)
        ax2.plot(time_hours, df_day['kd_real'], 'r--o', label='$k_d$ Real', markersize=3, alpha=0.6)
        ax2.plot(time_hours, df_day['P_fracao_difusa'], 'm--o', label='$k_d$ Persis', markersize=3)
        ax2.set_title(f"Dados Previstos - {self.model_name}")
        ax2.set_xlabel("Hora do Dia")
        ax2.set_ylim(0, 1.2)
        ax2.legend()

        # --- Imagem 3: Scatter kt vs kd (Física do Dia Transiente) ---
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(df_day['kt_real'], df_day['kd_real'], color='gray', alpha=0.6, label='Real', s=40)
        ax3.scatter(df_day['kt_pred'], df_day['kd_pred'], color='red', alpha=0.7, label='Pred', s=25)
        #ax3.scatter(df_day['P_kt'], df_day['P_fracao_difusa'], color='m', alpha=0.5, label='Pred', s=25)
        ax3.set_title("Correlação Física (Espalhamento)")
        ax3.set_xlabel("$k_t$")
        ax3.set_ylabel("$k_d$")
        ax3.set_xlim(0, 1.1)
        ax3.set_ylim(0, 1.1)
        ax3.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_transiente_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()