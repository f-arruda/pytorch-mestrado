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
        
        # Paleta de cores e marcadores dinâmicos para múltiplos modelos
        self.colors = ['b', 'g', 'r', 'c', 'orange', 'purple']
        self.markers = ['o', 's', '^', 'D', 'v', 'p']

    def _get_model_suffixes(self, df):
        """Função auxiliar para encontrar todos os sufixos de modelos nas colunas kt_pred."""
        kt_cols = [col for col in df.columns if col.startswith('kt_pred')]
        # Extrai o sufixo (ex: de 'kt_pred_LSTM' extrai '_LSTM', de 'kt_pred' extrai '')
        suffixes = [col[len('kt_pred'):] for col in kt_cols]
        return suffixes

    def plot_kt_kd_relationship(self, df):
        """
        Gera 3 plots comparativos: Observado, Previstos (Dinâmico) e Sobreposição.
        """
        suffixes = self._get_model_suffixes(df)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        scatter_kwargs = {'s': 10, 'alpha': 0.4, 'edgecolor': 'none'}
        xlims, ylims = (0, 1.1), (0, 1.1)

        # 1. Dados Observados
        axes[0].scatter(df['kt_real'], df['kd_real'], color='black', **scatter_kwargs)
        axes[0].set_title(f"Dados Observados\n({self.model_name})")
        axes[0].set_xlabel("$k_t$ (Real)")
        axes[0].set_ylabel("$k_d$ (Real)")
        axes[0].set_xlim(xlims); axes[0].set_ylim(ylims)

        # 2. Dados Previstos (Dinâmico para múltiplos modelos)
        for i, suff in enumerate(suffixes):
            c = self.colors[i % len(self.colors)]
            m = self.markers[i % len(self.markers)]
            model_label = f"Pred{suff.replace('_', ' ')}"
            if f'kd_pred{suff}' in df.columns:
                axes[1].scatter(df[f'kt_pred{suff}'], df[f'kd_pred{suff}'], color=c, marker=m, label=model_label, **scatter_kwargs)
        
        axes[1].set_title("Dados Previstos")
        axes[1].set_xlabel("$k_t$ (Previsto)")
        axes[1].set_xlim(xlims)
        axes[1].legend()

        # 3. Sobreposição
        axes[2].scatter(df['kt_real'], df['kd_real'], color='gray', label='Real', **scatter_kwargs)
        for i, suff in enumerate(suffixes):
            c = self.colors[i % len(self.colors)]
            m = self.markers[i % len(self.markers)]
            model_label = f"Pred{suff.replace('_', ' ')}"
            if f'kd_pred{suff}' in df.columns:
                axes[2].scatter(df[f'kt_pred{suff}'], df[f'kd_pred{suff}'], color=c, marker=m, label=model_label, s=8, alpha=0.5)
        
        axes[2].set_title("Sobreposição\n(Aderência Física)")
        axes[2].set_xlabel("$k_t$")
        axes[2].legend(loc='upper right', markerscale=2)
        axes[2].set_xlim(xlims)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "relacao_kt_kd_3_plots.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_clear_sky_day_analysis(self, df, min_samples=10):
        daily_stats = df.groupby(df.index.date)['kt_real'].agg(['mean', 'count'])
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        
        if valid_days.empty: return

        target_date = valid_days['mean'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        print(f"   ☀️ Dia selecionado ({len(df_day)} amostras): {target_date}")

        suffixes = self._get_model_suffixes(df)
        fig = plt.figure(figsize=(18, 5))
        time_hours = df_day.index.hour + df_day.index.minute/60
        
        # Imagem 1: Dados Originais (kt)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(time_hours, df_day['kt_real'], 'k-o', label='$k_t$ Real', markersize=5, linewidth=2)
        if 'P_kt' in df_day.columns:
            ax1.plot(time_hours, df_day['P_kt'], 'm--*', label='$k_t$ Persis', markersize=4, alpha=0.6)
        
        for i, suff in enumerate(suffixes):
            c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
            ax1.plot(time_hours, df_day[f'kt_pred{suff}'], f'{c}--{m}', label=f"Pred{suff.replace('_', ' ')}", markersize=4, alpha=0.8)
        
        ax1.set_title(f"Dados Originais - {target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.1); ax1.legend()

        # Imagem 2: Dados Previstos (kd)
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.plot(time_hours, df_day['kd_real'], 'r--o', label='$k_d$ Real', markersize=4, alpha=0.6, linewidth=2)
        if 'P_fracao_difusa' in df_day.columns:
            ax2.plot(time_hours, df_day['P_fracao_difusa'], 'm--*', label='$k_d$ Persis', markersize=3, alpha=0.6)
        
        for i, suff in enumerate(suffixes):
            if f'kd_pred{suff}' in df_day.columns:
                c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
                ax2.plot(time_hours, df_day[f'kd_pred{suff}'], f'{c}--{m}', label=f"Pred{suff.replace('_', ' ')}", markersize=3, alpha=0.8)
        
        ax2.set_title("Fração Difusa Prevista")
        ax2.set_xlabel("Hora do Dia"); ax2.legend()

        # Imagem 3: Correlação Física
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(df_day['kt_real'], df_day['kd_real'], color='gray', alpha=0.7, label='Real', s=40)
        for i, suff in enumerate(suffixes):
            if f'kd_pred{suff}' in df_day.columns:
                c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
                ax3.scatter(df_day[f'kt_pred{suff}'], df_day[f'kd_pred{suff}'], color=c, marker=m, alpha=0.7, label=f"Pred{suff.replace('_', ' ')}", s=25)
        
        ax3.set_title("Espalhamento kt vs kd (Dia Selecionado)")
        ax3.set_xlabel("$k_t$"); ax3.set_ylabel("$k_d$")
        ax3.set_xlim(0, 1.1); ax3.set_ylim(0, 1.1); ax3.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_ceu_claro_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_high_variability_day_analysis(self, df, min_samples=10):
        daily_stats = df.groupby(df.index.date)['kt_real'].agg(['std', 'count'])
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        if valid_days.empty: return

        target_date = valid_days['std'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        print(f"   ⛈️ Dia selecionado para alta variabilidade: {target_date}")

        suffixes = self._get_model_suffixes(df)
        time_hours = df_day.index.hour + df_day.index.minute/60
        fig = plt.figure(figsize=(18, 5))
        
        # Imagem 1: kt
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(time_hours, df_day['kt_real'], 'k-o', label='$k_t$ Real', markersize=5, linewidth=2, alpha=0.9)
        if 'P_kt' in df_day.columns: ax1.plot(time_hours, df_day['P_kt'], 'm--*', label='$k_t$ Persis', markersize=4, alpha=0.6)
        for i, suff in enumerate(suffixes):
            c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
            ax1.plot(time_hours, df_day[f'kt_pred{suff}'], f'{c}--{m}', label=f"Pred{suff.replace('_', ' ')}", markersize=4, alpha=0.8)
        ax1.set_title(f"Dados Originais (Variabilidade)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2); ax1.legend()

        # Imagem 2: kd
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.plot(time_hours, df_day['kd_real'], 'r--o', label='$k_d$ Real', markersize=4, alpha=0.6, linewidth=2)
        if 'P_fracao_difusa' in df_day.columns: ax2.plot(time_hours, df_day['P_fracao_difusa'], 'm--*', label='$k_d$ Persis', markersize=3, alpha=0.6)
        for i, suff in enumerate(suffixes):
            if f'kd_pred{suff}' in df_day.columns:
                c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
                ax2.plot(time_hours, df_day[f'kd_pred{suff}'], f'{c}--{m}', label=f"Pred{suff.replace('_', ' ')}", markersize=3, alpha=0.8)
        ax2.set_title("Fração Difusa Prevista")
        ax2.set_xlabel("Hora do Dia"); ax2.legend()

        # Imagem 3: Scatter
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(df_day['kt_real'], df_day['kd_real'], color='gray', alpha=0.5, label='Real', s=40)
        for i, suff in enumerate(suffixes):
            if f'kd_pred{suff}' in df_day.columns:
                c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
                ax3.scatter(df_day[f'kt_pred{suff}'], df_day[f'kd_pred{suff}'], color=c, marker=m, alpha=0.7, label=f"Pred{suff.replace('_', ' ')}", s=25)
        ax3.set_title("Espalhamento kt vs kd (Dia Transiente)")
        ax3.set_xlabel("$k_t$"); ax3.set_ylabel("$k_d$")
        ax3.set_xlim(0, 1.1); ax3.set_ylim(0, 1.1); ax3.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_variabilidade_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_overcast_day_analysis(self, df, min_samples=10):
        daily_stats = df.groupby(df.index.date)['kt_real'].agg(['mean', 'count'])
        valid_days = daily_stats[daily_stats['count'] >= min_samples]
        if valid_days.empty: return

        target_date = valid_days['mean'].idxmin()
        df_day = df[df.index.date == target_date].copy()
        print(f"   ☁️ Dia selecionado como Encoberto: {target_date}")

        suffixes = self._get_model_suffixes(df)
        time_hours = df_day.index.hour + df_day.index.minute/60
        fig = plt.figure(figsize=(18, 5))
        
        # Imagem 1: kt
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(time_hours, df_day['kt_real'], 'k-o', label='$k_t$ Real', markersize=5, linewidth=2)
        if 'P_kt' in df_day.columns: ax1.plot(time_hours, df_day['P_kt'], 'm--*', label='$k_t$ Persis', markersize=4, alpha=0.6)
        for i, suff in enumerate(suffixes):
            c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
            ax1.plot(time_hours, df_day[f'kt_pred{suff}'], f'{c}--{m}', label=f"Pred{suff.replace('_', ' ')}", markersize=4, alpha=0.8)
        ax1.set_title(f"Dados Originais (Encoberto)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2); ax1.legend()

        # Imagem 2: kd
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.plot(time_hours, df_day['kd_real'], 'r--o', label='$k_d$ Real', markersize=4, alpha=0.6, linewidth=2)
        if 'P_fracao_difusa' in df_day.columns: ax2.plot(time_hours, df_day['P_fracao_difusa'], 'm--*', label='$k_d$ Persis', markersize=3, alpha=0.6)
        for i, suff in enumerate(suffixes):
            if f'kd_pred{suff}' in df_day.columns:
                c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
                ax2.plot(time_hours, df_day[f'kd_pred{suff}'], f'{c}--{m}', label=f"Pred{suff.replace('_', ' ')}", markersize=3, alpha=0.8)
        ax2.set_title("Fração Difusa Prevista")
        ax2.set_xlabel("Hora do Dia"); ax2.legend()

        # Imagem 3: Scatter
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(df_day['kt_real'], df_day['kd_real'], color='gray', alpha=0.7, label='Real', s=40)
        for i, suff in enumerate(suffixes):
            if f'kd_pred{suff}' in df_day.columns:
                c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
                ax3.scatter(df_day[f'kt_pred{suff}'], df_day[f'kd_pred{suff}'], color=c, marker=m, alpha=0.7, label=f"Pred{suff.replace('_', ' ')}", s=25)
        ax3.set_title("Espalhamento kt vs kd (Céu Encoberto)")
        ax3.set_xlabel("$k_t$"); ax3.set_ylabel("$k_d$")
        ax3.set_xlim(0, 1.1); ax3.set_ylim(0, 1.1); ax3.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_overcast_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_transient_day_analysis(self, df, min_samples=10):
        def successive_variability(x): return np.abs(x.diff()).sum()
        daily_stats = df.groupby(df.index.date)['kt_real'].agg([successive_variability, 'mean', 'count'])
        valid_days = daily_stats[(daily_stats['count'] >= min_samples) & (daily_stats['mean'] > 0.3)]
        if valid_days.empty: return

        target_date = valid_days['successive_variability'].idxmax()
        df_day = df[df.index.date == target_date].copy()
        print(f"   🌦️ Dia Transiente selecionado: {target_date}")

        suffixes = self._get_model_suffixes(df)
        time_hours = df_day.index.hour + df_day.index.minute/60
        fig = plt.figure(figsize=(18, 5))
        
        # Imagem 1: kt
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(time_hours, df_day['kt_real'], 'k-o', label='$k_t$ Real', markersize=5, linewidth=2, alpha=0.9)
        if 'P_kt' in df_day.columns: ax1.plot(time_hours, df_day['P_kt'], 'm--*', label='$k_t$ Persis', markersize=4, alpha=0.6)
        for i, suff in enumerate(suffixes):
            c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
            ax1.plot(time_hours, df_day[f'kt_pred{suff}'], f'{c}--{m}', label=f"Pred{suff.replace('_', ' ')}", markersize=4, alpha=0.8)
        ax1.set_title(f"Dados Originais (Transiente)\n{target_date}")
        ax1.set_xlabel("Hora do Dia"); ax1.set_ylabel("Índice / Fração")
        ax1.set_ylim(0, 1.2); ax1.legend()

        # Imagem 2: kd
        ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
        ax2.plot(time_hours, df_day['kd_real'], 'r--o', label='$k_d$ Real', markersize=4, alpha=0.6, linewidth=2)
        if 'P_fracao_difusa' in df_day.columns: ax2.plot(time_hours, df_day['P_fracao_difusa'], 'm--*', label='$k_d$ Persis', markersize=3, alpha=0.6)
        for i, suff in enumerate(suffixes):
            if f'kd_pred{suff}' in df_day.columns:
                c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
                ax2.plot(time_hours, df_day[f'kd_pred{suff}'], f'{c}--{m}', label=f"Pred{suff.replace('_', ' ')}", markersize=3, alpha=0.8)
        ax2.set_title("Fração Difusa Prevista")
        ax2.set_xlabel("Hora do Dia"); ax2.legend()

        # Imagem 3: Scatter
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(df_day['kt_real'], df_day['kd_real'], color='gray', alpha=0.6, label='Real', s=40)
        for i, suff in enumerate(suffixes):
            if f'kd_pred{suff}' in df_day.columns:
                c, m = self.colors[i % len(self.colors)], self.markers[i % len(self.markers)]
                ax3.scatter(df_day[f'kt_pred{suff}'], df_day[f'kd_pred{suff}'], color=c, marker=m, alpha=0.7, label=f"Pred{suff.replace('_', ' ')}", s=25)
        ax3.set_title("Correlação Física (Espalhamento)")
        ax3.set_xlabel("$k_t$"); ax3.set_ylabel("$k_d$")
        ax3.set_xlim(0, 1.1); ax3.set_ylim(0, 1.1); ax3.legend()

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"analise_transiente_{target_date}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_statistical_metrics(self, df):
        print(f"   📊 Calculando métricas estatísticas para: {self.model_name}")
        suffixes = self._get_model_suffixes(df)
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

    def plot_scatter_validation(self, df):
        """
        Gera matriz de validação: Linha 1 para kt, Linha 2 para kd. N colunas para N modelos.
        """
        print(f"   📈 Gerando matriz de scatter de validação estatística...")
        suffixes = self._get_model_suffixes(df)
        n_models = len(suffixes) + 1
        if n_models == 0: return

        fig, axes = plt.subplots(2, n_models, figsize=(6 * (n_models), 12))
        # Se for apenas 1 modelo, o plt.subplots retorna um array 1D. Reformatamos para 2D.
        if n_models == 1: axes = axes.reshape(-1, 1)

        for i, suff in enumerate(suffixes):
            c = self.colors[i % len(self.colors)]
            model_tag = suff.replace('_', ' ') if suff else 'Base'

            # --- Linha 0: Avaliação do kt ---
            ax_kt = axes[0, i]
            y_true_kt, y_pred_kt = df['kt_real'], df[f'kt_pred{suff}']
            self._plot_single_scatter(ax_kt, y_true_kt, y_pred_kt, "$k_t$", model_tag, c)

            # --- Linha 1: Avaliação do kd ---
            ax_kd = axes[1, i]
            if f'kd_pred{suff}' in df.columns:
                y_true_kd, y_pred_kd = df['kd_real'], df[f'kd_pred{suff}']
                self._plot_single_scatter(ax_kd, y_true_kd, y_pred_kd, "$k_d$", model_tag, c)
            else:
                ax_kd.axis('off') # Desliga o eixo se não houver predição kd para esse modelo

        # PERSISTENCIA
        c = self.colors[i+1 % len(self.colors)]
        model_tag = suff.replace('_', ' ') if suff else 'Base'
        # --- Linha 0: Avaliação do kt ---
        ax_kt = axes[0, i+1]
        y_true_kt, y_pred_kt = df['kt_real'], df['P_kt']
        self._plot_single_scatter(ax_kt, y_true_kt, y_pred_kt, "$k_t$", "Persistencia_$k_t$", c)

        # --- Linha 1: Avaliação do kd ---
        ax_kd = axes[1, i+1]
        if f'kd_pred{suff}' in df.columns:
            y_true_kd, y_pred_kd = df['kd_real'], df['P_fracao_difusa']
            self._plot_single_scatter(ax_kd, y_true_kd, y_pred_kd, "$k_d$", "Persistencia_$k_t$", c)
        else:
                ax_kd.axis('off') # Desliga o eixo se não houver predição kd para esse modelo

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "scatter_validacao_estatistica.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_single_scatter(self, ax, y_true, y_pred, var_name, model_tag, color):
        mean_obs = np.mean(y_true)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        nrmse = (rmse / mean_obs) * 100 if mean_obs != 0 else 0
        r2 = r2_score(y_true, y_pred)
        mbe = np.mean(y_pred - y_true)

        ax.scatter(y_true, y_pred, color=color, alpha=0.3, s=10, label=f'Pred {model_tag}')
        ax.plot([0, 1.1], [0, 1.1], 'k--', linewidth=2, label='Ideal (1:1)')

        title_str = (f"Validação {var_name} - Pred{model_tag}\n"
                     f"RMSE: {rmse:.4f} | nRMSE: {nrmse:.2f}%\n"
                     f"R²: {r2:.4f} | MBE: {mbe:.4f}")
        ax.set_title(title_str, fontsize=11)
        ax.set_xlabel(f"{var_name} Observado")
        ax.set_ylabel(f"{var_name} Previsto")
        ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()