import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Tenta importar skill_metrics de forma segura
try:
    import skill_metrics as sm
    HAS_SKILL_METRICS = True
except ImportError:
    HAS_SKILL_METRICS = False
    print("⚠️ Aviso: 'skill_metrics' não instalado. Taylor Diagram será pulado.")

class SolarStatisticalAnalyzer:
    def __init__(self, df_combined, output_dir):
        self.df = df_combined.copy()
        self.save_path = output_dir
        os.makedirs(self.save_path, exist_ok=True)
        
        # Garante timestamp e cria colunas auxiliares
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Date'] = self.df['Timestamp'].dt.date 
        self.df['Hour'] = self.df['Timestamp'].dt.hour
        
        # Filtro Diurno: Como o CSV final não tem Zenith, usamos horário fixo (6h as 18h)
        # Se quiser ajustar, mude aqui.
        self.df_day = self.df[(self.df['Hour'] >= 6) & (self.df['Hour'] <= 18)].copy()

        # ================= ESTILOS ORIGINAIS =================
        self.markers_dict = {
            'Observado': {'Symbol': 'h', 'Size': 12, 'FaceColor': 'black', 'EdgeColor': 'black', 'style': '-'},
            'Persistencia': {'Symbol': 'X', 'Size': 9, 'FaceColor': 'gray', 'EdgeColor': 'gray', 'style': '--'},
            
            # Adicione aqui os nomes dos seus modelos se quiser cores fixas
            'LSTM_Bi_Attention': {'Symbol': 'o', 'Size': 10, 'FaceColor': 'white', 'EdgeColor': 'green', 'style': ':'},
            'GRU_Bi_Attention': {'Symbol': 's', 'Size': 10, 'FaceColor': 'white', 'EdgeColor': 'blue', 'style': ':'},
            'Teste_2': {'Symbol': 'D', 'Size': 10, 'FaceColor': 'white', 'EdgeColor': 'purple', 'style': '-.'}
        }
        
        # Cores padrão para Taylor Diagram
        self.COLS_COR = {'grid': '#8a8a8a', 'tick_labels': '#000000', 'title': '#000000'}
        self.COLS_STD = {'grid': '#8a8a8a', 'tick_labels': '#000000', 'ticks': '#8a8a8a', 'title': '#000000'}
        self.STYLES_RMS = {'color': '#8a8a8a', 'linestyle': '--'}
        
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})

    def _get_style(self, model_name):
        """Retorna o estilo do dicionário ou gera um padrão se não existir."""
        if model_name in self.markers_dict:
            return self.markers_dict[model_name]
        
        # Estilo genérico para modelos novos não cadastrados
        return {'Symbol': 'o', 'Size': 8, 'FaceColor': 'orange', 'EdgeColor': 'red', 'style': '-'}

    def _save_fig(self, fig, name):
        path = os.path.join(self.save_path, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close(fig)
        print(f"📊 Gráfico salvo: {path}")

    # ================= MÉTRICAS GLOBAIS =================
    def save_global_metrics(self):
        metrics = []
        
        # 1. Métricas dos Modelos
        for m in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == m]
            if sub.empty: continue
            
            obs, pred = sub['Observado'], sub['Previsto']
            
            rmse = np.sqrt(mean_squared_error(obs, pred))
            mae = mean_absolute_error(obs, pred)
            r2 = r2_score(obs, pred)
            
            ss = np.nan
            if 'Persistencia' in sub.columns:
                # Skill Score
                rmse_p = np.sqrt(mean_squared_error(sub['Observado'], sub['Persistencia']))
                if rmse_p != 0:
                    ss = (1 - (rmse/rmse_p)) * 100

            metrics.append({
                'Modelo': m,
                'RMSE (kW)': round(rmse, 4),
                'MAE (kW)': round(mae, 4),
                'R2': round(r2, 4),
                'Skill Score (%)': round(ss, 2)
            })
            
        # 2. Métricas da Persistência (Referência)
        if not self.df_day.empty:
            # Pega qualquer slice para calcular a persistência global
            sub = self.df_day.iloc[:len(self.df_day)//len(self.df_day['Modelo'].unique())]
            
            rmse_p = np.sqrt(mean_squared_error(sub['Observado'], sub['Persistencia']))
            mae_p = mean_absolute_error(sub['Observado'], sub['Persistencia'])
            r2_p = r2_score(sub['Observado'], sub['Persistencia'])
            
            metrics.append({
                'Modelo': 'Persistencia',
                'RMSE (kW)': round(rmse_p, 4),
                'MAE (kW)': round(mae_p, 4),
                'R2': round(r2_p, 4),
                'Skill Score (%)': 0.0
            })
        
        df_metrics = pd.DataFrame(metrics).sort_values('RMSE (kW)')
        df_metrics.to_csv(os.path.join(self.save_path, 'global_metrics.csv'), index=False)
        print("📊 Tabela de métricas salva.")

    # ================= PLOTS (PADRÃO ORIGINAL) =================

    def plot_boxplots_hourly(self):
        """Boxplots manuais com estilo customizado."""
        hours = sorted(self.df_day['Hour'].unique())
        models = sorted(self.df_day['Modelo'].unique())
        all_series = ['Observado', 'Persistencia'] + models
        
        fig, ax = plt.subplots(figsize=(16, 8))
        data, pos, colors = [], [], []
        
        # Para evitar duplicidade no Observado/Persistencia, pegamos de um modelo de referência
        ref_model = self.df_day['Modelo'].iloc[0]
        df_ref = self.df_day[self.df_day['Modelo'] == ref_model]
        
        curr = 1
        for h in hours:
            if curr != 1: curr += 1 # Espaçamento entre horas
            for m in all_series:
                if m == 'Observado':
                    vals = df_ref[df_ref['Hour'] == h]['Observado']
                    c = 'black'
                elif m == 'Persistencia':
                    vals = df_ref[df_ref['Hour'] == h]['Persistencia']
                    c = 'gray'
                else:
                    vals = self.df_day[(self.df_day['Hour'] == h) & (self.df_day['Modelo'] == m)]['Previsto']
                    # Usa cor do dicionário ou gera
                    style = self._get_style(m)
                    c = style.get('FaceColor', 'blue')
                
                if not vals.empty:
                    data.append(vals.dropna().values)
                    pos.append(curr)
                    colors.append(c)
                    curr += 1

        bplot = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, 
                           flierprops=dict(marker='o', markersize=3, alpha=0.5))
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            if color in ['black', 'gray']: patch.set_alpha(0.6)
        
        # Ajuste de ticks X para ficar no centro do grupo de horas
        group_center = [np.mean([p for p in pos if (p-1)//(len(all_series)+1) == i]) for i in range(len(hours))]
        ax.set_xticks(group_center)
        ax.set_xticklabels(hours)
        ax.set_xlabel("Hora do Dia")
        ax.set_ylabel("Potência (kW)")
        
        # Legenda Manual
        handles = [mlines.Line2D([],[], color='black' if m=='Observado' else ('gray' if m=='Persistencia' else self._get_style(m)['EdgeColor']), 
                                 marker='s', linestyle='None', label=m) for m in all_series]
        ax.legend(handles=handles, loc='upper left')
        self._save_fig(fig, "boxplot_hourly_distribution")

    def plot_error_by_hour_of_day(self):
        """RMSE por Hora."""
        # Calcula Erro Quadrático
        self.df_day['SE'] = (self.df_day['Observado'] - self.df_day['Previsto']) ** 2
        grouped = self.df_day.groupby(['Modelo', 'Hour'])['SE'].mean().reset_index()
        grouped['RMSE'] = np.sqrt(grouped['SE'])
        
        # Persistência
        ref_model = self.df_day['Modelo'].iloc[0]
        df_ref = self.df_day[self.df_day['Modelo'] == ref_model].copy()
        df_ref['SE_Pers'] = (df_ref['Observado'] - df_ref['Persistencia']) ** 2
        grouped_p = df_ref.groupby('Hour')['SE_Pers'].mean().reset_index()
        grouped_p['RMSE'] = np.sqrt(grouped_p['SE_Pers'])

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot Modelos
        for m in grouped['Modelo'].unique():
            sub = grouped[grouped['Modelo'] == m]
            style = self._get_style(m)
            ax.plot(sub['Hour'], sub['RMSE'], label=m, 
                    color=style['EdgeColor'], marker=style['Symbol'], linestyle='-')
            
        # Plot Persistência
        st_p = self.markers_dict['Persistencia']
        ax.plot(grouped_p['Hour'], grouped_p['RMSE'], label='Persistencia',
                color=st_p['EdgeColor'], marker=st_p['Symbol'], linestyle=st_p['style'], linewidth=2)
            
        ax.set_title('Perfil Diário de Erro (RMSE por Hora)')
        ax.set_xlabel('Hora do Dia')
        ax.set_ylabel('RMSE Médio (kW)')
        ax.set_xticks(range(6, 19))
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, 'profile_RMSE_Hourly')

    def plot_scenario_days(self):
        """Dias representativos."""
        # Encontra dias representativos baseado no primeiro modelo (dados reais são iguais pra todos)
        ref_model = self.df_day['Modelo'].iloc[0]
        df_base = self.df_day[self.df_day['Modelo'] == ref_model].copy()
        
        # Lógica simples para achar dias: 
        # Céu Claro = Alta Energia, Baixa Volatilidade (Desvio Padrão baixo)
        # Nublado = Alta Volatilidade
        daily_stats = df_base.groupby('Date')['Observado'].agg(['sum', 'std']).dropna()
        
        if daily_stats.empty: return

        # Dia Claro (Top 20% energia, menor desvio)
        top_energy = daily_stats.nlargest(int(len(daily_stats)*0.2), 'sum')
        clear_day = top_energy.nlargest(1, 'std').index[0] if not top_energy.empty else None
        
        # Dia Nublado (Maior desvio)
        cloudy_day = top_energy.nsmallest(1, 'std').index[0]
        
        scenarios = {'Ceu_Claro': clear_day, 'Nublado_Transiente': cloudy_day}
        
        for label, date_obj in scenarios.items():
            if date_obj is None: continue
            
            # Pega dados DO DIA INTEIRO (não só diurno) para o plot ficar bonito
            day_data = self.df[self.df['Date'] == date_obj].sort_values('Timestamp')
            if day_data.empty: continue

            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Observado e Persistência (apenas uma vez)
            ref_data = day_data[day_data['Modelo'] == ref_model]
            
            ax.plot(ref_data['Timestamp'], ref_data['Observado'], 
                    color='black', label='Observado', linewidth=2.5, zorder=10)
            
            st_p = self.markers_dict['Persistencia']
            ax.plot(ref_data['Timestamp'], ref_data['Persistencia'],
                    color=st_p['EdgeColor'], label='Persistencia', 
                    linestyle=st_p['style'], linewidth=1.5, zorder=5)
            
            # Modelos
            for m in day_data['Modelo'].unique():
                sub = day_data[day_data['Modelo'] == m]
                style = self._get_style(m)
                ax.plot(sub['Timestamp'], sub['Previsto'], label=m,
                        color=style['EdgeColor'], linestyle=style['style'], linewidth=1.5)
            
            ax.set_title(f'Cenário: {label.replace("_", " ")} ({date_obj})')
            ax.set_ylabel('Potência (kW)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.legend()
            ax.grid(True, alpha=0.3)
            self._save_fig(fig, f"scenario_{label}")

    def plot_scatter_hist(self):
        """Dashboard antigo: Histograma + Scatter para cada modelo."""
        # 1. Modelos
        for m in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == m]
            style = self._get_style(m)
            self._plot_single_dashboard(sub['Observado'], sub['Previsto'], m, style['EdgeColor'])
            
        # 2. Persistência
        if not self.df_day.empty:
            ref_model = self.df_day['Modelo'].iloc[0]
            sub = self.df_day[self.df_day['Modelo'] == ref_model]
            self._plot_single_dashboard(sub['Observado'], sub['Persistencia'], 'Persistencia', 'gray')

    def _plot_single_dashboard(self, obs, pred, name, color):
        """Gera o layout Histograma (Esq) + Scatter (Dir)."""
        error = pred - obs
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histograma
        sns.histplot(error, kde=True, color=color, ax=ax[0])
        ax[0].set_title(f"Distribuição de Erro: {name}")
        ax[0].set_xlabel("Erro (Previsto - Observado)")
        ax[0].axvline(0, color='k', linestyle='--')
        
        # Scatter
        sns.regplot(x=obs, y=pred, scatter_kws={'alpha':0.3, 'color':color}, 
                    line_kws={'color':'k'}, ax=ax[1])
        
        # Linha 1:1
        min_val = min(obs.min(), pred.min())
        max_val = max(obs.max(), pred.max())
        ax[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Identidade')
        
        ax[1].set_title(f"Dispersão: {name}")
        ax[1].set_xlabel("Observado (kW)")
        ax[1].set_ylabel("Previsto (kW)")
        ax[1].legend()
        
        self._save_fig(fig, f"scatter_{name}")

    def plot_taylor_diagram(self):
        """Diagrama de Taylor."""
        if not HAS_SKILL_METRICS or self.df_day.empty: return
        
        # Pivota para alinhar os dados
        pivot = self.df_day.pivot(index='Timestamp', columns='Modelo', values='Previsto')
        pivot['Observado'] = self.df_day.groupby('Timestamp')['Observado'].first()
        pivot['Persistencia'] = self.df_day.groupby('Timestamp')['Persistencia'].first()
        pivot.dropna(inplace=True)
        
        if pivot.empty: return

        ref = pivot['Observado'].values
        
        sdevs = [np.std(ref)]
        crmsds = [0]
        ccoefs = [1]
        labels = ['Ref']
        faces, edges, symbols = ['black'], ['black'], ['h']
        
        # Persistência
        per = pivot['Persistencia'].values
        sdevs.append(np.std(per))
        crmsds.append(np.sqrt(mean_squared_error(ref, per)))
        ccoefs.append(np.corrcoef(ref, per)[0, 1])
        labels.append('Persistencia')
        st_p = self.markers_dict['Persistencia']
        faces.append(st_p['FaceColor'])
        edges.append(st_p['EdgeColor'])
        symbols.append(st_p['Symbol'])
        
        # Modelos
        for m in self.df_day['Modelo'].unique():
            if m not in pivot.columns: continue
            pred = pivot[m].values
            sdevs.append(np.std(pred))
            crmsds.append(np.sqrt(mean_squared_error(ref, pred)))
            ccoefs.append(np.corrcoef(ref, pred)[0, 1])
            labels.append(m)
            
            st = self._get_style(m)
            faces.append(st['FaceColor'])
            edges.append(st['EdgeColor'])
            symbols.append(st['Symbol'])

        fig = plt.figure(figsize=(10, 10))
        sm.taylor_diagram(np.array(sdevs), np.array(crmsds), np.array(ccoefs),
                          markerLabel=labels, markerLegend='on',
                          markercolors={'face': faces, 'edge': edges},
                          markersymbol=symbols, 
                          markersize=10, 
                          colscor=self.COLS_COR, colsstd=self.COLS_STD,
                          styleRMS=self.STYLES_RMS['linestyle'], colRMS=self.STYLES_RMS['color'])
        self._save_fig(fig, "taylor_diagram")