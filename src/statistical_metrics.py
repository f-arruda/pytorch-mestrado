import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import skill_metrics as sm
import os

class SolarStatisticalAnalyzer:
    def __init__(self, df_combined, output_dir):
        self.df = df_combined.copy()
        self.save_path = output_dir
        os.makedirs(self.save_path, exist_ok=True)
        
        # Garante timestamp e cria coluna de DATA
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Date'] = self.df['Timestamp'].dt.date 
        
        # Filtro Diurno (Zenith < 80)
        if 'zenith' in self.df.columns:
            self.df_day = self.df[self.df['zenith'] < 80].copy()
        else:
            self.df_day = self.df.copy()

        # Dicionário de Estilos
        self.markers_dict = {
            'Observado': {'Symbol': 'h', 'Size': 12, 'FaceColor': 'black', 'EdgeColor': 'black', 'style': '-'},
            'Persistencia': {'Symbol': 'X', 'Size': 9, 'FaceColor': 'gray', 'EdgeColor': 'gray', 'style': '--'},
            
            # Seus modelos
            'LSTM_Bi_Attention': {'Symbol': 'o', 'Size': 10, 'FaceColor': 'white', 'EdgeColor': 'green', 'style': ':'},
            'GRU_Bi_Attention': {'Symbol': 's', 'Size': 10, 'FaceColor': 'white', 'EdgeColor': 'blue', 'style': ':'},
            'LSTM_Bi_Attention_PAZ': {'Symbol': 'v', 'Size': 10, 'FaceColor': 'cyan', 'EdgeColor': 'cyan', 'style': '-'},
            'EDLSTM_PAZTUVD': {'Symbol': 'D', 'Size': 10, 'FaceColor': 'orange', 'EdgeColor': 'red', 'style': '-.'}
        }
        
        self.COLS_COR = {'grid': '#8a8a8a', 'tick_labels': '#000000', 'title': '#000000'}
        self.COLS_STD = {'grid': '#8a8a8a', 'tick_labels': '#000000', 'ticks': '#8a8a8a', 'title': '#000000'}
        self.STYLES_RMS = {'color': '#8a8a8a', 'linestyle': '--'}
        
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 12})

    def _get_style(self, model_name):
        if model_name == 'Persistencia': 
            return self.markers_dict['Persistencia']
        
        for key in self.markers_dict:
            if key in model_name and key != 'Persistencia':
                return self.markers_dict[key]
        return {'Symbol': 'o', 'Size': 8, 'FaceColor': 'purple', 'EdgeColor': 'purple', 'style': '-'}

    def _save_fig(self, fig, name):
        path = os.path.join(self.save_path, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close(fig)
        print(f"📊 Salvo: {path}")

    # ==========================================================================
    #  MÉTRICAS GLOBAIS
    # ==========================================================================
    def save_global_metrics(self):
        metrics = []
        
        # 1. Métricas dos Modelos
        for m in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == m]
            obs, pred = sub['Observado'], sub['Previsto']
            
            rmse = np.sqrt(mean_squared_error(obs, pred))
            mae = mean_absolute_error(obs, pred)
            r2 = r2_score(obs, pred)
            
            ss = np.nan
            if 'Persistencia' in sub.columns:
                p_valid = sub.dropna(subset=['Persistencia'])
                if not p_valid.empty:
                    rmse_p = np.sqrt(mean_squared_error(p_valid['Observado'], p_valid['Persistencia']))
                    rmse_m = np.sqrt(mean_squared_error(p_valid['Observado'], p_valid['Previsto']))
                    ss = (1 - (rmse_m/rmse_p)) * 100 if rmse_p != 0 else np.nan

            metrics.append({
                'Modelo': m,
                'RMSE (kW)': round(rmse, 4),
                'MAE (kW)': round(mae, 4),
                'R2': round(r2, 4),
                'Skill Score (%)': round(ss, 2)
            })
            
        # 2. Métricas da Persistência
        if not self.df_day.empty:
            first_model = self.df_day['Modelo'].iloc[0]
            sub_p = self.df_day[self.df_day['Modelo'] == first_model].dropna(subset=['Persistencia'])
            
            if not sub_p.empty:
                rmse_p = np.sqrt(mean_squared_error(sub_p['Observado'], sub_p['Persistencia']))
                mae_p = mean_absolute_error(sub_p['Observado'], sub_p['Persistencia'])
                r2_p = r2_score(sub_p['Observado'], sub_p['Persistencia'])
                
                metrics.append({
                    'Modelo': 'Persistencia',
                    'RMSE (kW)': round(rmse_p, 4),
                    'MAE (kW)': round(mae_p, 4),
                    'R2': round(r2_p, 4),
                    'Skill Score (%)': 0.0
                })
        
        pd.DataFrame(metrics).sort_values('RMSE (kW)').to_csv(os.path.join(self.save_path, 'global_metrics.csv'), index=False)
        print("📊 Métricas (incluindo Persistência) salvas.")

    # ==========================================================================
    #  GRÁFICOS
    # ==========================================================================
    def plot_metrics_by_horizon(self):
        df = self.df_day
        horizons = sorted(df['Horizonte'].unique())
        
        # Prepara dados da persistência
        ref_model = df['Modelo'].iloc[0]
        df_ref = df[df['Modelo'] == ref_model]
        pers_metrics = {'pRMSE': [], 'pMBE': []}
        
        for h in horizons:
            sub_h = df_ref[df_ref['Horizonte'] == h].dropna(subset=['Persistencia'])
            if sub_h.empty:
                pers_metrics['pRMSE'].append(np.nan)
                pers_metrics['pMBE'].append(np.nan)
                continue
            obs_mean = sub_h['Observado'].mean()
            rmse_p = np.sqrt(mean_squared_error(sub_h['Observado'], sub_h['Persistencia']))
            mbe_p = np.mean(sub_h['Persistencia'] - sub_h['Observado'])
            pers_metrics['pRMSE'].append((rmse_p/obs_mean)*100 if obs_mean else np.nan)
            pers_metrics['pMBE'].append((mbe_p/obs_mean)*100 if obs_mean else np.nan)

        for metric in ['pRMSE', 'pMBE', 'Skill Score']:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for m in df['Modelo'].unique():
                sub = df[df['Modelo'] == m]
                x, y = [], []
                for h in horizons:
                    sub_h = sub[sub['Horizonte'] == h]
                    if sub_h.empty: continue
                    obs_mean = sub_h['Observado'].mean()
                    rmse = np.sqrt(mean_squared_error(sub_h['Observado'], sub_h['Previsto']))
                    
                    val = np.nan
                    if metric == 'pRMSE':
                        val = (rmse/obs_mean)*100 if obs_mean else np.nan
                    elif metric == 'pMBE':
                        mbe = np.mean(sub_h['Previsto'] - sub_h['Observado'])
                        val = (mbe/obs_mean)*100 if obs_mean else np.nan
                    elif metric == 'Skill Score':
                        if 'Persistencia' in sub_h.columns:
                             valid = sub_h.dropna(subset=['Persistencia'])
                             if not valid.empty:
                                 rmse_mod = np.sqrt(mean_squared_error(valid['Observado'], valid['Previsto']))
                                 rmse_per = np.sqrt(mean_squared_error(valid['Observado'], valid['Persistencia']))
                                 val = (1 - (rmse_mod/rmse_per))*100 if rmse_per else np.nan
                    x.append(h)
                    y.append(val)
                
                style = self._get_style(m)
                ax.plot(x, y, label=m, color=style['EdgeColor'], marker=style['Symbol'], linestyle=style['style'])

            if metric != 'Skill Score':
                st_p = self._get_style('Persistencia')
                ax.plot(horizons, pers_metrics[metric], label='Persistencia', 
                        color=st_p['EdgeColor'], marker=st_p['Symbol'], linestyle=st_p['style'], alpha=0.7)

            if metric == 'pMBE': ax.axhline(0, color='k', linestyle='--')
            ax.set_title(f"{metric} por Horizonte")
            ax.set_xlabel("Horizonte (h)")
            ax.set_ylabel(f"{metric} (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            self._save_fig(fig, f"lineplot_{metric}")

    def plot_boxplots_hourly(self):
        hours = sorted(self.df_day['Hour'].unique())
        models = sorted(self.df_day['Modelo'].unique())
        all_series = ['Observado', 'Persistencia'] + models
        
        fig, ax = plt.subplots(figsize=(16, 8))
        data, pos, colors = [], [], []
        
        ref_model = self.df_day['Modelo'].iloc[0]
        df_ref = self.df_day[self.df_day['Modelo'] == ref_model]
        
        curr = 1
        for h in hours:
            if curr != 1: curr += 1
            for m in all_series:
                if m == 'Observado':
                    vals = df_ref[df_ref['Hour'] == h]['Observado']
                    c = 'black'
                elif m == 'Persistencia':
                    vals = df_ref[df_ref['Hour'] == h]['Persistencia']
                    c = 'gray'
                else:
                    vals = self.df_day[(self.df_day['Hour'] == h) & (self.df_day['Modelo'] == m)]['Previsto']
                    c = self._get_style(m)['FaceColor']
                
                if not vals.empty:
                    data.append(vals.dropna().values)
                    pos.append(curr)
                    colors.append(c)
                    curr += 1

        bplot = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, 
                           flierprops=dict(marker='o', markersize=3, alpha=0.5))
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            if color == 'black' or color == 'gray': patch.set_alpha(0.6)
        
        group_center = [np.mean([p for p in pos if (p-1)//(len(all_series)+1) == i]) for i in range(len(hours))]
        ax.set_xticks(group_center)
        ax.set_xticklabels(hours)
        ax.set_xlabel("Hora do Dia")
        ax.set_ylabel("Potência (kW)")
        
        handles = [mlines.Line2D([],[], color=self._get_style(m)['EdgeColor'] if m not in ['Observado', 'Persistencia'] else ('black' if m=='Observado' else 'gray'), 
                                 marker='s', linestyle='None', label=m) for m in all_series]
        ax.legend(handles=handles, loc='upper left')
        self._save_fig(fig, "boxplot_hourly_distribution")

    def plot_taylor_diagram(self):
        fig = plt.figure(figsize=(10, 10))
        if self.df_day.empty: return
        
        ref_model = self.df_day['Modelo'].iloc[0]
        sub_ref = self.df_day[self.df_day['Modelo'] == ref_model].dropna(subset=['Observado', 'Persistencia'])
        
        ref_data = sub_ref['Observado'].values
        sdevs = [np.std(ref_data)]
        crmsds = [0]
        ccoefs = [1]
        labels = ['Ref']
        faces, edges, symbols = ['black'], ['black'], ['h']
        
        # Persistência
        per_data = sub_ref['Persistencia'].values
        sdevs.append(np.std(per_data))
        crmsds.append(np.sqrt(mean_squared_error(ref_data, per_data)))
        ccoefs.append(np.corrcoef(ref_data, per_data)[0, 1])
        labels.append('Persistencia')
        
        st_p = self._get_style('Persistencia')
        faces.append(st_p['FaceColor'])
        edges.append(st_p['EdgeColor'])
        symbols.append(st_p['Symbol'])
        
        # Modelos
        for m in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == m].dropna(subset=['Observado', 'Previsto'])
            if sub.empty: continue
            
            sdevs.append(np.std(sub['Previsto'].values))
            crmsds.append(np.sqrt(mean_squared_error(sub['Observado'].values, sub['Previsto'].values)))
            ccoefs.append(np.corrcoef(sub['Observado'].values, sub['Previsto'].values)[0, 1])
            labels.append(m)
            
            st = self._get_style(m)
            faces.append(st['FaceColor'])
            edges.append(st['EdgeColor'])
            symbols.append(st['Symbol'])

        sm.taylor_diagram(np.array(sdevs), np.array(crmsds), np.array(ccoefs),
                          markerLabel=labels, markerLegend='on',
                          markercolors={'face': faces, 'edge': edges},
                          markersymbol=symbols, 
                          markersize=10, 
                          colscor=self.COLS_COR, colsstd=self.COLS_STD,
                          styleRMS=self.STYLES_RMS['linestyle'], colRMS=self.STYLES_RMS['color'])
        self._save_fig(fig, "taylor_diagram")

    def plot_error_by_hour_of_day(self):
        print("📈 Gerando perfil de erro horário...")
        df_sun = self.df_day[(self.df_day['Hour'] >= 6) & (self.df_day['Hour'] <= 19)].copy()
        if df_sun.empty: return

        df_sun['SE'] = (df_sun['Observado'] - df_sun['Previsto']) ** 2
        grouped = df_sun.groupby(['Modelo', 'Hour'])['SE'].mean().reset_index()
        grouped['RMSE'] = np.sqrt(grouped['SE'])
        
        # Persistência
        ref_model = df_sun['Modelo'].iloc[0]
        df_ref = df_sun[df_sun['Modelo'] == ref_model].copy()
        df_ref['SE_Pers'] = (df_ref['Observado'] - df_ref['Persistencia']) ** 2
        grouped_p = df_ref.groupby('Hour')['SE_Pers'].mean().reset_index()
        grouped_p['RMSE'] = np.sqrt(grouped_p['SE_Pers'])

        fig, ax = plt.subplots(figsize=(12, 6))
        for m in grouped['Modelo'].unique():
            sub = grouped[grouped['Modelo'] == m]
            style = self._get_style(m)
            ax.plot(sub['Hour'], sub['RMSE'], label=m, 
                    color=style['EdgeColor'], marker=style['Symbol'], linestyle='-')
            
        st_p = self._get_style('Persistencia')
        ax.plot(grouped_p['Hour'], grouped_p['RMSE'], label='Persistencia',
                color=st_p['EdgeColor'], marker=st_p['Symbol'], linestyle=st_p['style'], linewidth=2)
            
        ax.set_title('Perfil Diário de Erro (RMSE por Hora)')
        ax.set_xlabel('Hora do Dia')
        ax.set_ylabel('RMSE Médio (kW)')
        ax.set_xticks(range(6, 20))
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save_fig(fig, 'profile_RMSE_Hourly')

    def plot_scenario_days(self):
        print("🔍 Buscando dias representativos...")
        date_clear, date_cloudy = self._find_representative_days()
        scenarios = {'Ceu_Claro': date_clear, 'Nublado_Transiente': date_cloudy}
        
        for label, date_obj in scenarios.items():
            if date_obj is None: continue
            
            day_data = self.df[self.df['Date'] == date_obj].sort_values('Timestamp')
            if day_data.empty: continue

            fig, ax = plt.subplots(figsize=(14, 7))
            
            first_model = day_data['Modelo'].unique()[0]
            ref_data = day_data[day_data['Modelo'] == first_model]
            
            ax.plot(ref_data['Timestamp'], ref_data['Observado'], 
                    color='black', label='Observado', linewidth=2.5, zorder=10)
            
            st_p = self._get_style('Persistencia')
            ax.plot(ref_data['Timestamp'], ref_data['Persistencia'],
                    color=st_p['EdgeColor'], label='Persistencia', 
                    linestyle=st_p['style'], linewidth=1.5, zorder=5)
            
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

    def _find_representative_days(self):
        if self.df.empty: return None, None
        first_model = self.df['Modelo'].unique()[0]
        df_base = self.df[self.df['Modelo'] == first_model].copy()
        
        daily_stats = []
        for date, group in df_base.groupby('Date'):
            if len(group) < 12: continue
            energy_sum = group['Observado'].sum()
            volatility = group['Observado'].diff().abs().mean()
            daily_stats.append({'Date': date, 'Energy': energy_sum, 'Volatility': volatility})
            
        stats_df = pd.DataFrame(daily_stats)
        if stats_df.empty: return None, None
        
        top_energy = stats_df.nlargest(int(len(stats_df)*0.2), 'Energy')
        clear_sky_day = top_energy.nsmallest(1, 'Volatility').iloc[0]['Date'] if not top_energy.empty else None
        cloudy_day = stats_df.nlargest(1, 'Volatility').iloc[0]['Date']
        return clear_sky_day, cloudy_day

    # ==========================================================================
    #  SCATTER PLOTS (VOLTANDO AO PADRÃO ANTIGO)
    # ==========================================================================
    def plot_scatter_hist(self):
        """Histograma + Scatter Plot individuais por modelo (incluindo Persistência)."""
        # 1. Plots dos Modelos
        for m in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == m]
            self._plot_single_dashboard(sub['Observado'], sub['Previsto'], m, self._get_style(m)['EdgeColor'])
            
        # 2. Plot da Persistência
        if not self.df_day.empty:
            ref_model = self.df_day['Modelo'].iloc[0]
            sub_p = self.df_day[self.df_day['Modelo'] == ref_model]
            self._plot_single_dashboard(sub_p['Observado'], sub_p['Persistencia'], 'Persistencia', 'gray')

    def _plot_single_dashboard(self, obs, pred, name, color):
        """Gera o layout antigo: Histograma (Esq) + Scatter (Dir)."""
        error = pred - obs
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histograma de Erros
        sns.histplot(error, kde=True, color=color, ax=ax[0])
        ax[0].set_title(f"Distribuição de Erro: {name}")
        ax[0].set_xlabel("Erro (Previsto - Observado)")
        ax[0].axvline(0, color='k', linestyle='--')
        
        # Scatter Plot (Regplot)
        sns.regplot(x=obs, y=pred, scatter_kws={'alpha':0.3, 'color':color}, 
                    line_kws={'color':'k'}, ax=ax[1])
        
        # Linha de Identidade 1:1
        min_val = min(obs.min(), pred.min())
        max_val = max(obs.max(), pred.max())
        ax[1].plot([min_val, max_val], [min_val, max_val], 'k--', label='Identidade')
        
        ax[1].set_title(f"Dispersão: {name}")
        ax[1].set_xlabel("Observado (kW)")
        ax[1].set_ylabel("Previsto (kW)")
        ax[1].legend()
        
        self._save_fig(fig, f"scatter_{name}")