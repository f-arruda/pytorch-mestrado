import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import skill_metrics as sm
import os

class SolarStatisticalAnalyzer:
    def __init__(self, df_combined, output_dir):
        self.df = df_combined
        self.save_path = output_dir
        os.makedirs(self.save_path, exist_ok=True)

        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['Date'] = self.df['Timestamp'].dt.date  # <--- Faltava isso!
        
        # Filtro Diurno (Zenith < 70)
        if 'zenith' in self.df.columns:
            self.df_day = self.df[self.df['zenith'] < 70].copy()
            print(f"☀️ Filtro Diurno aplicado: {len(self.df_day)} amostras.")
        else:
            self.df_day = self.df.copy()

        # Dicionário de Estilos (Baseado no seu script original)
        self.markers_dict = {
            'Observado': {'Symbol': 'h', 'Size': 12, 'FaceColor': 'black', 'EdgeColor': 'black', 'style': '-'},
            # Defina cores para seus modelos aqui
            'LSTM_Bi_Attention': {'Symbol': 'o', 'Size': 10, 'FaceColor': 'white', 'EdgeColor': 'green', 'style': ':'},
            'GRU_Bi_Attention': {'Symbol': 's', 'Size': 10, 'FaceColor': 'white', 'EdgeColor': 'blue', 'style': ':'},
            'LSTM_Bi_Attention_PAZ': {'Symbol': 'v', 'Size': 10, 'FaceColor': 'cyan', 'EdgeColor': 'cyan', 'style': '-'},
        }
        
        # Estilos globais (CORRIGIDO: Adicionado STYLES_RMS)
        self.COLS_COR = {'grid': '#8a8a8a', 'tick_labels': '#000000', 'title': '#000000'}
        self.COLS_STD = {'grid': '#8a8a8a', 'tick_labels': '#000000', 'ticks': '#8a8a8a', 'title': '#000000'}
        self.STYLES_RMS = {'color': '#8a8a8a', 'linestyle': '--'}  # <--- Faltava isso aqui!

    def _get_style(self, model_name):
        """Retorna estilo do dicionário ou gera um aleatório consistente."""
        # Busca exata ou parcial
        for key in self.markers_dict:
            if key == model_name:
                return self.markers_dict[key]
        
        # Fallback genérico se não achar
        import hashlib
        h = int(hashlib.sha256(model_name.encode()).hexdigest(), 16) % 0xFFFFFF
        c = f"#{h:06x}"
        return {'Symbol': 'd', 'Size': 10, 'FaceColor': c, 'EdgeColor': c, 'style': '-'}

    def _save_fig(self, fig, name):
        path = os.path.join(self.save_path, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='w')
        plt.close(fig)
        print(f"📊 Salvo: {path}")

    def save_global_metrics(self):
        metrics = []
        for m in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == m]
            obs, pred = sub['Observado'], sub['Previsto']
            
            rmse = np.sqrt(mean_squared_error(obs, pred))
            
            # Skill Score Global (vs Persistencia)
            ss = np.nan
            if 'Persistencia' in sub.columns:
                p_valid = sub.dropna(subset=['Persistencia'])
                if not p_valid.empty:
                    # Alinha índices
                    idx = p_valid.index
                    rmse_p = np.sqrt(mean_squared_error(p_valid['Observado'], p_valid['Persistencia']))
                    # Recalcula RMSE do modelo apenas nesses índices para ser justo
                    rmse_m = np.sqrt(mean_squared_error(p_valid['Observado'], sub.loc[idx, 'Previsto']))
                    
                    ss = (1 - (rmse_m/rmse_p)) * 100 if rmse_p != 0 else np.nan

            metrics.append({
                'Modelo': m,
                'RMSE': rmse,
                'MAE': mean_absolute_error(obs, pred),
                'R2': r2_score(obs, pred),
                'Skill Score (%)': ss
            })
        
        pd.DataFrame(metrics).to_csv(os.path.join(self.save_path, 'global_metrics.csv'), index=False)

    def plot_metrics_by_horizon(self):
        """Lineplots: pRMSE, pMBE, Skill Score."""
        df = self.df_day
        horizons = sorted(df['Horizonte'].unique())
        
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

            if metric == 'pMBE': ax.axhline(0, color='k', linestyle='--')
            ax.set_title(f"{metric} por Horizonte")
            ax.set_xlabel("Horizonte (h)")
            ax.set_ylabel(f"{metric} (%)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            self._save_fig(fig, f"lineplot_{metric}")

    def plot_boxplots_hourly(self):
        """Boxplot colorido por modelo."""
        hours = sorted(self.df_day['Hour'].unique())
        models = sorted(self.df_day['Modelo'].unique())
        all_series = ['Observado'] + models
        
        fig, ax = plt.subplots(figsize=(16, 8))
        data, pos, colors = [], [], []
        
        curr = 1
        for h in hours:
            if curr != 1: curr += 1
            for m in all_series:
                if m == 'Observado':
                    vals = self.df_day[self.df_day['Hour'] == h]['Observado']
                else:
                    vals = self.df_day[(self.df_day['Hour'] == h) & (self.df_day['Modelo'] == m)]['Previsto']
                
                if not vals.empty:
                    data.append(vals.dropna().values)
                    pos.append(curr)
                    colors.append(self._get_style(m)['FaceColor'])
                    curr += 1

        bplot = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, 
                           flierprops=dict(marker='o', markersize=3, alpha=0.5))
        
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
        for median in bplot['medians']:
            median.set_color('black')
            
        ax.set_xticks([np.mean([p for p in pos if (p-1)//(len(all_series)+1) == i]) for i in range(len(hours))])
        ax.set_xticklabels(hours)
        ax.set_xlabel("Hora (UTC-3)")
        
        handles = [mlines.Line2D([],[], color=self._get_style(m)['EdgeColor'], marker='s', linestyle='None', label=m) for m in all_series]
        ax.legend(handles=handles)
        self._save_fig(fig, "boxplot_hourly")

    def plot_taylor_diagram(self):
        """Diagrama de Taylor."""
        fig = plt.figure(figsize=(10, 10))
        
        # Referência (Observado)
        # Pega a referência do primeiro modelo (pois Observado é igual para todos)
        ref_model = self.df_day['Modelo'].iloc[0]
        ref_data = self.df_day[self.df_day['Modelo'] == ref_model]['Observado'].values
        
        sdevs = [np.std(ref_data)]
        crmsds = [0]
        ccoefs = [1]
        labels = ['Ref']
        
        # Configurações visuais (Listas)
        faces = ['black']
        edges = ['black']
        symbols = ['h']
        
        for m in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == m]
            pred, obs = sub['Previsto'].values, sub['Observado'].values
            
            sdevs.append(np.std(pred))
            crmsds.append(np.sqrt(mean_squared_error(obs, pred)))
            ccoefs.append(np.corrcoef(obs, pred)[0, 1])
            labels.append(m)
            
            st = self._get_style(m)
            faces.append(st['FaceColor'])
            edges.append(st['EdgeColor'])
            symbols.append(st['Symbol'])

        # Plotagem (Usando markersize fixo = 10 para evitar crash)
        sm.taylor_diagram(np.array(sdevs), np.array(crmsds), np.array(ccoefs),
                          markerLabel=labels, markerLegend='on',
                          markercolors={'face': faces, 'edge': edges},
                          markersymbol=symbols, 
                          markersize=10, 
                          styleOBS='-', colOBS='black',
                          titleOBS='Observado',
                          colscor=self.COLS_COR, colsstd=self.COLS_STD,
                          styleRMS=self.STYLES_RMS['linestyle'], colRMS=self.STYLES_RMS['color'])
        
        self._save_fig(fig, "taylor_diagram")

    def plot_scatter_hist(self):
        """Histograma + Scatter Plot individuais por modelo."""
        for m in self.df_day['Modelo'].unique():
            sub = self.df_day[self.df_day['Modelo'] == m]
            error = sub['Previsto'] - sub['Observado']
            
            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            style = self._get_style(m)
            c = style['EdgeColor']
            
            # Histograma
            sns.histplot(error, kde=True, color=c, ax=ax[0])
            ax[0].set_title(f"Erro: {m}")
            ax[0].set_xlabel("Erro (Previsto - Observado)")
            ax[0].axvline(0, color='k', linestyle='--')
            
            # Scatter
            sns.regplot(data=sub, x='Observado', y='Previsto', 
                        scatter_kws={'alpha':0.3, 'color':c}, line_kws={'color':'k'}, ax=ax[1])
            
            # Linha 1:1
            max_val = max(sub['Observado'].max(), sub['Previsto'].max())
            ax[1].plot([0, max_val], [0, max_val], 'k--')
            ax[1].set_title(f"Dispersão: {m}")
            
            self._save_fig(fig, f"scatter_{m}")

            # ==========================================================================
    #  FASE 1: DIAGNÓSTICO GRANULAR (NOVO)
    # ==========================================================================
    
    def plot_error_by_hour_of_day(self):
        """
        Plota o RMSE distribuído pelas horas do dia (0h - 23h).
        Isso responde: 'O modelo erra mais ao meio-dia ou no entardecer?'
        """
        print("📈 Gerando perfil de erro horário...")
        
        # Filtra apenas horas com sol (opcional, mas recomendado para evitar ruído noturno)
        # Assumindo que 6h as 19h tem sol
        df_sun = self.df[(self.df['Hour'] >= 5) & (self.df['Hour'] <= 20)].copy()
        
        df_sun['SE'] = (df_sun['Observado'] - df_sun['Previsto']) ** 2
        df_sun['AE'] = np.abs(df_sun['Observado'] - df_sun['Previsto'])
        
        # Agrupa por Modelo e Hora do Dia
        grouped = df_sun.groupby(['Modelo', 'Hour']).agg(
            RMSE=('SE', lambda x: np.sqrt(x.mean())),
            MAE=('AE', 'mean')
        ).reset_index()
        
        # Plot RMSE
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=grouped, x='Hour', y='RMSE', hue='Modelo', marker='o', linewidth=2)
        plt.title('Perfil Diário de Erro (RMSE por Hora)')
        plt.xlabel('Hora do Dia (Local)')
        plt.ylabel('RMSE (kW)')
        plt.xticks(range(5, 21)) # Mostra apenas horas de sol
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_path, 'profile_RMSE_Hourly.png'))
        plt.close()
        
        # Plot MAE (Opcional, bom para ver magnitude absoluta)
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=grouped, x='Hour', y='MAE', hue='Modelo', marker='s', linestyle='--', linewidth=2)
        plt.title('Perfil Diário de Erro Absoluto (MAE por Hora)')
        plt.xlabel('Hora do Dia (Local)')
        plt.ylabel('MAE (kW)')
        plt.xticks(range(5, 21))
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_path, 'profile_MAE_Hourly.png'))
        plt.close()


    def _find_representative_days(self):
        """
        Lógica interna para encontrar dias de Céu Claro e Dias Nublados automaticamente.
        Critério:
        - Céu Claro: Alta soma de energia + Baixa variabilidade (curva lisa).
        - Nublado/Transiente: Energia média + Alta variabilidade (zigue-zague).
        """
        # Agrupa dados observados por data
        # Pega apenas o primeiro modelo para analisar os dados observados (que são iguais para todos)
        first_model = self.df['Modelo'].unique()[0]
        df_base = self.df[self.df['Modelo'] == first_model].copy()
        
        daily_stats = []
        for date, group in df_base.groupby('Date'):
            if len(group) < 10: continue # Pula dias incompletos
            
            energy_sum = group['Observado'].sum()
            # Variabilidade = Desvio padrão da primeira derivada (mudança hora a hora)
            volatility = group['Observado'].diff().std()
            
            daily_stats.append({
                'Date': date,
                'Energy': energy_sum,
                'Volatility': volatility
            })
            
        stats_df = pd.DataFrame(daily_stats)
        if stats_df.empty: return None, None
        
        # Encontra Céu Claro: Alta Energia e Baixa Volatilidade (relativa à energia)
        # Ordena por energia decrescente
        stats_df = stats_df.sort_values('Energy', ascending=False)
        # Pega o top 10% de energia e escolhe o de menor volatilidade
        top_energy = stats_df.head(int(len(stats_df)*0.2)) 
        clear_sky_day = top_energy.sort_values('Volatility').iloc[0]['Date']
        
        # Encontra Sombreamento/Nublado: Volatilidade Máxima
        cloudy_day = stats_df.sort_values('Volatility', ascending=False).iloc[0]['Date']
        
        return clear_sky_day, cloudy_day

    def plot_scenario_days(self):
        """
        Gera curvas de geração (Obs vs Prev) para dias específicos encontrados automaticamente.
        """
        print("🔍 Buscando dias representativos (Céu Claro vs Nublado)...")
        date_clear, date_cloudy = self._find_representative_days()
        
        scenarios = {
            'Ceu_Claro': date_clear,
            'Nublado_Transiente': date_cloudy
        }
        
        for label, date_obj in scenarios.items():
            if date_obj is None: continue
            
            print(f"   📅 Plotando {label}: {date_obj}")
            
            # Filtra dados desse dia
            day_data = self.df[self.df['Date'] == date_obj].sort_values('Timestamp')
            
            plt.figure(figsize=(14, 7))
            
            # Plota Observado (Preto, linha grossa)
            # Pega de um modelo só para não duplicar linha
            one_model_data = day_data[day_data['Modelo'] == day_data['Modelo'].unique()[0]]
            plt.plot(one_model_data['Timestamp'], one_model_data['Observado'], 
                     color='black', label='Observado (Real)', linewidth=2.5, zorder=10)
            
            # Plota Previsão de cada Modelo
            sns.lineplot(data=day_data, x='Timestamp', y='Previsto', hue='Modelo', 
                         linewidth=1.5, alpha=0.8)
            
            plt.title(f'Análise de Cenário: {label.replace("_", " ")} ({date_obj})')
            plt.ylabel('Potência (kW)')
            plt.xlabel('Horário')
            
            # Formata eixo X para mostrar horas
            import matplotlib.dates as mdates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_path, f'scenario_{label}.png'))
            plt.close()

    # ==========================================================================
    #  FASE 1: DIAGNÓSTICO GRANULAR (NOVO)
    # ==========================================================================
    
    def plot_error_by_hour_of_day(self):
        """
        Plota o RMSE distribuído pelas horas do dia (0h - 23h).
        Isso responde: 'O modelo erra mais ao meio-dia ou no entardecer?'
        """
        print("📈 Gerando perfil de erro horário...")
        
        # Filtra apenas horas com sol (opcional, mas recomendado para evitar ruído noturno)
        # Assumindo que 6h as 19h tem sol
        df_sun = self.df[(self.df['Hour'] >= 5) & (self.df['Hour'] <= 20)].copy()
        
        df_sun['SE'] = (df_sun['Observado'] - df_sun['Previsto']) ** 2
        df_sun['AE'] = np.abs(df_sun['Observado'] - df_sun['Previsto'])
        
        # Agrupa por Modelo e Hora do Dia
        grouped = df_sun.groupby(['Modelo', 'Hour']).agg(
            RMSE=('SE', lambda x: np.sqrt(x.mean())),
            MAE=('AE', 'mean')
        ).reset_index()
        
        # Plot RMSE
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=grouped, x='Hour', y='RMSE', hue='Modelo', marker='o', linewidth=2)
        plt.title('Perfil Diário de Erro (RMSE por Hora)')
        plt.xlabel('Hora do Dia (Local)')
        plt.ylabel('RMSE (kW)')
        plt.xticks(range(5, 21)) # Mostra apenas horas de sol
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_path, 'profile_RMSE_Hourly.png'))
        plt.close()
        
        # Plot MAE (Opcional, bom para ver magnitude absoluta)
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=grouped, x='Hour', y='MAE', hue='Modelo', marker='s', linestyle='--', linewidth=2)
        plt.title('Perfil Diário de Erro Absoluto (MAE por Hora)')
        plt.xlabel('Hora do Dia (Local)')
        plt.ylabel('MAE (kW)')
        plt.xticks(range(5, 21))
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_path, 'profile_MAE_Hourly.png'))
        plt.close()

    # ==========================================================================
    #  FASE 2: ANÁLISE DE CENÁRIOS FÍSICOS (NOVO)
    # ==========================================================================

    def _find_representative_days(self):
        """
        Lógica interna para encontrar dias de Céu Claro e Dias Nublados automaticamente.
        Critério:
        - Céu Claro: Alta soma de energia + Baixa variabilidade (curva lisa).
        - Nublado/Transiente: Energia média + Alta variabilidade (zigue-zague).
        """
        # Agrupa dados observados por data
        # Pega apenas o primeiro modelo para analisar os dados observados (que são iguais para todos)
        first_model = self.df['Modelo'].unique()[0]
        df_base = self.df[self.df['Modelo'] == first_model].copy()
        
        daily_stats = []
        for date, group in df_base.groupby('Date'):
            if len(group) < 10: continue # Pula dias incompletos
            
            energy_sum = group['Observado'].sum()
            # Variabilidade = Desvio padrão da primeira derivada (mudança hora a hora)
            volatility = group['Observado'].diff().std()
            
            daily_stats.append({
                'Date': date,
                'Energy': energy_sum,
                'Volatility': volatility
            })
            
        stats_df = pd.DataFrame(daily_stats)
        if stats_df.empty: return None, None
        
        # Encontra Céu Claro: Alta Energia e Baixa Volatilidade (relativa à energia)
        # Ordena por energia decrescente
        stats_df = stats_df.sort_values('Energy', ascending=False)
        # Pega o top 10% de energia e escolhe o de menor volatilidade
        top_energy = stats_df.head(int(len(stats_df)*0.2)) 
        clear_sky_day = top_energy.sort_values('Volatility').iloc[0]['Date']
        
        # Encontra Sombreamento/Nublado: Volatilidade Máxima
        cloudy_day = stats_df.sort_values('Volatility', ascending=False).iloc[0]['Date']
        
        return clear_sky_day, cloudy_day

    def plot_scenario_days(self):
        """
        Gera curvas de geração (Obs vs Prev) para dias específicos encontrados automaticamente.
        """
        print("🔍 Buscando dias representativos (Céu Claro vs Nublado)...")
        date_clear, date_cloudy = self._find_representative_days()
        
        scenarios = {
            'Ceu_Claro': date_clear,
            'Nublado_Transiente': date_cloudy
        }
        
        for label, date_obj in scenarios.items():
            if date_obj is None: continue
            
            print(f"   📅 Plotando {label}: {date_obj}")
            
            # Filtra dados desse dia
            day_data = self.df[self.df['Date'] == date_obj].sort_values('Timestamp')
            
            plt.figure(figsize=(14, 7))
            
            # Plota Observado (Preto, linha grossa)
            # Pega de um modelo só para não duplicar linha
            one_model_data = day_data[day_data['Modelo'] == day_data['Modelo'].unique()[0]]
            plt.plot(one_model_data['Timestamp'], one_model_data['Observado'], 
                     color='black', label='Observado (Real)', linewidth=2.5, zorder=10)
            
            # Plota Previsão de cada Modelo
            sns.lineplot(data=day_data, x='Timestamp', y='Previsto', hue='Modelo', 
                         linewidth=1.5, alpha=0.8)
            
            plt.title(f'Análise de Cenário: {label.replace("_", " ")} ({date_obj})')
            plt.ylabel('Potência (kW)')
            plt.xlabel('Horário')
            
            # Formata eixo X para mostrar horas
            import matplotlib.dates as mdates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_path, f'scenario_{label}.png'))
            plt.close()