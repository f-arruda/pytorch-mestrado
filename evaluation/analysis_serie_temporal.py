import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import welch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class AnalisadorDeResiduos:
    def __init__(self, df, col_medido, col_previsto):
        """
        Inicializa a classe, remove valores nulos e calcula os resíduos.
        """
        self.df = df[[col_medido, col_previsto]].dropna().copy()
        self.col_medido = col_medido
        self.col_previsto = col_previsto
        
        # Calcula os resíduos (Medido - Previsto)
        self.df['residuos'] = self.df[col_medido] - self.df[col_previsto]
        
        # O uso do .flatten() resolve o erro do DTW garantindo que 
        # os arrays tenham formato (N,) e não (N, 1)
        self.arr_medido = self.df[col_medido].values.flatten()
        self.arr_previsto = self.df[col_previsto].values.flatten()
        self.arr_residuos = self.df['residuos'].values.flatten()

    def teste_whiteness(self, lags=10):
        resultado = acorr_ljungbox(self.arr_residuos, lags=[lags], return_df=True)
        p_value = resultado['lb_pvalue'].iloc[0]
        is_white_noise = p_value > 0.05
        return p_value, is_white_noise

    def teste_color(self):
        frequencias, psd = welch(self.arr_residuos)
        
        frequencias = frequencias[1:]
        psd = psd[1:]
        
        log_freq = np.log10(frequencias)
        log_psd = np.log10(psd)
        
        slope, _ = np.polyfit(log_freq, log_psd, 1)
        
        if abs(slope) < 0.5:
            cor = "Branco (Erros aleatórios independentes)"
        elif -1.5 < slope <= -0.5:
            cor = "Rosa (Memória de longo prazo)"
        else:
            cor = "Vermelho/Browniano (Forte persistência, Random Walk)"
            
        return slope, cor

    def teste_dtw(self):
        """Calcula a distância DTW (Dynamic Time Warping)"""
        # Transforma arrays de (N,) para (N, 1)
        medido_reshaped = self.arr_medido.reshape(-1, 1)
        previsto_reshaped = self.arr_previsto.reshape(-1, 1)
        
        distancia, _ = fastdtw(medido_reshaped, previsto_reshaped, dist=euclidean)
        return distancia

    def gerar_relatorio(self, lags=10):
        print("="*60)
        print("📊 RELATÓRIO DE ANÁLISE DE COMPORTAMENTO DO MODELO")
        print("="*60)
        
        p_value, is_white = self.teste_whiteness(lags)
        print("\n1️⃣ TESTE DE WHITENESS (Ljung-Box)")
        print(f"   p-value: {p_value:.5e}")
        if is_white:
            print("   ▶ Os resíduos SÃO Ruído Branco (Bom sinal!).")
        else:
            print("   ▶ Os resíduos NÃO SÃO Ruído Branco (Comum em persistência).")

        slope, cor = self.teste_color()
        print("\n2️⃣ TESTE DE COLOR (Espectro de Frequência)")
        print(f"   Inclinação (Slope): {slope:.2f}")
        print(f"   ▶ Cor: {cor}")

        # O DTW pode demorar se a série for gigantesca
        print("\n3️⃣ TESTE DTW (Calculando...)")
        dtw_dist = self.teste_dtw()
        print(f"   Distância DTW: {dtw_dist:.2f}")
        print("="*60)
        
        return {
            "whiteness_p_value": p_value,
            "is_white_noise": is_white,
            "color_slope": slope,
            "color_name": cor,
            "dtw_distance": dtw_dist
        }

    def plotar_analises(self, num_amostras=None):
        """
        Gera 4 gráficos para análise visual do comportamento do modelo e dos resíduos.
        num_amostras: Limita o número de pontos nos gráficos temporais para melhor visualização.
        """
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Análise Visual do Modelo', fontsize=16)
        
        df_plot = self.df.iloc[:num_amostras] if num_amostras else self.df
        
        # 1. Gráfico Medido vs Previsto
        axs[0, 0].plot(df_plot.index, df_plot[self.col_medido], label='Real (Medido)', color='#1f77b4', alpha=0.8)
        axs[0, 0].plot(df_plot.index, df_plot[self.col_previsto], label='Previsto', color='#ff7f0e', alpha=0.8, linestyle='--')
        axs[0, 0].set_title('Série Temporal: Real vs Previsto')
        axs[0, 0].set_ylabel('Valor')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Gráfico de Resíduos
        axs[0, 1].plot(df_plot.index, df_plot['residuos'], color='#d62728', alpha=0.7)
        axs[0, 1].axhline(0, color='black', linestyle='--', linewidth=1)
        axs[0, 1].set_title('Resíduos (Erros)')
        axs[0, 1].set_ylabel('Erro (Real - Previsto)')
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Gráfico de Autocorrelação (ACF) - Visualiza o Whiteness
        # Lags limitados a 40 ou ao tamanho máximo da série
        lags = min(40, len(self.arr_residuos) - 1)
        plot_acf(self.arr_residuos, ax=axs[1, 0], lags=lags, alpha=0.05, color='#2ca02c')
        axs[1, 0].set_title('Autocorrelação dos Resíduos (ACF)')
        axs[1, 0].set_xlabel('Lags')
        axs[1, 0].set_ylabel('Autocorrelação')
        axs[1, 0].grid(True, alpha=0.3)
        
        # 4. Gráfico do Espectro de Potência (PSD) - Visualiza o Color
        frequencias, psd = welch(self.arr_residuos)
        axs[1, 1].loglog(frequencias[1:], psd[1:], color='#9467bd')
        axs[1, 1].set_title('Densidade de Potência Espectral (Escala Log-Log)')
        axs[1, 1].set_xlabel('Frequência')
        axs[1, 1].set_ylabel('Potência (PSD)')
        axs[1, 1].grid(True, which="both", ls="--", alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()