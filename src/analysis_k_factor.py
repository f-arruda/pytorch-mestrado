import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

class KFactorAnalyzer:
    def __init__(self, model_name, output_root):
        # Cria pasta separada TIER_1 para não misturar com os gráficos de Potência
        self.model_name = model_name
        self.save_dir = os.path.join(output_root, "TIER_1_Fisica_Atmosferica", model_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
    def evaluate(self, y_pred_k, y_true_k, timestamps):
        print(f"      ☁️ [Física] Gerando gráficos de K para: {self.model_name}")
        
        # Garante alinhamento e formato
        min_len = min(len(y_pred_k), len(y_true_k), len(timestamps))
        df = pd.DataFrame({
            'K_Pred': y_pred_k.flatten()[:min_len],
            'K_Real': y_true_k.flatten()[:min_len]
        }, index=timestamps[:min_len])
        
        # Filtra dados noturnos/erros para o gráfico ficar limpo
        df = df[(df['K_Real'] >= 0.01) & (df['K_Real'] <= 1.5)]

        if df.empty:
            print("      ⚠️ Sem dados diurnos suficientes para análise de K.")
            return

        # 1. Scatter de Densidade (Onde o modelo acerta?)
        plt.figure(figsize=(7, 7))
        plt.hist2d(df['K_Real'], df['K_Pred'], bins=50, cmap='Blues', norm=LogNorm(), range=[[0, 1.2], [0, 1.2]])
        plt.colorbar(label='Densidade')
        plt.plot([0, 1.2], [0, 1.2], 'r--', label='Ideal')
        plt.xlabel("K Real"); plt.ylabel("K Previsto")
        plt.title(f"Aderência Física (K): {self.model_name}")
        plt.savefig(os.path.join(self.save_dir, "Scatter_K.png"), dpi=150)
        plt.close()

        # 2. Distribuição (O modelo sabe fazer extremos?)
        plt.figure(figsize=(10, 5))
        sns.kdeplot(df['K_Real'], fill=True, color='gray', alpha=0.3, label='Real')
        sns.kdeplot(df['K_Pred'], color='blue', linewidth=2, label='Modelo')
        plt.xlim(0, 1.2)
        plt.title("Distribuição: Real vs Modelo")
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "Distribuicao_K.png"), dpi=150)
        plt.close()

        # 3. Série Temporal (Zoom em alguns dias)
        daily_k = df.resample('D')['K_Real'].mean()
        # Tenta pegar 1 dia limpo e 1 nublado, se falhar pega o primeiro
        try:
            days = [daily_k[daily_k > 0.7].index[0], daily_k[daily_k < 0.4].index[0]]
        except:
            days = [df.index[0]]

        for day in days:
            d_str = day.strftime('%Y-%m-%d')
            sub = df.loc[d_str]
            if sub.empty: continue
            
            plt.figure(figsize=(10, 4))
            plt.plot(sub.index, sub['K_Real'], 'k-', label='Real')
            plt.plot(sub.index, sub['K_Pred'], 'r--', label='Modelo')
            plt.title(f"Dinâmica Atmosférica: {d_str}")
            plt.ylim(0, 1.3)
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, f"TimeSeries_{d_str}.png"), dpi=150)
            plt.close()