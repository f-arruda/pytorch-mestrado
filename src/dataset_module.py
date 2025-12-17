import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SolarEfficientDataset(Dataset):
    def __init__(self, df, input_tag, n_past, n_future):
        """
        df: DataFrame com índice datetime contínuo (use df.asfreq('H') antes se tiver buracos).
        input_tag: Lista de colunas de entrada (features).
        n_past: Tamanho da janela do passado (horas).
        n_future: Tamanho da janela do futuro (horas).
        """
        self.n_past = n_past
        self.n_future = n_future
        
        # 1. Converter para Float32 (Padrão PyTorch) e Numpy para velocidade máxima
        # Mantemos os dados brutos na memória (apenas uma cópia)
        self.data_input = torch.tensor(df[input_tag].values, dtype=torch.float32)
        self.data_target = torch.tensor(df[['Pot_BT']].values, dtype=torch.float32)
        self.timestamps = df.index
        
        print("Calculando índices válidos... (Isso roda uma vez só)")
        self.valid_indices = self._precompute_valid_indices(df, input_tag)
        print(f"Total de janelas válidas encontradas: {len(self.valid_indices)}")

    def _precompute_valid_indices(self, df, input_tag):
        """
        Substitui seu loop 'for' lento por operações vetorizadas rápidas.
        Retorna uma lista de inteiros onde é seguro começar uma janela.
        """
        valid_starts = []
        n_total = len(df)
        
        # Converte colunas para numpy para checagem rápida
        pot_bt = df['Pot_BT'].values
        inputs = df[input_tag].values
        
        # Criação de máscaras booleanas (True/False) para todo o dataset de uma vez
        
        # 1. Onde Pot_BT não é nulo?
        not_null_pot = ~np.isnan(pot_bt)
        
        # 2. Onde não tem '-1' (Sua regra específica)
        no_minus_one = (pot_bt != -1)
        
        # 3. Onde Inputs não são nulos? (Checa se há algum NaN na linha)
        not_null_input = ~np.isnan(inputs).any(axis=1)

        # Agora iteramos apenas índices inteiros, mas checamos as janelas matematicamente
        # Precisamos de espaço para n_past atrás e n_future na frente
        
        for i in range(self.n_past, n_total - self.n_future + 1):
            
            # --- Validação Vetorizada (Sua lógica original traduzida) ---
            
            # A. Validação do Passado (X)
            # Slice: [i - n_past : i]
            # Checa se todos os inputs nesse intervalo são válidos (não nulos)
            if not np.all(not_null_input[i - self.n_past : i]):
                continue # Pula se tiver NaN no passado
                
            # B. Validação do Futuro (Y)
            # Slice: [i : i + n_future]
            # Checa se Pot_BT não tem nulos E não tem -1 nesse intervalo
            future_pot = pot_bt[i : i + self.n_future]
            future_validity = not_null_pot[i : i + self.n_future] & no_minus_one[i : i + self.n_future]
            
            if not np.all(future_validity):
                continue # Pula se tiver NaN ou -1 no futuro
            
            # C. (Opcional) Checagem de Continuidade de Tempo
            # Se você já usou df.asfreq('H'), isso é garantido. 
            # Se não, precisaria checar: time[i] - time[i-n_past] == n_past hours
            
            # Se passou em tudo, salva o índice 'i' (que é o tempo 't' da previsão)
            valid_starts.append(i)
            
        return valid_starts

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Mágica do Lazy Loading:
        # Recebemos um índice virtual (0, 1, 2...) e traduzimos para o índice real do DF
        real_idx = self.valid_indices[idx]
        
        # Recorta os Tensors que já estão na memória
        # X: Do passado até agora (exclusivo)
        x = self.data_input[real_idx - self.n_past : real_idx]
        
        # Y: Do agora até o futuro
        y = self.data_target[real_idx : real_idx + self.n_future]
        
        return x, y