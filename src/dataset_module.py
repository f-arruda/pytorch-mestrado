import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SolarEfficientDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str, n_past: int, n_future: int):
        """
        Dataset profissional que separa explicitamente Features (X) e Target (Y).
        
        Args:
            df: DataFrame com índice datetime contínuo.
            feature_cols: Lista com os nomes das colunas de entrada (X).
            target_col: Nome da coluna alvo (Y).
            n_past: Tamanho da janela do passado.
            n_future: Tamanho da janela do futuro.
        """
        self.n_past = n_past
        self.n_future = n_future
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Validação básica
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Features faltando no DataFrame: {missing_features}")
        if target_col not in df.columns:
            raise ValueError(f"Target '{target_col}' não encontrado no DataFrame.")

        # Conversão para Tensor (Mantém na GPU/CPU memória apenas o necessário)
        # X: Apenas as colunas de feature
        self.data_input = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        # Y: Apenas a coluna de target
        self.data_target = torch.tensor(df[[target_col]].values, dtype=torch.float32)
        
        self.timestamps = df.index
        
        # Precomputação de índices (Lógica vetorizada mantida)
        self.valid_indices = self._precompute_valid_indices(df, feature_cols, target_col)
        print(f"✅ Dataset pronto. Amostras válidas: {len(self.valid_indices)}")

    def _precompute_valid_indices(self, df, feature_cols, target_col):
        valid_starts = []
        n_total = len(df)
        
        # Converte para numpy para velocidade
        targets = df[target_col].values
        inputs = df[feature_cols].values
        
        # Máscaras booleanas
        not_null_target = ~np.isnan(targets)
        no_minus_one = (targets != -1)
        not_null_input = ~np.isnan(inputs).any(axis=1)

        # Loop otimizado
        for i in range(self.n_past, n_total - self.n_future + 1):
            # Validação do Passado (X)
            if not np.all(not_null_input[i - self.n_past : i]):
                continue 
                
            # Validação do Futuro (Y)
            future_validity = not_null_target[i : i + self.n_future] & no_minus_one[i : i + self.n_future]
            if not np.all(future_validity):
                continue 
            
            valid_starts.append(i)
            
        return valid_starts

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        # X: Features do passado
        x = self.data_input[real_idx - self.n_past : real_idx]
        
        # Y: Target do futuro
        y = self.data_target[real_idx : real_idx + self.n_future]
        
        return x, y