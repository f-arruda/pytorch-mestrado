import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SolarEfficientDataset(Dataset):
    def __init__(self, df: pd.DataFrame, u_cols: list, y_past_cols: list, 
                 target_cols: list, aux_cols: list,
                 n_past: int, n_future: int):
        """
        Dataset profissional que separa explicitamente Entradas Exógenas (U) e Passado do Alvo (Y_past).
        
        Args:
            df: DataFrame com índice datetime contínuo.
            u_cols: Lista com os nomes das colunas exógenas (ex: temperatura, pressão).
            y_past_cols: Lista com os nomes das colunas de lags do alvo (ex: kt, kd passados).
            target_cols: Nome das colunas alvo (Y futuro, ex: kt, kd futuros).
            aux_cols: Colunas auxiliares para a loss física (GHI_cs, Zenite, etc).
            n_past: Tamanho da janela do passado.
            n_future: Tamanho da janela do futuro.
        """
        self.n_past = n_past
        self.n_future = n_future
        self.u_cols = u_cols
        self.y_past_cols = y_past_cols
        self.target_cols = target_cols
        self.aux_cols = aux_cols
        
        # Validação básica de colunas
        all_required_cols = u_cols + y_past_cols + target_cols + aux_cols + ['mask']
        missing_cols = [c for c in all_required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Colunas faltando no DataFrame: {missing_cols}")

        # Conversão para Tensor (Mantém na memória apenas o necessário)
        # U: Variáveis Exógenas
        self.data_u = torch.tensor(df[u_cols].values, dtype=torch.float32)
        # Y_past: Passado das variáveis que queremos prever
        self.data_y_past = torch.tensor(df[y_past_cols].values, dtype=torch.float32)
        # Y_future: Variáveis alvo no futuro
        self.data_target = torch.tensor(df[target_cols].values, dtype=torch.float32)
        
        # Mascará -> tensor
        self.mask = torch.tensor(df['mask'].values, dtype=torch.float32)
        # Aux -> tensor
        self.data_aux = torch.tensor(df[aux_cols].values, dtype=torch.float32)
        
        self.timestamps = df.index
        
        # Precomputação de índices (Lógica vetorizada mantida)
        self.valid_indices = self._precompute_valid_indices(df, u_cols, y_past_cols, target_cols)
        print(f"✅ Dataset pronto. Amostras válidas: {len(self.valid_indices)}")

    def _precompute_valid_indices(self, df, u_cols, y_past_cols, target_cols):
        valid_starts = []
        n_total = len(df)
        
        # Converte para numpy para velocidade no loop
        targets = df[target_cols].values
        u_inputs = df[u_cols].values
        y_inputs = df[y_past_cols].values
        mask = df['mask'].values
        
        # Máscaras booleanas contra NaNs (verifica se há algum NaN na linha inteira)
        not_null_target = ~np.isnan(targets).any(axis=1) if targets.ndim > 1 else ~np.isnan(targets)
        not_null_u = ~np.isnan(u_inputs).any(axis=1) if len(u_cols) > 0 else np.ones(n_total, dtype=bool)
        not_null_y = ~np.isnan(y_inputs).any(axis=1) if len(y_past_cols) > 0 else np.ones(n_total, dtype=bool)

        # Loop otimizado
        for i in range(self.n_past, n_total - self.n_future + 1):
            
            # 1. Validação do FUTURO (Obrigatório: o alvo deve ser válido)
            future_mask = mask[i : i + self.n_future]
            if np.sum(future_mask) < self.n_future:
                continue # Pula se o que queremos prever é noite ou dado ruim na máscara

            # 2. Validação do PASSADO (Exige contexto diurno no passado)
            past_mask = mask[i - self.n_past : i]
            if np.sum(past_mask) < self.n_past * 0.3:
                continue # Pula se não houver pelo menos 30% de contexto diurno válido

            # 3. Verificação de NaNs (Segurança nas janelas de tempo exatas)
            if not np.all(not_null_u[i - self.n_past : i]):
                continue 
            if not np.all(not_null_y[i - self.n_past : i]):
                continue 
            if not np.all(not_null_target[i : i + self.n_future]):
                continue
            
            valid_starts.append(i)
            
        return valid_starts

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        # U: Features exógenas do passado
        u_hist = self.data_u[real_idx - self.n_past : real_idx]
        
        # Y_past: Lags das variáveis alvo (Passado de kt e kd)
        y_hist = self.data_y_past[real_idx - self.n_past : real_idx]
        
        # Y_future: Target do futuro
        y_future = self.data_target[real_idx : real_idx + self.n_future]

        # Mask: mascara de dia e noite
        mask = self.mask[real_idx : real_idx + self.n_future]

        # Aux: variáveis físicas do futuro para a função de perda
        aux_future = self.data_aux[real_idx: real_idx + self.n_future]
        
        return u_hist, y_hist, y_future, mask, aux_future