import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SolarEfficientDatasetGroups(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, 
                 target_col: list, aux_col:list,
                 n_past: int, n_future: int,
                 group_col: str = None, 
                 target_group: int = None):
        """
        Dataset profissional que separa explicitamente Features (X) e Target (Y).
        
        Args:
            df: DataFrame com índice datetime contínuo.
            feature_cols: Lista com os nomes das colunas de entrada (X).
            target_col: Nome da coluna alvo (Y).
            n_past: Tamanho da janela do passado.
            n_future: Tamanho da janela do futuro.
            group_col: (Opcional) Nome da coluna que contém os IDs dos grupos.
            target_group: (Opcional) ID do grupo para filtrar as amostras.
        """
        self.n_past = n_past
        self.n_future = n_future
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.aux_col = aux_col
        
        # Salva as configurações de grupo
        self.group_col = group_col
        self.target_group = target_group
        
        # Validação básica
        missing_features = [c for c in feature_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Features faltando no DataFrame: {missing_features}")
        for col in target_col:
            if col not in df.columns:
                raise ValueError(f"Target '{col}' não encontrado no DataFrame.")
        if group_col is not None and group_col not in df.columns:
            raise ValueError(f"Coluna de grupo '{group_col}' não encontrada no DataFrame.")

        # Conversão para Tensor
        self.data_input = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.data_target = torch.tensor(df[target_col].values, dtype=torch.float32)
        self.mask = torch.tensor(df['mask'].values, dtype=torch.float32)
        self.data_aux = torch.tensor(df[aux_col].values, dtype=torch.float32)
        
        self.timestamps = df.index

        # Se houver coluna de grupo, armazena como Tensor para retornar no __getitem__
        if self.group_col is not None:
            self.groups_tensor = torch.tensor(df[self.group_col].values, dtype=torch.int32)
        else: 
            self.groups_tensor = None
        
        # Precomputação de índices
        self.valid_indices = self._precompute_valid_indices(df, feature_cols, target_col)
        
        msg_grupo = f" (Filtrado p/ Grupo {self.target_group})" if self.target_group is not None else ""
        print(f"✅ Dataset pronto. Amostras válidas{msg_grupo}: {len(self.valid_indices)}")

    def _precompute_valid_indices(self, df, feature_cols, target_col):
        valid_starts = []
        n_total = len(df)
        
        targets = df[target_col].values
        inputs = df[feature_cols].values
        mask = df['mask'].values
        
        # Puxa o array de grupos (numpy) apenas se a coluna foi definida
        groups_array = df[self.group_col].values if self.group_col is not None else None
        
        not_null_target = ~np.isnan(targets)
        not_null_input = ~np.isnan(inputs).any(axis=1)

        for i in range(self.n_past, n_total - self.n_future + 1):
            
            # --- NOVA CONDIÇÃO DE FILTRAGEM DE GRUPO ---
            # Se target_group foi definido, exige que TODOS os pontos de Y pertençam a ele
            if self.target_group is not None and groups_array is not None:
                # Compara o bloco futuro com o target_group
                if not np.all(groups_array[i : i + self.n_future] == self.target_group): continue
            
            # 1. Validação do FUTURO
            future_mask = mask[i : i + self.n_future]
            if np.sum(future_mask) < self.n_future: continue 

            # 2. Validação do PASSADO 
            past_mask = mask[i - self.n_past : i]
            if np.sum(past_mask) < self.n_past * 0.3: continue 

            # 3. Verificação de NaNs
            if not np.all(not_null_input[i - self.n_past : i]): continue 
            if not np.all(not_null_target[i : i + self.n_future]): continue
            
            valid_starts.append(i)
            
        return valid_starts

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        x = self.data_input[real_idx - self.n_past : real_idx]
        y = self.data_target[real_idx : real_idx + self.n_future]
        mask = self.mask[real_idx : real_idx + self.n_future]
        aux_future = self.data_aux[real_idx: real_idx + self.n_future]
        
        # Retorna o grupo apenas se a coluna foi configurada
        if self.groups_tensor is not None:
            group = self.groups_tensor[real_idx]
            return x, y, mask, aux_future, group
        else: 
            return x, y, mask, aux_future