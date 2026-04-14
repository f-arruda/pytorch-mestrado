import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureAttention(nn.Module):
    def __init__(self, input_size):
        super(FeatureAttention, self).__init__()
        
        # Uma rede neural pequena para calcular os "scores" de importância
        # Recebe o input e decide o peso de cada feature
        self.attn = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, input_size)
        )

    def forward(self, x):
        """
        x shape: (Batch, Seq_Len, Features)
        """
        # 1. Calcula os scores de atenção
        # scores shape: (Batch, Seq_Len, Features)
        scores = self.attn(x)
        
        # 2. Transforma em probabilidades (0 a 1) via Softmax
        # Aplicamos softmax na dimensão das features (dim=2)
        # Para dizer: "Neste instante t, a soma da importância das features é 1.0"
        weights = F.softmax(scores, dim=2)
        
        # 3. Aplica os pesos na entrada original (Element-wise multiplication)
        # Se o peso da Temp for 0.1, o valor da Temp é reduzido em 90%
        weighted_input = x * weights
        
        return weighted_input, weights