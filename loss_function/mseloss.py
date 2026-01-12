import torch
import torch.nn as nn

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.base_mse = nn.MSELoss(reduction='none') # 'none' para podermos aplicar a máscara antes da média

    def forward(self, output, target, mask=None):
        # Calcula o erro quadrado para todos os pontos (Dia e Noite)
        squared_errors = (output - target) ** 2
        
        if mask is not None:
            # Aplica a máscara: Zera o erro onde for noite (mask=0)
            # O mask deve ter o mesmo shape do output/target
            masked_errors = squared_errors * mask
            
            # Calcula a média dividindo apenas pelo número de amostras válidas (Dia)
            # Soma dos erros / Soma da máscara (quantidade de '1's)
            loss = masked_errors.sum() / (mask.sum() + 1e-8) # epsilon para não dividir por zero
        else:
            # Se não tiver máscara, média simples global
            loss = squared_errors.mean()
            
        return loss