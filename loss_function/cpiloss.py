import torch
import torch.nn as nn

class CPILoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CPILoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):

        # Garantir que são 1D para cálculos estatísticos
        pred_flat = output.view(-1)
        target_flat = target.view(-1)
        n = pred_flat.size(0)
        
        # --- 1. Cálculo do NRMSE (Eq. 4 e 5) ---
        mse = torch.mean((pred_flat - target_flat) ** 2)
        rmse = torch.sqrt(mse)

        target_mean = torch.mean(target_flat)
        nrmse = rmse / (target_mean + self.epsilon)

        # --- 2. Preparação para KSI e OVER (Baseado em CDFs) ---
        # Para tornar a comparação de CDFs diferenciável, ordenamos os vetores.
        # A integral da diferença absoluta entre CDFs é aproximada pela
        # média das diferenças absolutas dos valores ordenados.
        pred_sorted, _ = torch.sort(pred_flat)
        target_sorted, _ = torch.sort(target_flat)

        # Diferença absoluta entre as distribuições ordenadas
        diff_sorted = torch.abs(pred_sorted - target_sorted)

        # --- 3. Cálculo do VC (Variability Coefficient) (Eq. 8) ---
        # VC = max(D_max, D_hat_max) - min(D_min, D_hat_min) [cite: 859]
        max_val = torch.max(torch.max(target_flat), torch.max(pred_flat))
        min_val = torch.min(torch.min(target_flat), torch.min(pred_flat))
        vc = max_val - min_val + self.epsilon # Epsilon para evitar divisão por zero

        # Fator crítico de normalização mencionado nas Eq. 9 e 10 
        # O artigo usa o termo 1.63 * sqrt(n)
        critical_factor = 1.63 * torch.sqrt(torch.tensor(n, dtype=torch.float32, device=output.device))
        
        # Denominador comum para KSI e OVER (baseado na Eq. 9 e 10)
        # Nota: O artigo normaliza a integral. A integral discretizada é a soma * largura do passo (1/n).
        # Aqui usamos a média direta (diff_sorted.mean()) que implicitamente lida com o 1/n da integral.
        # Ajustamos a escala conforme a fórmula: 100 / (1.63 * sqrt(n) * VC)
        scaling_factor = 100.0 / (critical_factor * vc)

        # --- 4. Cálculo do KSI (Eq. 9) ---
        # Integral da diferença das CDFs 
        # Aproximamos a integral pela soma das diferenças ordenadas
        integral_approx = torch.sum(diff_sorted) 
        ksi = scaling_factor * integral_approx

        # --- 5. Cálculo do OVER (Eq. 10) ---
        # Integral apenas das partes que excedem o valor crítico 
        # O valor crítico dentro da integral é o limiar estatístico
        critical_limit = critical_factor * vc # Este é o valor limite para significância estatística
        
        # Como estamos somando diferenças discretas, precisamos adaptar o limiar para a escala da soma
        # Uma aproximação comum para Loss é penalizar as grandes desviações de distribuição
        excess = torch.clamp(diff_sorted - (critical_limit / n), min=0.0)
        over = scaling_factor * torch.sum(excess)   

        # --- 6. Cálculo do CPI (Eq. 11) ---
        # CPI = (KSI + OVER + 2 * NRMSE) / 4
        cpi = (ksi + over + (2*nrmse))/4

        # Passo 4: Retornar a média (redução)
        return cpi
    
