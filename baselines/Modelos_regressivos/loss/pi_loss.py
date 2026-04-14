import torch
import torch.nn as nn
from loss_function.cpiloss import CPILoss
import numpy as np

class PhysicsGuidedLoss(nn.Module):
    def __init__(self, lambda_hard=0.1, lambda_soft=0.1, data_loss_type='mse'):
        super(PhysicsGuidedLoss, self).__init__()
        self.lambda_hard = lambda_hard
        self.lambda_soft = lambda_soft
        self.data_loss_type = data_loss_type
        
        if self.data_loss_type == 'cpi':
            self.data_loss_fn = CPILoss()

        elif self.data_loss_type == 'mse':
            self.data_loss_fn = nn.MSELoss(reduction='none')

    def forward(self, output, target, 
                aux_future, mask=None):
        
        device = output.device

        # Separa as variáveis: index 0 = kt, index 1 = kd (fração difusa)
        pred_kt = output[:, :, 0]
        pred_kd = output[:, :, 1]

        original_kt = target[:, :, 0]
        original_kd = target[:, :, 1]

        eps = 1e-8

        # --- 1. Erro de Dados (MSE com Máscara) ---
        if self.data_loss_type == 'mse':
            mse_raw = self.data_loss_fn(output, target)
            if mask is not None:
                data_loss = (mse_raw * mask).sum() / (mask.sum() * output.size(-1) + eps)
            else:
                data_loss = mse_raw.mean()
        elif self.data_loss_type == 'cpi':
            loss_kt = self.data_loss_fn(pred_kt, original_kt, mask)
            loss_kd = self.data_loss_fn(pred_kd, original_kd, mask)
            #data_loss = (loss_kt + loss_kd) / 2
            data_loss = torch.sqrt(loss_kt**2 + loss_kd**2)
        
        #=============================
        #   --- Quality Control ---
        #=============================
        

        ghi_cs = aux_future[:, :, 0]
        cos_zenith = aux_future[:, :, 1]
        elevation = aux_future[:, :, 2] * 90
        ghi_extra = aux_future[:, :, 3]

        sin_alpha = torch.sin(torch.deg2rad(elevation))
        ghi = pred_kt * ghi_extra
        dhi = pred_kd * ghi
        dni = (ghi - dhi) / (cos_zenith + eps)
        G = ghi_extra / (cos_zenith + eps)

        # Aplicação da Máscara e Médias Individuais (Logica PINN)
        def masked_mean(error_tensor):
            if mask is not None:
                # Remove a dimensão extra da máscara se necessário
                m = mask.squeeze(-1) if mask.dim() > error_tensor.dim() else mask
                return (error_tensor * m).sum() / (m.sum() + eps)
            return error_tensor.mean()

        #========================
        #   --- Hard Limits ---
        #========================

        # 0 < X < (100 + 1.5 * G * sin(alpha)^1.2)
        limit = 100 + 1.5 * G * (sin_alpha ** 1.2)
        err_ghi_limits = (torch.relu(ghi-limit) + torch.relu(0 - ghi))
        err_dhi_limits =  (torch.relu(dhi-limit) + torch.relu(0 - dhi))
        err_dni_limits = (torch.relu(dni-G) + torch.relu(0 - dni))
        err_diffuse_fraction = (torch.relu(pred_kd - 1.1) + torch.relu(0 - pred_kd))

        # Consistência de Fechamento (Closure check)
        sum_components = dhi + (dni * sin_alpha)
        closure_ratio = torch.abs((ghi - sum_components) / (ghi + eps))
        limit_val = torch.where(elevation > 15, torch.tensor(0.08, device=device), torch.tensor(0.15, device=device))
        
        components = torch.relu(50 - sum_components)
        closure = torch.relu(closure_ratio - limit_val)

        err_consistence_check = (components + closure)

        # Loss mean masked
        loss_ghi_limits = masked_mean(err_ghi_limits)
        loss_dhi_limits = masked_mean(err_dhi_limits)
        loss_dni_limits = masked_mean(err_dni_limits)
        loss_diffuse_fraction = masked_mean(err_diffuse_fraction)
        loss_consistence_check = masked_mean(err_consistence_check)

        #========================
        #   --- Soft Limits ---
        #========================
        # Teste de GHI mínimo relativo à elevação
        lhs_7 = ghi / (G * sin_alpha + eps)
        rhs_7 = (elevation - 10) / 10000
        err_ghi_min_slope = (torch.relu(rhs_7 - lhs_7))
        # Consistência de Difusa
        term_const = -1 + (1.05 / 0.95)
        lhs_9 = ((dni * sin_alpha) / (ghi + eps))
        rhs_9 = ((ghi / ghi_cs) + term_const)
        err_diffuse_consistency = (torch.relu(lhs_9 - rhs_9))
        # Critério de Rejeição 
        check_ghi_6 = ghi / (ghi_cs + eps)
        check_dhi_6 = dhi / (ghi + eps)
        err_reject_ratio = (torch.relu(check_ghi_6-0.85) + torch.relu(check_dhi_6-0.85))

        # Loss mean masked
        loss_ghi_min_slope = masked_mean(err_ghi_min_slope)
        loss_diffuse_consistency = masked_mean(err_diffuse_consistency)
        loss_reject_ratio = masked_mean(err_reject_ratio)
        
        # --- Loss dos Hard Limits ---
        loss_hard_limits = loss_ghi_limits + loss_dhi_limits + loss_dni_limits + \
                            loss_diffuse_fraction + loss_consistence_check
        # --- Loss dos Softs Limits ---
        loss_soft_limits = loss_ghi_min_slope + loss_diffuse_consistency + \
                           loss_reject_ratio
        
        # --- Loss Function ---
        # L_function = L_data + Lambda_1 * hard_limits + Lambda_2 * soft_limits
        loss_function = data_loss + (self.lambda_hard * loss_hard_limits) + (self.lambda_soft * loss_soft_limits)
        
        #===============================================
        #   --- Salvando informações para logging --- 
        #===============================================
        loss_dict = {
            'mse_stat': data_loss.detach().item(),
            'check_ghi': loss_ghi_limits.detach().item(),
            'check_dhi': loss_dhi_limits.detach().item(),
            'check_dni': loss_dni_limits.detach().item(),
            'check_diffuse_fraction': loss_diffuse_fraction.detach().item(),
            'check_consistence': loss_consistence_check.detach().item(),
            'check_overcast_condition':loss_ghi_min_slope.detach().item(),
            'check_maximum_direct_fraction':loss_diffuse_consistency.detach().item(),
            'check_tracker_off':loss_reject_ratio.detach().item(),
            'check_loss_physics':(loss_hard_limits + loss_soft_limits).detach().item(),
        }
   
        return loss_function, loss_dict