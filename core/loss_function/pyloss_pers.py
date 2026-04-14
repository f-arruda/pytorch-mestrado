import torch
import torch.nn as nn
from core.loss_function.cpiloss import CPILoss
import numpy as np

class PhysicsGuidedLossPers(nn.Module):
    def __init__(self, 
                 lambda_hard=0.1, lambda_soft=0.1, lambda_pers=0.1,
                 data_loss_type='mse'):
        super(PhysicsGuidedLossPers, self).__init__()
        self.lambda_hard = lambda_hard
        self.lambda_soft = lambda_soft
        self.lambda_pers = lambda_pers
        self.data_loss_type = data_loss_type
        
        if self.data_loss_type == 'cpi':
            self.data_loss_fn = CPILoss()
        elif self.data_loss_type == 'mse':
            self.data_loss_fn = nn.MSELoss(reduction='none')

    def forward(self, output, target, 
                aux_future, mask=None, x_past=None):
        
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
                expanded_mask = mask.unsqueeze(-1).expand_as(output) if mask.dim() < output.dim() else mask
                data_loss = (mse_raw * expanded_mask).sum() / (expanded_mask.sum() + eps)
            else:
                data_loss = mse_raw.mean()
        elif self.data_loss_type == 'cpi':
            loss_kt_val = self.data_loss_fn(pred_kt, original_kt, mask)
            loss_kd_val = self.data_loss_fn(pred_kd, original_kd, mask)
            data_loss = torch.sqrt(loss_kt_val**2 + loss_kd_val**2 + eps)
        
        #=============================
        #   --- Quality Control ---
        #=============================
        ghi_cs = aux_future[:, :, 0]
        cos_zenith = aux_future[:, :, 1]
        elevation = aux_future[:, :, 2] * 90
        ghi_extra = aux_future[:, :, 3]

        sin_alpha = torch.sin(torch.deg2rad(elevation))
        
        def safe_denom(denom, eps=1e-8):
            # Garante divisor não-nulo sem modificar valores físicos negativos ou positivos.
            return torch.where(torch.abs(denom) < eps, torch.sign(denom + 1e-15) * eps, denom)
            
        ghi = pred_kt * ghi_extra
        dhi = pred_kd * ghi
        
        dni = (ghi - dhi) / safe_denom(cos_zenith)
        G = ghi_extra / safe_denom(cos_zenith)

        def masked_mean(error_tensor):
            if mask is not None:
                m = mask.squeeze(-1) if mask.dim() > error_tensor.dim() else mask
                return (error_tensor * m).sum() / (m.sum() + eps)
            return error_tensor.mean()

        #========================
        #   --- Hard Limits ---
        #========================
        # O expoente 1.2 gera NaN se sin_alpha < 0 (noite). A equação BSRN só existe fisicamente de dia.
        sin_alpha_pow = torch.where(sin_alpha > 0, sin_alpha ** 1.2, torch.zeros_like(sin_alpha))
        limit = 100 + 1.5 * G * sin_alpha_pow
        err_ghi_limits = (torch.relu(ghi-limit) + torch.relu(0 - ghi))
        err_dhi_limits =  (torch.relu(dhi-limit) + torch.relu(0 - dhi))
        err_dni_limits = (torch.relu(dni-G) + torch.relu(0 - dni))
        err_diffuse_fraction = (torch.relu(pred_kd - 1.1) + torch.relu(0 - pred_kd))

        sum_components = dhi + (dni * sin_alpha)
        closure_ratio = torch.abs((ghi - sum_components) / safe_denom(ghi))
        limit_val = torch.where(elevation > 15, torch.tensor(0.08, device=device), torch.tensor(0.15, device=device))
        
        components = torch.relu(50 - sum_components)
        closure = torch.relu(closure_ratio - limit_val)
        err_consistence_check = (components + closure)

        loss_ghi_limits = masked_mean(err_ghi_limits)
        loss_dhi_limits = masked_mean(err_dhi_limits)
        loss_dni_limits = masked_mean(err_dni_limits)
        loss_diffuse_fraction = masked_mean(err_diffuse_fraction)
        loss_consistence_check = masked_mean(err_consistence_check)

        #========================
        #   --- Soft Limits ---
        #========================
        lhs_7 = ghi / safe_denom(G * sin_alpha)
        rhs_7 = (elevation - 10) / 10000
        err_ghi_min_slope = (torch.relu(rhs_7 - lhs_7))
        term_const = -1 + (1.05 / 0.95)
        lhs_9 = ((dni * sin_alpha) / safe_denom(ghi))
        rhs_9 = ((ghi / safe_denom(ghi_cs)) + term_const)
        err_diffuse_consistency = (torch.relu(lhs_9 - rhs_9))
        check_ghi_6 = ghi / safe_denom(ghi_cs)
        check_dhi_6 = dhi / safe_denom(ghi)
        err_reject_ratio = (torch.relu(check_ghi_6-0.85) + torch.relu(check_dhi_6-0.85))

        loss_ghi_min_slope = masked_mean(err_ghi_min_slope)
        loss_diffuse_consistency = masked_mean(err_diffuse_consistency)
        loss_reject_ratio = masked_mean(err_reject_ratio)
        
        loss_hard_limits = loss_ghi_limits + loss_dhi_limits + loss_dni_limits + \
                            loss_diffuse_fraction + loss_consistence_check
        loss_soft_limits = loss_ghi_min_slope + loss_diffuse_consistency + \
                           loss_reject_ratio
        
        #===========================
        #   --- Loss Persistência ---
        #===========================
        loss_persistencia = torch.tensor(0.0, device=device)
        
        if x_past is not None and self.lambda_pers > 0:
            # Assumindo que x_past possui [Batch, SeqLen, Features]
            # e que features 0=kt, 1=kd como em target e output.
            # Pegamos o valor do último timestep de entrada.
            last_kt = x_past[:, -1:, 0]
            last_kd = x_past[:, -1:, 1]
            
            # Calculamos a métrica de erro (ex: MSE) da persistência vs target
            pers_err_kt = (last_kt - original_kt)**2
            pers_err_kd = (last_kd - original_kd)**2
            model_err_kt = (pred_kt - original_kt)**2
            model_err_kd = (pred_kd - original_kd)**2
            
            # Punição se o erro do modelo é maior que o erro da persistência.
            # Isso pune o modelo por não superar a persistência em previsão.
            err_pers = torch.relu(model_err_kt - pers_err_kt) + torch.relu(model_err_kd - pers_err_kd)
            loss_persistencia = masked_mean(err_pers)

        # L_function = L_data + Lambda_1 * hard_limits + Lambda_2 * soft_limits + Lambda_3 * pers_limits
        loss_function = data_loss + (self.lambda_hard * loss_hard_limits) + \
                        (self.lambda_soft * loss_soft_limits) + (self.lambda_pers * loss_persistencia)
        
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
            'check_persistence': loss_persistencia.detach().item(),
        }
   
        return loss_function, loss_dict
