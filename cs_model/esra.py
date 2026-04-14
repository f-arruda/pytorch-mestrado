import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Union

class ESRA:
    def __init__ (self, 
                  apparent_zenith: pd.Series, 
                  dni_extra: pd.Series, 
                  linke_turbidity: Union[pd.Series, float], 
                  altitude: float):
        
        # Garante que as séries tenham índice datetime se possível
        self.apparent_zenith = apparent_zenith
        self.dni_extra = dni_extra
        self.linke_turbidity = linke_turbidity
        self.altitude = altitude # Altitude do local em metros (ex: 570)

    def _esra_model(self, gepsilon, sza, eelevation, ltf):
        """
        Núcleo matemático do modelo ESRA.
        eelevation: Altitude do SITE (metros).
        sza: Ângulo Zenital Solar (graus).
        """
        # Garante arrays para evitar erros com floats escalares
        sza = np.atleast_1d(sza)
        gepsilon = np.atleast_1d(gepsilon)
        
        # Constantes e correções
        Io = gepsilon
        th_s = 0.5 
        
        # Pressão baseada na altitude do local (eelevation em metros)
        P_Po = np.exp(-eelevation / 8434.5)
     
        # Refração atmosférica para corrigir ângulo zenital
        sea = 90 - sza
        sea_rad = np.deg2rad(sea)
        delta_sea = 0.061359 * (180 / np.pi) * ((0.1594 + (sea_rad) * 1.123 + ((sea_rad) ** 2) * 0.065656) /
                                              (1 + (sea_rad) * 28.9344 + ((sea_rad) ** 2) * 277.3971))
        sea = sea + delta_sea
        sea[sea <= 0] = 0
        sza = 90 - sea # Zenital corrigido
     
        # Seno da elevação
        csa = np.sin(np.deg2rad(sea))
        mask_low = sea < th_s
        csa[mask_low] = np.pi * sea[mask_low] / 180
        csa[csa < 0] = 0
     
        # Massa de ar (m)
        m_denom = csa + (0.50572 * (sea + 6.07995) ** (-1.6364))
        m = P_Po / np.maximum(m_denom, 1e-6)
        m[m > 37.9] = 37.9 
     
        # Espalhamento Rayleigh
        Rs = np.zeros_like(m)
        mask_m20 = m <= 20
        Rs[mask_m20] = 1 / (6.6296 + (1.7513 * m[mask_m20]) - (0.1202 * m[mask_m20] ** 2) + (0.0065 * m[mask_m20] ** 3) - (0.00013 * m[mask_m20] ** 4))
        Rs[~mask_m20] = 1 / (10.4 + (0.718 * m[~mask_m20]))
     
        # Transmitância direta
        Trb = np.exp(-0.8662 * ltf * P_Po * Rs)
     
        # Coeficientes baseados na elevação máxima do array
        max_sea = np.max(sea) if sea.size > 0 else 0
        if max_sea > 30:
            L00, L01, L02 = -1.7349E-2, -5.8985E-3, 6.8868E-4
            L10, L11, L12 = 1.0258, -1.2196E-1, 1.9229E-3
            L20, L21, L22, L23 = -7.2178E-3, 1.3086E-1, -2.8405E-3, 0
        elif max_sea > 15:
            L00, L01, L02 = -8.2193E-3, 4.5643E-4, 6.7916E-5
            L10, L11, L12 = 8.9233E-1, -1.9991E-1, 9.9741E-3
            L20, L21, L22, L23 = 2.5428E-1, 2.6140E-1, -1.7020E-2, 0
        else:
            L00, L01, L02, L23 = -1.1656E-3, 1.8408E-4, -4.8754E-7, 0 
            L10, L11, L12 = 7.4095E-1, -2.2427E-1, 1.5314E-2
            L20, L21, L22 = 3.4959E-1, 7.2313E-1, -1.2305E-1

        # Polinômios
        C0 = L00 + (L01 * ltf * P_Po) + (L02 * (ltf * P_Po) ** 2)
        C1 = L10 + (L11 * ltf * P_Po) + (L12 * (ltf * P_Po) ** 2)
        if max_sea > 30:
            C2 = L20 + (L21 * ltf * P_Po) + (L22 * (ltf * P_Po) ** 2) + (L23 * (ltf * P_Po) ** 3)
        else:
            C2 = L20 + (L21 * ltf * P_Po) + (L22 * (ltf * P_Po) ** 2)
     
        Fb = C0 + (C1 * csa) + (C2 * csa ** 2)
     
        ESRA_DNI = Io * Trb * Fb
        ESRA_DNI[ESRA_DNI < 0] = 0
     
        # Transmitância difusa
        Trd = (-1.5843E-2) + (3.0543E-2 * ltf) + (3.797E-4 * ltf ** 2)
        A0 = (2.6463E-1) + (-6.1581E-2 * ltf) + (3.1408E-3 * ltf ** 2)
        
        # Correção A0
        cond_val = A0 * Trd
        replace_val = np.divide(2E-3, Trd, out=np.zeros_like(Trd), where=Trd!=0)
        A0 = np.where(cond_val < 2E-3, replace_val, A0)
        A1 = (2.0402) + (1.8945E-2 * ltf) + (-1.1161E-2 * ltf ** 2)
        A2 = (-1.3025) + (3.9231E-2 * ltf) + (8.5079E-3 * ltf ** 2)
        Fd = A0 + (A1 * csa) + (A2 * csa ** 2)
        Fd = np.where(sea > 0, Fd, 1)
     
        ESRA_DHI = Io * Trd * Fd
        ESRA_DHI[ESRA_DHI < 0] = 0
        ESRA_GHI = ESRA_DNI + ESRA_DHI
     
        # Recálculo DNI consistente
        csa_safe = np.where(csa < 1e-4, 1e-4, csa)
        ESRA_DNI_Recalc = (ESRA_GHI - ESRA_DHI) / csa_safe
        ESRA_DNI_Recalc[ESRA_DNI_Recalc < 0] = 0
        IO = Io * csa
     
        return ESRA_GHI, ESRA_DNI_Recalc, ESRA_DHI, IO

    def get_esra_clearsky(self):
        """Retorna DataFrame com GHI, DNI, DHI calculados."""
        sza_arr = self.apparent_zenith.values
        gepsilon_arr = self.dni_extra.values
        
        # Prepara TL (Series ou float)
        if isinstance(self.linke_turbidity, (pd.Series, pd.DataFrame)):
            ltf_arr = self.linke_turbidity.values
        else:
            ltf_arr = self.linke_turbidity
            
        # IMPORTANTE: Aqui passamos a altitude do local (fixa)
        site_elevation_arr = np.full_like(sza_arr, self.altitude)
        
        esra_ghi, esra_dni, esra_dhi, _ = self._esra_model(
            gepsilon=gepsilon_arr,
            sza=sza_arr,
            eelevation=site_elevation_arr, 
            ltf=ltf_arr
        )
        
        # Limpeza final
        df_result = pd.DataFrame({
            'ghi': np.nan_to_num(esra_ghi, nan=0.0),
            'dni': np.nan_to_num(esra_dni, nan=0.0),
            'dhi': np.nan_to_num(esra_dhi, nan=0.0)
        }, index=self.apparent_zenith.index)
        
        return df_result

    def _objective_function(self, tl, sza_arr, dni_extra_arr, ghi_measured_arr):
        """Função de erro (RMSE) para o otimizador."""
        if tl < 1.0 or tl > 15.0: return 1e6 # Limites físicos

        tl_val = float(tl)
        site_elevation_arr = np.full_like(sza_arr, self.altitude)
        
        esra_ghi, _, _, _ = self._esra_model(
            gepsilon=dni_extra_arr, sza=sza_arr,
            eelevation=site_elevation_arr, ltf=tl_val
        )
        
        mask_valid = np.isfinite(ghi_measured_arr) & np.isfinite(esra_ghi)
        if np.sum(mask_valid) == 0: return 1e6
        
        diff = ghi_measured_arr[mask_valid] - esra_ghi[mask_valid]
        return np.sqrt(np.mean(diff ** 2))

    def get_clear_days_pearson(self, df_measured, threshold=0.97, min_ratio=0.8, min_hours=5):
        """
        Identifica dias claros baseando-se na forma (Correlação de Pearson) e magnitude (Ratio).
        Retorna uma lista de objetos datetime.date.
        """
        # Cria modelo de referência com TL neutro (3.0)
        esra_ref = ESRA(self.apparent_zenith, self.dni_extra, 3.0, self.altitude)
        df_ref = esra_ref.get_esra_clearsky()
        
        sza = self.apparent_zenith.reindex(df_measured.index)
        ele = 90 - sza
        clear_dates = []
        
        for day in np.unique(df_measured.index.date):
            mask = (df_measured.index.date == day) & (ele > 10)
            obs = df_measured.loc[mask, 'ghi']
            ref = df_ref.loc[mask, 'ghi']
            
            valid = np.isfinite(obs) & np.isfinite(ref)
            obs = obs[valid]
            ref = ref[valid]
            
            if len(obs) >= min_hours:
                # 1. Forma (Correlação)
                if np.std(obs) == 0 or np.std(ref) == 0: continue # Evita erro em correlação
                corr = np.corrcoef(obs, ref)[0, 1]
                
                # 2. Magnitude (Energia total vs Teórico)
                ratio = obs.sum() / ref.sum() if ref.sum() > 0 else 0
                
                if corr >= threshold and ratio >= min_ratio:
                    clear_dates.append(day)
                    
        return clear_dates

    def optimize_linke_turbidity(self, df_measured: pd.DataFrame, inplace: bool = True, use_pearson: bool = True):
        """
        Calcula o melhor TL para os dias claros e interpola para os demais.
        use_pearson=False: Confia que df_measured['is_clear_moment'] já está preenchido corretamente.
        """
        if not isinstance(df_measured.index, pd.DatetimeIndex):
            raise ValueError("O índice deve ser DatetimeIndex.")

        # Prepara dados auxiliares
        sza_ref = self.apparent_zenith.reindex(df_measured.index).ffill()
        dni_extra_ref = self.dni_extra.reindex(df_measured.index).ffill()
        
        # --- SELEÇÃO DE DIAS ---
        days_to_optimize = []
        if use_pearson:
            # Se solicitado, calcula internamente (padrão)
            days_to_optimize = self.get_clear_days_pearson(df_measured)
        elif 'is_clear_moment' in df_measured.columns:
            # Se usar manual, pega os dias onde 'is_clear_moment' é True
            # Agrupamos por dia e verificamos se há flags de clareza
            daily_check = df_measured.groupby(df_measured.index.date)['is_clear_moment'].any()
            days_to_optimize = daily_check[daily_check].index
        else:
            # Fallback: tenta tudo
            days_to_optimize = np.unique(df_measured.index.date)

        print(f"Otimizando TL para {len(days_to_optimize)} dias selecionados...")

        daily_tl = {}
        all_days = np.unique(df_measured.index.date)
        
        # --- LOOP DE OTIMIZAÇÃO ---
        for day in all_days:
            # Se o dia NÃO foi selecionado como claro, marcamos NaN para interpolar depois
            if day not in days_to_optimize:
                daily_tl[day] = np.nan
                continue

            # Otimiza apenas os dias selecionados
            ele_ref = 90 - sza_ref
            mask_day = (df_measured.index.date == day) & (ele_ref > 15)
            
            sub_ghi = df_measured.loc[mask_day, 'ghi'].values
            sub_sza = sza_ref.loc[mask_day].values
            sub_dni_extra = dni_extra_ref.loc[mask_day].values
            
            valid_idx = np.isfinite(sub_ghi) & np.isfinite(sub_sza) & np.isfinite(sub_dni_extra)
            sub_ghi = sub_ghi[valid_idx]
            sub_sza = sub_sza[valid_idx]
            sub_dni_extra = sub_dni_extra[valid_idx]

            if len(sub_ghi) > 5:
                res = minimize_scalar(
                    self._objective_function, 
                    bounds=(1.0, 10.0), 
                    args=(sub_sza, sub_dni_extra, sub_ghi), 
                    method='bounded'
                )
                if res.success:
                    daily_tl[day] = res.x
                else:
                    daily_tl[day] = np.nan 
            else:
                daily_tl[day] = np.nan

        # --- INTERPOLAÇÃO TEMPORAL ---
        series_daily = pd.Series(daily_tl)
        # Preenche os dias nublados (NaN) interpolando entre os dias claros vizinhos
        series_daily_interp = series_daily.interpolate(method='linear', limit_direction='both')
        series_daily_interp = series_daily_interp.fillna(3.0) # Segurança final

        # Expande de diário para horário
        temp_date_series = pd.Series(df_measured.index.date, index=df_measured.index)
        tl_final_series = temp_date_series.map(series_daily_interp)
        
        if inplace:
            self.linke_turbidity = tl_final_series
            print(f"Otimização concluída. Média TL: {tl_final_series.mean():.2f}")
        
        return tl_final_series