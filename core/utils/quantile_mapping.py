import numpy as np

"""
def quantile_mapping(obs, mod):
    # Remove NaNs para o cálculo das distribuições
    obs_clean = obs.dropna().values
    mod_clean = mod.dropna().values
    
    # 1. Ordenar os dados para criar as distribuições empíricas
    obs_sorted = np.sort(obs_clean)
    mod_sorted = np.sort(mod_clean)
    
    # 2. Criar o eixo de probabilidades (0 a 1)
    prob_obs = np.linspace(0, 1, len(obs_sorted))
    prob_mod = np.linspace(0, 1, len(mod_sorted))
    
    # 3. Encontrar o percentil de cada valor original do modelo
    # Fazemos uma interpolação: dado um valor x, qual sua probabilidade P?
    percentis_mod = np.interp(mod, mod_sorted, prob_mod)
    
    # 4. Encontrar o valor correspondente na distribuição real para esse percentil
    # Inversa: dada uma probabilidade P, qual o valor x na distribuição real?
    mod_corrigido = np.interp(percentis_mod, prob_obs, obs_sorted)
    
    return mod_corrigido"""


class SolarQuantileMapping:
    def __init__(self):
        """
        Inicializa o corretor de viés empírico.
        """
        self.obs_train_sorted = None
        self.mod_train_sorted = None
        self.prob_obs = None
        self.prob_mod = None
        self._is_fitted = False

    #def fit(self, obs_train, mod_train):
    def fit(self, obs_train):
        """
        Aprende as distribuições empíricas usando APENAS os dados de treinamento.
        """
        # Garantir que sejam arrays numpy 1D
        obs_train = np.asarray(obs_train).flatten()
        #mod_train = np.asarray(mod_train).flatten()
        
        # Remove NaNs
        obs_clean = obs_train[~np.isnan(obs_train)]
        #mod_clean = mod_train[~np.isnan(mod_train)]
        
        # 1. Ordenar os dados do TREINO
        self.obs_train_sorted = np.sort(obs_clean)
        #self.mod_train_sorted = np.sort(mod_clean)
        
        # 2. Criar o eixo de probabilidades do TREINO
        self.prob_obs = np.linspace(0, 1, len(self.obs_train_sorted))
        #self.prob_mod = np.linspace(0, 1, len(self.mod_train_sorted))
        
        self._is_fitted = True
        return self

    def transform(self, mod_test, mask_teste=None):
        """
        Aplica o mapeamento aos novos dados de teste.
        Se mask_teste for fornecida (1 para dia, 0 para noite), garante zero à noite.
        """
        if not self._is_fitted:
            raise RuntimeError("Você deve chamar o método .fit() com os dados de treino antes do .transform()")
            
        mod_test_array = np.asarray(mod_test).flatten()
        
        # 3. Qual o percentil da nova predição com base na curva do TREINO?
        # É aqui que evitamos o vazamento de dados!
        #percentis_mod = np.interp(mod_test_array, self.mod_train_sorted, self.prob_mod)
        
        # 4. Qual o valor real correspondente a esse percentil?
        #mod_corrigido = np.interp(percentis_mod, self.prob_obs, self.obs_train_sorted)
        mod_corrigido = np.interp(mod_test_array, self.prob_obs, self.obs_train_sorted)
        # 5. Tratamento de limites (não gerar valor negativo)
        mod_corrigido = np.maximum(mod_corrigido, 0.0)
        
        # 6. Aplicação da máscara física (Dilema da Noite)
        if mask_teste is not None:
            mask_array = np.asarray(mask_teste).flatten()
            mod_corrigido = mod_corrigido * mask_array
            
        # Retorna no mesmo formato (shape) da entrada original
        return mod_corrigido.reshape(np.shape(mod_test))