import torch
import torch.nn as nn

class MIMOPolynomialModel(nn.Module):
    def __init__(self, n_u, n_y, n_a, n_b, n_c=0, n_future = 1, hidden_dim=None, bias=True):
        """
        n_u: Número de entradas (MIMO)
        n_y: Número de saídas (MIMO)
        n_a: Ordem dos lags de saída (y)
        n_b: Ordem dos lags de entrada (u)
        n_c: Ordem dos lags de erro (e) - Para ARMAX/ARMA
        n_future: Quantos passos à frente o modelo deve prever
        hidden_dim: Se definido, transforma o modelo em não-linear (NARX)
        """
        super().__init__()
        self.n_a, self.n_b, self.n_c = n_a, n_b, n_c
        self.n_future = n_future
        self.n_y = n_y
        
        # Cálculo da dimensão de entrada total do regressor
        # Entradas u: (n_b + 1) pois inclui o tempo atual u(k)
        self.input_dim = (n_y * n_a) + (n_u * (n_b)) + (n_y * n_c)

        self.output_dim = n_y * n_future
        
        if hidden_dim is None:
            # Estrutura Linear Clássica
            self.regressor = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        else:
            # Estrutura Não-Linear (NARX/NARMAX)
            self.regressor = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.output_dim)
            )

    def forward(self, u_hist, y_hist, e_hist=None):
        """
        u_hist: [batch, n_u * (n_b + 1)]
        y_hist: [batch, n_y * n_a]
        e_hist: [batch, n_y * n_c] (opcional)
        """
        inputs = []
        if self.n_a > 0:inputs.append(y_hist.reshape(y_hist.size(0), -1))
        if self.n_b >= 0:inputs.append(u_hist.reshape(u_hist.size(0), -1))
        if self.n_c > 0 and e_hist is not None:inputs.append(e_hist.reshape(e_hist.size(0), -1))
        
        # Concatena todos os regressores em um único vetor para a rede/linear
        x = torch.cat(inputs, dim=1)
        out = self.regressor(x)
        return out.view(-1, self.n_future, self.n_y)    

# --- Factory Function para Seleção Rápida ---

def create_mimo_model(model_type, n_u, n_y, n_future=1, order=2, hidden_dim=None):
    """
    Facilitador para criar arquiteturas específicas.
    """
    if model_type.upper() == 'ARX':
        # AutoRegressive with Exogenous: y + u
        return MIMOPolynomialModel(n_u, n_y, n_a=order, n_b=order, n_future=n_future, n_c=0, hidden_dim=hidden_dim)
    
    elif model_type.upper() == 'ARMAX':
        # ARX + Moving Average: y + u + e
        return MIMOPolynomialModel(n_u, n_y, n_a=order, n_b=order, n_future=n_future, n_c=order, hidden_dim=hidden_dim)
    
    elif model_type.upper() == 'AR':
        # AutoRegressive puro: apenas o passado de y
        return MIMOPolynomialModel(n_u, n_y, n_a=order, n_b=-1, n_future=n_future, n_c=0, hidden_dim=hidden_dim)
    
    elif model_type.upper() == 'ARMA':
        # Apenas y + erro: y + e
        return MIMOPolynomialModel(n_u, n_y, n_a=order, n_b=-1, n_future=n_future, n_c=order, hidden_dim=hidden_dim)
    
    elif model_type.upper() == 'FIR':
        # Finite Impulse Response: apenas u
        return MIMOPolynomialModel(n_u, n_y, n_a=0, n_b=order, n_future=n_future, n_c=0, hidden_dim=hidden_dim)
    
    else:
        raise ValueError("Tipo de modelo não reconhecido.")