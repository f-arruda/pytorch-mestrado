import torch.nn as nn
from models.feature_attention import FeatureAttention

# --- 1. ENCODER ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes: list, cell_type='lstm', bidirectional=False, 
                 dropout_prob=0, use_feature_attention=False):
        super(Encoder, self).__init__()
        
        self.cell_type = cell_type.strip().lower()
        self.rnn_layers = nn.ModuleList()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_feature_attention = use_feature_attention
        self.dropout=nn.Dropout(dropout_prob)

        # --- 1. Inicializa Feature Attention (Novo) ---
        if self.use_feature_attention:
            self.feat_att = FeatureAttention(input_size)
        else:
            self.feat_att = None

        current_input = input_size
        
        for h_size in hidden_sizes:
            if self.cell_type == 'gru':
                layer = nn.GRU(current_input, h_size, batch_first=True, bidirectional=bidirectional)
            else:
                layer = nn.LSTM(current_input, h_size, batch_first=True, bidirectional=bidirectional)
            
            self.rnn_layers.append(layer)
            
            current_input = h_size*self.num_directions

    def forward(self, x):
        """
        x: (Batch, Seq_Len, Features)
        """
        current_input = x

        # --- Passo 1: Aplica Feature Attention (Se ativado) ---
        if self.use_feature_attention and self.feat_att is not None:
            # O XAI capturará os pesos internamente via Hook, 
            # não precisamos retorná-los explicitamente aqui.
            current_input, _ = self.feat_att(current_input)

        all_states = []
        
        for layer in self.rnn_layers:
            output, state = layer(current_input)
            output = self.dropout(output)
            
            # --- FUSÃO DE ESTADOS (CRUCIAL PARA O DECODER SIMPLES) ---
            if self.bidirectional:
                if self.cell_type == 'lstm':
                    h_n, c_n = state
                    # h_n: (Num_Layers*Directions, Batch, Hidden)
                    # Separar e somar as direções para voltar ao tamanho original
                    h_n = h_n.view(1, 2, h_n.size(1), h_n.size(2))
                    h_final = h_n[:, 0, :, :] + h_n[:, 1, :, :] # Soma Forward + Backward
                    
                    c_n = c_n.view(1, 2, c_n.size(1), c_n.size(2))
                    c_final = c_n[:, 0, :, :] + c_n[:, 1, :, :]
                    
                    state_to_save = (h_final, c_final)
                else: # GRU
                    h_n = state.view(1, 2, state.size(1), state.size(2))
                    state_to_save = h_n[:, 0, :, :] + h_n[:, 1, :, :]
            else:
                state_to_save = state

            all_states.append(state_to_save)
            current_input = output
            
        # Retorna TUPLA: (Tensor saída, Lista de estados)
        return current_input, all_states