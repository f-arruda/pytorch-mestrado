import torch.nn as nn

# --- 2. DECODER ---
class Decoder(nn.Module):
    def __init__(self, hidden_sizes: list, output_seq_len, output_dim=1, cell_type='lstm', dropout_prob=0):
        super(Decoder, self).__init__()
        self.output_seq_len = output_seq_len
        self.cell_type = cell_type.strip().lower()
        self.rnn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob)

        current_input = hidden_sizes[-1]
        
        for h_size in hidden_sizes:
            if self.cell_type == 'gru':
                self.rnn_layers.append(nn.GRU(current_input, h_size, batch_first=True))
            else:
                self.rnn_layers.append(nn.LSTM(current_input, h_size, batch_first=True))
            current_input = h_size
            
        self.fc = nn.Linear(current_input, output_dim)

    def forward(self, encoder_states):
        # Pega o estado da última camada para iniciar
        last_state = encoder_states[-1]
        
        # Verifica se é Tupla (LSTM) ou Tensor (GRU)
        if isinstance(last_state, tuple):
            context_vector = last_state[0].squeeze(0) # Pega h_n
        else:
            context_vector = last_state.squeeze(0) # Pega h_n
            
        # Repeat Vector
        decoder_input = context_vector.unsqueeze(1).repeat(1, self.output_seq_len, 1)
        
        current_input = self.dropout(decoder_input)
        
        for i, layer in enumerate(self.rnn_layers):
            state = encoder_states[i]
            output, _ = layer(current_input, state)
            output = self.dropout(output)
            current_input = output
            
        prediction = self.fc(current_input)
        # Retorna TENSOR
        return prediction
