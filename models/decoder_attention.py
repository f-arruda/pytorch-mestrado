import torch
import torch.nn as nn
from .attention import Attention

class AttentionDecoder(nn.Module):
    def __init__(self, hidden_sizes: list, output_seq_len, output_dim=1, cell_type='lstm', encoder_bidirectional=False, dropout_prob=0):
        super(AttentionDecoder, self).__init__()
        self.output_seq_len = output_seq_len
        self.cell_type = cell_type.strip().lower()
        self.hidden_sizes = hidden_sizes
        self.rnn_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob)

        # Atenção precisa saber o tamanho da última camada do encoder/decoder
        last_hidden_size = hidden_sizes[-1]
        enc_directions = 2 if encoder_bidirectional else 1

        self.attention = Attention(last_hidden_size, enc_directions)
        
        context_size = last_hidden_size*enc_directions
   
        current_input_size = last_hidden_size + context_size # Input + Context
        
        for h_size in hidden_sizes:
            if self.cell_type == 'gru':
                self.rnn_layers.append(nn.GRU(current_input_size, h_size, batch_first=True))
            else:
                self.rnn_layers.append(nn.LSTM(current_input_size, h_size, batch_first=True))
            
            # Apenas a primeira camada recebe a concatenação com atenção. 
            # As próximas recebem apenas o output da camada anterior.
            current_input_size = h_size 
            
        self.fc = nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, encoder_states, encoder_outputs):
        # encoder_states: Lista de estados iniciais
        # encoder_outputs: [Batch, Seq_Len, Hidden] (Necessário para atenção)
        
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        decoder_input = torch.zeros(batch_size, 1, self.hidden_sizes[-1]).to(device)
        outputs = []
        
 
        current_states = [s for s in encoder_states] 
        
        for t in range(self.output_seq_len):
            last_layer_state = current_states[-1]
            if isinstance(last_layer_state, tuple):
                h_last = last_layer_state[0]
            else:
                h_last = last_layer_state
                
            context, _ = self.attention(h_last, encoder_outputs)
            decoder_input = self.dropout(decoder_input)
            rnn_input = torch.cat((decoder_input, context), dim=2)
            
            for i, layer in enumerate(self.rnn_layers):
                if i > 0: rnn_input = decoder_input
                out, new_state = layer(rnn_input, current_states[i])
                out = self.dropout(out)
                current_states[i] = new_state
                decoder_input = out 
            
            step_output = self.fc(decoder_input)
            outputs.append(step_output)
            
        return torch.cat(outputs, dim=1)