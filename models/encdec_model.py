import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from models.decoder_attention import AttentionDecoder

class EncDecModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_seq_len, output_dim=1, 
                 cell_type='lstm', use_attention=False, bidirectional = False,
                 dropout_prob=0):
        super(EncDecModel, self).__init__()

        self.use_attention=use_attention

        self.encoder = Encoder(input_size, hidden_sizes, cell_type,bidirectional,dropout_prob)

        if use_attention:
            self.decoder = AttentionDecoder(hidden_sizes, output_seq_len, output_dim, cell_type,
                                            encoder_bidirectional=bidirectional,
                                            dropout_prob=dropout_prob)
        else:
            self.decoder = Decoder(hidden_sizes, output_seq_len, output_dim, cell_type,
                                   dropout_prob=dropout_prob)

    def forward(self, x):
        # O Encoder retorna (encoder_outputs, encoder_states)
        encoder_outputs, encoder_states = self.encoder(x)
        
        if self.use_attention:
            # Decoder com Atenção precisa dos outputs
            predictions = self.decoder(encoder_states, encoder_outputs)
        else:
            # Decoder sem Atenção precisa apenas dos estados
            predictions = self.decoder(encoder_states)
            
        return predictions