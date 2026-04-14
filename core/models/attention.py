import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, enc_directions=1):
        super(Attention, self).__init__()
        # O input da atenção é: [Decoder_State (H)] + [Encoder_Output (H * Directions)]
        combined_dim = hidden_size + (hidden_size * enc_directions)
        
        self.attn = nn.Linear(combined_dim, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, Batch, Hidden]
        # encoder_outputs: [Batch, Seq, Hidden * Directions]
        
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        
        hidden_expanded = hidden.permute(1, 0, 2).expand(batch_size, seq_len, -1)
        
        # Concatena estado do decoder com output (gordo) do encoder
        energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), 2)))
        
        attention = self.v(energy).squeeze(2)
        alpha = F.softmax(attention, dim=1).unsqueeze(1)
        
        context_vector = torch.bmm(alpha, encoder_outputs)
        
        return context_vector, alpha