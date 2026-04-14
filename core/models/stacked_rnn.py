import torch
import torch.nn as nn

# Mapeamento para selecionar o módulo PyTorch correto
CELL_MAPPING = {
    'lstm': nn.LSTM,
    'gru': nn.GRU
}

class Stacked_RNN(nn.Module):
    """
    Modelo RNN Stacked Modular, suportando diferentes hidden_sizes e
    escolha entre LSTM e GRU.
    """
    def __init__(self, input_size, hidden_sizes: list, output_size, cell_type='lstm',
                 bidirectional = False):
        super(Stacked_RNN, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.cell_type = cell_type.lower()
        self.num_directions = 2 if bidirectional else 1
        # 1. Seleção da Célula
        RNN_Cell = CELL_MAPPING.get(self.cell_type)
        if not RNN_Cell:
            raise ValueError(f"Tipo de célula '{cell_type}' não suportado. Escolha 'lstm' ou 'gru'.")
        
        # O input_size da primeira camada é o 'input_size' do dataset
        current_input_size = input_size
        self.rnn_layers = nn.ModuleList()
        
        # 2. Construção Dinâmica das Camadas
        for hidden_size in hidden_sizes:
            rnn_layer = RNN_Cell( # <-- Usa a célula selecionada (LSTM ou GRU)
                input_size=current_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional
            )
            self.rnn_layers.append(rnn_layer)
            current_input_size = hidden_size*self.num_directions
        
        # 3. Camada Densa Final
        final_hidden_size = hidden_sizes[-1]
        if bidirectional:
            self.fc = nn.Linear(final_hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(final_hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        current_input = x
        
        # 1. Loop pela Pilha de RNNs (Conexão Manual)
        for i, rnn_layer in enumerate(self.rnn_layers):
            
            # 💡 Melhoria Crítica: Inicializa h0/c0 no dispositivo do MÓDULO atual
            device = rnn_layer.weight_ih_l0.device 
            h_size = self.hidden_sizes[i]

            # 2. Inicialização do Estado Oculto
            # O GRU só tem h0 (Hidden State); o LSTM tem h0 e c0 (Cell State).
            h0 = torch.zeros(1*self.num_directions, batch_size, h_size).to(device)
            
            if self.cell_type == 'lstm':
                c0 = torch.zeros(1*self.num_directions, batch_size, h_size).to(device)
                initial_state = (h0, c0) # Para LSTM, passa a tupla (h0, c0)
            else: # GRU
                initial_state = h0 # Para GRU, passa apenas h0
            
            # 3. Passagem pela RNN
            output_i, state_i = rnn_layer(current_input, initial_state)
            
            # A saída da camada atual é a entrada da próxima
            current_input = output_i

        # 4. Saída Final
        last_step_output = current_input[:, -1, :] 
        prediction = self.fc(last_step_output)

        return prediction