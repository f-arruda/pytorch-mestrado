import torch
import torch.nn as nn

class LSTM_stacked(nn.Module):
    """
    Modelo LSTM com suporte a:
    1. Quantidade Ilimitada de Camadas (baseado no tamanho da lista hidden_sizes).
    2. Hidden Sizes Diferentes para cada camada.
    """
    def __init__(self, input_size, hidden_sizes: list, output_size, bidirectional=False):
        super(LSTM_stacked, self).__init__()
        
        # Quantidade de unidades lstm em cada layer oculto
        self.hidden_sizes = hidden_sizes
        # Quantidade de layers ocultos
        self.num_layers = len(hidden_sizes)
        
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1
        # O input_size da primeira camada é o 'input_size' do dataset
        current_input_size = input_size
        
        # Usamos nn.ModuleList para armazenar as camadas LSTM
        # nn.ModuleList garante que o PyTorch registre todas as camadas
        self.lstm_layers = nn.ModuleList()
        
        # 1. Construção Dinâmica das Camadas
        for i, hidden_size in enumerate(hidden_sizes):
            lstm_layer = nn.LSTM(
                input_size=current_input_size,
                hidden_size=hidden_size,
                num_layers=1,        # Sempre 1, pois estamos empilhando manualmente
                batch_first=True,
                bidirectional=self.bidirectional
            )
            self.lstm_layers.append(lstm_layer)
            
            # A entrada da PRÓXIMA camada é a saída (hidden_size) da camada ATUAL
            current_input_size = hidden_size*num_directions
        
        # 2. Camada Densa Final
        # Recebe a saída da ÚLTIMA camada da pilha (hidden_sizes[-1])
        final_hidden_size = hidden_sizes[-1]
        self.fc = nn.Linear(final_hidden_size, output_size)

    def forward(self, x):
        # x_shape = (Batch Size, Seq Len, Input Size)
        batch_size = x.size(0)
        current_input = x # O input da primeira camada é o x
        
        # 1. Loop pela Pilha de LSTMs (Conexão Manual)
        for i, lstm_layer in enumerate(self.lstm_layers):
            
            # A dimensão do estado oculto deve ser (1, Batch Size, Hidden Size da Camada i)
            # 💡 Melhoria Crítica: Inicializa h0/c0 no dispositivo do MÓDULO atual
            device = lstm_layer.weight_ih_l0.device 
            h_size = self.hidden_sizes[i]

            h0 = torch.zeros(1, batch_size, h_size).to(device)
            c0 = torch.zeros(1, batch_size, h_size).to(device)
            
            # Passagem pela LSTM:
            # output_i: (Batch, Seq_Len, Hidden_Size) -> Usado como entrada para a próxima camada
            # hn_i, cn_i: Ignorados, a menos que você queira o estado final de TODAS as camadas
            output_i, _ = lstm_layer(current_input, (h0, c0))
            
            # A saída da camada atual é a entrada da próxima
            current_input = output_i

        # 2. Saída Final: Usa a saída da ÚLTIMA LSTM (current_input)
        
        # last_step_output: Saída do último passo de tempo
        last_step_output = current_input[:, -1, :] # shape (batch_size, final_hidden_size)

        # 3. Densa Layer
        prediction = self.fc(last_step_output) # shape (batch_size, output_size)

        return prediction