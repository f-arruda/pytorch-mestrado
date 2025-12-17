import torch
import torch.nn as nn

#criar o objeto que será usado para construção do modelo
class LSTM_stacked (nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_stacked, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Contrução da LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True # Formato (Batch size, Sequencia, Feature)
        )

        # Layer Dense do keras
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x_shape = (Batch Size, Seq Len, Input Size)
        batch_size=x.size(0)
        
        #inicialização do estado oculto
        # identifica onde os tensores da lstm estão (cpu ou gpu)
        # para que todas as etapas seguintes do modelo estejam locadas no mesmo lugares
        device= self.lstm.weight_ih_l0.device # dispositivo do tensor (CPU/GPU) 
        h0=torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0=torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # passagem pela lstm
        output, (hn,cn) = self.lstm(x, (h0,c0))

        #saida final
        last_step_output= output[:, -1,:] #shape (batch_size, hidden_size)

        #dense layer e as previsões
        prediction = self.fc(last_step_output) # shape (batch_size, output_size)

        return prediction
