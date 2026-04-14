import os
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime

class ExperimentManager:
    def __init__(self, base_dir='experiments', model_name='Model', config=None):
        """
        Cria automaticamente a pasta do experimento.
        Args:
            base_dir: Pasta raiz (ex: 'experiments')
            model_name: Nome base (ex: 'GRU_Bidirectional')
            config: Dicionário com os hiperparâmetros (para salvar no JSON)
        """
        # Gera timestamp único: Ex: 2023-10-27_15-30-00
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Nome da pasta: Ex: experiments/2023-10-27_15-30_GRU_Bidirectional
        self.run_name = f"{timestamp}_{model_name}"
        self.exp_dir = os.path.join(base_dir, self.run_name)
        
        # Cria a pasta
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f"📁 Experimento iniciado em: {self.exp_dir}")
        
        # Salva a configuração (CRUCIAL para reproduzir depois)
        if config:
            self.save_config(config)

    def save_config(self, config):
        """Salva os hiperparâmetros em um arquivo JSON legível."""
        # Filtra objetos que não são serializáveis em JSON (como torch.device)
        config_to_save = {k: str(v) for k, v in config.items()}
        
        with open(os.path.join(self.exp_dir, 'config.json'), 'w') as f:
            json.dump(config_to_save, f, indent=4)

    def save_model(self, model, filename='best_model.pt'):
        """Salva os pesos do modelo (.pt)."""
        save_path = os.path.join(self.exp_dir, filename)
        torch.save(model.state_dict(), save_path)
        # print(f"💾 Modelo salvo: {save_path}")

    def save_plot(self, history, filename='learning_curve.png'):
        """Gera e salva o gráfico de Loss."""
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Treino Loss')
        plt.plot(history['val_loss'], label='Validação Loss')
        plt.title(f'Curva de Aprendizado - {self.run_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.exp_dir, filename)
        plt.savefig(save_path)
        plt.close() # Fecha para liberar memória
        print(f"📈 Gráfico salvo em: {save_path}")

    def save_metrics(self, metrics_dict, filename='metrics.json'):
        """Salva métricas finais em texto."""
        config_to_save = {k: str(v) for k, v in metrics_dict.items()}
        with open(os.path.join(self.exp_dir, filename), 'w') as f:
            json.dump(config_to_save, f, indent=4)