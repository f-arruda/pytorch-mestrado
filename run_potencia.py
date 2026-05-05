import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.train import main
from domains.previsao_ceu.refactored_module import build_preprocessing_pipeline
from domains.previsao_potencia.dataset import SolarEfficientDataset

import yaml

def load_config():
    with open('configs/potencia_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG_POTENCIA = load_config()

if __name__ == "__main__":
    print("Iniciando treinamento de Previsão de Potência...")
    main(CONFIG_POTENCIA, build_preprocessing_pipeline, SolarEfficientDataset)
