import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.train import main
from core.preprocessing.pipeline import build_preprocessing_pipeline
from domains.previsao_ceu.dataset import SolarEfficientDataset

import yaml

def load_config():
    with open('configs/ceu_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

CONFIG_CEU = load_config()

if __name__ == "__main__":
    print("Iniciando treinamento de Previsão de Condição de Céu...")
    main(CONFIG_CEU, build_preprocessing_pipeline, SolarEfficientDataset)
