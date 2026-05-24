# Repository Analysis & LLM Memory Guide: `Remodelacao_mestrado`

## 1. Visão Geral da Arquitetura
Este repositório contém um pipeline completo para treinamento e avaliação de modelos de machine learning aplicados à previsão de geração de energia solar e irradiância solar. O projeto depende fortemente do PyTorch para a construção dos modelos e utiliza uma mistura de características físicas e estatísticas para as previsões.

**Padrão de Design Principal:** O sistema é dividido em componentes genéricos na pasta `core/` e componentes específicos na pasta `domains/` (`previsao_ceu` e `previsao_potencia`). Houve recentemente uma mudança arquitetural voltada para os **Princípios SOLID**, especificamente modularizando as classes monolíticas de pré-processamento em pipelines compatíveis com o `scikit-learn`.

## 2. Estrutura de Diretórios e Módulos Principais

### `core/`
O núcleo central para a lógica compartilhada entre todos os domínios.
- **`preprocessing/`**: Contém o pipeline de pré-processamento refatorado e compatível com SOLID.
  - `pipeline.py`: Define a função factory `build_preprocessing_pipeline` usando `sklearn.pipeline.Pipeline`.
  - `transformers.py`: Contém as classes transformadoras modulares e isoladas (`DataSanitizer`, `SolarPositionCalculator`, `ClearSkyEstimator`, `PhysicalFeatureGenerator`, etc.).
- **`models/`**: Arquiteturas de redes neurais.
  - `encdec_model.py`: Ponto de entrada principal para a arquitetura encoder-decoder.
  - `encoder.py`, `decoder.py`, `decoder_attention.py`: Subcomponentes do modelo sequence-to-sequence.
  - `attention.py`, `feature_attention.py`: Mecanismos de atenção.
- **`loss_function/`**: Funções de custo customizadas e baseadas em física (`pyloss.py`, `cpiloss.py`, `pyloss_pers.py`).
- **`utils/`**: Utilitários para treinamento e experimentação (`early_stopping.py`, `experiment_manager.py`, `xai.py` para Inteligência Artificial Explicável).
- **`train.py` & `lightning_wrapper.py`**: Loop principal de execução de treinamento e integração com PyTorch Lightning.

### `domains/`
Implementações de domínios específicos que herdam ou encapsulam as funcionalidades do `core`.
- **`previsao_ceu/` (Previsão do Índice de Céu Claro)**:
  - Contém `dataset.py`, `postprocessing.py` e os arquivos mais antigos de `preprocessing.py`.
- **`previsao_potencia/` (Previsão de Potência Solar)**:
  - Já atualizado para usar o novo pipeline de pré-processamento do núcleo. O `preprocessing.py` atua como um wrapper simples: `from core.preprocessing.pipeline import build_preprocessing_pipeline as SolarPreprocessor`.
  - Contém `dataset.py`, `postprocessing.py`.

### `evaluation/`
Scripts para análise pós-treinamento.
- `analysis_power.py`, `main_analysis_sky.py`, `analysis_k_factor.py`: Geração de relatórios, diagramas de Taylor, gráficos de dispersão e análise de séries temporais.
- `metrics/`: Módulos estatísticos para o cálculo de métricas.

### `configs/`
Arquivos YAML armazenando hiperparâmetros, configurações de features e arquiteturas da rede.

### Nível Raiz
- `run_ceu.py` & `run_potencia.py`: Pontos de entrada para iniciar o processo de treinamento/inferência.
- `README.md`: Contém uma boa visão geral das classes, embora algumas documentações de pré-processamento ainda possam refletir o design mais antigo do `SolarPreprocessor` monolítico em vez do novo `core/preprocessing/pipeline.py`.

## 3. Refatorações Recentes e Estado Atual (A "Memória")
- **Refatoração de Pré-Processamento SOLID**: O código base abandonou recentemente os arquivos monolíticos grandes `preprocessing.py` em cada domínio. Em vez disso, um pipeline modular, compatível com o padrão `scikit-learn`, foi construído em `core/preprocessing/`. O domínio `previsao_potencia` já migrou com sucesso para usar esse novo pipeline.
- **Princípios de Design**: O foco agora é aplicar os princípios de "Responsabilidade Única" (Single Responsibility) e "Aberto/Fechado" (Open-Closed). Por exemplo, os cálculos da posição solar e as estimativas de céu claro agora são etapas isoladas e distintas no pipeline, ao invés de métodos acumulados em uma mesma classe.

## 4. Diretrizes para Interações Futuras (Guia LLM)
1. **Utilize o Core Pipeline**: Ao modificar o pré-processamento dos dados, **NÃO** adicione métodos nas classes monolíticas de domínio. Em vez disso, crie ou modifique um transformador em `core/preprocessing/transformers.py` e adicione-o ao `core/preprocessing/pipeline.py`.
2. **Boas Práticas de PyTorch**: Os modelos estão estruturados em `core/models/`. Mantenha a separação de responsabilidades entre modelagem sequencial (LSTMs) e camadas de atenção. Utilize `lightning_wrapper.py` nos loops de treinamento sempre que aplicável.
3. **Loss Functions (Funções de Perda)**: Novas penalidades guiadas por física devem ser adicionadas dentro de `core/loss_function/`.
4. **Executando o Código**: Use `run_potencia.py` ou `run_ceu.py` como entrypoints de execução.
5. **Configurações**: Sempre verifique os arquivos dentro de `configs/` antes de realizar o hardcoding de hiperparâmetros ou nomes de variáveis. O código deve continuar lendo dinamicamente desses `.yaml`.
6. **Documentação**: Tente manter o arquivo `README.md` atualizado na raiz caso ocorram grandes mudanças estruturais (como a futura exclusão de preprocessors defasados no `previsao_ceu`).
