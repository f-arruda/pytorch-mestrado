# Solar Forecasting Project Documentation

Este repositório contém uma infraestrutura completa para o treinamento, validação e avaliação de modelos de aprendizado de máquina aplicados à previsão de geração de energia solar e irradiância. 

A arquitetura do projeto separa o pré-processamento físico rigoroso dos dados (utilizando modelos de céu claro e correções térmicas), a criação de datasets customizados para PyTorch e a montagem flexível de diferentes algoritmos preditivos (como Encoder-Decoder com mecanismos de atenção).

---

## 📂 Estrutura Principal do Repositório

O repositório é composto pelas seguintes pastas e arquivos descritos no fluxograma de uso geral:

- **`run_ceu.py` / `run_potencia.py`** : Scripts principais de execução. Carregam as configurações (YAML), instanciam classes de domínio e invocam o treinamento principal.
- **`core/`** : Lógica principal compartilhada, incluindo `train.py` (execução principal), `models/` (implementações dos módulos neurais agregados na classe `EncDecModel`), `utils/` (ferramentas acessórias como `EarlyStopping`), e `loss_function/` (funções de custo customizadas guiadas por física).
- **`domains/`** : Contém domínios específicos (ex: `previsao_ceu`, `previsao_potencia`), cada um com seu respectivo `preprocessing.py` (processamento de dados metereológicos) e `dataset.py` (carregamento de dados em tensores PyTorch).
- **`evaluation/`** : Lógica de avaliação e métricas, como a obtenção de métricas estatísticas em `evaluation/metrics/statistical_metrics.py`.
- **`configs/`** : Arquivos de configuração em YAML que ditam as hiperparametrizações, features mapeadas, configurações de pré-processamento e rede.

---

## 🛠️ Detalhamento de Classes e Funções

Abaixo segue um levantamento das classes mais importantes desenvolvidas na base de código, acompanhados de seus respectivos atributos/argumentos necessários para instanciação ou uso.

### 1. `SolarPreprocessor` (em `domains/<dominio>/preprocessing.py`)

Esta é a classe fundamental para limpar e tratar os dados crus, processar e calcular atributos físicos baseados em posição solar, modelos de céu claro e normalização de features. Esta classe estende `BaseEstimator` e `TransformerMixin` facilitando usabilidade igual ao `scikit-learn`.

| Atributo / Parâmetro | Tipo / Descrição |
| :--- | :--- |
| `latitude` | `float` - Latitude da localização dos painéis solares. |
| `longitude` | `float` - Longitude da localização dos painéis solares. |
| `altitude` | `float` (default=0) - Altitude da localização. |
| `timezone` | `str` (default='UTC') - Fuso horário do conjunto de dados (ex: 'Etc/GMT+3'). |
| `nominal_power` | `float` (default=156.0) - Potência nominal em kW do sistema. |
| `start_year` | `int` (default=2018) - Ano inicial para considerar a degradação temporal do equipamento. |
| `cs_model` | `str` (default='esra') - Modelo adotado para o cálculo do clear_sky (ex: 'esra', 'perez'). |
| `degradation_rate`| `float` (default=0.05) - Taxa de degradação anual estimada. |
| `column_mapping` | `Dict[str, str]` - Dicionário descrevendo a variação entre colunas originais do CSV e nomenclaturas internas requeridas para o Pipeline. |
| `features_to_scale` | `List[str]` - Lista de variáveis metereológicas/features que deverão ser mapeadas via `MinMaxScaler`. |
| `target_col` | `str` (default='power') - Descrição se o target interno visa 'power' (potência) ou 'sky' (índice kt). |
| `kasten_corr` | `bool` (default=False) - Habilita a correção de irradiação difusa com anel de sombreamento (Kasten). |

**Funções principais:** `fit(X, y)` (ajuste e otimização dos parâmetros de U0/U1 caso solicitado), `transform(X)` (cálculo iterativo do zenit, azimuth, ESRA turbidez, K factors, limites físicos etc.).


### 2. `SolarEfficientDataset` (em `domains/<dominio>/dataset.py`)

Dataset do PyTorch com responsabilidade de alinhar janelas passadas e futuras, remover horários desnecessários (noites absolutas baseadas em mask ou cloud_enhancement), tratar NaNs e transformar dados tabulares numéricos em tensores.

| Atributo / Parâmetro | Tipo / Descrição |
| :--- | :--- |
| `df` | `pd.DataFrame` - DataFrame já preprocessado possuindo índices de DateTime. |
| `feature_cols` | `list` - Nomes em formato `str` contendo todas colunas X que serão inseridas no período passado do modelo. |
| `target_col` | `list` - Nomes listando a(s) coluna(s) Y que se deseja predizer no período futuro. |
| `aux_col` | `list` - Features auxiliares adicionais provenientes de leis de modelos matemáticos empíricos atrelados ao futuro. |
| `n_past` | `int` - O comprimento do lag histórico do tamanho da sequência em unidades de tempo. |
| `n_future` | `int` - A quantidade progressiva de steps para predição da rede em unidades temporais. |
| `cloud_enhancement`| `bool` (default=False) - Flag para filtragem opcional extra se o que for prever é corrompido, e para ativar a coluna `cloud_enh`. |
| `group_col` | `str` (default=None) - Opcional. Representa os tensores agrupadores, comumente retornados no método `__getitem__`. |


### 3. `EncDecModel` (em `core/models/encdec_model.py`)

Construtor macro da Rede Neural, une blocos arquiteturais do Encoder e do Decoder, com possíveis injeções de mecanismos de atenção temporal e atenção em features.

| Atributo / Parâmetro | Tipo / Descrição |
| :--- | :--- |
| `input_size` | `int` - O número de características dimensionais de X (corresponde a `len(feature_cols)`). |
| `hidden_sizes`| `list` - Array indicando o tamanho da Hidden Layer por camada do modelo (ex: `[300]`). |
| `output_seq_len`| `int` - Tamanho da sequência das features geradas nas saídas. |
| `output_dim` | `int` (default=1) - Tamanho das dimensões em cada saída (corresponde a `len(target_col)`). |
| `cell_type` | `str` (default='lstm') - Célula para memória recorrente. (ex: 'lstm', 'gru' ou 'rnn'). |
| `use_attention` | `bool` (default=False) - Aciona a classe `AttentionDecoder` de `models.decoder_attention` ao invés de  `Decoder` baseado nos context vectors puros. |
| `bidirectional` | `bool` (default=False) - Força o encoder passar nas duas direções do vetor de tempo. |
| `dropout_prob` | `float` (default=0) - Regularização entre as células. |
| `use_feature_attention`| `bool` (default=False) - Modifica o encoder para empregar análise focada em prever pesos nas colunas específicas no instante passado. |


### 4. `EarlyStopping` (em `core/utils/early_stopping.py`)

Utilitário essencial para impedir o sobreajuste (overfitting) travando o modelo no melhor estado obtido no decorrer das `epochs`.

| Atributo / Parâmetro | Tipo / Descrição |
| :--- | :--- |
| `patience` | `int` (default=7) - Número delimitador de epochs corridas sem melhoramento visível no *validation loss* (loss mínima) antes de interrupção forçada. |
| `verbose` | `bool` (default=False) - Autoriza visualização do rastreamento a cada epoch do counter em `sys.stdout`. |
| `delta` | `float` (default=0) - Taxa de mitigação exigida. Diferenças entre a nova *validation loss* calculada em relação a última `best_score` que sejam menores que `delta` não registram "melhoramento verdadeiro". |
| `path` | `str` (default='checkpoint.pt') - Caminhos absolutos/relativos onde armazeamos fisicamente local com salvamento dos pesos do melhor *state_dict* do PyTorch. |


### 5. `SolarStatisticalAnalyzer` (em `evaluation/metrics/statistical_metrics.py`)

Automatiza o plotting comparativo (Diagrama de Taylor, Scatter Plots e Histogramas, perfis diurnos de medições de RMS/MAE) em cima das inferências resultadas do modelo vs Observação + Baseline de persistência clássica de meteorologia.

| Atributo / Parâmetro | Tipo / Descrição |
| :--- | :--- |
| `df_combined` | `pd.DataFrame` - Requer contendo ao decorrer de si uma coluna `Timestamp`, uma coluna `Modelo`, a coluna do medidor `Observado` e por fim `Previsto`. Para skill scores requer também modelo chamado `Persistencia`. |
| `output_dir` | `str` - Caminho base do repositório/pasta onde gráficos serão postos em exportação nativa `.png` (gera e varre caminhos inexistentes). |

**Funções de apoio em uso:** `save_global_metrics()` salva em CSV. `plot_boxplots_hourly()` cria variação ao redor das horas. `plot_taylor_diagram()` mede covariância usando pacote modular `skill_metrics`.

## Estratégias e Scripts

- A arquitetura utiliza arquivos de configuração YAML unificados dentro da pasta `configs/` (ex: `ceu_config.yaml` e `potencia_config.yaml`). Os scripts de entrada (`run_ceu.py` e `run_potencia.py`) carregam essas configurações e as repassam para a função principal de treinamento (`core/train.py`). Isso orienta o `preprocessor` de *from* -> *to* nos nomes na base de dados sem chumbá-las no script em si e permite definir facilmente os hiperparâmetros e os targets `kt` em céu vs potência elétrica direta.
