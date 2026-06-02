# Especificações Técnicas e Stack Tecnológico: Previsão Solar e Irradiância

Este documento apresenta a especificação técnica detalhada de todas as tecnologias, bibliotecas, ferramentas de infraestrutura e padrões de software implementados no repositório `Remodelacao_mestrado`. O objetivo é servir como uma referência técnica exata sobre o ecossistema computacional utilizado durante o período do estágio sanduíche, detalhando versões, propósitos e integrações.

---

## 1. Arquitetura de Execução, Infraestrutura e Containerização

Para garantir a reprodutibilidade científica total dos experimentos e a compatibilidade de hardware com aceleração por placas gráficas dedicadas (GPUs), todo o ecossistema de software está isolado sob um ambiente de contêiner.

### 1.1 O Contêiner de Execução (NVIDIA NGC)
*   **Imagem Base**: `nvcr.io/nvidia/pytorch:25.10-py3`
*   **Propósito**: Esta imagem é mantida e otimizada mensalmente pela NVIDIA. Ela contém uma instalação de alta performance do **PyTorch**, integrada nativamente com:
    *   **NVIDIA CUDA Toolkit**: Driver de aceleração computacional em GPU.
    *   **cuDNN (CUDA Deep Neural Network library)**: Biblioteca de primitivas otimizada para redes neurais profundas (convoluções, recorrências e pooling).
    *   **TensorRT**: Otimizador de inferência para implantação em produção.
*   **Compilador Python**: Python 3.10+ integrado na imagem NGC, com suporte a otimizações de vetorização SIMD de CPU e suporte de drivers de kernel CUDA.

### 1.2 Virtualização e Portabilidade (`Dockerfile`)
O ambiente de desenvolvimento é orquestrado por um `Dockerfile` localizado na raiz do projeto, que estende a imagem base da NVIDIA executando as seguintes ações automáticas:
1.  **Configuração de Logs**: Define a variável de ambiente `ENV TF_CPP_MIN_LOG_LEVEL=2` para silenciar logs verbosos de depuração e focar em avisos críticos.
2.  **Gerenciamento de Dependências**: Copia o arquivo `requirements.txt` da máquina hospedeira para a pasta `/tmp/requirements.txt` do contêiner e executa a instalação via gerenciador de pacotes `pip` atualizado:
    ```bash
    /usr/bin/python3 -m pip install --upgrade pip
    /usr/bin/python3 -m pip install -r /tmp/requirements.txt
    ```
3.  **Portas de Rede**: Expõe a porta de rede TCP `8888` para permitir conexões externas ao servidor de notebooks interativos.
4.  **Entrypoint (Jupyter Notebook)**: Por padrão, o contêiner inicia o Jupyter Notebook no diretório de trabalho `/workspace`, permitindo prototipação rápida e visualização de análises estatísticas:
    ```bash
    CMD ["/usr/bin/python3", "-m", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
    ```

---

## 2. Frameworks de Aprendizado de Máquina e Modelagem Recorrente

A construção, treinamento e inferência dos modelos autoregressivos sequence-to-sequence são estruturadas em camadas baseadas em tensores acelerados por GPU.

### 2.1 PyTorch (Deep Learning Core)
*   **Papel**: Fornece o motor de diferenciação automática (`autograd`) e a abstração de tensores. 
*   **Aplicação**: Toda a rede neural (Encoder base de LSTMs, Decoder com pesos Bahdanau e a camada customizada `FeatureAttention`) é programada estendendo a classe base `torch.nn.Module`.
*   **Diferenciação Customizada**: Utilizado para criar funções de custo diferenciáveis complexas (como a ordenação em lote em `torch.sort` para o cálculo da integral Kolmogorov-Smirnov da perda `CPILoss`).

### 2.2 PyTorch Lightning (Orquestrador de Treino)
*   **Papel**: Separa a lógica puramente científica da rede neural da infraestrutura de loops de treino e validação, garantindo código limpo e modular.
*   **Componentes Utilizados**:
    *   `LightningModule`: Encapsula a arquitetura da rede, o passo de treino (`training_step`), validação (`validation_step`) e a configuração de otimizadores de gradiente.
    *   `LightningDataModule`: Encapsula o pipeline de carregamento de dados, dividindo a base de dados temporal em conjuntos de treino, validação e teste com seus respectivos `DataLoaders`.
    *   `Trainer`: O executor central do loop de treino. Configurado para rodar automaticamente na GPU com precisão mista automática (FP16/AMP) para economizar memória de vídeo.
    *   `Callbacks`:
        *   `EarlyStopping`: Callback configurável que monitora o erro de validação (`val_loss`) e encerra o treinamento prematuramente (ex: se não houver melhora por 10 ou 100 épocas consecutivas) para evitar *overfitting*.
        *   `ModelCheckpoint`: Salva periodicamente o estado dos pesos que obtiveram a menor perda no conjunto de validação (`best_model.pth`).

### 2.3 Optuna (Ajuste Automático de Hiperparâmetros)
*   **Papel**: Framework de otimização de hiperparâmetros de nível de produção.
*   **Aplicação**: Substitui a busca exaustiva em grade (*Grid Search*) por algoritmos inteligentes de otimização bayesiana (como o *Tree-structured Parzen Estimator* - TPE).
*   **Função**: Automatiza a exploração de:
    *   Número de camadas ocultas recorrentes e dimensão oculta do LSTM.
    *   Taxa de aprendizado (*learning rate*) do otimizador Adam.
    *   Fração de regularização de dropout.
    *   Pesos de Lagrange das penalidades geofísicas ($\lambda_{hard}$ e $\lambda_{soft}$).
    *   Callback de poda (*pruning*): Encerra automaticamente sessões de treino cujas curvas de aprendizado iniciais se mostram estatisticamente inferiores à média histórica das rodadas anteriores.

---

## 3. Pré-Processamento Físico-Atmosférico e Engenharia de Atributos

A sanitização e engenharia de variáveis contam com bibliotecas geofísicas altamente especializadas da comunidade de energia solar internacional.

### 3.1 PVLib Python (Cinemática Solar e Céu Claro)
*   **Papel**: Biblioteca de referência mundial para modelagem de sistemas de energia solar fotovoltaica.
*   **Aplicações no Pipeline**:
    *   **Cálculo da Posição Solar**: Estima a órbita do sol a partir de coordenadas geográficas terrestres ($\phi, \lambda, z$) gerando a elevação ($\alpha$), ângulo zenital ($\theta_z$) e azimute ($\gamma_s$).
    *   **Radiação Extraterrestre**: Fornece o valor analítico da irradiância extraterrestre horizontal ($G_{0h}$) ajustada pela excentricidade diária da órbita terrestre.
    *   **Massa de Ar Óptica**: Calcula a massa de ar ótica absoluta ($m_{abs}$) considerando a refração atmosférica local e a razão de pressões barométricas.
    *   **Estimativa de Céu Claro**: Engine matemática auxiliar para o modelo Ineichen e de apoio à formulação ESRA.

### 3.2 Scipy (Otimizações Numéricas Não-Lineares)
*   **Papel**: Ferramenta científica para resolução de sistemas de equações e regressões matemáticas complexas.
*   **Aplicações no Pipeline**:
    *   `scipy.optimize.minimize_scalar`: Utilizado no módulo `ClearSkyEstimator` para rodar o algoritmo inverso de calibração do Fator de Turbidez de Linke ($TL$), minimizando o RMSE em dias ensolarados.
    *   `scipy.optimize.curve_fit`: Executa regressão não-linear por mínimos quadrados nos dados históricos para calibrar os coeficientes de condutividade e convecção térmica ($U_0$ e $U_1$) que determinam a temperatura física do silício ($T_{cell}$) com o vento.

### 3.3 Scikit-Learn (Pipelines Modulares SOLID)
*   **Papel**: Padronização dos fluxos de dados e tratamento de variáveis.
*   **Aplicações no Pipeline**:
    *   `sklearn.pipeline.Pipeline`: Agrupa as classes de transformação física, térmica, BSRN e temporal em uma sequência unificada e robusta de execução.
    *   `BaseEstimator` e `TransformerMixin`: Classes herdadas para todas as etapas customizadas de transformação de dados (`DataSanitizer`, `SolarPositionCalculator`, `ClearSkyEstimator`, `PhysicalFeatureGenerator`, `KastenCorrector`, `QualityControlFilter`, `TemporalFeatureGenerator`), garantindo compatibilidade nativa com os métodos `.fit()`, `.transform()` e `.fit_transform()`.
    *   `MinMaxScaler`: Escalonador estatístico para comprimir os vetores de atributos na escala restrita $[0, 1]$ exigida pela inicialização de redes LSTMs.

---

## 4. Rastreamento Científico, Monitoramento de Carbono e Green AI

Para alinhar o projeto com as melhores práticas globais de reprodutibilidade e responsabilidade ecológica, integraram-se sistemas de auditoria experimental contínuos.

### 4.1 MLflow (Gerenciamento de Experimentos)
*   **Papel**: Plataforma de ciclo de vida completo de Machine Learning.
*   **Aplicações**:
    *   **Rastreamento de Hiperparâmetros**: Registra automaticamente todos os parâmetros lidos do arquivo YAML de configuração de cada treino.
    *   **Métricas de Aprendizado**: Registra curvas de perda de treino/validação por época e métricas estatísticas finais (RMSE, MAE, R², Skill Score) no banco SQLite local `mlflow.db`.
    *   **Registro de Artefatos**: Armazena os modelos treinados ótimos (`best_model.pth`), os arquivos de escalonamento estatístico serializados (`scaler_X.pkl`, `scaler_Y.pkl`) e as curvas de aprendizado geradas em imagem (`learning_curve.png`).
    *   **Reprodutibilidade**: Rastreia a hash do commit Git associado à rodada de treino.

### 4.2 CodeCarbon (Monitoramento Ecológico - Green AI)
*   **Papel**: Biblioteca open-source desenvolvida por institutos científicos para rastrear a pegada ecológica de algoritmos de computação pesada.
*   **Funcionamento**:
    *   Mede o consumo elétrico dos núcleos de processamento (CPU) e das placas de vídeo aceleradoras (GPU) por meio de interfaces de telemetria nativas de hardware (como NVIDIA NVML e arquivos RAPL em CPUs Intel/AMD).
    *   Mapeia a quantidade de energia consumida em quilowatts-hora ($kWh$).
    *   Com base na matriz energética local (fator de emissões nacional de carbono da rede elétrica brasileira), converte o consumo em quilogramas de dióxido de carbono equivalente ($CO_2e$) liberados na atmosfera.
    *   Os logs gerados são persistidos em formato `.csv` e registrados no MLflow.

---

## 5. Visualização Estatística e Inteligência Artificial Explicável (XAI)

### 5.1 Captum (Engine de Explicabilidade Geofísica)
*   **Papel**: Biblioteca oficial do ecossistema PyTorch para interpretabilidade de modelos complexos.
*   **Aplicações**:
    *   **Integrated Gradients (IG)**: Executa a atribuição de importância acumulando gradientes em relação a referências neutras. Utilizado para quantificar o peso temporal de cada hora passada (lags) nos horizontes futuros.
    *   **Feature Ablation**: Remove dinamicamente variáveis do fluxo de inferência para medir o impacto direto no erro final, gerando um ranqueamento de importância geofísica de variáveis.

### 5.2 SkillMetrics (Validação Meteorológica)
*   **Papel**: Biblioteca python especializada na compilação de diagramas de desempenho físico-meteorológico.
*   **Aplicação**: Gerador de diagramas de Taylor customizados que fundem estatisticamente e graficamente a correlação de Pearson ($r$), o desvio padrão normalizado ($\sigma$) e a raiz do desvio quadrático médio centrado ($cRMSD$).

### 5.3 Seaborn e Matplotlib (Motores Gráficos Científicos)
*   **Papel**: Geração de visualizações estáticas de nível acadêmico de alta definição.
*   **Visualizações Geradas**:
    *   Boxplots de distribuição de erros horários.
    *   Perfis de RMSE diurnos (curvas diárias de erro).
    *   Gráficos comparativos de séries temporais para cenários de céu claro e nublado/transiente.

### 5.4 Outras Dependências Científicas
*   **umap-learn**: Redução de dimensionalidade não-linear (UMAP). Usado para projetar o espaço latente de alta dimensão gerado pelas memórias do Encoder em duas dimensões, permitindo visualizar graficamente se o modelo separa representações de céu claro de céus severamente encobertos.
*   **statsmodels**: Biblioteca para modelos estatísticos clássicos. Usada na análise de estacionariedade de séries temporais, cálculo de funções de autocorrelação (ACF/PACF) para determinar o comprimento ótimo da janela de lags ($n_{past} = 24$).
*   **pyyaml**: Carregador de estruturas complexas em arquivos YAML (`configs/ceu_config.yaml` e `configs/potencia_config.yaml`), mantendo a separação entre hiperparâmetros de treino e o código Python.
