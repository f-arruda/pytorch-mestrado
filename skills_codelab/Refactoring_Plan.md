# Refactoring Plan (@architect)

## 1. Problema Atual
A classe `SolarPreprocessor` atual possui quase 800 linhas, sendo uma "God Class". Ela agrupa limpeza de dados, física solar, controle de qualidade (QC), persistência de lags, normalização e escalonamento num único bloco (`fit`/`transform`). Além de ferir o *Single Responsibility Principle* (SRP), torna difícil testar e realizar manutenção. Há também duplicação direta desse arquivo entre os domínios `previsao_ceu` e `previsao_potencia`.

## 2. Objetivo da Refatoração
Criar um pipeline scikit-learn (`sklearn.pipeline.Pipeline`) modular, composto por transformadores que realizam uma única etapa lógica, de forma sequencial. Os transformadores devem preservar estritamente toda a matemática e regras de negócio originais.

## 3. Módulos Isolados (Transformers)

1. **`DataCleaningTransformer`**: 
   - Responsabilidade: Renomear colunas baseadas num `column_mapping`, garantir índice temporal (datetime), tratar fuso horário (tz_localize/tz_convert), remover NaTs e remover duplicatas, além de filtrar pelo `start_year`.
2. **`SolarPositionTransformer`**:
   - Responsabilidade: Instanciar `pvlib.location.Location` e calcular ângulos solares (zenith, azimuth, elevation, apparent_zenith), massa de ar (am_abs) e irradiação extra-terrestre.
3. **`ClearSkyTransformer`**:
   - Responsabilidade: Otimizar o Linke Turbidity diariamente e executar os modelos de céu claro (ESRA ou Perez), gerando `ghi_cs`, `dhi_cs`, `dni_cs`. Contém as funções `_is_clear` e `_otimization_LT`.
4. **`PhysicalFeaturesTransformer`**:
   - Responsabilidade: Calcular o clearness index (`kt`), as frações difusas e diretas, o fator `k`, variabilidade temporal (`QS`, `VS_cdfn`), corrigir dhi/dni usando modelo Kasten (opcional) e calcular `temp_cell` e `pot_cs`. Esse módulo também será o responsável por rodar o `_fit_thermal_parameters` no `.fit()`.
5. **`QualityControlTransformer`**:
   - Responsabilidade: Executar os 10 testes lógicos do controle de qualidade da BSRN (limites físicos, testes de consistência) e registrar quais testes causaram reprovação na coluna `qc_reprovals`.
6. **`FeatureEngineeringTransformer`**:
   - Responsabilidade: Criar variáveis de lag (`P1`, `P2`, `P3`), persistência, calcular degradação por ano e outras normalizações simples.
7. **`ScalerTransformer`**:
   - Responsabilidade: Aplicar o `MinMaxScaler` apenas nas colunas definidas (`features_to_scale`), rodando `.fit()` no treino.

## 4. Estrutura de Arquivos
- `refactored_pipeline/transformers.py`: Contém a definição das 7 classes acima que herdam de `BaseEstimator` e `TransformerMixin`.
- `refactored_pipeline/pipeline.py`: Contém a função `build_solar_pipeline()` que instancia e encadeia esses passos.
