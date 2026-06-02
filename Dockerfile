# Dockerfile

# ==============================================================================
# 1. ESTÁGIO BASE: Configuração de GPU e Instalação de Dependências Comuns
# ==============================================================================
FROM nvcr.io/nvidia/pytorch:25.10-py3 AS base

# Configurações de ambiente para Python, TensorFlow e CUDA
ENV TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instala ferramentas básicas de sistema adicionais
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia a lista de dependências para instalação
COPY requirements.txt /tmp/requirements.txt

# Instala as dependências usando Pip Cache Mount do BuildKit para acelerar compilações
RUN --mount=type=cache,target=/root/.cache/pip \
    /usr/bin/python3 -m pip install --upgrade pip && \
    /usr/bin/python3 -m pip install -r /tmp/requirements.txt

WORKDIR /workspace

# ==============================================================================
# 2. ESTÁGIO DE PRODUÇÃO / EXECUÇÃO: Imagem enxuta para rodar treinamentos em lote
# ==============================================================================
FROM base AS runner

# Copia o código-fonte necessário para execução em produção/lote
COPY core/ /workspace/core/
COPY domains/ /workspace/domains/
COPY configs/ /workspace/configs/
COPY evaluation/ /workspace/evaluation/
COPY run_ceu.py run_potencia.py /workspace/

# Executa por padrão o pipeline de céu claro (pode ser sobrescrito no comando de execução)
CMD ["python3", "run_ceu.py"]

# ==============================================================================
# 3. ESTÁGIO DE DESENVOLVIMENTO: Jupyter Notebook e ambiente interativo
# ==============================================================================
FROM base AS dev

# Expor a porta padrão do Jupyter
EXPOSE 8888

# Inicia o Jupyter Notebook por padrão no diretório de trabalho /workspace
CMD ["/usr/bin/python3", "-m", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
