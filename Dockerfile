# Dockerfile

# 1. IMAGEM BASE (NGC OTIMIZADA)
# Usa a imagem NGC que já contém TensorFlow 2.17.0, CUDA e cuDNN configurados.
FROM nvcr.io/nvidia/pytorch:25.10-py3

# 2. DEFINIÇÃO DE AMBIENTE
# Reduz mensagens de log verbosas do TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=2

# 3. INSTALAÇÃO DE DEPENDÊNCIAS FALTANTES
# Copia o requirements.txt para o container
COPY requirements.txt /tmp/requirements.txt

# Usa o PIP da imagem NGC para instalar o que falta
# Garante a versão mais recente do pip e instala as dependências
RUN /usr/bin/python3 -m pip install --upgrade pip && \
    /usr/bin/python3 -m pip install -r /tmp/requirements.txt

# 4. CONFIGURAÇÃO FINAL
# Expor a porta padrão do Jupyter
EXPOSE 8888

# 5. COMANDO DE EXECUÇÃO PADRÃO
# Inicia o Jupyter Notebook por padrão no diretório de trabalho /workspace
CMD ["/usr/bin/python3", "-m", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
