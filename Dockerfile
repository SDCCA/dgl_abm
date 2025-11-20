# Base of Miniconda3
FROM continuumio/miniconda3:latest

# Working directory
WORKDIR /app

# Copy environment information/source code
COPY environment.yaml .
COPY . .

# Create/activate conda environment on opening Bash
RUN conda env create -f environment.yaml -n dgl_abm_cpu \
    && echo "conda activate dgl_abm_cpu" >> ~/.bashrc
SHELL ["bash", "-c"]

# Install package in editable mode
RUN source ~/.bashrc && conda activate dgl_abm_cpu && pip install -e .

# Open shell by default
CMD ["bash"]