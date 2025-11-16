#!/bin/bash
set -e
echo "Starting Academic Graph Project Setup with Conda (Linux version)..."

echo "Creating directory structure..."
mkdir -p ../data/raw  ../data/processed

echo "Creating conda environment from environment.yml..."
conda create -n 4222_Project python=3.10 -y

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate 4222_Project

# CUDA 11.8
echo "Downloading CUDA..."
conda install cudatoolkit=11.8 -c nvidia -y

echo "Downloading basic data science packages..."
conda install numpy=1.23 pandas=1.5 scipy=1.9 scikit-learn=1.2 -y

echo "Downloading visualization packages.."
conda install matplotlib=3.6 seaborn=0.12 jupyter -y

echo "Downloading data preprocessing packages..."
conda install networkx requests=2.28 beautifulsoup4=4.11 lxml=4.9 tqdm=4.64 pyyaml=6.0 -y

echo "Downloading PyTorch..."
conda install pytorch=2.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "Downloading PyG relative packages..."
pip install torch-geometric==2.5.3 torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

echo "Downloading DGL..."
conda install -c dglteam/label/cu118 dgl -y

echo "Downloading other relative packages..."
pip install py2neo sentence-transformers==2.2.2 transformers==4.25.1 nltk==3.8.1 spacy==3.7.2 huggingface-hub==0.16.4

echo "Downloading remaining tools..."
pip install plotly==5.13.0 pyunpack==0.3.0 patool python-dateutil==2.8.2 chardet==5.1.0 pillow==9.4.0 openpyxl==3.1.0 joblib==1.2.0 wandb==0.15.0 tensorboard==2.12.0

echo "Downloading OpenAI and relative packages..."
conda install openai python-dotenv -y

echo "Downloading NLTK dataset and spaCy model..."
python -c "import nltk; nltk.download(\"punkt\"); nltk.download(\"stopwords\"); nltk.download(\"wordnet\")" || echo "Warning: NLTK download had issues but continuing..."
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

python ../verify_environment.py

echo ""
echo "Setup completed successfully!"
echo "To deactivate the environment: conda deactivate"