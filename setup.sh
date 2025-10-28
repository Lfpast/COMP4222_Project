#!/bin/bash
echo "Starting Academic Graph Project Setup with Conda..."


# 创建目录结构
echo "Creating directory structure..."
mkdir -p data/raw data/processed
# 创建conda环境
echo "Creating conda environment from environment.yml..."
conda create -n 4222_Project python=3.10 -y

# 激活环境
echo "Activating conda environment..."
conda activate 4222_Project

# 下载各种安装包
echo "Downloading CUDA..."
conda install cudatoolkit=11.8 -c nvidia -y

echo "Downloading basic data science packages..."
conda install numpy=1.23 pandas=1.5 scipy=1.9 scikit-learn=1.2 -y

echo "Downloading visualization packages.."
conda install matplotlib=3.6 seaborn=0.12 jupyter -y

echo "Downloading data preprocessing packages..."
conda install networkx requests=2.28 beautifulsoup4=4.11 lxml=4.9 tqdm=4.64 pyyaml=6.0 -y

echo "Downloading PyTorch..."
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "Downloading PyG relative packages..."
conda install pyg=2.3.0 -c pyg -y
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

echo "Downloading DGL..."
conda install dgl=1.1.0 -c dglteam -y

echo "Downloading other relative packages..."
pip install py2neo sentence-transformers==2.2.2 transformers==4.25.1 nltk==3.8.1 spacy==3.7.2 huggingface-hub==0.16.4

echo "Downloading remaining tools..."
pip install plotly==5.13.0 pyunpack==0.3.0 patool python-dateutil==2.8.2 chardet==5.1.0 pillow==9.4.0 openpyxl==3.1.0 joblib==1.2.0 wandb==0.15.0 tensorboard==2.12.0


echo "Downloading NLTK dataset and spaCy model..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
conda install -c conda-forge spacy-model-en_core_web_sm

# 进行验证
python verify_environment.py

# 下载Aminer数据集
echo "Downloading Aminer dataset..."
python download_data.py

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate academic_graph_project"
echo "2. Start Neo4j database (if using Neo4j)"
echo "3. Run: python scripts/data_collection.py"
echo "4. Run: python scripts/neo4j_import.py"
echo "5. Run: python scripts/han_model.py"
echo ""
echo "To deactivate the environment: conda deactivate"