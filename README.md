# COMP 4222 Group Project

## 📋 Project Overview
Academic Citation Network Analysis using Heterogeneous Attention Networks (HAN)

This project builds a complete pipeline for:
- Data collection and preprocessing from Aminer citation network
- Graph database construction with Neo4j
- HAN model training for node embeddings
- Embedding quality evaluation and hyperparameter tuning

## 🚀 Quick Start

### 1. Environment Setup
```bash
bash setup.sh
```

This will:
- Create conda environment `4222_Project`
- Install all required dependencies (PyTorch, DGL, Neo4j drivers, etc.)
- Verify the environment

### 2. Install Neo4j Database
Download and install Neo4j Desktop from [Neo4j Official Website](https://neo4j.com/download/)

**Neo4j Configuration:**
- Create a new database
- Set username: `neo4j`
- Set password: `Jackson050609` (or update in scripts)
- Start the database service

### 3. Data Pipeline

#### Option A: One-Click Execution (Recommended)
```bash
bash data.sh
```

This automated script:
- Downloads Aminer citation dataset
- Processes raw data (papers, authors, keywords, citations)
- Generates CSV files ready for Neo4j import

#### Option B: Step-by-Step Execution
```bash
# Step 1: Download dataset
python download_data.py

# Step 2: Process data
python process_data.py
```

### 4. Import Data to Neo4j
```bash
python neo4j_import.py
```

This will:
- Create graph schema (constraints, indexes)
- Import ~630K papers, ~100K authors, ~57K keywords
- Build ~4.8M relationships (citations, authorship, keywords)

### 5. Train HAN Model

Navigate to `training/` directory:
```bash
cd training

# Windows
run.bat trial1

# Linux/Mac
bash run.sh trial1
```

This will:
- Train HAN model on citation network
- Generate node embeddings
- Evaluate embedding quality
- Provide hyperparameter tuning recommendations

See [`training/README.md`](training/README.md) for detailed training instructions.

## 📁 Project Structure

```
COMP_4222_Project/
├── README.md                 # This file
├── setup.sh                  # Environment setup script
├── verify_environment.py     # Environment verification
├── environment.yml           # Conda environment config
├── requirements.txt          # Python dependencies│
├── download_data.py          # Data download script
├── process_data.py           # Data processing script
├── data.sh                   # Automated data pipeline
├── neo4j_import.py           # Neo4j batch import script
├── check_neo4j_status.py     # Neo4j connection checker
└── training/                 # HAN model training
    ├── README.md             # Training documentation
    ├── han_model.py          # HAN model implementation
    ├── validate_embeddings.py # Embedding evaluation
    └── run.sh                # Linux/Mac automation
```

## 🛠️ Main Components

### Data Processing Pipeline

1. **`download_data.py`** - Download Aminer dataset
   - Automatic download with fallback URLs
   - ZIP/GZIP extraction
   - Sample data generation if download fails

2. **`process_data.py`** - Process raw data
   - Parse Aminer text format
   - Extract keywords from paper titles
   - Generate 6 CSV files for Neo4j import

3. **`data.sh`** - One-click automation
   - Runs complete download → process pipeline
   - Error detection and reporting

### Graph Database

4. **`neo4j_import.py`** - Import to Neo4j
   - Batch import optimization (5K nodes, 10K edges per batch)
   - Schema creation (constraints, indexes)
   - ~10-15 minutes for full dataset

5. **`check_neo4j_status.py`** - Verify Neo4j connection

### HAN Model Training

6. **`training/han_model.py`** - HAN model training
   - Heterogeneous graph neural network
   - Multi-head attention mechanism
   - Unsupervised embedding learning

7. **`training/validate_embeddings.py`** - Evaluation
   - Clustering quality (Silhouette, Calinski-Harabasz, Davies-Bouldin)
   - Similarity metrics
   - Loss trend analysis
   - Automated hyperparameter tuning suggestions

8. **`training/run.bat` / `run.sh`** - Automation
   - Configurable hyperparameters
   - Training + evaluation pipeline
   - Multi-experiment management

## ⚙️ Requirements

### System Requirements
- Python 3.10+
- 8GB+ RAM (16GB recommended for full dataset)
- ~5GB disk space for data and models

### Key Dependencies
- **PyTorch** 2.0+ (CPU version)
- **DGL** 1.0+ (Deep Graph Library)
- **Neo4j** Desktop 2.x
- **sentence-transformers** (all-MiniLM-L6-v2)
- **py2neo** (Neo4j Python driver)
- **pandas**, **numpy**, **scikit-learn**

See `requirements.txt` and `environment.yml` for complete list.

## 📊 Dataset Information

**Source:** Aminer Citation Network (ACM subset)
- **Papers:** 629,814 (1987-2009)
- **Authors:** 99,749
- **Keywords:** 56,580
- **Citations:** ~4.8M relationships
- **Format:** Custom text format with markers (#*, #@, #t, #c, #index, #%)

## 🎯 Workflow Summary

```
1. Setup Environment       → bash setup.sh
2. Verify Setup           → python verify_environment.py
3. Download & Process     → bash data.sh
4. Start Neo4j            → Neo4j Desktop
5. Import Data            → python neo4j_import.py
6. Check Import           → python check_neo4j_status.py
7. Train Model            → cd training && run.bat trial1
8. Review Results         → Check evaluation_report.json
9. Tune & Retrain         → Adjust hyperparameters → run.bat trial2
```

## 📈 Model Training Details

**HAN Architecture:**
- Input: 384-dim sentence embeddings (Sentence-BERT)
- Hidden: 256-dim (configurable)
- Output: 128-dim node embeddings (configurable)
- Attention: 4-head GAT (configurable)

**Training:**
- Unsupervised learning with L2 regularization
- Adam optimizer
- Default: 10K papers, 100 epochs, lr=0.001
- Training time: ~15-30 minutes (CPU)

**Evaluation Metrics:**
- Clustering: Silhouette Score, Calinski-Harabasz, Davies-Bouldin
- Similarity: Cosine similarity (Top-1, Top-5)
- Convergence: Loss variance, improvement rate

See [`training/README.md`](training/README.md) for hyperparameter tuning guide.

## 🎓 Downstream Applications

Trained embeddings can be used for:
- **Paper Classification** - Categorize papers by research area
- **Citation Prediction** - Recommend papers to cite
- **Author Collaboration** - Find potential collaborators
- **Topic Clustering** - Discover research trends
- **Similarity Search** - Find related papers

**Load Embeddings:**
```python
import torch
checkpoint = torch.load('training/models/trial1/han_embeddings.pth')
embeddings = checkpoint['embeddings']  # {'paper': tensor, 'author': tensor, 'keyword': tensor}
id_maps = checkpoint['id_maps']        # ID mappings
```

## 👥 Team Members
- JIA Yusheng | 21035474 | yjiaag@connect.ust.hk
- YANG Yuhan  | 21039755 | yyangfv@connect.ust.hk
- WANG Daci   | 20944185 | dwangbg@connect.ust.hk

## 📝 Notes

- **Neo4j Password:** Default is `Jackson050609`, update in scripts if different
- **Sample Size:** Use smaller samples (e.g., 5K-10K papers) for quick testing
- **Memory:** Reduce `SAMPLE_SIZE` if out of memory during training
- **GPU:** Install CUDA versions of PyTorch/DGL for GPU acceleration
- **Data Source:** Aminer URLs may be unstable, script includes fallback options

## 🔗 References

- [Neo4j Documentation](https://neo4j.com/docs/)
- [DGL Documentation](https://docs.dgl.ai/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Aminer Dataset](https://www.aminer.org/citation)


