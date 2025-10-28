# COMP 4222 Group Project

## 📋 Project Overview
Academic Graph Neural Network Analysis Project

## 🚀 Quick Start

### 1. Environment Setup
Run `setup.sh` to configure the project environment:

```bash
bash setup.sh
```

Or manually create the environment using conda:

```bash
conda env create -f environment.yml
conda activate 4222_Project
```

### 2. Install Neo4j Database
Visit [Neo4j Official Website](https://neo4j.com/download/) to download and install Neo4j client

### 3. Data Preparation

#### Method 1: One-Click Execution (Recommended)
Use the automated script to complete data download and processing:

```bash
bash data.sh
```


#### Method 2: Step-by-Step Execution
If you need to run each step individually:

**Step 3.1: Download Data**
```bash
python data_download.py
```

**Step 3.2: Process Data**
```bash
python process_data.py
```

If automatic download fails, refer to the manual download guide in the script, or manually place data files in the `data/raw/` directory.

## 📁 Project Structure
```
.
├── data_download.py      # Data download script
├── process_data.py       # Data processing script
├── data.sh               # One-click data pipeline script
├── neo4j_import.py       # Neo4j data import
├── han_model.py          # HAN model training
├── verify_environment.py # Environment verification
├── setup.sh              # Environment setup script
├── environment.yml       # Conda environment config
└── requirements.txt      # Python dependencies
```

## 🛠️ Main Scripts

### Automation Scripts
- `data.sh` - One-click data download and processing
  - Automatically runs download and processing pipeline
  - Error detection and status reporting
  - Cross-platform support (Linux/Mac/Windows)

### Data Processing Pipeline
1. `download_data.py` - Download dataset to `data/raw/`
   - Automatically download Aminer dataset
   - Support ZIP and GZIP extraction
   - Optional sample data creation

2. `process_data.py` - Process data and save to `data/processed/`
   - Load data from raw directory
   - Clean and transform data
   - Generate CSV files (papers, authors, keywords, etc.)

### Other Scripts
- `setup.sh` - Environment setup script
- `verify_environment.py` - Environment verification script
- `neo4j_import.py` - Import data to Neo4j
- `han_model.py` - HAN model training

## ⚙️ Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- Neo4j Database
- Other dependencies: see `requirements.txt` and `environment.yml`

## 👥 Team Members
- JIA Yusheng | 21035474 | yjiaag@connect.ust.hk
- YANG Yuhan  | 21039755 | yyangfv@connect.ust.hk
- WANG Daci   | 20944185 | dwangbg@connect.ust.hk


## 📝 Workflow

1. **Setup Environment**: `bash setup.sh`
2. **Verify Environment**: `python verify_environment.py`
3. **Prepare Data**: `bash data.sh`
4. **Start Neo4j**: Start Neo4j database service
5. **Import Data**: `python neo4j_import.py`
6. **Train Model**: `python han_model.py`


