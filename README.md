# Enhanced Academic Recommendation System Using Multimodal Heterogeneous Graphs and GraphRAG 

## Purpose
This project aims to build an academic recommender and retrieval system based on heterogeneous graph neural networks (HAN) and GraphRAG (Graph Retrieval-Augmented Generation). It integrates Neo4j graph database, HAN model training, and LLM-powered retrieval to provide advanced recommendations and context-aware academic search.

## Dataset
We use the ACM Citation Network V12 dataset to support our GraphRAG System.  
https://opendata.aminer.cn/dataset/ACM-Citation-network-V12.zip  
We perform some feature selection rather than using all of the attributes. You can download it here.  
https://hkustconnect-my.sharepoint.com/:u:/g/personal/yjiaag_connect_ust_hk/EXJjZsR9qrZNkOcDPJVskgcBxYZA6xsGoG3AMEudWfqsyg?e=Jt2AVq  
Put it inside the /data/raw directory and start running the entire pipeline.  


## Repository Structure
```
Project/
├── data/
│   ├── process_data.py                 # Data preprocessing script
│   └── analyze_time_distribution.py    # Paper time distribution analysis
├── GraphRAG/
│   ├── graph_rag_system.py             # Main GraphRAG engine
│   ├── evaluate_retrieval.py           # HAN vs SBERT retrieval evaluation
│   └── test_graph_rag.py               # Full test suite for system
├── training/
│   └── han_model.py                    # HAN model training script
├── scripts/
│   ├── setup.sh                        # Environment and dependency setup
│   ├── data_processing.sh              # Data preprocessing pipeline
│   ├── neo4j_pipeline.sh               # Neo4j import and subgraph selection
│   ├── train_han_model.sh              # HAN model training pipeline
│   └── graph_rag.sh                    # Unified GraphRAG system runner
├── neo4j_import.py                     # Neo4j import pipeline
├── subgraph_exporter.py                # Subgraph selection/export
├── verify_environment.py               # Environment verification
└── README.md                           # Project documentation
```

## Setup & Usage Steps
1. **Environment Setup**
   - Run `scripts/setup.sh` to create the conda environment and install all dependencies.
   - Verify environment with `verify_environment.py`.

2. **Data Processing**
   - Run `scripts/data_processing.sh` to preprocess raw data and generate processed CSVs.
   - Analyze paper time distribution for insights.

3. **Neo4j Import & Subgraph Selection**
   - Run `scripts/neo4j_pipeline.sh` to import processed data into Neo4j and select focused subgraphs.

4. **HAN Model Training**
   - Run `scripts/train_han_model.sh` to train the HAN model and generate embeddings.

5. **GraphRAG System & Evaluation**
   - Run `scripts/graph_rag.sh` for:
     - Full system test suite (`test`)
     - Retrieval evaluation (`evaluate`)
     - Custom query (`query`)
   - All LLM outputs are logged to `GraphRAG/llm_output.log`.

## Python Script Functions
- **neo4j_import.py**: Imports processed CSV data into Neo4j, creates constraints, and verifies import.
- **subgraph_exporter.py**: Selects and exports a focused subgraph from Neo4j for downstream tasks.
- **verify_environment.py**: Checks Python environment and package installation.
- **data/process_data.py**: Converts raw JSONL data to processed CSVs for graph import.
- **data/analyze_time_distribution.py**: Analyzes the distribution of papers over time.
- **training/han_model.py**: Trains the HAN model and saves embeddings for link prediction.
- **GraphRAG/graph_rag_system.py**: Implements the GraphRAG engine for academic search and recommendation.
- **GraphRAG/evaluate_retrieval.py**: Compares HAN and SBERT retrieval performance.
- **GraphRAG/test_graph_rag.py**: Runs all system tests and logs LLM outputs.

## Team Members
| Name         | Student ID | Email                  |
|--------------|------------|------------------------|
| JIA Yusheng  | 21035474   | yjiaag@connect.ust.hk  |
| Wang Daci    | 20944185   | dwangbg@connect.ust.hk |
| Yang Yuhan   | 21039755   | yyangfv@connect.ust.hk |

## Additional Notes
- All configuration except API keys is managed via shell scripts in `scripts/`.
- API keys (e.g., DEEPSEEK_API_KEY) should be stored in a `.env` file in the local desktop, you cannot see it.
- All Python scripts use argparse for parameter parsing; no hardcoded defaults.
- The project is modular and easy to extend for new models or graph types.
- For troubleshooting, check `GraphRAG/llm_output.log` and Neo4j connection status.

## License
This project is for academic use only.

---

