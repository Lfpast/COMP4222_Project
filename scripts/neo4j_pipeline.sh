#!/bin/bash

# Configuration - Neo4j Connection
NEO4J_URI="neo4j://127.0.0.1:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="12345678"


# Configuration - Data Paths
DATA_DIR="../data/processed"
OUTPUT_DIR="../data/focused_v1"

# Configuration - Batch Size for Import
BATCH_SIZE=1000000  # You can adjust this value based on your memory and performance needs

echo "Importing processed data to neo4j database..."
python ../neo4j_import.py \
    --uri "$NEO4J_URI" \
    --username "$NEO4J_USERNAME" \
    --password "$NEO4J_PASSWORD" \
    --data_dir "$DATA_DIR" \
    --batch_size "$BATCH_SIZE"

echo "Selecting seed and citated papers..."
python ../subgraph_exporter.py \
    --uri "$NEO4J_URI" \
    --username "$NEO4J_USERNAME" \
    --password "$NEO4J_PASSWORD" \
    --output_dir "$OUTPUT_DIR"