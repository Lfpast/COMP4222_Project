#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")/.."

# ============================================================================
# GraphRAG System - Unified Execution Script
# ============================================================================
# This script runs all GraphRAG-related functionalities:
# 1. Test recommender system and GraphRAG engine
# 2. Evaluate retrieval performance (HAN vs SBERT)
# 3. Run complete system test suite
# ============================================================================

# ============================================================================
# Configuration - Neo4j Database Connection
# ============================================================================
NEO4J_URI="neo4j://127.0.0.1:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="12345678"

# ============================================================================
# Configuration - Model Path
# ============================================================================
MODEL_PATH="training/models/link_prediction_v4/han_embeddings.pth"

# ============================================================================
# Configuration - GraphRAG Query Parameters
# ============================================================================
TOP_K_SEEDS=5              # Number of seed papers to retrieve
ENABLE_GRAPH_RETRIEVAL=1   # Enable graph retrieval (1=enabled, 0=disabled)

# ============================================================================
# Configuration - Testing Parameters
# ============================================================================
TEST_TOP_K=5               # Number of recommendations for testing
EVAL_TOP_K=10              # Number of recommendations for evaluation

# ============================================================================
# Usage Information
# ============================================================================
print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Available commands:"
    echo "  test       - Run complete system test suite (Recommender + GraphRAG + Evaluation)"
    echo "  evaluate   - Evaluate HAN vs SBERT retrieval performance"
    echo "  query      - Run a single GraphRAG query (requires QUERY environment variable)"
    echo ""
    echo "Examples:"
    echo "  $0 test                # Run all tests"
    echo "  $0 evaluate            # Evaluate retrieval performance"
    echo "  QUERY='graph neural networks' $0 query  # Query example"
    echo ""
}

# ============================================================================
# Main Logic
# ============================================================================
COMMAND=${1:-test}

case "$COMMAND" in
    test)
        echo "=========================================="
        echo "Running GraphRAG System Test Suite"
        echo "=========================================="
        python GraphRAG/test_graph_rag.py \
            --neo4j_uri "$NEO4J_URI" \
            --neo4j_username "$NEO4J_USERNAME" \
            --neo4j_password "$NEO4J_PASSWORD" \
            --model_path "$MODEL_PATH" \
            --top_k "$TEST_TOP_K"
        ;;
    
    evaluate)
        echo "=========================================="
        echo "Evaluating HAN vs SBERT Retrieval Performance"
        echo "=========================================="
        python GraphRAG/evaluate_retrieval.py \
            --neo4j_uri "$NEO4J_URI" \
            --neo4j_username "$NEO4J_USERNAME" \
            --neo4j_password "$NEO4J_PASSWORD" \
            --model_path "$MODEL_PATH" \
            --top_k "$EVAL_TOP_K"
        ;;
    
    query)
        if [ -z "$QUERY" ]; then
            echo "Error: Please set the QUERY environment variable"
            echo "Example: QUERY='graph neural networks' $0 query"
            exit 1
        fi
        
        echo "=========================================="
        echo "Running GraphRAG Query"
        echo "Query: $QUERY"
        echo "=========================================="
        python -c "
import os
import sys
sys.path.insert(0, 'GraphRAG')
from dotenv import load_dotenv
from graph_rag_system import AcademicRecommender, GraphRAGEngine

load_dotenv()

# Load API key from environment variable
api_key = os.environ.get('DEEPSEEK_API_KEY')
if not api_key:
    print('Error: Please set DEEPSEEK_API_KEY in .env file')
    sys.exit(1)

# Initialize system
recommender = AcademicRecommender(
    model_path='$MODEL_PATH',
    neo4j_uri='$NEO4J_URI',
    neo4j_username='$NEO4J_USERNAME',
    neo4j_password='$NEO4J_PASSWORD'
)

engine = GraphRAGEngine(recommender, api_key)

# Execute query
answer = engine.query(
    '$QUERY',
    top_k_seeds=$TOP_K_SEEDS,
    enable_graph_retrieval=bool($ENABLE_GRAPH_RETRIEVAL)
)

print('\n' + '='*70)
print('Query Result:')
print('='*70)
print(answer)
"
        ;;
    
    help|--help|-h)
        print_usage
        ;;
    
    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        print_usage
        exit 1
        ;;
esac
