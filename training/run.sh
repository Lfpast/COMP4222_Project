#!/bin/bash
# ============================================================================
# HAN Model Training & Evaluation Pipeline
# ============================================================================
# Usage: bash run.sh [trial_name]
# Example: bash run.sh trial2
#
# This script automatically trains the HAN model and evaluates the embeddings.
# You can customize hyperparameters by editing the variables below.
# ============================================================================

# ============================================================================
# HYPERPARAMETER CONFIGURATION
# ============================================================================

# Trial Configuration
TRIAL_NAME="${1:-trial1}"  # Use command line argument or default to "trial1"
SAVE_DIR="models/${TRIAL_NAME}"

# Neo4j Connection
NEO4J_URI="neo4j://127.0.0.1:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="Jackson050609"

# Data Configuration
SAMPLE_SIZE=10000  # Number of papers to use (set to 0 or empty for all papers)

# Model Architecture
IN_DIM=384         # Input dimension (Sentence-BERT embedding size, don't change)
HIDDEN_DIM=256     # Hidden layer dimension (try: 128, 256, 512)
OUT_DIM=128        # Output embedding dimension (try: 64, 128, 256)
NUM_HEADS=4        # Number of attention heads (try: 2, 4, 8)

# Training Hyperparameters
EPOCHS=100         # Number of training epochs (try: 50, 100, 200)
LEARNING_RATE=0.001  # Learning rate (try: 0.0001, 0.001, 0.005, 0.01)

# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

echo "========================================================================"
echo "🚀 HAN Model Training & Evaluation Pipeline"
echo "========================================================================"
echo ""
echo "📋 Configuration:"
echo "   Trial Name: ${TRIAL_NAME}"
echo "   Save Directory: ${SAVE_DIR}"
echo "   Sample Size: ${SAMPLE_SIZE}"
echo "   Model Architecture: IN=${IN_DIM}, HIDDEN=${HIDDEN_DIM}, OUT=${OUT_DIM}, HEADS=${NUM_HEADS}"
echo "   Training: EPOCHS=${EPOCHS}, LR=${LEARNING_RATE}"
echo ""

# Create save directory
mkdir -p "${SAVE_DIR}"

# ============================================================================
# STEP 1: Train HAN Model
# ============================================================================

echo "========================================================================"
echo "📚 STEP 1/2: Training HAN Model"
echo "========================================================================"
echo ""

# Run training with command line arguments
python han_model.py \
    ${SAMPLE_SIZE} \
    ${EPOCHS} \
    ${LEARNING_RATE} \
    ${SAVE_DIR} \
    ${HIDDEN_DIM} \
    ${OUT_DIM} \
    ${NUM_HEADS}

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Training failed with exit code ${TRAIN_EXIT_CODE}"
    echo "Pipeline stopped."
    exit 1
fi

# ============================================================================
# STEP 2: Evaluate Embeddings
# ============================================================================

echo ""
echo "========================================================================"
echo "📊 STEP 2/2: Evaluating Embeddings"
echo "========================================================================"
echo ""

# Run evaluation
python validate_embeddings.py "${SAVE_DIR}/han_embeddings.pth"

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Evaluation failed with exit code ${EVAL_EXIT_CODE}"
    exit 1
fi

# ============================================================================
# PIPELINE COMPLETED
# ============================================================================

echo ""
echo "========================================================================"
echo "✅ Pipeline Completed Successfully!"
echo "========================================================================"
echo ""
echo "📁 Output files in ${SAVE_DIR}/:"
echo "   • han_embeddings.pth - Trained model and embeddings"
echo "   • summary.json - Training statistics"
echo "   • evaluation_report.json - Evaluation metrics and recommendations"
echo "   • visualizations/*.png - Embedding visualizations"
echo ""
echo "📋 Next steps:"
echo "   1. Review evaluation_report.json for hyperparameter tuning suggestions"
echo "   2. Check visualizations/ folder for embedding quality plots"
echo "   3. Adjust hyperparameters in this script and re-run if needed"
echo ""
echo "🔄 To run with different hyperparameters:"
echo "   Edit the variables at the top of run.sh"
echo "   Or run: bash run.sh trial2 (creates new trial directory)"
echo ""
