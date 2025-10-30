# HAN Model Training & Evaluation

Automated pipeline for training Heterogeneous Attention Network (HAN) models and evaluating embeddings.

## 📁 Files

- `han_model.py` - HAN model training script
- `validate_embeddings.py` - Embedding quality evaluation script  
- `run.bat` - Windows automation script
- `run.sh` - Linux/Mac automation script

## 🚀 Quick Start

```bash
bash run.sh            # Use default parameters
bash run.sh trial2     # Create new experiment
```

## ⚙️ Hyperparameter Configuration

Edit variables at the top of `run.bat` (Windows) or `run.sh` (Linux/Mac):

### Data
```bash
SAMPLE_SIZE=10000      # Number of papers to train on (0 for all data)
```

### Model Architecture
```bash
HIDDEN_DIM=256         # Hidden layer dimension (128, 256, 512)
OUT_DIM=128            # Output embedding dimension (64, 128, 256)
NUM_HEADS=4            # Attention heads (2, 4, 8)
```

### Training
```bash
EPOCHS=100             # Training epochs (50, 100, 200)
LEARNING_RATE=0.001    # Learning rate (0.0001, 0.001, 0.01)
```

## 📊 Output Files

After running, files are generated in `models/<trial_name>/`:

```
models/trial1/
├── han_embeddings.pth          # Trained model and embeddings
├── summary.json                # Training statistics
├── evaluation_report.json      # Evaluation metrics + tuning suggestions ⭐
└── visualizations/             # Visualization plots
    ├── paper_embeddings.png
    ├── author_embeddings.png
    ├── keyword_embeddings.png
    └── loss_curve.png
```

**Key File**: `evaluation_report.json` contains quality metrics and hyperparameter tuning recommendations.

## 💡 Hyperparameter Tuning

1. **Check recommendations** in `evaluation_report.json`
2. **Adjust parameters** based on suggestions:
   - Loss not converged → Increase `EPOCHS`
   - Slow improvement → Increase `LEARNING_RATE` or `HIDDEN_DIM`
   - Low clustering quality (Silhouette < 0.3) → Increase `OUT_DIM` or `EPOCHS`
   - Low similarity (< 0.5) → Increase `NUM_HEADS` or `HIDDEN_DIM`
3. **Run new experiment** with adjusted parameters

### Recommended Configurations

**Quick Test** (5-10 min):
```bash
SAMPLE_SIZE=5000, EPOCHS=50, HIDDEN_DIM=128, OUT_DIM=64, NUM_HEADS=2
```

**Standard Training** (15-30 min):
```bash
SAMPLE_SIZE=10000, EPOCHS=100, HIDDEN_DIM=256, OUT_DIM=128, NUM_HEADS=4
```

**High Quality** (1-2 hours):
```bash
SAMPLE_SIZE=50000, EPOCHS=200, HIDDEN_DIM=512, OUT_DIM=256, NUM_HEADS=8
```

## 🔬 Individual Components

### Train Only
```bash
python han_model.py 10000 100 0.001 models/trial1 256 128 4
# Args: sample_size epochs lr save_dir hidden_dim out_dim num_heads
```

### Evaluate Only
```bash
python validate_embeddings.py models/trial1/han_embeddings.pth
```

## 📈 Training Progress

Real-time progress bar during training:
```
Training: 45%|████████▌     | 45/100 [00:23<00:28] Loss: 2.3456, Time: 0.52s, ETA: 0:00:28
```

## ❓ FAQ

**Q: Out of memory?**  
A: Reduce `SAMPLE_SIZE`, `HIDDEN_DIM`, or `OUT_DIM`

**Q: Training too slow?**  
A: Reduce `SAMPLE_SIZE` or `EPOCHS`

**Q: How to compare experiments?**  
A: Compare `evaluation_report.json` files across different trials

## 🎓 Next Steps

Use trained embeddings for:
- Paper classification
- Citation link prediction
- Author collaboration recommendation
- Research topic clustering

Load embeddings:
```python
import torch
checkpoint = torch.load('models/trial1/han_embeddings.pth')
embeddings = checkpoint['embeddings']  # {'paper': tensor, 'author': tensor, 'keyword': tensor}
id_maps = checkpoint['id_maps']
```

