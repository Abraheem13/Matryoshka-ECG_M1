# Matryoshka ECG: Adaptive-Fidelity Arrhythmia Classification

> **Paper:** Matryoshka ECG: Adaptive-Fidelity Arrhythmia Classification for Wearable-to-Cloud Deployment  
> **Target:** IEEE Journal of Biomedical and Health Informatics (J-BHI)

## 3-Day Execution Plan

### Day 1: Setup + Data Pipeline + MRL Implementation ← YOU ARE HERE
```bash
# Step 1: Run the setup script (creates env, downloads PTB-XL ~2.6GB)
chmod +x setup_day1.sh
./setup_day1.sh

# Step 2: Activate environment
source venv/bin/activate

# Step 3: Preprocess PTB-XL (extract signals, compute labels, normalize)
python scripts/preprocess_ptbxl.py

# Step 4: Verify everything works (data + model forward pass)
python scripts/verify_data.py

# Step 5: Start training MRL model (runs overnight)
python scripts/train_day1.py --config configs/mrl_resnet1d.yaml
```

### Day 2: Baseline Training + Cross-Dataset Evaluation
```bash
# Train fixed-dimension baselines (one model per dim)
for DIM in 16 32 64 128 256 512; do
    python scripts/train_day1.py --config configs/mrl_resnet1d.yaml \
        --no-mrl --fixed-dim $DIM --run-name baseline_dim${DIM}
done

# Train MRL with Inception1D backbone
python scripts/train_day1.py --config configs/mrl_resnet1d.yaml \
    --backbone inception1d --run-name mrl_inception1d

# SVD baseline (post-hoc compression of full-dim model)
python scripts/svd_baseline.py  # Day 2 script

# Cross-dataset evaluation on CPSC 2018
python scripts/eval_cpsc2018.py  # Day 2 script
```

### Day 3: Results Analysis + Pareto Plots + Deployment Simulation
```bash
# Generate all figures for the paper
python scripts/generate_figures.py

# Compute FLOPs, latency, memory at each nesting dim
python scripts/deployment_analysis.py

# Generate LaTeX tables
python scripts/generate_tables.py
```

## Project Structure

```
matryoshka-ecg/
├── configs/
│   └── mrl_resnet1d.yaml        # Experiment configuration
├── data/
│   ├── dataset.py                # PyTorch Dataset + DataModule
│   ├── processed/                # Preprocessed .npy files (after step 3)
│   ├── raw/                      # Downloaded PTB-XL (after step 1)
│   └── ptbxl_raw -> ...          # Symlink to extracted data
├── losses/
│   └── matryoshka_loss.py        # MRL loss + Standard loss
├── models/
│   ├── xresnet1d.py              # XResNet1D + Inception1D backbones
│   └── mrl_ecg_model.py          # Complete model (backbone + head)
├── utils/
│   └── metrics.py                # Evaluation metrics (AUC, F1, etc.)
├── scripts/
│   ├── preprocess_ptbxl.py       # Data preprocessing
│   ├── verify_data.py            # Verification script
│   └── train_day1.py             # Training script
├── results/
│   ├── checkpoints/              # Model checkpoints
│   ├── figures/                  # Generated plots
│   └── logs/                     # TensorBoard logs
├── setup_day1.sh                 # One-click setup script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Implementation Details

### Matryoshka Representation Learning (MRL)

The core innovation: a single model produces embeddings that are useful at **any prefix dimension**. The first 16 dimensions carry enough information for a smartwatch, the first 64 for a smartphone, and the full 512 for a clinical workstation.

**Loss function:**
```
L_total = (1/|M|) * Σ_{m ∈ M} w_m * BCE(classifier_m(z[:m]), y)
```
where M = {16, 32, 64, 128, 256, 512} and each `classifier_m` is an independent linear layer operating on the first `m` dimensions.

### PTB-XL Dataset

- **21,799** 12-lead ECGs, 10 seconds, 100/500 Hz
- **5 superdiagnostic classes:** NORM, MI, STTC, CD, HYP
- Official train/val/test splits (folds 1-8 / 9 / 10)
- Primary metric: **macro-averaged AUC-ROC**

## Training Tips

```bash
# Quick test run (2 epochs, smaller model)
python scripts/train_day1.py --config configs/mrl_resnet1d.yaml \
    --backbone xresnet1d50 --epochs 2 --batch-size 64

# Monitor training
tensorboard --logdir results/logs

# Resume from checkpoint (modify train script if needed)
# Checkpoints auto-saved every 10 epochs + best model
```

## GPU Memory Requirements

| Backbone     | Batch 64 | Batch 128 | Batch 256 |
|-------------|----------|-----------|-----------|
| xresnet1d50  | ~3 GB    | ~5 GB     | ~9 GB     |
| xresnet1d101 | ~5 GB    | ~8 GB     | ~14 GB    |
| inception1d  | ~2 GB    | ~3 GB     | ~5 GB     |

If GPU OOM: reduce batch_size or use `--backbone inception1d`.
