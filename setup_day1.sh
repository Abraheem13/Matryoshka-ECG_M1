#!/bin/bash
# ============================================================
# MATRYOSHKA ECG - DAY 1 COMPLETE SETUP
# Paper: "Matryoshka ECG: Adaptive-Fidelity Arrhythmia 
#         Classification for Wearable-to-Cloud Deployment"
# Target: IEEE J-BHI
# ============================================================

set -e  # Exit on any error

echo "=============================================="
echo "  MATRYOSHKA ECG - Day 1 Setup"
echo "=============================================="

# ----------------------------------------------------------
# 1. Create project structure
# ----------------------------------------------------------
echo "[1/6] Creating project structure..."
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p configs data/ptbxl data/raw models losses utils scripts results
mkdir -p results/figures results/checkpoints results/logs

# ----------------------------------------------------------
# 2. Create virtual environment & install dependencies
# ----------------------------------------------------------
echo "[2/6] Setting up Python environment..."

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

pip install \
    torch>=2.0.0 \
    torchvision \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    wfdb>=4.1.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    tqdm>=4.65.0 \
    PyYAML>=6.0 \
    tensorboard>=2.13.0 \
    h5py>=3.8.0 \
    ast-comments>=1.0.0

echo "   ✓ All packages installed"

# ----------------------------------------------------------
# 3. Download PTB-XL dataset
# ----------------------------------------------------------
echo "[3/6] Downloading PTB-XL dataset (~2.6 GB)..."

cd data/raw

if [ ! -f "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip" ]; then
    wget -q --show-progress \
        https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
    echo "   ✓ Download complete"
else
    echo "   ✓ Dataset already downloaded"
fi

# ----------------------------------------------------------
# 4. Extract dataset
# ----------------------------------------------------------
echo "[4/6] Extracting dataset..."

if [ ! -d "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" ]; then
    unzip -q ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
    echo "   ✓ Extraction complete"
else
    echo "   ✓ Dataset already extracted"
fi

# Create a symlink for easier access
cd "$PROJECT_DIR"
ln -sf data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 data/ptbxl_raw

echo "   ✓ Symlink created at data/ptbxl_raw"

# ----------------------------------------------------------
# 5. Verify installation
# ----------------------------------------------------------
echo "[5/6] Verifying installation..."

python3 -c "
import torch
import wfdb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
print(f'   PyTorch:       {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU:           {torch.cuda.get_device_name(0)}')
print(f'   NumPy:         {np.__version__}')
print(f'   Pandas:        {pd.__version__}')
print('   ✓ All imports successful')
"

# ----------------------------------------------------------
# 6. Verify dataset
# ----------------------------------------------------------
echo "[6/6] Verifying PTB-XL dataset..."

python3 -c "
import pandas as pd
import os

base_path = 'data/ptbxl_raw'
if os.path.exists(base_path):
    df = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'))
    print(f'   Records found:  {len(df)}')
    print(f'   Columns:        {len(df.columns)}')
    print(f'   Folds (1-10):   {sorted(df.strat_fold.unique())}')
    print('   ✓ PTB-XL dataset verified')
else:
    print('   ✗ Dataset not found - check extraction')
"

echo ""
echo "=============================================="
echo "  ✓ DAY 1 SETUP COMPLETE"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python scripts/preprocess_ptbxl.py"
echo "  3. python scripts/verify_data.py"
echo "  4. python scripts/train_day1.py --config configs/mrl_resnet1d.yaml"
echo ""
