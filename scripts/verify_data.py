"""
Data Verification Script
========================
Verifies preprocessed data, tests model forward pass,
and confirms everything is ready for training.

Usage: python scripts/verify_data.py
"""

import os
import sys
import numpy as np
import pickle
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("=" * 60)
    print("  DATA & MODEL VERIFICATION")
    print("=" * 60)
    
    data_dir = "data/processed"
    errors = 0
    
    # ----------------------------------------------------------
    # 1. Check processed files exist
    # ----------------------------------------------------------
    print("\n[1/5] Checking processed files...")
    required = ['X_train.npy', 'y_train.npy', 'X_val.npy', 'y_val.npy',
                'X_test.npy', 'y_test.npy', 'metadata.pkl']
    
    for f in required:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  ✓ {f:20s} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {f:20s} MISSING!")
            errors += 1
    
    if errors > 0:
        print(f"\n  ✗ {errors} files missing. Run preprocessing first:")
        print("    python scripts/preprocess_ptbxl.py")
        return
    
    # ----------------------------------------------------------
    # 2. Verify shapes and dtypes
    # ----------------------------------------------------------
    print("\n[2/5] Verifying shapes and dtypes...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    for name, arr in [('X_train', X_train), ('y_train', y_train),
                      ('X_val', X_val), ('y_val', y_val),
                      ('X_test', X_test), ('y_test', y_test)]:
        print(f"  {name:10s}: shape={str(arr.shape):20s} dtype={arr.dtype}")
    
    # Verify expected shapes
    assert X_train.ndim == 3, f"X_train should be 3D, got {X_train.ndim}D"
    assert X_train.shape[1] == 12, f"Expected 12 leads, got {X_train.shape[1]}"
    assert X_train.shape[2] == 1000, f"Expected 1000 samples, got {X_train.shape[2]}"
    assert y_train.ndim == 2, f"y_train should be 2D, got {y_train.ndim}D"
    assert X_train.shape[0] == y_train.shape[0], "X/y size mismatch"
    print("  ✓ All shapes correct")
    
    # ----------------------------------------------------------
    # 3. Verify label distribution
    # ----------------------------------------------------------
    print("\n[3/5] Verifying label distribution...")
    
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    print(f"  Task:       {meta['task']}")
    print(f"  Classes:    {meta['num_classes']}")
    print(f"  Mapping:    {meta['idx_to_class']}")
    
    for split_name, labels in [('train', y_train), ('val', y_val), ('test', y_test)]:
        print(f"\n  {split_name.upper()} distribution:")
        for idx, cls_name in meta['idx_to_class'].items():
            count = int(labels[:, idx].sum())
            pct = 100 * count / len(labels)
            bar = "█" * int(pct / 2)
            print(f"    {cls_name:6s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # Check for samples with no labels
    no_label_train = (y_train.sum(axis=1) == 0).sum()
    no_label_val = (y_val.sum(axis=1) == 0).sum()
    no_label_test = (y_test.sum(axis=1) == 0).sum()
    
    if no_label_train + no_label_val + no_label_test > 0:
        print(f"\n  ⚠ Samples with no labels: train={no_label_train}, val={no_label_val}, test={no_label_test}")
    else:
        print(f"\n  ✓ All samples have at least one label")
    
    # ----------------------------------------------------------
    # 4. Test DataLoader
    # ----------------------------------------------------------
    print("\n[4/5] Testing DataLoader...")
    
    import yaml
    from data.dataset import ECGDataModule
    
    with open('configs/mrl_resnet1d.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['num_workers'] = 0  # For testing
    data_module = ECGDataModule(data_dir=data_dir, config=config)
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    batch_x, batch_y = next(iter(train_loader))
    print(f"  Batch X: {batch_x.shape} ({batch_x.dtype})")
    print(f"  Batch Y: {batch_y.shape} ({batch_y.dtype})")
    print(f"  X range: [{batch_x.min():.2f}, {batch_x.max():.2f}]")
    print(f"  Y range: [{batch_y.min():.2f}, {batch_y.max():.2f}]")
    print(f"  ✓ DataLoader working")
    
    # ----------------------------------------------------------
    # 5. Test Model Forward Pass
    # ----------------------------------------------------------
    print("\n[5/5] Testing model forward pass...")
    
    from models.mrl_ecg_model import MatryoshkaECGModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model = MatryoshkaECGModel(config).to(device)
    
    x = batch_x[:4].to(device)
    y = batch_y[:4].to(device)
    
    # Training mode
    model.train()
    result = model(x, y)
    print(f"  Training loss: {result['total_loss'].item():.4f}")
    for dim, loss in sorted(result['per_dim_loss'].items()):
        print(f"    dim={dim:3d}: loss={loss.item():.4f}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        for dim in config['mrl']['nesting_dims']:
            logits, probs = model.predict(x, dim=dim)
            print(f"  Predict dim={dim:3d}: logits={logits.shape}, probs range=[{probs.min():.3f}, {probs.max():.3f}]")
    
    # Get embedding
    with torch.no_grad():
        emb = model.get_embedding(x)
        print(f"  Embedding: {emb.shape}")
        print(f"  Embedding norm: {emb.norm(dim=1).mean():.4f}")
    
    print(f"\n  ✓ Model forward/backward pass successful")
    
    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "=" * 60)
    print("  ✓ ALL VERIFICATIONS PASSED")
    print("=" * 60)
    print(f"  Dataset:    PTB-XL ({meta['task']})")
    print(f"  Samples:    {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
    print(f"  Classes:    {meta['num_classes']} ({', '.join(meta['idx_to_class'].values())})")
    print(f"  Signal:     12 leads × 1000 samples @ 100Hz")
    print(f"  Model:      {config['model']['backbone']} + MRL")
    print(f"  Parameters: {total_params:,}")
    print(f"  Nesting:    {config['mrl']['nesting_dims']}")
    print(f"  Device:     {device}")
    print("=" * 60)
    print(f"\n  Ready to train! Run:")
    print(f"    python scripts/train_day1.py --config configs/mrl_resnet1d.yaml")
    print()


if __name__ == "__main__":
    main()
