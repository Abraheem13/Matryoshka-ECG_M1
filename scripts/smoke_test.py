"""
Quick Smoke Test
=================
Tests all components with synthetic data (no download needed).
Run this immediately after cloning to verify code integrity.

Usage: python scripts/smoke_test.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_backbones():
    """Test all backbone architectures."""
    from models.xresnet1d import create_backbone
    
    print("[1/4] Testing backbones...")
    x = torch.randn(4, 12, 1000)
    
    for name in ['xresnet1d50', 'xresnet1d101', 'inception1d']:
        model = create_backbone(name, input_channels=12, embedding_dim=512)
        out = model(x)
        assert out.shape == (4, 512), f"{name}: expected (4,512), got {out.shape}"
        
        # Test gradient flow
        loss = out.sum()
        loss.backward()
        print(f"  ✓ {name}: {x.shape} -> {out.shape}, grad OK")
    
    print()


def test_mrl_loss():
    """Test Matryoshka loss."""
    from losses.matryoshka_loss import MatryoshkaLoss, StandardLoss
    
    print("[2/4] Testing MRL loss...")
    
    nesting_dims = [16, 32, 64, 128, 256, 512]
    mrl = MatryoshkaLoss(nesting_dims=nesting_dims, num_classes=5, multi_label=True)
    
    embedding = torch.randn(8, 512, requires_grad=True)
    targets = torch.randint(0, 2, (8, 5)).float()
    
    result = mrl(embedding, targets)
    
    assert 'total_loss' in result
    assert 'per_dim_loss' in result
    assert len(result['per_dim_loss']) == len(nesting_dims)
    
    result['total_loss'].backward()
    assert embedding.grad is not None
    print(f"  ✓ MRL loss: {result['total_loss'].item():.4f}")
    
    # Test predictions at each dim
    for dim in nesting_dims:
        logits, probs = mrl.get_predictions(embedding.detach(), dim=dim)
        assert probs.shape == (8, 5)
        assert (probs >= 0).all() and (probs <= 1).all()
    print(f"  ✓ Predictions at all {len(nesting_dims)} dims OK")
    
    # Test weight strategies
    for strategy in ['equal', 'linear', 'exponential']:
        mrl_s = MatryoshkaLoss(nesting_dims=nesting_dims, num_classes=5, 
                                weight_strategy=strategy)
        r = mrl_s(embedding.detach(), targets)
        print(f"  ✓ Strategy '{strategy}': loss={r['total_loss'].item():.4f}")
    
    # Test standard loss baseline
    std = StandardLoss(embedding_dim=64, num_classes=5)
    emb_small = torch.randn(8, 64)
    r = std(emb_small, targets)
    print(f"  ✓ StandardLoss: loss={r['total_loss'].item():.4f}")
    print()


def test_full_model():
    """Test complete MRL ECG model."""
    import yaml
    from models.mrl_ecg_model import MatryoshkaECGModel, FixedDimECGModel
    
    print("[3/4] Testing full models...")
    
    # Use xresnet1d50 for speed
    config = {
        'model': {'backbone': 'xresnet1d50', 'input_channels': 12,
                  'base_embedding_dim': 512, 'num_classes': 5, 'dropout': 0.3},
        'mrl': {'enabled': True, 'nesting_dims': [16, 32, 64, 128, 256, 512],
                'loss_weights': 'equal'},
        'training': {'label_smoothing': 0.1}
    }
    
    # MRL model
    model = MatryoshkaECGModel(config)
    x = torch.randn(4, 12, 1000)
    y = torch.randint(0, 2, (4, 5)).float()
    
    result = model(x, y)
    result['total_loss'].backward()
    print(f"  ✓ MRL model: train loss={result['total_loss'].item():.4f}")
    
    model.eval()
    with torch.no_grad():
        emb = model.get_embedding(x)
        assert emb.shape == (4, 512)
        
        for dim in [16, 64, 512]:
            logits, probs = model.predict(x, dim=dim)
            assert probs.shape == (4, 5)
    print(f"  ✓ MRL model: inference at all dims OK")
    
    # Fixed-dim baseline
    model_fixed = FixedDimECGModel(config, fixed_dim=64)
    result = model_fixed(x, y)
    result['total_loss'].backward()
    print(f"  ✓ Fixed-dim model (64): loss={result['total_loss'].item():.4f}")
    print()


def test_metrics():
    """Test evaluation metrics."""
    from utils.metrics import compute_all_metrics, MetricTracker
    
    print("[4/4] Testing metrics...")
    
    y_true = np.array([[1,0,0,1,0], [0,1,0,0,1], [1,0,1,0,0], 
                       [0,0,0,1,1], [1,1,0,0,0]], dtype=np.float32)
    y_prob = np.random.uniform(0, 1, (5, 5)).astype(np.float32)
    
    metrics = compute_all_metrics(y_true, y_prob)
    
    assert 'macro_auc' in metrics
    assert 'macro_f1' in metrics
    assert 0 <= metrics['macro_auc'] <= 1
    print(f"  ✓ Metrics: AUC={metrics['macro_auc']:.4f}, F1={metrics['macro_f1']:.4f}")
    
    # Test tracker
    tracker = MetricTracker()
    tracker.update(0, {'auc': 0.8, 'f1': 0.7})
    tracker.update(1, {'auc': 0.85, 'f1': 0.75})
    
    best_epoch, best_val = tracker.get_best('auc', mode='max')
    assert best_epoch == 1
    assert best_val == 0.85
    print(f"  ✓ MetricTracker: best AUC={best_val:.4f} at epoch {best_epoch}")
    print()


def main():
    print("=" * 60)
    print("  SMOKE TEST - Matryoshka ECG")
    print("=" * 60)
    print()
    
    try:
        test_backbones()
        test_mrl_loss()
        test_full_model()
        test_metrics()
        
        print("=" * 60)
        print("  ✓ ALL SMOKE TESTS PASSED")
        print("=" * 60)
        print("\n  Code is correct. Proceed with:")
        print("  1. ./setup_day1.sh        (download data)")
        print("  2. python scripts/preprocess_ptbxl.py")
        print("  3. python scripts/verify_data.py")
        print("  4. python scripts/train_day1.py")
        print()
        
    except Exception as e:
        print(f"\n  ✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
