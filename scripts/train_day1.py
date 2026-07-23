"""
Day 1 Training Script - Matryoshka ECG
========================================
Trains MRL model on PTB-XL and evaluates at all nesting dimensions.

Usage:
    python scripts/train_day1.py --config configs/mrl_resnet1d.yaml
    python scripts/train_day1.py --config configs/mrl_resnet1d.yaml --backbone xresnet1d50
    python scripts/train_day1.py --config configs/mrl_resnet1d.yaml --no-mrl --fixed-dim 64
"""

import os
import sys
import argparse
import time
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ECGDataModule
from models.mrl_ecg_model import MatryoshkaECGModel, FixedDimECGModel
from utils.metrics import compute_all_metrics, print_metrics, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train Matryoshka ECG Model")
    parser.add_argument('--config', type=str, default='configs/mrl_resnet1d.yaml')
    parser.add_argument('--backbone', type=str, default=None, 
                        help='Override backbone (xresnet1d50/xresnet1d101/inception1d)')
    parser.add_argument('--no-mrl', action='store_true', 
                        help='Disable MRL, use standard loss')
    parser.add_argument('--fixed-dim', type=int, default=None,
                        help='Fixed embedding dim for baseline')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    return parser.parse_args()


def setup_device(config, args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def create_optimizer(model, config):
    """Create optimizer with weight decay."""
    train_cfg = config['training']
    
    # Separate parameters that should/shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bn' in name or 'bias' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': train_cfg.get('weight_decay', 0.01)},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    lr = train_cfg.get('learning_rate', 0.001)
    
    if train_cfg.get('optimizer', 'adamw') == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=lr)
    
    return optimizer


def create_scheduler(optimizer, config, num_batches_per_epoch):
    """Create learning rate scheduler with warmup."""
    train_cfg = config['training']
    epochs = train_cfg.get('epochs', 50)
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    
    # Cosine annealing with warmup
    warmup_steps = warmup_epochs * num_batches_per_epoch
    total_steps = epochs * num_batches_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, 
                    device, config, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    per_dim_losses = defaultdict(float)
    num_batches = 0
    clip_norm = config['training'].get('gradient_clip_norm', 1.0)
    use_amp = config['training'].get('mixed_precision', True) and device.type == 'cuda'
    
    pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1} [Train]", leave=False)
    
    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with autocast():
                result = model(x, y)
                loss = result['total_loss']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            result = model(x, y)
            loss = result['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        for dim, dim_loss in result['per_dim_loss'].items():
            per_dim_losses[dim] += dim_loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    avg_loss = total_loss / num_batches
    avg_dim_losses = {d: v / num_batches for d, v in per_dim_losses.items()}
    
    return avg_loss, avg_dim_losses


@torch.no_grad()
def evaluate(model, dataloader, device, config, nesting_dims=None):
    """
    Evaluate model on a dataset.
    Returns metrics at each nesting dimension.
    """
    model.eval()
    
    mrl_enabled = config.get('mrl', {}).get('enabled', True)
    
    if nesting_dims is None:
        if mrl_enabled:
            nesting_dims = config.get('mrl', {}).get('nesting_dims', [16, 32, 64, 128, 256, 512])
        else:
            nesting_dims = [config.get('model', {}).get('base_embedding_dim', 512)]
    
    # Collect all predictions and labels
    all_embeddings = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    use_amp = config['training'].get('mixed_precision', True) and device.type == 'cuda'
    
    for x, y in tqdm(dataloader, desc="  Evaluating", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        if use_amp:
            with autocast():
                embedding = model.get_embedding(x)
                result = model.head(embedding, y)
        else:
            embedding = model.get_embedding(x)
            result = model.head(embedding, y)
        
        total_loss += result['total_loss'].item()
        all_embeddings.append(embedding.cpu())
        all_targets.append(y.cpu())
        num_batches += 1
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    avg_loss = total_loss / num_batches
    
    # Compute metrics at each nesting dimension
    results = {'loss': avg_loss}
    
    for dim in nesting_dims:
        # Get predictions at this dim
        logits, probs = model.head.get_predictions(all_embeddings.to(device), dim=dim)
        probs_np = probs.cpu().numpy()
        
        metrics = compute_all_metrics(all_targets, probs_np)
        results[dim] = metrics
    
    return results


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
    }, path)


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply CLI overrides
    if args.backbone:
        config['model']['backbone'] = args.backbone
    if args.no_mrl:
        config['mrl']['enabled'] = False
    if args.fixed_dim:
        config['model']['base_embedding_dim'] = args.fixed_dim
        config['mrl']['enabled'] = False
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup
    device = setup_device(config, args)
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.get('seed', 42))
    
    mrl_enabled = config.get('mrl', {}).get('enabled', True)
    nesting_dims = config['mrl']['nesting_dims'] if mrl_enabled else [config['model']['base_embedding_dim']]
    
    # Run name
    if args.run_name:
        run_name = args.run_name
    else:
        backbone = config['model']['backbone']
        mode = "mrl" if mrl_enabled else f"fixed{config['model']['base_embedding_dim']}"
        run_name = f"{backbone}_{mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Directories
    log_dir = os.path.join(config['logging']['log_dir'], run_name)
    ckpt_dir = config['logging']['checkpoint_dir']
    fig_dir = config['logging']['figure_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Print header
    print("\n" + "=" * 70)
    print("  MATRYOSHKA ECG TRAINING")
    print("=" * 70)
    print(f"  Run:         {run_name}")
    print(f"  Device:      {device}")
    print(f"  Backbone:    {config['model']['backbone']}")
    print(f"  MRL:         {'ENABLED' if mrl_enabled else 'DISABLED'}")
    if mrl_enabled:
        print(f"  Nesting:     {nesting_dims}")
    else:
        print(f"  Embed dim:   {config['model']['base_embedding_dim']}")
    print(f"  Epochs:      {config['training']['epochs']}")
    print(f"  Batch size:  {config['training']['batch_size']}")
    print(f"  LR:          {config['training']['learning_rate']}")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading data...")
    data_module = ECGDataModule(data_dir="data/processed", config=config)
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    
    # Load metadata for class names
    meta_path = "data/processed/metadata.pkl"
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        class_names = [metadata['idx_to_class'][i] for i in range(metadata['num_classes'])]
    else:
        class_names = [f"class_{i}" for i in range(config['model']['num_classes'])]
    
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    print(f"  Classes: {class_names}")
    
    # Create model
    print("\n[2/4] Creating model...")
    if mrl_enabled:
        model = MatryoshkaECGModel(config)
    else:
        model = FixedDimECGModel(config, fixed_dim=config['model']['base_embedding_dim'])
    
    model = model.to(device)
    
    # Optimizer, scheduler, scaler
    print("\n[3/4] Setting up training...")
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler() if (config['training'].get('mixed_precision', True) and device.type == 'cuda') else None
    
    # Tensorboard
    writer = SummaryWriter(log_dir)
    
    # Save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Metric tracking
    tracker = MetricTracker()
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = config['training'].get('early_stopping_patience', 10)
    
    # ============================================================
    # Training Loop
    # ============================================================
    print("\n[4/4] Training...\n")
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # --- Train ---
        train_loss, train_dim_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, 
            device, config, epoch
        )
        
        # --- Validate ---
        val_results = evaluate(model, val_loader, device, config, nesting_dims)
        
        # Get primary metric (macro AUC at largest dim)
        primary_dim = nesting_dims[-1]
        val_auc = val_results[primary_dim]['macro_auc']
        val_f1 = val_results[primary_dim]['macro_f1']
        val_loss = val_results['loss']
        
        epoch_time = time.time() - epoch_start
        
        # --- Log ---
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/macro_auc', val_auc, epoch)
        writer.add_scalar('val/macro_f1', val_f1, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Log per-dimension metrics
        for dim in nesting_dims:
            dim_auc = val_results[dim]['macro_auc']
            dim_f1 = val_results[dim]['macro_f1']
            writer.add_scalar(f'val/auc_dim{dim}', dim_auc, epoch)
            writer.add_scalar(f'val/f1_dim{dim}', dim_f1, epoch)
            
            if dim in train_dim_losses:
                writer.add_scalar(f'train/loss_dim{dim}', train_dim_losses[dim], epoch)
        
        tracker.update(epoch, {'val_auc': val_auc, 'val_f1': val_f1, 
                               'val_loss': val_loss, 'train_loss': train_loss})
        
        # --- Print epoch summary ---
        print(f"  Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Time: {epoch_time:.1f}s", end="")
        
        # --- Check for best model ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_results,
                os.path.join(ckpt_dir, f'{run_name}_best.pt')
            )
            print(" ★ BEST", end="")
        else:
            patience_counter += 1
        
        print()  # newline
        
        # Print per-dim AUC every 5 epochs
        if (epoch + 1) % 5 == 0 and mrl_enabled:
            print("  Per-dimension AUC:")
            for dim in nesting_dims:
                auc = val_results[dim]['macro_auc']
                bar = "█" * int(auc * 30)
                print(f"    dim={dim:3d}: {auc:.4f} |{bar}")
            print()
        
        # Save periodic checkpoint
        save_every = config['logging'].get('save_every_n_epochs', 10)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_results,
                os.path.join(ckpt_dir, f'{run_name}_epoch{epoch+1}.pt')
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  ⚠ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # ============================================================
    # Final Test Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("  FINAL TEST EVALUATION")
    print("=" * 70)
    
    # Load best model
    best_ckpt = os.path.join(ckpt_dir, f'{run_name}_best.pt')
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Loaded best model from epoch {ckpt['epoch']+1}")
    
    test_results = evaluate(model, test_loader, device, config, nesting_dims)
    
    print(f"\n  Test Loss: {test_results['loss']:.4f}\n")
    
    # Print results for each nesting dimension
    all_test_metrics = {}
    for dim in nesting_dims:
        print(f"  ─── Dimension {dim} ───")
        print_metrics(test_results[dim], dim=dim, class_names=class_names)
        all_test_metrics[dim] = test_results[dim]
        print()
    
    # ============================================================
    # Save Final Results
    # ============================================================
    results_summary = {
        'run_name': run_name,
        'config': config,
        'best_epoch': best_epoch + 1,
        'best_val_auc': best_val_auc,
        'test_results': all_test_metrics,
        'nesting_dims': nesting_dims,
        'class_names': class_names,
        'training_history': tracker.history,
    }
    
    results_path = os.path.join('results', f'{run_name}_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_summary, f)
    
    # Also save a readable summary
    summary_path = os.path.join('results', f'{run_name}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Matryoshka ECG - Results Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Run:        {run_name}\n")
        f.write(f"Backbone:   {config['model']['backbone']}\n")
        f.write(f"MRL:        {'Enabled' if mrl_enabled else 'Disabled'}\n")
        f.write(f"Best epoch: {best_epoch+1}\n")
        f.write(f"Best val AUC: {best_val_auc:.4f}\n\n")
        
        f.write(f"Test Results by Dimension:\n")
        f.write(f"{'-'*50}\n")
        f.write(f"{'Dim':>6s} | {'AUC':>8s} | {'F1':>8s} | {'AP':>8s}\n")
        f.write(f"{'-'*50}\n")
        for dim in nesting_dims:
            m = all_test_metrics[dim]
            f.write(f"{dim:6d} | {m['macro_auc']:8.4f} | {m['macro_f1']:8.4f} | {m['macro_ap']:8.4f}\n")
    
    print(f"\n  Results saved to: {results_path}")
    print(f"  Summary saved to: {summary_path}")
    print(f"  TensorBoard:      tensorboard --logdir {config['logging']['log_dir']}")
    
    writer.close()
    
    print("\n" + "=" * 70)
    print("  ✓ TRAINING COMPLETE")
    print(f"  Best Val AUC: {best_val_auc:.4f} (epoch {best_epoch+1})")
    
    if mrl_enabled:
        print(f"\n  Test AUC by nesting dimension:")
        for dim in nesting_dims:
            auc = all_test_metrics[dim]['macro_auc']
            bar = "█" * int(auc * 40)
            print(f"    dim={dim:3d}: {auc:.4f} |{bar}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
