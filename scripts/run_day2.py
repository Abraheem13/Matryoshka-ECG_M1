import os, sys, argparse, pickle, yaml, time, copy, glob
import numpy as np
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import ECGDataModule
from models.mrl_ecg_model import MatryoshkaECGModel, FixedDimECGModel
from utils.metrics import compute_all_metrics

def train_fixed_baseline(config, fixed_dim, device, train_loader, val_loader, run_name):
    print(f"\n{'='*60}\n  Training FIXED-DIM baseline: dim={fixed_dim}\n{'='*60}")
    cfg = copy.deepcopy(config)
    cfg['model']['base_embedding_dim'] = fixed_dim
    cfg['mrl']['enabled'] = False
    model = FixedDimECGModel(cfg, fixed_dim=fixed_dim).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    epochs = cfg['training']['epochs']
    warmup_steps = cfg['training'].get('warmup_epochs', 3) * len(train_loader)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s/max(warmup_steps,1) if s < warmup_steps else 0.5*(1+np.cos(np.pi*(s-warmup_steps)/max(total_steps-warmup_steps,1))))
    best_val_auc, best_state, patience_counter = 0.0, None, 0
    patience = cfg['training'].get('early_stopping_patience', 8)
    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for x, y in tqdm(train_loader, desc=f"  E{epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(x, y)
            result['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            total_loss += result['total_loss'].item(); n += 1
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                _, probs = model.head.get_predictions(model.backbone(x))
                all_p.append(probs.detach().cpu().numpy()); all_t.append(y.numpy())
        metrics = compute_all_metrics(np.concatenate(all_t), np.concatenate(all_p))
        val_auc = metrics['macro_auc']
        is_best = val_auc > best_val_auc
        if is_best: best_val_auc = val_auc; best_state = copy.deepcopy(model.state_dict()); patience_counter = 0
        else: patience_counter += 1
        print(f"  E{epoch+1:3d} | Loss:{total_loss/n:.4f} | AUC:{val_auc:.4f}{' *BEST*' if is_best else ''}")
        if patience_counter >= patience: print(f"  Early stop at epoch {epoch+1}"); break
    torch.save({'model_state_dict': best_state, 'dim': fixed_dim, 'best_val_auc': best_val_auc}, f"results/checkpoints/{run_name}_best.pt")
    model.load_state_dict(best_state)
    return model, best_val_auc

def eval_model(model, test_loader, device):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            _, probs = model.head.get_predictions(model.backbone(x))
            all_p.append(probs.detach().cpu().numpy()); all_t.append(y.numpy())
    return compute_all_metrics(np.concatenate(all_t), np.concatenate(all_p))

def svd_baseline(mrl_model, test_loader, device, dims):
    print(f"\n{'='*60}\n  SVD Compression Baseline\n{'='*60}")
    mrl_model.eval()
    all_e, all_t = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="  Embeddings"):
            all_e.append(mrl_model.get_embedding(x.to(device)).cpu().numpy()); all_t.append(y.numpy())
    E, T = np.concatenate(all_e), np.concatenate(all_t)
    mean = E.mean(axis=0); centered = E - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    results = {}
    for dim in dims:
        proj = centered @ Vt[:dim].T
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0))
        clf.fit(proj, T)
        probs = clf.predict_proba(proj)
        if isinstance(probs, list): probs = np.column_stack([p[:,1] for p in probs])
        metrics = compute_all_metrics(T, probs)
        results[dim] = metrics
        print(f"  SVD dim={dim:3d}: AUC={metrics['macro_auc']:.4f}")
    return results

def gen_figures(mrl_r, fixed_r, svd_r, dims, save_dir="results/figures"):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    mrl_a = [mrl_r[d]['macro_auc'] for d in dims]
    fix_a = [fixed_r.get(d,{}).get('macro_auc',0) for d in dims]
    svd_a = [svd_r.get(d,{}).get('macro_auc',0) for d in dims]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(dims, mrl_a, 'o-', color='#534AB7', lw=2.5, ms=8, label='MRL (ours)', zorder=5)
    ax.plot(dims, fix_a, 's--', color='#D85A30', lw=2, ms=7, label='Fixed-dim (separate models)')
    ax.plot(dims, svd_a, '^--', color='#1D9E75', lw=2, ms=7, label='SVD compression')
    ax.set_xlabel('Embedding Dimension', fontsize=13); ax.set_ylabel('Macro AUC', fontsize=13)
    ax.set_title('Accuracy-Efficiency Tradeoff: MRL vs Baselines', fontsize=14)
    ax.legend(fontsize=11, loc='lower right'); ax.set_xscale('log', base=2)
    ax.set_xticks(dims); ax.set_xticklabels([str(d) for d in dims])
    low = min(min(mrl_a), min(fix_a), min(svd_a)); ax.set_ylim(bottom=low-0.02)
    ax.grid(True, alpha=0.3); plt.tight_layout()
    fig.savefig(f'{save_dir}/fig1_pareto_frontier.png', dpi=300)
    fig.savefig(f'{save_dir}/fig1_pareto_frontier.pdf'); plt.close()
    print(f"  Saved fig1_pareto_frontier.png/pdf")
    # Heatmap
    import seaborn as sns
    classes = ['CD','HYP','MI','NORM','STTC']
    hm = np.array([[mrl_r[d]['per_class_auc'].get(j,0) for j in range(5)] for d in dims])
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(hm, annot=True, fmt='.4f', cmap='YlOrRd', xticklabels=classes,
                yticklabels=[f'd={d}' for d in dims], vmin=0.78, vmax=0.95, ax=ax, linewidths=0.5)
    ax.set_title('Per-class AUC Across Nesting Dimensions (MRL)', fontsize=14)
    plt.tight_layout(); fig.savefig(f'{save_dir}/fig3_perclass_heatmap.png', dpi=300)
    fig.savefig(f'{save_dir}/fig3_perclass_heatmap.pdf'); plt.close()
    print(f"  Saved fig3_perclass_heatmap.png/pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-epochs', type=int, default=25)
    parser.add_argument('--baseline-dims', nargs='+', type=int, default=[16, 32, 64, 128, 256, 512])
    parser.add_argument('--skip-trained', action='store_true', default=True)
    args = parser.parse_args()
    with open('configs/mrl_resnet1d.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['training']['epochs'] = args.baseline_epochs
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    nesting_dims = config['mrl']['nesting_dims']
    data_module = ECGDataModule(data_dir="data/processed", config=config)
    train_loader, val_loader, test_loader = data_module.get_dataloaders()
    with open('data/processed/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    class_names = [metadata['idx_to_class'][i] for i in range(metadata['num_classes'])]

    # Load MRL results
    print("\n[1/4] Loading MRL results...")
    with open('results/test_results_final.pkl', 'rb') as f:
        mrl_data = pickle.load(f)
    mrl_results = {r['dim']: r for r in mrl_data['results_table']}

    # Train fixed baselines
    print("\n[2/4] Training fixed-dim baselines...")
    fixed_results = {}
    for dim in args.baseline_dims:
        ckpt_path = f"results/checkpoints/baseline_dim{dim}_best.pt"
        if os.path.exists(ckpt_path) and args.skip_trained:
            print(f"\n  Baseline dim={dim} already exists, loading...")
            cfg = copy.deepcopy(config); cfg['model']['base_embedding_dim'] = dim
            model = FixedDimECGModel(cfg, fixed_dim=dim).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            metrics = eval_model(model, test_loader, device)
            fixed_results[dim] = metrics
            print(f"  dim={dim}: Test AUC={metrics['macro_auc']:.4f}")
            continue
        model, _ = train_fixed_baseline(config, dim, device, train_loader, val_loader, f"baseline_dim{dim}")
        metrics = eval_model(model, test_loader, device)
        fixed_results[dim] = metrics
        print(f"  dim={dim}: Test AUC={metrics['macro_auc']:.4f}")

    # SVD baseline
    print("\n[3/4] SVD baseline...")
    mrl_ckpts = glob.glob('results/checkpoints/*mrl*_best.pt')
    mrl_model = MatryoshkaECGModel(config).to(device)
    ckpt = torch.load(mrl_ckpts[0], map_location=device, weights_only=False)
    mrl_model.load_state_dict(ckpt['model_state_dict']); mrl_model.eval()
    svd_results = svd_baseline(mrl_model, test_loader, device, nesting_dims)

    # Comparison table
    print(f"\n{'='*70}\n  PAPER TABLE: MRL vs Fixed-Dim vs SVD\n{'='*70}")
    print(f"\n  {'Dim':>5s} | {'MRL AUC':>9s} | {'Fixed AUC':>10s} | {'SVD AUC':>9s} | {'MRL-Fixed':>10s}")
    print(f"  {'-'*55}")
    for dim in nesting_dims:
        m = mrl_results[dim]['macro_auc']; f = fixed_results.get(dim,{}).get('macro_auc',0)
        s = svd_results.get(dim,{}).get('macro_auc',0)
        diff = m - f; sign = '+' if diff >= 0 else ''
        print(f"  {dim:5d} | {m:9.4f} | {f:10.4f} | {s:9.4f} | {sign}{diff:9.4f}")
    m16 = mrl_results[16]['macro_auc']; m512 = mrl_results[512]['macro_auc']
    f16 = fixed_results.get(16,{}).get('macro_auc',0)
    print(f"\n  KEY FINDINGS:")
    print(f"  1. MRL d=16 ({m16:.4f}) vs Fixed d=16 ({f16:.4f}): gap={abs(m16-f16):.4f}")
    print(f"  2. MRL d=16 ({m16:.4f}) vs MRL d=512 ({m512:.4f}): only {abs(m16-m512):.4f} drop, 32x compression")
    print(f"  3. One MRL model replaces {len(nesting_dims)} separate models")

    # Figures
    print("\n[4/4] Generating figures...")
    gen_figures(mrl_results, fixed_results, svd_results, nesting_dims)

    pickle.dump({'mrl': mrl_results, 'fixed': fixed_results, 'svd': svd_results,
                 'nesting_dims': nesting_dims, 'class_names': class_names},
                open('results/day2_all_results.pkl', 'wb'))
    print(f"\n{'='*70}\n  DAY 2 COMPLETE\n{'='*70}\n")

if __name__ == "__main__":
    main()
