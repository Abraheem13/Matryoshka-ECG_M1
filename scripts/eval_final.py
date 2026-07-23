import os, sys, glob, pickle, yaml, numpy as np, torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import ECGDataModule
from models.mrl_ecg_model import MatryoshkaECGModel
from utils.metrics import compute_all_metrics, print_metrics

def main():
    with open('configs/mrl_resnet1d.yaml', 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    nesting_dims = config['mrl']['nesting_dims']
    ckpt_dir = config['logging']['checkpoint_dir']
    best_files = glob.glob(os.path.join(ckpt_dir, '*_best.pt'))
    if not best_files:
        print("No checkpoint found!"); sys.exit(1)
    best_ckpt = best_files[0]
    print(f"Loading checkpoint: {best_ckpt}")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    print(f"Best model from epoch {ckpt['epoch'] + 1}")
    model = MatryoshkaECGModel(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    data_module = ECGDataModule(data_dir="data/processed", config=config)
    _, _, test_loader = data_module.get_dataloaders()
    with open('data/processed/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    class_names = [metadata['idx_to_class'][i] for i in range(metadata['num_classes'])]
    print("\nRunning test evaluation...")
    all_embeddings, all_targets = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="  Test"):
            x = x.to(device, non_blocking=True)
            embedding = model.get_embedding(x)
            all_embeddings.append(embedding.cpu())
            all_targets.append(y)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    print("\n" + "=" * 70)
    print("  FINAL TEST RESULTS")
    print("=" * 70)
    results_table = []
    for dim in nesting_dims:
        logits, probs = model.head.get_predictions(all_embeddings.to(device), dim=dim)
        probs_np = probs.detach().cpu().numpy()
        metrics = compute_all_metrics(all_targets, probs_np, class_names=class_names)
        results_table.append({'dim': dim, 'macro_auc': metrics['macro_auc'], 'macro_f1': metrics['macro_f1'], 'macro_ap': metrics['macro_ap'], 'per_class_auc': metrics['per_class_auc']})
        print(f"\n  --- Dimension {dim} ---")
        print_metrics(metrics, dim=dim, class_names=class_names)
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE (for paper)")
    print("=" * 70)
    print(f"\n  {'Dim':>6s} | {'AUC':>8s} | {'F1':>8s} | {'AP':>8s} | {'Storage':>10s} | {'Ratio':>6s}")
    print(f"  {'-'*58}")
    for r in results_table:
        storage = r['dim'] * 4
        ratio = f"{512 / r['dim']:.0f}x"
        print(f"  {r['dim']:6d} | {r['macro_auc']:8.4f} | {r['macro_f1']:8.4f} | {r['macro_ap']:8.4f} | {storage:>7d} B | {ratio:>6s}")
    print(f"\n  Per-class AUC:")
    print(f"  {'Dim':>6s} | " + " | ".join(f"{c:>6s}" for c in class_names))
    print(f"  {'-'*58}")
    for r in results_table:
        aucs = " | ".join(f"{r['per_class_auc'].get(i, 0):6.4f}" for i in range(len(class_names)))
        print(f"  {r['dim']:6d} | {aucs}")
    best_small = max(results_table[:3], key=lambda x: x['macro_auc'])
    full_dim = results_table[-1]
    auc_drop = full_dim['macro_auc'] - best_small['macro_auc']
    print(f"\n  KEY FINDING:")
    print(f"  dim={best_small['dim']} achieves {best_small['macro_auc']:.4f} AUC")
    print(f"  dim=512 achieves {full_dim['macro_auc']:.4f} AUC")
    print(f"  AUC difference: {abs(auc_drop):.4f} ({abs(auc_drop)*100:.2f}%)")
    print(f"  Storage reduction: {512/best_small['dim']:.0f}x smaller embedding")
    print(f"  -> A single MRL model replaces {len(nesting_dims)} separate models!")
    results_path = 'results/test_results_final.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump({'results_table': results_table, 'class_names': class_names, 'nesting_dims': nesting_dims, 'best_epoch': ckpt['epoch'] + 1}, f)
    print(f"\n  Results saved to: {results_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
