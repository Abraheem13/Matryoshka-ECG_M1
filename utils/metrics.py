"""
Evaluation Metrics
==================
Macro-AUC, macro-F1, per-class AUC for PTB-XL evaluation.
Follows the official PTB-XL benchmarking protocol.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, 
    average_precision_score, classification_report
)
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def compute_macro_auc(y_true, y_prob):
    """
    Compute macro-averaged AUC-ROC.
    This is the PRIMARY metric for PTB-XL benchmarking.
    
    Args:
        y_true: (N, C) ground truth multi-hot labels
        y_prob: (N, C) predicted probabilities
    
    Returns:
        macro_auc: float
        per_class_auc: dict of {class_idx: auc}
    """
    num_classes = y_true.shape[1]
    per_class_auc = {}
    valid_aucs = []
    
    for i in range(num_classes):
        if y_true[:, i].sum() > 0 and y_true[:, i].sum() < len(y_true):
            try:
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                per_class_auc[i] = auc
                valid_aucs.append(auc)
            except ValueError:
                per_class_auc[i] = 0.0
        else:
            per_class_auc[i] = 0.0
    
    macro_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    return macro_auc, per_class_auc


def compute_macro_f1(y_true, y_prob, threshold=0.5):
    """
    Compute macro-averaged F1 score.
    
    Args:
        y_true: (N, C) ground truth
        y_prob: (N, C) predicted probabilities
        threshold: classification threshold
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Per-sample at least one label
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    return macro_f1, micro_f1


def compute_all_metrics(y_true, y_prob, class_names=None, threshold=0.5):
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: (N, C) np.ndarray
        y_prob: (N, C) np.ndarray
        class_names: list of class names (optional)
        threshold: classification threshold
        
    Returns:
        Dictionary with all metrics
    """
    # AUC
    macro_auc, per_class_auc = compute_macro_auc(y_true, y_prob)
    
    # F1
    macro_f1, micro_f1 = compute_macro_f1(y_true, y_prob, threshold)
    
    # Average Precision
    try:
        macro_ap = average_precision_score(y_true, y_prob, average='macro')
    except ValueError:
        macro_ap = 0.0
    
    # Subset accuracy (exact match)
    y_pred = (y_prob >= threshold).astype(int)
    subset_acc = accuracy_score(y_true, y_pred)
    
    metrics = {
        'macro_auc': macro_auc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'macro_ap': macro_ap,
        'subset_accuracy': subset_acc,
        'per_class_auc': per_class_auc,
    }
    
    # Add class names if available
    if class_names:
        for idx, name in enumerate(class_names):
            metrics[f'auc_{name}'] = per_class_auc.get(idx, 0.0)
    
    return metrics


def print_metrics(metrics, dim=None, class_names=None):
    """Pretty print metrics."""
    prefix = f"[dim={dim}] " if dim else ""
    
    print(f"  {prefix}Macro AUC:      {metrics['macro_auc']:.4f}")
    print(f"  {prefix}Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"  {prefix}Macro AP:       {metrics['macro_ap']:.4f}")
    print(f"  {prefix}Subset Acc:     {metrics['subset_accuracy']:.4f}")
    
    if class_names:
        print(f"  {prefix}Per-class AUC:")
        for idx, name in enumerate(class_names):
            auc = metrics['per_class_auc'].get(idx, 0.0)
            print(f"    {name:6s}: {auc:.4f}")


class MetricTracker:
    """Track metrics across training epochs."""
    
    def __init__(self):
        self.history = {}
    
    def update(self, epoch, metrics_dict, prefix=""):
        """Store metrics for an epoch."""
        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                continue  # Skip nested dicts
            full_key = f"{prefix}{key}" if prefix else key
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append((epoch, value))
    
    def get_best(self, metric_name, mode='max'):
        """Get best epoch for a metric."""
        if metric_name not in self.history:
            return None, None
        
        values = self.history[metric_name]
        if mode == 'max':
            best_epoch, best_val = max(values, key=lambda x: x[1])
        else:
            best_epoch, best_val = min(values, key=lambda x: x[1])
        
        return best_epoch, best_val
    
    def get_latest(self, metric_name):
        """Get latest value for a metric."""
        if metric_name not in self.history or not self.history[metric_name]:
            return None
        return self.history[metric_name][-1][1]
