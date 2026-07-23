"""
PTB-XL Data Preprocessing Script
=================================
Loads raw PTB-XL data, extracts labels, applies preprocessing,
and saves processed numpy arrays for fast training.

Usage: python scripts/preprocess_ptbxl.py
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import pickle
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path="configs/mrl_resnet1d.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_data(df, sampling_rate, path):
    """Load raw ECG waveforms using wfdb."""
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f.replace('.dat', ''))) 
                for f in tqdm(df.filename_lr, desc="Loading 100Hz signals")]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f.replace('.dat', ''))) 
                for f in tqdm(df.filename_hr, desc="Loading 500Hz signals")]
    data = np.array([signal for signal, meta in data])
    return data


def compute_label_aggregation(df, task="superdiagnostic"):
    """
    Aggregate SCP codes into diagnostic categories.
    
    Tasks:
        - superdiagnostic: 5 classes (NORM, MI, STTC, CD, HYP)
        - diagnostic: 44 classes
        - subdiagnostic: 23 classes
        - form: 19 classes
        - rhythm: 12 classes
    """
    # Load SCP statements mapping
    agg_df = pd.read_csv(os.path.join(df.attrs['path'], 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def aggregate_diagnostic(y_dic):
        """Map SCP codes to superdiagnostic classes."""
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                if task == "superdiagnostic":
                    cat = agg_df.loc[key].diagnostic_class
                elif task == "subdiagnostic":
                    cat = agg_df.loc[key].diagnostic_subclass
                elif task == "diagnostic":
                    cat = key
                else:
                    cat = key
                if isinstance(cat, str):
                    tmp.append(cat)
        return list(set(tmp))
    
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)
    return df


def encode_labels(df, task="superdiagnostic"):
    """
    Multi-hot encode the diagnostic labels.
    Returns label matrix and class mapping.
    """
    # Get all unique classes
    all_classes = set()
    for classes in df['diagnostic_superclass']:
        all_classes.update(classes)
    all_classes = sorted(list(all_classes))
    
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    # Create multi-hot encoding
    num_samples = len(df)
    num_classes = len(all_classes)
    labels = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    for i, classes in enumerate(df['diagnostic_superclass']):
        for c in classes:
            if c in class_to_idx:
                labels[i, class_to_idx[c]] = 1.0
    
    return labels, class_to_idx, idx_to_class


def normalize_signals(data):
    """Z-score normalization per lead, per sample."""
    normalized = np.zeros_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):  # per lead
            lead = data[i, :, j]
            mean = np.mean(lead)
            std = np.std(lead)
            if std > 1e-8:
                normalized[i, :, j] = (lead - mean) / std
            else:
                normalized[i, :, j] = lead - mean
    return normalized


def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=100, order=4):
    """Apply bandpass filter to remove baseline wander and high-freq noise."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    filtered = np.zeros_like(data)
    for i in tqdm(range(data.shape[0]), desc="Filtering"):
        for j in range(data.shape[2]):
            filtered[i, :, j] = filtfilt(b, a, data[i, :, j])
    return filtered


def main():
    config = load_config()
    
    base_path = config['dataset']['path']
    sampling_rate = config['dataset']['sampling_rate']
    task = config['dataset']['task']
    test_fold = config['dataset']['test_fold']
    val_fold = config['dataset']['val_fold']
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  PTB-XL Preprocessing")
    print("=" * 60)
    print(f"  Base path:      {base_path}")
    print(f"  Sampling rate:  {sampling_rate} Hz")
    print(f"  Task:           {task}")
    print(f"  Test fold:      {test_fold}")
    print(f"  Val fold:       {val_fold}")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # Step 1: Load metadata
    # ----------------------------------------------------------
    print("\n[1/6] Loading metadata...")
    df = pd.read_csv(os.path.join(base_path, 'ptbxl_database.csv'), index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    df.attrs['path'] = base_path
    print(f"   Total records: {len(df)}")
    
    # ----------------------------------------------------------
    # Step 2: Load raw signals
    # ----------------------------------------------------------
    print("\n[2/6] Loading raw ECG signals...")
    data = load_raw_data(df, sampling_rate, base_path)
    print(f"   Signal shape: {data.shape}")  # (N, 1000, 12) for 100Hz
    
    # ----------------------------------------------------------
    # Step 3: Compute labels
    # ----------------------------------------------------------
    print("\n[3/6] Computing diagnostic labels...")
    df = compute_label_aggregation(df, task=task)
    
    # Filter out samples with no diagnostic label
    valid_mask = df['diagnostic_superclass'].apply(lambda x: len(x) > 0)
    df = df[valid_mask]
    data = data[valid_mask.values]
    
    labels, class_to_idx, idx_to_class = encode_labels(df, task=task)
    
    num_classes = len(class_to_idx)
    print(f"   Valid samples: {len(df)}")
    print(f"   Num classes:   {num_classes}")
    print(f"   Classes:       {class_to_idx}")
    
    # Print class distribution
    print("\n   Class distribution:")
    for cls_name, cls_idx in sorted(class_to_idx.items()):
        count = int(labels[:, cls_idx].sum())
        pct = 100 * count / len(labels)
        print(f"     {cls_name:6s}: {count:5d} ({pct:.1f}%)")
    
    # ----------------------------------------------------------
    # Step 4: Normalize
    # ----------------------------------------------------------
    print("\n[4/6] Normalizing signals...")
    data = normalize_signals(data)
    print(f"   Signal stats: mean={data.mean():.4f}, std={data.std():.4f}")
    
    # ----------------------------------------------------------
    # Step 5: Split into train/val/test using official folds
    # ----------------------------------------------------------
    print("\n[5/6] Splitting data...")
    
    # Get fold assignments
    folds = df.strat_fold.values
    
    train_mask = np.isin(folds, list(range(1, val_fold)))  # folds 1-8
    val_mask = folds == val_fold                              # fold 9
    test_mask = folds == test_fold                             # fold 10
    
    X_train, y_train = data[train_mask], labels[train_mask]
    X_val, y_val = data[val_mask], labels[val_mask]
    X_test, y_test = data[test_mask], labels[test_mask]
    
    print(f"   Train: {X_train.shape[0]:5d} samples")
    print(f"   Val:   {X_val.shape[0]:5d} samples")
    print(f"   Test:  {X_test.shape[0]:5d} samples")
    
    # ----------------------------------------------------------
    # Step 6: Save processed data
    # ----------------------------------------------------------
    print("\n[6/6] Saving processed data...")
    
    # Transpose to (N, Channels, Length) format for PyTorch Conv1d
    X_train = X_train.transpose(0, 2, 1).astype(np.float32)  # (N, 12, 1000)
    X_val = X_val.transpose(0, 2, 1).astype(np.float32)
    X_test = X_test.transpose(0, 2, 1).astype(np.float32)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save metadata
    metadata = {
        'task': task,
        'sampling_rate': sampling_rate,
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_leads': 12,
        'signal_length': X_train.shape[2],
        'train_size': X_train.shape[0],
        'val_size': X_val.shape[0],
        'test_size': X_test.shape[0],
        'class_distribution': {
            cls_name: int(labels[:, cls_idx].sum()) 
            for cls_name, cls_idx in class_to_idx.items()
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Save shapes for quick reference
    with open(os.path.join(output_dir, 'shapes.txt'), 'w') as f:
        f.write(f"X_train: {X_train.shape}\n")
        f.write(f"y_train: {y_train.shape}\n")
        f.write(f"X_val:   {X_val.shape}\n")
        f.write(f"y_val:   {y_val.shape}\n")
        f.write(f"X_test:  {X_test.shape}\n")
        f.write(f"y_test:  {y_test.shape}\n")
        f.write(f"\nClasses: {class_to_idx}\n")
    
    total_size_mb = (X_train.nbytes + X_val.nbytes + X_test.nbytes + 
                     y_train.nbytes + y_val.nbytes + y_test.nbytes) / 1e6
    
    print(f"   Total saved: {total_size_mb:.1f} MB")
    print(f"   Output dir:  {output_dir}/")
    
    print("\n" + "=" * 60)
    print("  ✓ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n  Final shapes:")
    print(f"    X_train: {X_train.shape}  (N, 12_leads, 1000_samples)")
    print(f"    y_train: {y_train.shape}  (N, {num_classes}_classes)")
    print(f"\n  Ready for training!")


if __name__ == "__main__":
    main()
