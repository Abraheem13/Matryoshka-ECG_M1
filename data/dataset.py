"""
ECG Dataset
============
PyTorch Dataset for preprocessed PTB-XL data with augmentations.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    ECG Dataset for PTB-XL.
    
    Expects preprocessed .npy files with shape:
        X: (N, 12, 1000) - 12-lead ECG, 1000 samples at 100Hz
        y: (N, C) - multi-hot encoded labels
    """
    
    def __init__(self, X_path, y_path, augment=False, config=None):
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)
        self.augment = augment
        self.config = config or {}
        
        aug_cfg = self.config.get('preprocessing', {}).get('augmentation', {})
        self.noise_std = aug_cfg.get('gaussian_noise_std', 0.01)
        self.scale_range = aug_cfg.get('random_scale', [0.95, 1.05])
        self.shift_max = aug_cfg.get('random_shift', 10)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (12, 1000)
        y = self.y[idx]
        
        if self.augment:
            x = self._augment(x)
        
        return torch.from_numpy(x), torch.from_numpy(y)
    
    def _augment(self, x):
        """Apply data augmentation to ECG signal."""
        # 1. Gaussian noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
            x = x + noise
        
        # 2. Random amplitude scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            x = x * scale
        
        # 3. Random temporal shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            x = np.roll(x, shift, axis=-1)
        
        # 4. Random lead dropout (zero out 1-2 leads) 
        if np.random.random() < 0.1:
            num_drop = np.random.randint(1, 3)
            drop_leads = np.random.choice(x.shape[0], num_drop, replace=False)
            x[drop_leads] = 0.0
        
        return x


class ECGDataModule:
    """
    Data module that creates train/val/test dataloaders.
    """
    
    def __init__(self, data_dir="data/processed", config=None):
        self.data_dir = data_dir
        self.config = config or {}
        
        train_cfg = config.get('training', {})
        self.batch_size = train_cfg.get('batch_size', 128)
        self.num_workers = train_cfg.get('num_workers', 4)
        self.pin_memory = train_cfg.get('pin_memory', True)
    
    def get_datasets(self):
        """Create train/val/test datasets."""
        import os
        
        train_ds = ECGDataset(
            os.path.join(self.data_dir, 'X_train.npy'),
            os.path.join(self.data_dir, 'y_train.npy'),
            augment=True,
            config=self.config
        )
        
        val_ds = ECGDataset(
            os.path.join(self.data_dir, 'X_val.npy'),
            os.path.join(self.data_dir, 'y_val.npy'),
            augment=False,
            config=self.config
        )
        
        test_ds = ECGDataset(
            os.path.join(self.data_dir, 'X_test.npy'),
            os.path.join(self.data_dir, 'y_test.npy'),
            augment=False,
            config=self.config
        )
        
        return train_ds, val_ds, test_ds
    
    def get_dataloaders(self):
        """Create train/val/test dataloaders."""
        from torch.utils.data import DataLoader
        
        train_ds, val_ds, test_ds = self.get_datasets()
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            test_ds, 
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader, test_loader
