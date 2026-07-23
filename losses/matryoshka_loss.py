"""
Matryoshka Representation Learning (MRL) Loss
==============================================
Implementation of the nested loss from:
"Matryoshka Representation Learning" (Kusupati et al., NeurIPS 2022)

The key idea: train a single model where the first m dimensions of 
the embedding are independently useful for m ∈ {16, 32, 64, 128, 256, 512}.
This is achieved by applying classification losses at each nesting dimension.

Total loss = Σ_{m ∈ M} w_m · L(classifier_m(z[:m]), y)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


class MatryoshkaLoss(nn.Module):
    """
    Matryoshka multi-granularity loss.
    
    For each nesting dimension m, applies a separate classification loss
    on the first m dimensions of the embedding. The total loss is a 
    weighted sum across all nesting dimensions.
    
    Args:
        nesting_dims: List of nesting dimensions [16, 32, 64, 128, 256, 512]
        num_classes: Number of output classes
        weight_strategy: How to weight losses ("equal", "linear", "exponential")
        label_smoothing: Label smoothing factor
        multi_label: Whether to use BCE (multi-label) or CE (single-label)
    """
    
    def __init__(
        self,
        nesting_dims: List[int] = [16, 32, 64, 128, 256, 512],
        num_classes: int = 5,
        weight_strategy: str = "equal",
        label_smoothing: float = 0.1,
        multi_label: bool = True  # PTB-XL is multi-label
    ):
        super().__init__()
        
        self.nesting_dims = sorted(nesting_dims)
        self.num_classes = num_classes
        self.multi_label = multi_label
        
        # Create independent classification heads for each nesting dim
        self.classifiers = nn.ModuleDict({
            str(dim): nn.Linear(dim, num_classes)
            for dim in self.nesting_dims
        })
        
        # Initialize classifiers
        for clf in self.classifiers.values():
            nn.init.xavier_uniform_(clf.weight)
            nn.init.zeros_(clf.bias)
        
        # Compute loss weights
        self.loss_weights = self._compute_weights(weight_strategy)
        
        # Loss function
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing, reduction='mean'
            )
        
        self.label_smoothing = label_smoothing
    
    def _compute_weights(self, strategy: str) -> Dict[int, float]:
        """Compute per-dimension loss weights."""
        n = len(self.nesting_dims)
        
        if strategy == "equal":
            weights = {d: 1.0 for d in self.nesting_dims}
        
        elif strategy == "linear":
            # Larger dims get higher weight
            total = sum(range(1, n + 1))
            weights = {d: (i + 1) / total 
                      for i, d in enumerate(self.nesting_dims)}
        
        elif strategy == "exponential":
            # Exponentially increasing weight
            raw = {d: 2 ** i for i, d in enumerate(self.nesting_dims)}
            total = sum(raw.values())
            weights = {d: v / total for d, v in raw.items()}
        
        else:
            raise ValueError(f"Unknown weight strategy: {strategy}")
        
        # Normalize so weights sum to len(nesting_dims) for stable gradients
        total = sum(weights.values())
        scale = n / total
        weights = {d: w * scale for d, w in weights.items()}
        
        return weights
    
    def forward(
        self, 
        embedding: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Matryoshka loss across all nesting dimensions.
        
        Args:
            embedding: (batch, max_dim) full embedding from backbone
            targets: (batch, num_classes) multi-hot labels or (batch,) class indices
            
        Returns:
            Dictionary with:
                'total_loss': weighted sum of all per-dim losses
                'per_dim_loss': dict of {dim: loss_value}
                'per_dim_logits': dict of {dim: logits} for evaluation
        """
        total_loss = torch.tensor(0.0, device=embedding.device, dtype=embedding.dtype)
        per_dim_loss = {}
        per_dim_logits = {}
        
        for dim in self.nesting_dims:
            # Truncate embedding to first `dim` dimensions
            truncated = embedding[:, :dim]
            
            # Classify using dim-specific head
            logits = self.classifiers[str(dim)](truncated)
            
            # Apply label smoothing for BCE
            if self.multi_label and self.label_smoothing > 0:
                smoothed_targets = targets * (1 - self.label_smoothing) + \
                                   self.label_smoothing / self.num_classes
                loss = self.criterion(logits, smoothed_targets)
            else:
                loss = self.criterion(logits, targets)
            
            # Accumulate weighted loss
            weight = self.loss_weights[dim]
            total_loss = total_loss + weight * loss
            
            per_dim_loss[dim] = loss.detach()
            per_dim_logits[dim] = logits.detach()
        
        # Average over number of dimensions
        total_loss = total_loss / len(self.nesting_dims)
        
        return {
            'total_loss': total_loss,
            'per_dim_loss': per_dim_loss,
            'per_dim_logits': per_dim_logits
        }
    
    def get_predictions(self, embedding: torch.Tensor, dim: Optional[int] = None):
        """
        Get predictions at a specific nesting dimension.
        If dim is None, returns predictions at the largest dimension.
        """
        if dim is None:
            dim = self.nesting_dims[-1]
        
        assert dim in self.nesting_dims, \
            f"dim={dim} not in nesting_dims={self.nesting_dims}"
        
        truncated = embedding[:, :dim]
        logits = self.classifiers[str(dim)](truncated)
        
        if self.multi_label:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=-1)
        
        return logits, probs


class StandardLoss(nn.Module):
    """
    Standard (non-Matryoshka) classification loss.
    Used for baseline fixed-dimension models.
    
    Args:
        embedding_dim: Input embedding dimension
        num_classes: Number of output classes
        multi_label: Whether to use BCE (multi-label) or CE
    """
    
    def __init__(self, embedding_dim=512, num_classes=5, 
                 label_smoothing=0.1, multi_label=True):
        super().__init__()
        
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.multi_label = multi_label
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        if multi_label:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing, reduction='mean'
            )
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, embedding, targets):
        logits = self.classifier(embedding)
        
        if self.multi_label and self.label_smoothing > 0:
            smoothed = targets * (1 - self.label_smoothing) + \
                       self.label_smoothing / self.num_classes
            loss = self.criterion(logits, smoothed)
        else:
            loss = self.criterion(logits, targets)
        
        return {
            'total_loss': loss,
            'per_dim_loss': {embedding.shape[1]: loss.detach()},
            'per_dim_logits': {embedding.shape[1]: logits.detach()}
        }
    
    def get_predictions(self, embedding, dim=None):
        logits = self.classifier(embedding)
        if self.multi_label:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=-1)
        return logits, probs


if __name__ == "__main__":
    # Quick test
    print("Testing MatryoshkaLoss...")
    
    batch_size = 8
    embedding_dim = 512
    num_classes = 5
    nesting_dims = [16, 32, 64, 128, 256, 512]
    
    mrl_loss = MatryoshkaLoss(
        nesting_dims=nesting_dims,
        num_classes=num_classes,
        weight_strategy="equal",
        multi_label=True
    )
    
    # Fake data
    embedding = torch.randn(batch_size, embedding_dim)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    
    result = mrl_loss(embedding, targets)
    
    print(f"  Total loss: {result['total_loss'].item():.4f}")
    for dim, loss in result['per_dim_loss'].items():
        print(f"  Dim {dim:3d} loss: {loss.item():.4f}")
    
    # Test prediction at different dims
    for dim in nesting_dims:
        logits, probs = mrl_loss.get_predictions(embedding, dim=dim)
        print(f"  Dim {dim:3d} predictions shape: {probs.shape}")
    
    print("\n  ✓ MatryoshkaLoss test passed!")
    
    # Test standard loss
    print("\nTesting StandardLoss...")
    std_loss = StandardLoss(embedding_dim=64, num_classes=5)
    embedding_small = torch.randn(batch_size, 64)
    result = std_loss(embedding_small, targets)
    print(f"  Total loss: {result['total_loss'].item():.4f}")
    print("  ✓ StandardLoss test passed!")
