"""
Matryoshka ECG Model
=====================
Complete model combining backbone + MRL/Standard classification head.
"""

import torch
import torch.nn as nn
from models.xresnet1d import create_backbone
from losses.matryoshka_loss import MatryoshkaLoss, StandardLoss


class MatryoshkaECGModel(nn.Module):
    """
    Complete Matryoshka ECG classification model.
    
    Architecture:
        Input (B, 12, 1000) 
        -> Backbone (XResNet1D / Inception1D) 
        -> Embedding (B, embedding_dim)
        -> MRL Head: {dim -> classifier(z[:dim]) for dim in nesting_dims}
        -> Per-dimension predictions
    
    Args:
        config: Full experiment config dict
    """
    
    def __init__(self, config):
        super().__init__()
        
        model_cfg = config.get('model', {})
        mrl_cfg = config.get('mrl', {})
        
        self.backbone_name = model_cfg.get('backbone', 'xresnet1d101')
        self.input_channels = model_cfg.get('input_channels', 12)
        self.embedding_dim = model_cfg.get('base_embedding_dim', 512)
        self.num_classes = model_cfg.get('num_classes', 5)
        self.dropout = model_cfg.get('dropout', 0.3)
        
        self.mrl_enabled = mrl_cfg.get('enabled', True)
        self.nesting_dims = mrl_cfg.get('nesting_dims', [16, 32, 64, 128, 256, 512])
        
        # Create backbone
        self.backbone = create_backbone(
            name=self.backbone_name,
            input_channels=self.input_channels,
            embedding_dim=self.embedding_dim,
            dropout=self.dropout
        )
        
        # Create classification head
        if self.mrl_enabled:
            self.head = MatryoshkaLoss(
                nesting_dims=self.nesting_dims,
                num_classes=self.num_classes,
                weight_strategy=mrl_cfg.get('loss_weights', 'equal'),
                label_smoothing=config.get('training', {}).get('label_smoothing', 0.1),
                multi_label=True
            )
            print(f"  MRL head: dims={self.nesting_dims}")
        else:
            self.head = StandardLoss(
                embedding_dim=self.embedding_dim,
                num_classes=self.num_classes,
                label_smoothing=config.get('training', {}).get('label_smoothing', 0.1),
                multi_label=True
            )
            print(f"  Standard head: dim={self.embedding_dim}")
        
        # Print total params
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        print(f"  Total params: {total:,} (backbone: {backbone_params:,}, head: {head_params:,})")
    
    def forward(self, x, targets=None):
        """
        Forward pass.
        
        Args:
            x: (batch, 12, 1000) ECG signal
            targets: (batch, num_classes) multi-hot labels (optional)
            
        Returns:
            If targets provided: loss dict
            If no targets: embedding
        """
        embedding = self.backbone(x)  # (batch, embedding_dim)
        
        if targets is not None:
            return self.head(embedding, targets)
        
        return embedding
    
    def get_embedding(self, x):
        """Get the full embedding without classification."""
        return self.backbone(x)
    
    def predict(self, x, dim=None):
        """
        Get predictions at a specific nesting dimension.
        
        Args:
            x: (batch, 12, 1000) ECG signal
            dim: nesting dimension (None = use full embedding)
            
        Returns:
            logits, probabilities
        """
        embedding = self.backbone(x)
        return self.head.get_predictions(embedding, dim=dim)


class FixedDimECGModel(nn.Module):
    """
    Fixed-dimension baseline model.
    Trains a backbone with a specific (small) embedding dimension.
    Used for comparison against MRL.
    """
    
    def __init__(self, config, fixed_dim=64):
        super().__init__()
        
        model_cfg = config.get('model', {})
        
        self.backbone = create_backbone(
            name=model_cfg.get('backbone', 'xresnet1d101'),
            input_channels=model_cfg.get('input_channels', 12),
            embedding_dim=fixed_dim,  # Use fixed small dim
            dropout=model_cfg.get('dropout', 0.3)
        )
        
        self.head = StandardLoss(
            embedding_dim=fixed_dim,
            num_classes=model_cfg.get('num_classes', 5),
            label_smoothing=config.get('training', {}).get('label_smoothing', 0.1),
            multi_label=True
        )
        
        self.fixed_dim = fixed_dim
    
    def forward(self, x, targets=None):
        embedding = self.backbone(x)
        if targets is not None:
            return self.head(embedding, targets)
        return embedding
    
    def predict(self, x, dim=None):
        embedding = self.backbone(x)
        return self.head.get_predictions(embedding)


if __name__ == "__main__":
    import yaml
    
    with open('configs/mrl_resnet1d.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("Testing MatryoshkaECGModel")
    print("=" * 50)
    
    model = MatryoshkaECGModel(config)
    
    x = torch.randn(4, 12, 1000)
    y = torch.randint(0, 2, (4, 5)).float()
    
    # Training forward pass
    result = model(x, y)
    print(f"\n  Total loss: {result['total_loss'].item():.4f}")
    
    # Inference at different dims
    model.eval()
    with torch.no_grad():
        for dim in [16, 64, 256, 512]:
            logits, probs = model.predict(x, dim=dim)
            print(f"  Predict at dim={dim}: probs shape={probs.shape}")
    
    print("\n  ✓ Model test passed!")
