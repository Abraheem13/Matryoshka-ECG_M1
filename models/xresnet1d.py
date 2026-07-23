"""
XResNet1D - 1D Residual Network for ECG Classification
=======================================================
Adapted from the fastai/tsai xresnet1d implementation.
This is the SOTA backbone for PTB-XL classification.

Architecture: Conv1d-based ResNet with pre-activation blocks,
squeeze-and-excitation, and global average pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Building Blocks
# ============================================================

class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation block for 1D signals."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class ConvBlock1d(nn.Module):
    """Conv1d + BatchNorm + ReLU block."""
    
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, 
                 groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, 
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch)
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class ResBlock1d(nn.Module):
    """
    Pre-activation residual block for 1D signals.
    Uses bottleneck design for deeper networks.
    """
    
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, se=True):
        super().__init__()
        
        self.conv1 = ConvBlock1d(in_ch, mid_ch, kernel_size=1, stride=1)
        self.conv2 = ConvBlock1d(mid_ch, mid_ch, kernel_size=3, stride=stride)
        self.conv3 = ConvBlock1d(mid_ch, out_ch, kernel_size=1, stride=1, act=False)
        
        self.se = SEBlock1d(out_ch) if se else nn.Identity()
        
        # Skip connection
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool1d(stride, stride=stride, ceil_mode=True) if stride > 1 else nn.Identity(),
                ConvBlock1d(in_ch, out_ch, kernel_size=1, stride=1, act=False)
            )
        else:
            self.shortcut = nn.Identity()
        
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)
        out = out + identity
        out = self.act(out)
        return out


# ============================================================
# XResNet1D Backbone
# ============================================================

class XResNet1D(nn.Module):
    """
    XResNet1D backbone for ECG signals.
    
    Produces a fixed-size embedding from variable-length 1D input.
    The classification head is NOT included here - that's handled
    by the MRL wrapper.
    
    Args:
        input_channels: Number of input channels (12 for 12-lead ECG)
        base_filters: Base number of filters (default: 64)
        layers: Number of blocks per stage (e.g., [3,4,23,3] for resnet101)
        embedding_dim: Output embedding dimension
    """
    
    def __init__(self, input_channels=12, base_filters=64, 
                 layers=[3, 4, 23, 3], embedding_dim=512, dropout=0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        bf = base_filters
        
        # --- Stem: 3 conv layers to process raw ECG ---
        self.stem = nn.Sequential(
            ConvBlock1d(input_channels, bf // 2, kernel_size=7, stride=2),
            ConvBlock1d(bf // 2, bf // 2, kernel_size=3, stride=1),
            ConvBlock1d(bf // 2, bf, kernel_size=3, stride=1),
            nn.MaxPool1d(3, stride=2, padding=1)
        )
        
        # --- Residual stages ---
        # Stage 1: bf -> bf*4
        self.stage1 = self._make_stage(bf, bf, bf * 4, layers[0], stride=1)
        # Stage 2: bf*4 -> bf*8
        self.stage2 = self._make_stage(bf * 4, bf * 2, bf * 8, layers[1], stride=2)
        # Stage 3: bf*8 -> bf*16
        self.stage3 = self._make_stage(bf * 8, bf * 4, bf * 16, layers[2], stride=2)
        # Stage 4: bf*16 -> bf*32
        self.stage4 = self._make_stage(bf * 16, bf * 8, bf * 32, layers[3], stride=2)
        
        # --- Global pooling + embedding projection ---
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(bf * 32, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _make_stage(self, in_ch, mid_ch, out_ch, num_blocks, stride=1):
        layers = [ResBlock1d(in_ch, mid_ch, out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock1d(out_ch, mid_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 12, 1000) - 12-lead ECG
        Returns:
            embedding: (batch, embedding_dim)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)  # (batch, bf*32)
        x = self.dropout(x)
        x = self.fc(x)                # (batch, embedding_dim)
        x = self.bn(x)
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Inception1D Backbone (alternative)
# ============================================================

class InceptionBlock1d(nn.Module):
    """Single Inception block with multiple kernel sizes."""
    
    def __init__(self, in_ch, out_ch, bottleneck_ch=32):
        super().__init__()
        
        self.bottleneck = ConvBlock1d(in_ch, bottleneck_ch, kernel_size=1)
        
        self.branch1 = ConvBlock1d(bottleneck_ch, out_ch, kernel_size=9, act=False)
        self.branch2 = ConvBlock1d(bottleneck_ch, out_ch, kernel_size=19, act=False)
        self.branch3 = ConvBlock1d(bottleneck_ch, out_ch, kernel_size=39, act=False)
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            ConvBlock1d(in_ch, out_ch, kernel_size=1, act=False)
        )
        
        self.bn = nn.BatchNorm1d(out_ch * 4)
        self.act = nn.ReLU(inplace=True)
        
        # Shortcut
        self.shortcut = ConvBlock1d(in_ch, out_ch * 4, kernel_size=1, act=False) \
                        if in_ch != out_ch * 4 else nn.Identity()
    
    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([
            self.branch1(b),
            self.branch2(b),
            self.branch3(b),
            self.branch_pool(x)
        ], dim=1)
        out = self.bn(out)
        out = out + self.shortcut(x)
        out = self.act(out)
        return out


class Inception1D(nn.Module):
    """Inception1D backbone for ECG."""
    
    def __init__(self, input_channels=12, embedding_dim=512, 
                 num_blocks=6, base_filters=32, dropout=0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        blocks = []
        in_ch = input_channels
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** min(i, 3))
            blocks.append(InceptionBlock1d(in_ch, out_ch))
            in_ch = out_ch * 4  # 4 branches concatenated
            if (i + 1) % 2 == 0:
                blocks.append(nn.MaxPool1d(2, stride=2))
        
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
    
    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Factory function
# ============================================================

def create_backbone(name="xresnet1d101", input_channels=12, 
                    embedding_dim=512, dropout=0.3):
    """
    Create a 1D backbone by name.
    
    Options:
        - xresnet1d50:  XResNet1D with [3, 4, 6, 3] blocks  (~15M params)
        - xresnet1d101: XResNet1D with [3, 4, 23, 3] blocks (~28M params)
        - inception1d:  Inception1D with 6 blocks             (~5M params)
    """
    configs = {
        'xresnet1d50':  {'layers': [3, 4, 6, 3],  'base_filters': 64},
        'xresnet1d101': {'layers': [3, 4, 23, 3], 'base_filters': 64},
        'inception1d':  None  # separate class
    }
    
    if name not in configs:
        raise ValueError(f"Unknown backbone: {name}. Choose from {list(configs.keys())}")
    
    if name == 'inception1d':
        model = Inception1D(
            input_channels=input_channels,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
    else:
        cfg = configs[name]
        model = XResNet1D(
            input_channels=input_channels,
            base_filters=cfg['base_filters'],
            layers=cfg['layers'],
            embedding_dim=embedding_dim,
            dropout=dropout
        )
    
    print(f"  Created {name}: {model.get_num_params():,} parameters")
    return model


if __name__ == "__main__":
    # Quick test
    for name in ['xresnet1d50', 'xresnet1d101', 'inception1d']:
        model = create_backbone(name, input_channels=12, embedding_dim=512)
        x = torch.randn(4, 12, 1000)
        out = model(x)
        print(f"  {name}: input {x.shape} -> embedding {out.shape}")
        print()
