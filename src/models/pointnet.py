"""
PointNet and PointNet2 (SSG) for semantic segmentation of 3D point clouds.

Reference
---------
- Qi et al., "PointNet: Deep Learning on Point Sets" (CVPR 2017)
- Qi et al., "PointNet++: Deep Hierarchical Feature Learning" (NeurIPS 2017)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
#  Shared building blocks
# ──────────────────────────────────────────────

class SharedMLP(nn.Module):
    """1-D convolution acting point-wise (shared MLP)."""

    def __init__(self, channels: list[int], bn: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TNet(nn.Module):
    """Spatial-transformer network that predicts a k×k transform."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.mlp = SharedMLP([k, 64, 128, 1024])
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Linear(256, k * k),
        )
        # Initialise to identity
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.fc[-1].bias.data.copy_(torch.eye(k).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, k, N) → transform : (B, k, k)"""
        feat = self.mlp(x)                        # (B, 1024, N)
        feat = feat.max(dim=2)[0]                  # (B, 1024)
        mat = self.fc(feat).view(-1, self.k, self.k)
        return mat


# ──────────────────────────────────────────────
#  PointNet Segmentation
# ──────────────────────────────────────────────

class PointNetSegmentation(nn.Module):
    """PointNet for per-point semantic segmentation.

    Input  : (B, N, C)  C = num input channels (e.g. 9 for XYZ+RGB+norm)
    Output : (B, N, num_classes) logits
    """

    def __init__(
        self,
        input_channels: int = 9,
        num_classes: int = 13,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Input transform
        self.tnet_input = TNet(k=input_channels)

        # MLP1
        self.mlp1 = SharedMLP([input_channels, 64, 64])

        # Feature transform
        self.tnet_feat = TNet(k=64)

        # MLP2
        self.mlp2 = SharedMLP([64, 128, 1024])

        # Segmentation head  (concat local 64 + global 1024 → 1088)
        self.seg_head = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(True),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, N, C)
        Returns
        -------
        logits       : (B, N, num_classes)
        feat_transform : (B, 64, 64) for regularisation loss
        """
        B, N, C = x.shape
        x = x.transpose(1, 2)  # (B, C, N)

        # Input transform
        t_input = self.tnet_input(x)       # (B, C, C)
        x = torch.bmm(t_input, x)         # (B, C, N)

        # MLP1 → local features
        x = self.mlp1(x)                   # (B, 64, N)
        local_feat = x

        # Feature transform
        t_feat = self.tnet_feat(x)         # (B, 64, 64)
        x = torch.bmm(t_feat, x)          # (B, 64, N)

        # MLP2 → global features
        x = self.mlp2(x)                   # (B, 1024, N)
        global_feat = x.max(dim=2, keepdim=True)[0]  # (B, 1024, 1)
        global_feat = global_feat.expand(-1, -1, N)   # (B, 1024, N)

        # Concat local + global
        x = torch.cat([local_feat, global_feat], dim=1)  # (B, 1088, N)

        # Segmentation head
        logits = self.seg_head(x)           # (B, num_classes, N)
        logits = logits.transpose(1, 2)     # (B, N, num_classes)

        return logits, t_feat


# ──────────────────────────────────────────────
#  PointNet++ building blocks
# ──────────────────────────────────────────────

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest-point sampling.

    xyz   : (B, N, 3)
    npoint: number of centroids
    Returns indices (B, npoint)
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[
            torch.arange(B, device=device), farthest
        ].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid_xyz) ** 2).sum(dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def ball_query(
    radius: float, nsample: int,
    xyz: torch.Tensor, new_xyz: torch.Tensor,
) -> torch.Tensor:
    """Ball query – find nsample points within radius of each centroid.

    xyz     : (B, N, 3)
    new_xyz : (B, S, 3)  centroids
    Returns : (B, S, nsample) indices into xyz
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    device = xyz.device

    # Pairwise distances
    dists = torch.cdist(new_xyz, xyz)  # (B, S, N)

    # Mask out-of-radius
    dists[dists > radius] = 1e10

    # Sort and take top-nsample
    sorted_dists, idx = dists.sort(dim=-1)
    idx = idx[:, :, :nsample]  # (B, S, nsample)

    # Fill if fewer than nsample in ball
    first = idx[:, :, 0:1].expand_as(idx)
    mask = sorted_dists[:, :, :nsample] >= 1e10
    idx[mask] = first[mask]

    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    points : (B, N, C)
    idx    : (B, ..., ) long indices
    Returns: (B, ..., C)
    """
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.ndim - 1)
    expand_shape = [B] + list(idx.shape[1:])
    batch_idx = torch.arange(B, device=points.device).view(view_shape).expand(expand_shape)
    return points[batch_idx, idx]


class SetAbstraction(nn.Module):
    """PointNet++ Set Abstraction (SSG)."""

    def __init__(
        self, npoint: int, radius: float, nsample: int,
        in_channel: int, mlp_channels: list[int],
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers: list[nn.Module] = []
        last_ch = in_channel + 3  # relative XYZ is concat'd
        for ch in mlp_channels:
            layers.append(nn.Conv2d(last_ch, ch, 1))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU(True))
            last_ch = ch
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor | None):
        """
        xyz  : (B, N, 3)
        feat : (B, N, D) or None
        Returns new_xyz (B, npoint, 3), new_feat (B, npoint, D')
        """
        # Sample centroids
        idx = farthest_point_sample(xyz, self.npoint)   # (B, npoint)
        new_xyz = index_points(xyz, idx)                 # (B, npoint, 3)

        # Group neighbours
        group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, group_idx) - new_xyz.unsqueeze(2)

        if feat is not None:
            grouped_feat = index_points(feat, group_idx)
            grouped = torch.cat([grouped_xyz, grouped_feat], dim=-1)
        else:
            grouped = grouped_xyz

        # (B, S, nsample, D) → (B, D, nsample, S) for Conv2d
        grouped = grouped.permute(0, 3, 2, 1)
        new_feat = self.mlp(grouped)           # (B, D', nsample, S)
        new_feat = new_feat.max(dim=2)[0]      # (B, D', S)
        new_feat = new_feat.permute(0, 2, 1)   # (B, S, D')

        return new_xyz, new_feat


class FeaturePropagation(nn.Module):
    """PointNet++ Feature Propagation via distance-weighted interpolation."""

    def __init__(self, in_channel: int, mlp_channels: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_ch = in_channel
        for ch in mlp_channels:
            layers.append(nn.Conv1d(last_ch, ch, 1))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(True))
            last_ch = ch
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1: torch.Tensor, xyz2: torch.Tensor,
        feat1: torch.Tensor | None, feat2: torch.Tensor,
    ) -> torch.Tensor:
        """
        xyz1 : (B, N, 3) – target (high-res)
        xyz2 : (B, S, 3) – source (low-res)
        feat1: (B, N, D1) or None – skip features from encoder
        feat2: (B, S, D2) – features to interpolate
        Returns (B, N, D')
        """
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]

        if S == 1:
            interp_feat = feat2.expand(-1, N, -1)
        else:
            dists = torch.cdist(xyz1, xyz2)                          # (B, N, S)
            topk_dists, topk_idx = dists.topk(3, dim=-1, largest=False)  # (B, N, 3)
            topk_dists = topk_dists.clamp(min=1e-8)
            weights = 1.0 / topk_dists
            weights = weights / weights.sum(dim=-1, keepdim=True)    # (B, N, 3)

            interp_feat = torch.zeros(B, N, feat2.shape[2], device=xyz1.device)
            for k in range(3):
                gathered = index_points(feat2, topk_idx[:, :, k])    # (B, N, D2)
                interp_feat += weights[:, :, k:k+1] * gathered

        if feat1 is not None:
            combined = torch.cat([feat1, interp_feat], dim=-1)
        else:
            combined = interp_feat

        combined = combined.transpose(1, 2)   # (B, D, N)
        out = self.mlp(combined)              # (B, D', N)
        return out.transpose(1, 2)            # (B, N, D')


# ──────────────────────────────────────────────
#  PointNet++ Segmentation (SSG)
# ──────────────────────────────────────────────

class PointNet2Segmentation(nn.Module):
    """PointNet++ SSG for per-point semantic segmentation.

    Input  : (B, N, C)  C = num input channels
    Output : (B, N, num_classes)
    """

    def __init__(
        self,
        input_channels: int = 9,
        num_classes: int = 13,
        dropout: float = 0.3,
    ):
        super().__init__()
        extra = input_channels - 3  # non-XYZ channels

        # Encoder
        self.sa1 = SetAbstraction(1024, 0.1, 32, extra, [32, 32, 64])
        self.sa2 = SetAbstraction(256,  0.2, 64, 64,    [64, 64, 128])
        self.sa3 = SetAbstraction(64,   0.4, 128, 128,  [128, 128, 256])
        self.sa4 = SetAbstraction(16,   0.8, 64,  256,  [256, 256, 512])

        # Decoder
        self.fp4 = FeaturePropagation(256 + 512, [256, 256])
        self.fp3 = FeaturePropagation(128 + 256, [256, 256])
        self.fp2 = FeaturePropagation(64 + 256,  [256, 128])
        self.fp1 = FeaturePropagation(extra + 128, [128, 128, 128])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x : (B, N, C)
        Returns logits (B, N, num_classes), None
        """
        B, N, C = x.shape
        xyz0 = x[:, :, :3]
        feat0 = x[:, :, 3:] if C > 3 else None

        # Encoder
        xyz1, feat1 = self.sa1(xyz0, feat0)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        xyz3, feat3 = self.sa3(xyz2, feat2)
        xyz4, feat4 = self.sa4(xyz3, feat3)

        # Decoder
        feat3_up = self.fp4(xyz3, xyz4, feat3, feat4)
        feat2_up = self.fp3(xyz2, xyz3, feat2, feat3_up)
        feat1_up = self.fp2(xyz1, xyz2, feat1, feat2_up)
        feat0_up = self.fp1(xyz0, xyz1, feat0, feat1_up)

        # Classify
        logits = self.classifier(feat0_up.transpose(1, 2))  # (B, C, N)
        logits = logits.transpose(1, 2)                       # (B, N, C)
        return logits, None


# ──────────────────────────────────────────────
#  Loss utilities
# ──────────────────────────────────────────────

def pointnet_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    feat_transform: torch.Tensor | None = None,
    reg_weight: float = 0.001,
) -> torch.Tensor:
    """Cross-entropy + optional feature-transform regularisation."""
    B, N, C = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, C), labels.reshape(-1))
    if feat_transform is not None:
        k = feat_transform.shape[1]
        eye = torch.eye(k, device=feat_transform.device).unsqueeze(0)
        aat = torch.bmm(feat_transform, feat_transform.transpose(1, 2))
        diff = aat - eye
        reg = (diff ** 2).sum() / B
        return ce + reg_weight * reg
    return ce


# ──────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────

def build_model(
    name: str = "pointnet",
    input_channels: int = 9,
    num_classes: int = 13,
    dropout: float = 0.3,
) -> nn.Module:
    """Construct a segmentation model by name."""
    name = name.lower()
    if name == "pointnet":
        return PointNetSegmentation(input_channels, num_classes, dropout)
    elif name in ("pointnet2", "pointnet++"):
        return PointNet2Segmentation(input_channels, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model: {name}")
