"""L2 三個模態的編碼器。全部用現成架構，不自創模型。

- FEncoder：統計向量走 MLP，封包序列走 1D-CNN（FS-Net 以序列建模封包長度與方向的路線，
  這裡用較輕的卷積版本），兩者串接後投影。
- PEncoder：TextCNN 式 byte CNN（Wang et al., ISI 2017 的 1D-CNN 流量分類路線）。
- CEncoder：MLP。

變長序列一律做 padding mask：payload 中位數只有 34–42 bytes、
封包序列多數短於 SEQ_LEN，Conv1d 的 bias 會讓 padding 區段輸出非零常數，
max-pool 前必須把 padding 位置蓋成 -inf。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.features.view_p import PAD_TOKEN, VOCAB_SIZE


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, out_dim), nn.ReLU()]
    return nn.Sequential(*layers)


def _masked_max(feats: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """feats (batch, ch, L)、pad (batch, L) True=padding。呼叫端保證每列至少一個非 padding。"""
    return feats.masked_fill(pad.unsqueeze(1), float("-inf")).max(dim=2).values


class FEncoder(nn.Module):
    def __init__(self, stats_dim: int, seq_channels: int, out_dim: int = 64,
                 conv_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.stats = _mlp(stats_dim, (128,), out_dim, dropout)
        self.conv = nn.Sequential(
            nn.Conv1d(seq_channels, conv_channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, 3, padding=1), nn.ReLU(),
        )
        self.proj = nn.Sequential(nn.Linear(out_dim + conv_channels, out_dim), nn.ReLU())

    def forward(self, stats: torch.Tensor, seq: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(seq.shape[1], device=seq.device)
        pad = positions.unsqueeze(0) >= seq_len.clamp(min=1).unsqueeze(1)
        s = _masked_max(self.conv(seq.transpose(1, 2)), pad)
        return self.proj(torch.cat([self.stats(stats), s], dim=1))


class PEncoder(nn.Module):
    def __init__(self, out_dim: int = 64, embed_dim: int = 64,
                 kernel_sizes: tuple[int, ...] = (3, 5, 7), num_filters: int = 100, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_TOKEN)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
                                    for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(num_filters * len(kernel_sizes), out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        e = self.embed(tokens).transpose(1, 2)
        feats = torch.cat([torch.relu(conv(e)) for conv in self.convs], dim=1)
        pooled = _masked_max(feats, tokens == PAD_TOKEN)   # EOS 保證每列至少一個非 PAD
        return torch.relu(self.proj(self.dropout(pooled)))


class CEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = _mlp(in_dim, (128,), out_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
