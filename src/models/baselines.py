"""E2 神經網路對照組（固定後補入，論文須標明）。

- FSNetBaseline：FS-Net（Liu et al., IEEE INFOCOM 2019）的做法套在本專案的封包序列上。
  雙向 GRU 編碼器讀前 SEQ_LEN 包的逐包特徵，雙向 GRU 解碼器從編碼向量重建封包長度
  序列（重建損失即原論文的 reconstruction mechanism），分類特徵取編碼與解碼的最終
  狀態 [z_e, z_d, z_e⊙z_d, |z_e−z_d|]。與原作的差異：輸入是 9 個連續通道而非離散化
  的封包長度 embedding；只吃封包序列，不用統計向量、payload 與主機情境。
- KitNET：Kitsune（Mirsky et al., NDSS 2018）的偵測核心。以特徵相關距離做階層分群
  （每群至多 max_cluster 維），每群一個小自編碼器，再以一個輸出自編碼器吃各群的
  重建 RMSE；只用良性資料訓練，異常分數是輸出層 RMSE。輸入用觀測窗層級的 F 統計
  + C 主機情境（C 本身就是 AfterImage 式的多時間尺度統計）。與原作的差異：原作逐封包
  線上訓練一次，這裡以批次訓練多個 epoch。只能比偵測層，沒有攻擊分類。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FSNetBaseline(nn.Module):
    def __init__(self, seq_channels: int, n_classes: int = 6, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, recon_weight: float = 1.0):
        super().__init__()
        self.modalities = "s"
        self.recon_weight = recon_weight
        self.encoder = nn.GRU(seq_channels, hidden, layers, batch_first=True, bidirectional=True,
                              dropout=dropout)
        z_dim = 2 * layers * hidden
        self.decoder = nn.GRU(z_dim, hidden, layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.reconstruct = nn.Linear(2 * hidden, 1)
        self.dense = nn.Sequential(nn.Linear(4 * z_dim, 128), nn.SELU(), nn.Dropout(dropout))
        self.tactic_head = nn.Linear(128, n_classes)
        self.det_head = nn.Linear(128, 1)
        self.aux_loss = None

    @staticmethod
    def _final_state(h: torch.Tensor) -> torch.Tensor:
        return h.transpose(0, 1).reshape(h.shape[1], -1)        # (batch, layers*2*hidden)

    def forward(self, batch: dict, return_gates: bool = False):
        seq, lengths = batch["f_seq"], batch["f_seq_len"].clamp(min=1)
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_enc = self.encoder(packed)
        z_e = self._final_state(h_enc)
        dec_in = z_e.unsqueeze(1).expand(-1, seq.shape[1], -1)
        dec_out, h_dec = self.decoder(pack_padded_sequence(dec_in, lengths.cpu(), batch_first=True,
                                                           enforce_sorted=False))
        z_d = self._final_state(h_dec)
        if self.training:
            out, _ = pad_packed_sequence(dec_out, batch_first=True, total_length=seq.shape[1])
            valid = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)
            err = (self.reconstruct(out).squeeze(-1) - seq[:, :, 0]) ** 2
            self.aux_loss = self.recon_weight * err[valid].mean()
        h = self.dense(torch.cat([z_e, z_d, z_e * z_d, (z_e - z_d).abs()], dim=1))
        out = (self.tactic_head(h), self.det_head(h).squeeze(1))
        return (out, None) if return_gates else out


def _cluster_features(X: np.ndarray, max_cluster: int) -> list[list[int]]:
    """KitNET 的 feature mapper：以 1−|相關係數| 為距離做 single-linkage 分群，
    遞迴切開大於 max_cluster 的群。常數欄位的相關係數視為 0。"""
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.nan_to_num(np.corrcoef(X, rowvar=False))
    dist = np.clip(1.0 - np.abs(corr), 0.0, None)
    np.fill_diagonal(dist, 0.0)
    root = to_tree(linkage(squareform(dist, checks=False), method="single"))
    groups: list[list[int]] = []

    def split(node):
        if node.get_count() <= max_cluster:
            groups.append(node.pre_order())
        else:
            split(node.get_left())
            split(node.get_right())

    split(root)
    return groups


class _AE(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float):
        super().__init__()
        h = max(1, int(np.ceil(dim * hidden_ratio)))
        self.enc, self.dec = nn.Linear(dim, h), nn.Linear(h, dim)

    def forward(self, x):
        return torch.sigmoid(self.dec(torch.sigmoid(self.enc(x))))


class KitNET:
    def __init__(self, max_cluster: int = 10, hidden_ratio: float = 0.75, lr: float = 1e-2,
                 epochs: int = 30, batch_size: int = 256, seed: int = 42):
        self.max_cluster, self.hidden_ratio = max_cluster, hidden_ratio
        self.lr, self.epochs, self.batch_size, self.seed = lr, epochs, batch_size, seed

    def _norm(self, X):
        return torch.tensor(np.clip((X - self.lo) / self.span, 0.0, 1.0), dtype=torch.float32)

    def _ensemble_rmse(self, x: torch.Tensor) -> torch.Tensor:
        cols = [torch.sqrt(F.mse_loss(ae(x[:, g]), x[:, g], reduction="none").mean(1))
                for ae, g in zip(self.ensemble, self.groups)]
        return torch.stack(cols, dim=1)

    def fit(self, X_benign: np.ndarray) -> "KitNET":
        torch.manual_seed(self.seed)
        self.lo, hi = X_benign.min(0), X_benign.max(0)
        self.span = np.where(hi > self.lo, hi - self.lo, 1.0)
        self.groups = _cluster_features(X_benign, self.max_cluster)
        self.ensemble = [_AE(len(g), self.hidden_ratio) for g in self.groups]
        self.output = _AE(len(self.groups), self.hidden_ratio)
        x = self._norm(X_benign)
        params = [p for ae in self.ensemble + [self.output] for p in ae.parameters()]
        opt = torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(x)).split(self.batch_size):
                xb = x[idx]
                loss = sum(F.mse_loss(ae(xb[:, g]), xb[:, g]) for ae, g in zip(self.ensemble, self.groups))
                r = self._ensemble_rmse(xb).detach()
                loss = loss + F.mse_loss(self.output(r), r)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        r = self._ensemble_rmse(self._norm(X))
        return torch.sqrt(F.mse_loss(self.output(r), r, reduction="none").mean(1)).numpy()
