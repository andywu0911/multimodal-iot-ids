"""L2 多模態分類器：任意模態子集（f/p/c）+ 融合 + 雙輸出頭。

融合：
- 'gmu'（主模型）：Gated Multimodal Unit（Arevalo et al., ICLR Workshop 2017）推廣到
  三個模態：h_m = tanh(W_m e_m)，閘門 z = softmax_m(W_z [e_f; e_p; e_c])（逐維度、
  跨模態 softmax），融合表徵 h = Σ_m z_m ⊙ h_m。閘門權重可直接讀出，用來回答
  「加密流量時模型比較依賴哪個模態」（E3）。
- 'concat'：early concat，消融對照。

輸出：分類頭（L1 六類）與偵測頭（攻擊 vs 良性，1 個 logit）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoders import CEncoder, FEncoder, PEncoder


class MultimodalIDS(nn.Module):
    def __init__(self, modalities: str, fusion: str, stats_dim: int, seq_channels: int,
                 ctx_dim: int, n_classes: int = 6, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        assert modalities and set(modalities) <= set("fpc"), modalities
        assert fusion in ("gmu", "concat"), fusion
        self.modalities = "".join(m for m in "fpc" if m in modalities)
        self.fusion = fusion
        enc = {"f": lambda: FEncoder(stats_dim, seq_channels, embed_dim, dropout=dropout),
               "p": lambda: PEncoder(embed_dim, dropout=dropout),
               "c": lambda: CEncoder(ctx_dim, embed_dim, dropout)}
        self.encoders = nn.ModuleDict({m: enc[m]() for m in self.modalities})
        k = len(self.modalities)

        if fusion == "gmu" and k > 1:
            self.transform = nn.ModuleDict({m: nn.Linear(embed_dim, embed_dim) for m in self.modalities})
            self.gate = nn.Linear(embed_dim * k, embed_dim * k)
            fused_dim = embed_dim
        else:
            fused_dim = embed_dim * k

        self.tactic_head = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(64, n_classes))
        self.det_head = nn.Sequential(nn.Linear(fused_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, batch: dict, return_gates: bool = False):
        e = {}
        if "f" in self.modalities:
            e["f"] = self.encoders["f"](batch["f_stats"], batch["f_seq"], batch["f_seq_len"])
        if "p" in self.modalities:
            e["p"] = self.encoders["p"](batch["p"])
        if "c" in self.modalities:
            e["c"] = self.encoders["c"](batch["c"])
        embs = [e[m] for m in self.modalities]

        gates = None
        if self.fusion == "gmu" and len(embs) > 1:
            h = torch.stack([torch.tanh(self.transform[m](e[m])) for m in self.modalities], dim=1)
            z = self.gate(torch.cat(embs, dim=1)).view(h.shape)       # (batch, k, d)
            gates = torch.softmax(z, dim=1)
            fused = (gates * h).sum(dim=1)
        else:
            fused = torch.cat(embs, dim=1)

        out = (self.tactic_head(fused), self.det_head(fused).squeeze(1))
        if return_gates:
            # 每個模態的平均閘門權重 (batch, k)，供 E3 分析
            return out, (gates.mean(dim=2) if gates is not None else None)
        return out
