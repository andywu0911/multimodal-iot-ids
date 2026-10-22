"""View F 封包序列的數值邊界：錯了不會拋例外，只會讓特定樣本變成 NaN，
而 NaN 在不同後端的 max-pool 行為不一致（MPS 會略過、CPU 會傳遞），
結果可能在一台機器上看起來正常、換一台就壞。"""
import numpy as np
import pandas as pd

from src.features.view_f import build_view_f_seq


def test_negative_iat_from_reordered_timestamps_stays_finite():
    df = pd.DataFrame({"pkt_lengths": [[60, 60, 60]], "pkt_iat": [[0.0, -0.002, 0.01]],
                       "pkt_dir": [[1, 0, 1]], "tcp_flags": [[2, 18, 16]]})
    seq, lengths = build_view_f_seq(df, seq_len=4)
    assert np.isfinite(seq).all()
    assert seq[0, 1, 2] == 0.0
    assert lengths[0] == 3
