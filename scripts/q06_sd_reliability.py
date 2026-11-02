"""Q6 · 구간별 '의견 갈림'(평가자 간 SD)의 반분 신뢰도. 세 정의: raw / 평가자 보정 / 보정 + 평균 효과 제거."""
import numpy as np, pandas as pd
from q_common import *
rng = np.random.default_rng(0)
d = d960(ratings("mean"), human_only=True)

def resid_on_mean(sd, mean):
    X = np.c_[np.ones(len(mean)), mean, mean ** 2]; beta, *_ = np.linalg.lstsq(X, sd, rcond=None); return sd - X @ beta

def sd_split_half(df, c, mode, n_iter=100, min_raters=6):
    x = df[["segment_id", "unit", "user", c]].dropna()
    if mode != "raw":
        _, _, b = additive_residuals(df, c); x = x.assign(**{c: x[c] - b.reindex(x.user).to_numpy()})
    groups = [(u, v[c].to_numpy()) for (s, u), v in x.groupby(["segment_id", "unit"]) if len(v) >= min_raters]
    units = np.array([u for u, _ in groups]); rs = []
    for _ in range(n_iter):
        A = np.empty((len(groups), 2)); B = np.empty((len(groups), 2))
        for i, (_, v) in enumerate(groups):
            pm = rng.permutation(len(v)); h = len(v) // 2; va, vb = v[pm[:h]], v[pm[h:]]
            A[i] = va.std(ddof=1), va.mean(); B[i] = vb.std(ddof=1), vb.mean()
        sa, sb = A[:, 0], B[:, 0]
        if mode == "adj_mean": sa, sb = resid_on_mean(sa, A[:, 1]), resid_on_mean(sb, B[:, 1])
        sa = sa - pd.Series(sa).groupby(units).transform("mean").to_numpy(); sb = sb - pd.Series(sb).groupby(units).transform("mean").to_numpy()
        rs.append(np.corrcoef(sa, sb)[0, 1])
    rh = float(np.mean(rs)); return 2 * rh / (1 + rh), len(groups)
rows = {}
for c in NINE:
    r = {}
    for mode in ("raw", "adj", "adj_mean"):
        rel, n = sd_split_half(d, c, mode); r[mode] = round(rel, 3)
    r["n_segments"] = n; rows[SHORT[c]] = r
T = pd.DataFrame(rows).T; print("== 평가자 간 SD 의 반분 신뢰도 (단위 내, Spearman–Brown) ==\n", T.to_string())
# 참고: SD 와 평균의 관계 (천장 효과 크기)
g = d.groupby("segment_id")[CONV].agg(["mean", "std"]).dropna(); print("\n참고 convincing: corr(SD, mean) = %.2f, corr(SD, |mean−4|) = %.2f" % (g["mean"].corr(g["std"]), (g["mean"] - 4).abs().corr(g["std"])))
conv_rel = rows["unconvincing→convincing"]["adj_mean"]
verdict = dict(conv_sd_reliability=conv_rel, pass_h3cd=bool(conv_rel >= 0.30), dims_pass=[k for k, v in rows.items() if v["adj_mean"] >= 0.30])
print("\n판정:", verdict)
save("q06_sd_reliability", dict(table=rows, verdict=verdict))
