# %% [markdown]
# # 05. (초안·미실행) 같은 단위 사람 연주 쌍: 지각 라벨 차 vs Aria 거리 vs 손 특징 차
#
# - **질문**: 같은 악보 구간을 친 두 사람의 연주가 **청취자에게 다르게 들린 만큼**(단위 내 신뢰도가 높은 차원의 |라벨 차|),
#   Aria 임베딩 거리도 멀어지는가? 손 특징 1개의 차이와 비교하면?
# - **입력**: `data/interim/pp_aria_emb.npz`(`scripts/pp_aria_embed.py`), `pp_segments_clean.parquet`, `pp_features.parquet`, `feas/normalization.npz`
# - **범위**: D960 2·3악장, 사람 연주만(Score 제외), 악보 쪽 매치율 ≥ 0.8. 평가자 54명, 단위 내 신뢰도(사람만) ≥ 0.5 차원 7개(01 §4.2).
# - **통계**: 쌍 단위 값을 단위 안에서 백분위 순위로 바꾼 뒤 풀링해 Spearman. 영가설은 "단위 안에서 연주자의 라벨을 섞어도 같다"(순열 1,000회).
# - **사전 예측 (실행 전에 적음, 2026-10-24)**
#   1. 페달 차원(dry→wet, clean→blurred)은 Aria 거리와 양의 상관 — 04에서 페달 제거를 짚었음(D899/3 0.97).
#   2. fast→slow는 약함 — Aria 학습 증강에 tempo ±20% 불변이 들어 있음(Bradshaw 2025 §3).
#   3. 대응 손 특징 1개(페달 비율, log 템포 등)가 Aria와 같거나 나음.
#   예측과 다르게 나오면 그쪽이 더 흥미롭다. 어느 쪽이든 n_units·n_pairs 와 함께 보고한다.

# %%
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.eda import paths, percepiano_io as pp

L, S = pp.LABELS19, pp.SHORT19
SHORT = dict(zip(L, S))
# 01 §4.2 (D960 · 사람만) 단위 내 신뢰도 ≥ 0.5 인 차원과, 대응시킬 손 특징 1개
DIMS = {
    "Pedal_Sparse/dry_Saturated/wet": "pedal_down_ratio",
    "Pedal_Clean_Blurred": "pedal_down_ratio",
    "Music_Making_Fast_paced_Slow_paced": "log_tempo",
    "Timing_Stable_Unstable": "tempo_logstd",
    "Timbre_Shallow_Rich": "pedal_down_ratio",
    "Articulation_Soft_cushioned_Hard_solid": "vel_mean",
    "Articulation_Long_Short": "articulation_median",
    "Interpretation_Unsatisfactory/doubtful_Convincing": "vel_range",  # 신뢰도 0.40 — 참고용(≈0 예상)
}
HAND_ALL = ["log_tempo", "tempo_logstd", "timing_std_ms", "articulation_median", "vel_std", "vel_range", "pedal_down_ratio"]

seg = pd.read_parquet(paths.INTERIM / "pp_segments_clean.parquet")
feat = pd.read_parquet(paths.INTERIM / "pp_features.parquet")
z = np.load(paths.INTERIM / "pp_aria_emb.npz")
norm = np.load(paths.FEAS / "normalization.npz")
E = (z["emb"].astype(np.float64) - norm["mean"]) / norm["std"]
E /= np.linalg.norm(E, axis=1, keepdims=True)
emb = dict(zip([str(s) for s in z["segment_id"]], E))
print("임베딩", len(emb), "| 2048 토큰에서 잘린 구간", int((z["n_tokens"] >= 2048).sum()))

d = seg[seg.piece.str.startswith("Schubert_D960") & ~seg.is_score].copy()
d = d.merge(feat.drop(columns=["piece", "unit", "performer", "is_score", "n_bars"]), on="segment_id")
d = d[(d.match / d.score_notes) >= 0.8]
d = d[d.segment_id.isin(emb)].copy()
d["log_tempo"] = np.log2(d.tempo_median_bpm)
d = d.dropna(subset=HAND_ALL)
print("구간", len(d), "| 단위", d.unit.nunique(), "| 단위당 연주 중앙값", d.groupby("unit").size().median())

# %% [markdown]
# ## 1. 쌍 만들기
# 단위 안 손 특징은 단위 내 표준화(사람 연주 기준) 뒤 유클리드 거리로 `hand_all` 도 만든다.

# %%
d = d.reset_index(drop=True)
Y = d[list(DIMS)].to_numpy(float)
F = d[[f for f in DIMS.values()]].to_numpy(float)
Hz = d[HAND_ALL].to_numpy(float)
for unit, idx in d.groupby("unit").indices.items():
    Hz[idx] = (Hz[idx] - Hz[idx].mean(0)) / (Hz[idx].std(0) + 1e-9)
V = np.stack([emb[s] for s in d.segment_id])

ri, rj, un = [], [], []
for unit, idx in d.groupby("unit").indices.items():
    if len(idx) < 4:
        continue
    for i, j in combinations(idx.tolist(), 2):
        ri.append(i); rj.append(j); un.append(unit)
ri, rj = np.asarray(ri), np.asarray(rj)
P = pd.DataFrame({"unit": un, "d_aria": 1.0 - np.einsum("ij,ij->i", V[ri], V[rj]),
                  "d_hand_all": np.linalg.norm(Hz[ri] - Hz[rj], axis=1)})
for k, (lab, f) in enumerate(DIMS.items()):
    P[f"dl_{SHORT[lab]}"] = np.abs(Y[ri, k] - Y[rj, k])
    P[f"dh_{SHORT[lab]}"] = np.abs(F[ri, k] - F[rj, k])
print("쌍", len(P), "| 단위", P.unit.nunique())

# %% [markdown]
# ## 2. 단위 내 순위 풀링 Spearman + 순열 검정
# 순열: 단위 안에서 연주자의 라벨 벡터를 섞고(쌍 구조는 고정) |라벨 차|를 다시 계산한다.

# %%
unit_codes = pd.factorize(P.unit)[0]

def within_rank(v):
    return pd.Series(v).groupby(unit_codes).rank(pct=True).to_numpy()

def rho(x, y):
    return float(spearmanr(within_rank(x), within_rank(y))[0])

groups = [idx for idx in d.groupby("unit").indices.values()]

def perm_rhos(x, k, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed)
    y = Y[:, k].copy()
    out = np.empty(n_iter)
    rx = within_rank(x)
    for t in range(n_iter):
        yp = y.copy()
        for idx in groups:
            yp[idx] = y[rng.permutation(idx)]
        out[t] = spearmanr(rx, within_rank(np.abs(yp[ri] - yp[rj])))[0]
    return out

rows = []
for k, (lab, f) in enumerate(DIMS.items()):
    s = SHORT[lab]
    obs = rho(P.d_aria.to_numpy(), P[f"dl_{s}"].to_numpy())
    null = perm_rhos(P.d_aria.to_numpy(), k, n_iter=1000, seed=k)
    rows.append({"차원": s, "손 특징": f, "ρ Aria": obs,
                 "p(순열)": (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1),
                 "ρ 손특징 1개": rho(P[f"dh_{s}"].to_numpy(), P[f"dl_{s}"].to_numpy()),
                 "ρ 손특징 7개": rho(P.d_hand_all.to_numpy(), P[f"dl_{s}"].to_numpy()),
                 "n_units": P.unit.nunique(), "n_pairs": len(P)})
RES = pd.DataFrame(rows)
print(RES.round(3).to_string(index=False))

# %% [markdown]
# ## 3. Pereval식 '전형성' 점수가 평가적 차원과 상관하는가
# 구간마다 같은 단위의 다른 사람 연주들까지의 평균 코사인 거리(전형성의 반대) 를 구하고,
# 라벨의 단위 내 편차(연주 − 단위 평균)와 상관을 본다. 천장(단위 내 신뢰도 ≤ 0.5)상 convincing 은 ≈0 이 예상된다.

# %%
typ = {}
for unit, g in d.groupby("unit"):
    ids = g.segment_id.tolist()
    for s in ids:
        others = [emb[o] for o in ids if o != s]
        typ[s] = float(np.mean([1.0 - emb[s] @ o for o in others])) if others else np.nan
d["typicality_dist"] = d.segment_id.map(typ)
dev = d[L] - d.groupby("unit")[L].transform("mean")
tz = d.typicality_dist - d.groupby("unit").typicality_dist.transform("mean")
out = pd.Series({SHORT[c]: spearmanr(tz, dev[c], nan_policy="omit")[0] for c in L}).sort_values()
print("단위 내 편차 상관 (전형성 거리 vs 라벨):")
print(out.round(2).to_string())

# %% [markdown]
# ## 4. 정리 (실행 후 작성)
# - [사실] …
# - [가설] …
# - 사전 예측 1·2·3 과의 대조: …
