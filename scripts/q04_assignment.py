"""Q4 · 평가자 배정 교란 (D960 사람 연주)."""
import numpy as np, pandas as pd
from q_common import *
rng = np.random.default_rng(0)
d = d960(ratings("mean"), human_only=True)
seg = d.drop_duplicates("segment_id")[["segment_id", "unit", "performer"]].set_index("segment_id")

# ① 평가자 × 연주자 빈도표의 편중: 단위 안에서 연주자 라벨을 섞은 순열과 비교 (카이제곱 통계량)
def chi2(perf_of_seg):
    t = pd.crosstab(d.user, d.segment_id.map(perf_of_seg)); E = np.outer(t.sum(1), t.sum(0)) / t.values.sum()
    return float(((t.values - E) ** 2 / E).sum())
obs = chi2(seg.performer)
null = []
for _ in range(1000):
    p = seg.performer.copy()
    for u, idx in seg.groupby("unit").groups.items(): p.loc[idx] = rng.permutation(p.loc[idx].to_numpy())
    null.append(chi2(p))
p1 = float((np.sum(np.array(null) >= obs) + 1) / 1001)
print(f"① 평가자×연주자 편중: χ² 관측 {obs:.0f}, 순열 평균 {np.mean(null):.0f}, p = {p1:.3f}")

# ② 평가자 주효과 보정 전후 단위 내 신뢰도
rows = {}
for c in NINE:
    e, a, b = additive_residuals(d, c); adj = d.loc[e.index, ["segment_id", "unit", "user"]].copy()
    adj[c] = d.loc[e.index, c] - b.reindex(adj.user).to_numpy()
    rows[SHORT[c]] = dict(raw=round(within_unit_split_half(d, c), 3), rater_adjusted=round(within_unit_split_half(adj, c), 3))
    rows[SHORT[c]]["delta"] = round(rows[SHORT[c]]["rater_adjusted"] - rows[SHORT[c]]["raw"], 3)
T = pd.DataFrame(rows).T; print("\n② 평가자 주효과 보정 전후 단위 내 신뢰도\n", T.to_string())

# ③ B1·B2 가 섞인 구간: 두 집단 평균 차이가 무작위 반분보다 큰가
mixed = d.groupby("segment_id").block.nunique(); mixed = mixed[mixed == 2].index
zs = {}
for c in NINE:
    z = []
    for s_ in mixed:
        g = d[(d.segment_id == s_)][["block", c]].dropna()
        if g.block.nunique() < 2 or len(g) < 6: continue
        obs_d = abs(g.groupby("block")[c].mean().diff().iloc[-1]); n1 = (g.block == "B1(0-28)").sum()
        v = g[c].to_numpy(); nd = [abs(v[pm[:n1]].mean() - v[pm[n1:]].mean()) for pm in (rng.permutation(len(v)) for _ in range(200))]
        z.append((obs_d - np.mean(nd)) / (np.std(nd) + 1e-9))
    zs[SHORT[c]] = dict(n_seg=len(z), mean_z=round(float(np.mean(z)), 2), frac_z_gt2=round(float(np.mean(np.array(z) > 2)), 3))
print("\n③ 혼합 구간에서 B1−B2 평균 차이의 z (무작위 반분 기준)\n", pd.DataFrame(zs).T.to_string())
verdict = dict(perm_p=p1, pass_assignment=bool(p1 >= .05), max_abs_delta=float(T.delta.abs().max()), pass_adjust=bool(T.delta.abs().max() <= 0.05))
verdict["use_rater_adjusted_labels"] = not (verdict["pass_assignment"] and verdict["pass_adjust"])
print("\n판정:", verdict)
save("q04_assignment", dict(verdict=verdict, chi2_obs=obs, chi2_null_mean=float(np.mean(null)), reliability=rows, mixed_block=zs))
