"""Q12 · α·ρ 재측정(16↔8마디 겹침), 연주자 잎-하나-빼기, 검정력 모의."""
import sys, json, numpy as np, pandas as pd
from q_common import *
sys.path.insert(0, str(ROOT / "scripts")); import x2_rq3_layers as x2
d = x2.load_ratings(); F = x2.load_features()
seg = d.drop_duplicates("segment_id")[["segment_id", "unit", "is_score"]]; seg = seg[seg.segment_id.isin(F.index)].reset_index(drop=True)
hum = seg[~seg.is_score].segment_id; Xz = (F.loc[seg.segment_id] - F.loc[hum].mean()) / (F.loc[hum].std() + 1e-9)
C = x2.decompose(seg, Xz, False).set_index("segment_id"); info = pd.DataFrame([pp.parse_name(s) for s in C.index]).set_index("segment_id")
C = C.join(info[["piece", "performer", "n_bars", "segment"]])
# ① 겹침 재측정: 16마디 k ↔ 8마디 2k−1, 2k (같은 연주자) 의 α, ‖ρ‖, ‖r‖ 평균
pairs = []
for sid, r in C[C.n_bars == 16].iterrows():
    stem = sid.rsplit("_", 2)[0].replace("16bars", "8bars"); kids = [f"{stem}_{r.performer}_{k:02d}" for k in (2 * r.segment - 1, 2 * r.segment)]; kids = [k for k in kids if k in C.index]
    if len(kids) == 2: pairs.append(dict(alpha16=r.alpha, alpha8=C.loc[kids, "alpha"].mean(), rho16=r.rho_norm, rho8=C.loc[kids, "rho_norm"].mean(), r16=r.r_norm, r8=C.loc[kids, "r_norm"].mean()))
P = pd.DataFrame(pairs)
from scipy.stats import spearmanr
retest = {k: round(float(spearmanr(P[f"{k}16"], P[f"{k}8"]).correlation), 3) for k in ("alpha", "rho", "r")}
print("① 겹침 재측정 (16마디 vs 8마디 두 개 평균, n=%d): Spearman" % len(P), retest)
# ② 8마디만 / 전체
print("② 세그먼트: 전체 %d (8마디 %d, 16마디 %d), 단위 %d" % (len(C), (C.n_bars == 8).sum(), (C.n_bars == 16).sum(), C.unit.nunique()))
# ③ 연주자 잎-하나-빼기: α 좌표의 안정성 (남은 연주자들의 α 가 전체 기준과 얼마나 같은가)
lopo = {}
for p in sorted(C.performer.unique()):
    seg_p = seg[~seg.segment_id.isin(C.index[C.performer == p])]
    Cp = x2.decompose(seg_p, Xz, False).set_index("segment_id"); j = Cp.index.intersection(C.index)
    lopo[p] = dict(alpha=round(float(spearmanr(Cp.loc[j, "alpha"], C.loc[j, "alpha"]).correlation), 3), rho=round(float(spearmanr(Cp.loc[j, "rho_norm"], C.loc[j, "rho_norm"]).correlation), 3))
L_ = pd.DataFrame(lopo).T; print("③ 연주자 잎-하나-빼기: α 상관 최소 %.3f, ‖ρ‖ 상관 최소 %.3f" % (L_.alpha.min(), L_.rho.min()))
# ④ 검정력 모의: 참 상관 0.3, 신뢰도 r_x(‖ρ‖ 재측정), r_y(SD 반분, Q6), n 세그먼트, 단위 내 순위 풀링 → 관측 ≥ 0.10 & p<.05 비율
q6 = json.load(open(OUT / "q06_sd_reliability.json"))["table"]["unconvincing→convincing"]["adj_mean"]
rng = np.random.default_rng(0); n = int((C.n_bars == 8).sum())
def power(rx, ry, true=0.3, reps=400):
    hits = 0
    for _ in range(reps):
        tx, ty = rng.multivariate_normal([0, 0], [[1, true], [true, 1]], n).T
        ox = np.sqrt(max(rx, 1e-6)) * tx + np.sqrt(1 - max(rx, 1e-6)) * rng.normal(size=n); oy = np.sqrt(max(ry, 1e-6)) * ty + np.sqrt(1 - max(ry, 1e-6)) * rng.normal(size=n)
        r = spearmanr(ox, oy); hits += (r.correlation >= 0.10) and (r.pvalue < .05)
    return hits / reps
pw = {"Q6 실측 SD 신뢰도 (%.2f)" % q6: round(power(retest["rho"], max(q6, 0)), 2), "SD 신뢰도 0.30 가정": round(power(retest["rho"], 0.30), 2), "SD 신뢰도 0.50 가정": round(power(retest["rho"], 0.50), 2)}
print("④ 검정력 (참 ρ=0.3, ‖ρ‖ 신뢰도 %.2f, n=%d):" % (retest["rho"], n), pw)
verdict = dict(retest=retest, pass_retest=bool(retest["alpha"] >= 0.5 and retest["rho"] >= 0.5), lopo_min_alpha=float(L_.alpha.min()), lopo_min_rho=float(L_.rho.min()), power=pw, pass_power=bool(list(pw.values())[0] >= 0.8))
print("\n판정:", verdict); save("q12_alpha_rho", verdict)
