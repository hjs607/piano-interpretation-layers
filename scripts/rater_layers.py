"""두 층 추정(완료 실험 R5 의 코드, 컨테이너 exp/rater_layers.py 를 폴더 구조에 맞춰 옮김. 2026-10-25)
① 공유 축: 차원별 신뢰도(ICC, 단위 내 split-half) ② 개인 층: 평가자별 축 민감도 w, 설득력 선호 β, 유형(k=2).
결과 보고: reports/rater_layers_2026-10-25.md.   실행  .venv/bin/python -I scripts/rater_layers.py
"""
import json
import sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eda import paths, percepiano_io as pp  # noqa: E402

RESULTS = ROOT / "reports" / "rater_layers"; RESULTS.mkdir(parents=True, exist_ok=True)
L, S = pp.LABELS19, pp.SHORT19; SHORT = dict(zip(L, S))
DESC = ["Timing_Stable_Unstable", "Articulation_Long_Short", "Articulation_Soft_cushioned_Hard_solid",
        "Pedal_Sparse/dry_Saturated/wet", "Pedal_Clean_Blurred", "Timbre_Shallow_Rich",
        "Dynamic_Little_dynamic_range_Large_dynamic_range", "Music_Making_Fast_paced_Slow_paced"]
CONV = "Interpretation_Unsatisfactory/doubtful_Convincing"
MIN_SEG = 40
rng = np.random.default_rng(0)

p = paths.INTERIM / "pp_ratings_clean.parquet"
r = pd.read_parquet(p) if p.exists() else pp.clean(pp.load_raw())
d = r[r.piece.str.startswith("Schubert_D960")].copy()
cnt = d.groupby("user").size(); raters = cnt[cnt >= MIN_SEG].index
d = d[d.user.isin(raters)].reset_index(drop=True)
print(f"D960 평가자 {len(raters)}명(≥{MIN_SEG}구간), 평가 {len(d)}, 구간 {d.segment_id.nunique()}")

def icc_oneway(df, value):
    x = df[["segment_id", value]].dropna(); g = x.groupby("segment_id")[value]
    n_i = g.size().to_numpy(float); a, N = len(n_i), n_i.sum(); grand = x[value].mean()
    msb = (n_i * (g.mean().to_numpy() - grand) ** 2).sum() / (a - 1)
    msw = ((x[value] - g.transform("mean")) ** 2).sum() / (N - a)
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw), (msb - msw) / msb, k0

def split_half_within_unit(df, value, n_iter=100, seed=0):
    g0 = np.random.default_rng(seed); x = df[["segment_id", "unit", "user", value]].dropna()
    groups = [(u, v[value].to_numpy()) for (s, u), v in x.groupby(["segment_id", "unit"]) if len(v) >= 4]
    units = np.array([u for u, _ in groups]); rs = []
    for _ in range(n_iter):
        a = np.empty(len(groups)); b = np.empty(len(groups))
        for i, (_, v) in enumerate(groups):
            pm = g0.permutation(len(v)); h = len(v) // 2; a[i], b[i] = v[pm[:h]].mean(), v[pm[h:]].mean()
        sa = pd.Series(a).groupby(units).transform("mean").to_numpy(); sb = pd.Series(b).groupby(units).transform("mean").to_numpy()
        rs.append(np.corrcoef(a - sa, b - sb)[0, 1])
    rh = float(np.mean(rs)); return 2 * rh / (1 + rh)

hum = d[~d.is_score]
rows = []
for c in L:
    i1, ik, k0 = icc_oneway(d, c); w = split_half_within_unit(hum, c)
    grade = "공유 축 (강)" if w >= 0.7 else "공유 축" if w >= 0.5 else "약한 공유" if w >= 0.3 else "개인 층"
    rows.append({"dim": SHORT[c], "ICC1_single": round(i1, 2), "ICC1k_mean": round(ik, 2), "k0": round(k0, 1),
                 "within_unit_rel_human": round(w, 2), "grade": grade})
AX = pd.DataFrame(rows); AX.to_csv(RESULTS / "layer1_axes.csv", index=False)
print("\n== ① 공유 축: 차원별 신뢰도 (D960, 사람 연주) =="); print(AX.to_string(index=False))

def loo_consensus(dim):
    m = d.pivot_table(index="segment_id", columns="user", values=dim)
    s = m.sum(1); n = m.notna().sum(1)
    loo = (s.to_numpy()[:, None] - np.nan_to_num(m.to_numpy())) / np.maximum(n.to_numpy()[:, None] - 1, 1)
    return m, pd.DataFrame(loo, index=m.index, columns=m.columns).where(m.notna())

def slopes(dim, seg_mask=None):
    m, loo = loo_consensus(dim)
    if seg_mask is not None:
        m, loo = m.loc[seg_mask], loo.loc[seg_mask]
    out = {}
    for u in m.columns:
        y, x = m[u].to_numpy(), loo[u].to_numpy(); ok = ~np.isnan(y) & ~np.isnan(x)
        if ok.sum() >= 15 and np.std(x[ok]) > 0:
            out[u] = float(np.polyfit(x[ok], y[ok], 1)[0])
    return pd.Series(out)

Wmat = pd.DataFrame({SHORT[c]: slopes(c) for c in L}); Wmat.to_csv(RESULTS / "layer2_rater_sensitivity.csv")
segs = d.segment_id.unique(); stab = {}
for c in L:
    rs = []
    for _ in range(50):
        pm = rng.permutation(segs); h = len(pm) // 2
        a, b = slopes(c, pm[:h]), slopes(c, pm[h:]); j = a.index.intersection(b.index)
        if len(j) >= 10: rs.append(float(np.corrcoef(a[j], b[j])[0, 1]))
    stab[SHORT[c]] = float(np.mean(rs))
SENS = pd.DataFrame({"mean_w": Wmat.mean(), "sd_w": Wmat.std(), "split_half_r": pd.Series(stab)}).round(2)
print("\n== ② (a) 축 민감도 w =="); print(SENS.to_string())

Z = {c: loo_consensus(c)[1] for c in DESC}
mconv = d.pivot_table(index="segment_id", columns="user", values=CONV)
def pref(seg_mask=None):
    out = {}
    for u in mconv.columns:
        y = mconv[u]; X = pd.DataFrame({c: Z[c][u] for c in DESC})
        if seg_mask is not None: y, X = y.loc[seg_mask], X.loc[seg_mask]
        ok = y.notna() & X.notna().all(1)
        if ok.sum() < 30: continue
        Xs = (X[ok] - X[ok].mean()) / (X[ok].std() + 1e-9); ys = (y[ok] - y[ok].mean()) / (y[ok].std() + 1e-9)
        A = np.c_[np.ones(ok.sum()), Xs.to_numpy()]
        beta, *_ = np.linalg.lstsq(A, ys.to_numpy(), rcond=None); out[u] = beta[1:]
    return pd.DataFrame(out, index=[SHORT[c] for c in DESC]).T
B = pref(); B.to_csv(RESULTS / "layer2_rater_preference.csv")
rs = []
for _ in range(50):
    pm = rng.permutation(segs); h = len(pm) // 2
    a, b = pref(pm[:h]), pref(pm[h:]); j = a.index.intersection(b.index)
    if len(j) >= 10: rs.append([float(np.corrcoef(a.loc[j, c], b.loc[j, c])[0, 1]) for c in a.columns])
PREF = pd.DataFrame({"mean_beta": B.mean(), "sd_beta": B.std(), "split_half_r": np.mean(rs, 0)}).round(2)
print(f"\n== ② (b) 선호 모형 (평가자 {len(B)}명) =="); print(PREF.to_string())
from scipy.cluster.vq import kmeans2
Bz = (B - B.mean()) / (B.std() + 1e-9); labs = []
for s in range(20):
    _, lab = kmeans2(Bz.to_numpy(), 2, seed=s, minit="++"); labs.append(lab)
ref = labs[0]; agree = [max(np.mean(lab == ref), np.mean(lab != ref)) for lab in labs[1:]]
_, lab = kmeans2(Bz.to_numpy(), 2, seed=0, minit="++")
prof = B.groupby(lab).mean().round(2); prof["n"] = pd.Series(lab).value_counts().sort_index().values
print("\n== 평가자 유형 (k=2), 재실행 일치율 %.2f ==" % np.mean(agree)); print(prof.to_string())
json.dump({"n_raters": int(len(raters)), "axes": AX.to_dict("records"), "sensitivity": SENS.to_dict("index"),
           "preference": PREF.to_dict("index"), "cluster_agreement": float(np.mean(agree)),
           "cluster_profiles": prof.to_dict("index")}, open(RESULTS / "rater_layers_summary.json", "w"), ensure_ascii=False, indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); order = AX.sort_values("within_unit_rel_human").dim
ax[0].barh(order, AX.set_index("dim").loc[order, "within_unit_rel_human"], color="#2a78d6"); ax[0].axvline(0.5, ls="--", c="gray"); ax[0].set_title("shared axes: within-unit reliability (D960, human)"); ax[0].set_xlim(0, 1)
cols = [SHORT[c] for c in DESC]
ax[1].boxplot([B[c].dropna() for c in cols], vert=False, tick_labels=cols, showfliers=False); ax[1].axvline(0, c="gray"); ax[1].set_title("per-rater convincing preference (standardized)")
fig.tight_layout(); fig.savefig(RESULTS / "rater_layers.png", dpi=130); print("saved figure")
