"""Q7 · 감도 w 와 선호 β 재추정: 사람 연주만, 평가자 보정, 단위 내 중심화, 평가자별 표준화."""
import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.cluster.vq import kmeans2
from q_common import *
rng = np.random.default_rng(0)
d = rater_adjusted(d960(ratings("mean"), human_only=True), L)
unit_of = d.drop_duplicates("segment_id").set_index("segment_id").unit
segs = np.array(sorted(d.segment_id.unique()))

def prep(c, standardize):
    dd = d.copy()
    if standardize: dd[c] = dd.groupby("user")[c].transform(lambda v: (v - v.mean()) / (v.std() + 1e-9))
    m, loo = loo_consensus(dd, c); u = unit_of.reindex(m.index)
    um = loo.mean(1).groupby(u).transform("mean")           # 단위 평균 (합의 기준)
    return m.sub(um, axis=0), loo.sub(um, axis=0)
def slopes(m, loo, seg_mask=None, min_n=15):
    if seg_mask is not None: m, loo = m.loc[m.index.isin(seg_mask)], loo.loc[loo.index.isin(seg_mask)]
    out = {}
    for u in m.columns:
        y, x = m[u].to_numpy(), loo[u].to_numpy(); ok = ~np.isnan(y) & ~np.isnan(x)
        if ok.sum() >= min_n and np.std(x[ok]) > 0: out[u] = float(np.polyfit(x[ok], y[ok], 1)[0])
    return pd.Series(out)
def split_half(fn, n=50):
    rs = []
    for _ in range(n):
        pm = rng.permutation(segs); h = len(pm) // 2; a, b = fn(pm[:h]), fn(pm[h:]); j = a.index.intersection(b.index)
        if len(j) >= 10: rs.append(float(np.corrcoef(a[j], b[j])[0, 1]))
    return float(np.mean(rs))
res = {}; Wraw = {}; Wstd = {}
for c in L:
    r = {}
    for std in (False, True):
        m, loo = prep(c, std); w = slopes(m, loo); r[f"mean_w_{'std' if std else 'raw'}"] = round(w.mean(), 2); r[f"sd_w_{'std' if std else 'raw'}"] = round(w.std(), 2)
        r[f"splithalf_{'std' if std else 'raw'}"] = round(split_half(lambda s: slopes(m, loo, s)), 2); (Wstd if std else Wraw)[SHORT[c]] = w
    res[SHORT[c]] = r
T = pd.DataFrame(res).T; print("== 축 민감도 w (단위 내, 평가자 보정): raw vs 평가자별 표준화 ==\n", T.to_string())
Wr, Ws = pd.DataFrame(Wraw), pd.DataFrame(Wstd)
rsd = d.groupby("user")[L].std().mean(1)
pc1 = lambda W: float(np.linalg.svd((W - W.mean()).dropna().to_numpy() / (W.std().to_numpy() + 1e-9), compute_uv=False)[0] ** 2 / ((W - W.mean()).dropna().to_numpy() / (W.std().to_numpy() + 1e-9) ** 2).sum()) if False else None
def pc1share(W):
    Z = ((W - W.mean()) / (W.std() + 1e-9)).dropna(); s = np.linalg.svd(Z.to_numpy(), compute_uv=False); return float(s[0] ** 2 / (s ** 2).sum())
print("\nw 의 평가자 간 1주성분 몫: raw %.2f, std %.2f · corr(평균 w_raw, 평가자 자기 SD) = %.2f · corr(평균 w_std, 자기 SD) = %.2f" % (
    pc1share(Wr), pc1share(Ws), Wr.mean(1).corr(rsd.reindex(Wr.index)), Ws.mean(1).corr(rsd.reindex(Ws.index))))
# Q5 자기 일관성(재검사 r)과 w 의 상관
q3 = pd.read_csv(OUT / "q03_raters.csv").set_index("user").retest_r.dropna()
print("corr(평균 w_std, 재검사 r) (n=%d) = %.2f" % (len(q3), Ws.mean(1).reindex(q3.index).corr(q3)))

# β: 설득력(표준화·보정·단위 내) ~ EXEC 8 합의(단위 내 중심화, LOO)
mC, looC = prep(CONV, True); Z = {c: prep(c, False)[1] for c in EXEC}
def pref(seg_mask=None):
    out = {}
    for u in mC.columns:
        y = mC[u]; X = pd.DataFrame({c: Z[c][u] for c in EXEC})
        if seg_mask is not None: y, X = y[y.index.isin(seg_mask)], X[X.index.isin(seg_mask)]
        ok = y.notna() & X.notna().all(1)
        if ok.sum() < 30: continue
        Xs = (X[ok] - X[ok].mean()) / (X[ok].std() + 1e-9); A = np.c_[np.ones(ok.sum()), Xs.to_numpy()]
        beta = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)[0]; out[u] = beta[1:]
    return pd.DataFrame(out, index=[SHORT[c] for c in EXEC]).T
B = pref(); sh = []
for _ in range(50):
    pm = rng.permutation(segs); h = len(pm) // 2; a, b = pref(pm[:h]), pref(pm[h:]); j = a.index.intersection(b.index)
    if len(j) >= 10: sh.append([float(np.corrcoef(a.loc[j, c], b.loc[j, c])[0, 1]) for c in a.columns])
PB = pd.DataFrame({"mean_beta": B.mean(), "sd_beta": B.std(), "splithalf": np.mean(sh, 0)}).round(2)
print(f"\n== 설득력 선호 β (평가자 {len(B)}명, 단위 내·보정·표준화) ==\n", PB.to_string())
Bz = ((B - B.mean()) / (B.std() + 1e-9)).to_numpy()
bic = {k: float(GaussianMixture(k, covariance_type="diag", random_state=0, n_init=5).fit(Bz).bic(Bz)) for k in (1, 2, 3)}
labs = [kmeans2(Bz, 2, seed=s, minit="++")[1] for s in range(20)]; agree = float(np.mean([max(np.mean(l == labs[0]), np.mean(l != labs[0])) for l in labs[1:]]))
print("\n유형: BIC", {k: round(v, 1) for k, v in bic.items()}, "· k=2 재실행 일치율 %.2f" % agree)
shared = ["dry→wet", "clean→blurred", "fast→slow", "stable→unstable", "long→short*", "cushioned→solid", "shallow→rich", "flat→spacious"]
n_w = int((T.loc[shared, "splithalf_std"] >= 0.5).sum()); n_b = int((PB.splithalf >= 0.3).sum())
verdict = dict(w_trait=bool(n_w >= 3), n_shared_axes_w_ge_05=n_w, beta_keep=bool(n_b >= 4), n_beta_ge_03=n_b, types_keep=bool(agree >= 0.9 and bic[2] < bic[1]), bic=bic, kmeans_agree=agree)
print("\n판정:", verdict)
save("q07_w_beta", dict(w=res, beta=PB.to_dict("index"), verdict=verdict, pc1_raw=pc1share(Wr), pc1_std=pc1share(Ws)))
