"""신뢰도·분산성분·재표집 검정 유틸리티."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats


def icc_oneway(df: pd.DataFrame, target: str, value: str) -> dict:
    """불균형 일원 무선효과 ICC (ANOVA 추정).

    k0 = (N - Σn_i²/N)/(a-1) 를 쓰는 표준 불균형 보정.
    ICC(1)   = (MSB - MSW) / (MSB + (k0-1) MSW)
    ICC(1,k) = (MSB - MSW) / MSB   (k = k0 명의 평균에 대한 신뢰도)
    """
    d = df[[target, value]].dropna()
    g = d.groupby(target)[value]
    n_i = g.size().to_numpy(float)
    a, N = len(n_i), n_i.sum()
    grand = d[value].mean()
    ssb = (n_i * (g.mean().to_numpy() - grand) ** 2).sum()
    ssw = ((d[value] - g.transform("mean")) ** 2).sum()
    msb, msw = ssb / (a - 1), ssw / (N - a)
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    s2_t = max((msb - msw) / k0, 0.0)
    return {
        "icc1": (msb - msw) / (msb + (k0 - 1) * msw),
        "icck": (msb - msw) / msb,
        "k0": k0,
        "var_target": s2_t,
        "var_error": msw,
        "n_targets": a,
        "n_obs": int(N),
    }


def spearman_brown(r: float, k: float) -> float:
    return k * r / (1 + (k - 1) * r)


def implied_k(icc1: float, icck: float) -> float:
    """ICC(1)와 ICC(1,k)가 주어졌을 때 Spearman–Brown 이 암시하는 k."""
    return icck * (1 - icc1) / (icc1 * (1 - icck))


def per_target_reliability(var_target: float, var_error: float, k_i: np.ndarray) -> np.ndarray:
    """구간별 평균 점수의 신뢰도 rel_i = σ²_t / (σ²_t + σ²_e / k_i)."""
    k_i = np.asarray(k_i, float)
    return var_target / (var_target + var_error / k_i)


def split_half_reliability(df: pd.DataFrame, target: str, rater: str, value: str,
                           n_iter: int = 200, seed: int = 0, min_per_half: int = 2) -> dict:
    """구간마다 평가자를 무작위로 반분 → 두 반쪽 평균의 상관 → Spearman–Brown(2배)."""
    rng = np.random.default_rng(seed)
    d = df[[target, rater, value]].dropna()
    groups = [g[value].to_numpy() for _, g in d.groupby(target) if len(g) >= 2 * min_per_half]
    rs = []
    for _ in range(n_iter):
        a, b = [], []
        for v in groups:
            p = rng.permutation(len(v))
            h = len(v) // 2
            a.append(v[p[:h]].mean())
            b.append(v[p[h:]].mean())
        rs.append(np.corrcoef(a, b)[0, 1])
    r = float(np.mean(rs))
    return {"r_half": r, "sb_full": spearman_brown(r, 2), "r_half_sd": float(np.std(rs)), "n_targets": len(groups)}


def exact_perm_corr(x, y, method: str = "spearman") -> tuple[float, float]:
    """n ≤ 9 정도에서 모든 순열을 열거한 양측 p값."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = {"spearman": lambda a, b: stats.spearmanr(a, b)[0],
         "kendall": lambda a, b: stats.kendalltau(a, b)[0],
         "pearson": lambda a, b: stats.pearsonr(a, b)[0]}[method]
    obs = f(x, y)
    perms = itertools.permutations(range(len(y)))
    null = np.array([f(x, y[list(p)]) for p in perms])
    return float(obs), float((np.abs(null) >= abs(obs) - 1e-12).mean())


def loo_range(x, y, method: str = "pearson") -> tuple[float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    vals = [f(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))]
    return float(min(vals)), float(max(vals))


def cluster_bootstrap_corr(df: pd.DataFrame, x: str, y: str, cluster: str,
                           n_boot: int = 2000, seed: int = 0, method: str = "spearman") -> tuple[float, float, float]:
    """클러스터(같은 연주의 겹치는 구간 등) 단위 재표집으로 상관의 95% CI."""
    rng = np.random.default_rng(seed)
    d = df[[x, y, cluster]].dropna()
    f = (lambda a, b: stats.spearmanr(a, b)[0]) if method == "spearman" else (lambda a, b: stats.pearsonr(a, b)[0])
    obs = f(d[x], d[y])
    idx = {c: g.index.to_numpy() for c, g in d.groupby(cluster)}
    keys = list(idx)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        rows = np.concatenate([idx[keys[i]] for i in pick])
        s = d.loc[rows]
        boots.append(f(s[x], s[y]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def se_reference(sd: float, n: int) -> float:
    return sd / math.sqrt(n)


def two_way_additive(d: pd.DataFrame, target: str, rater: str, value: str, n_iter: int = 50):
    """y = μ + target + rater + e 를 교대 평균(최소제곱)으로 적합. 불균형 교차 설계용.

    반환: (target 효과 Series, rater 효과 Series, 잔차 분산 σ²_e)
    """
    y = d[value].to_numpy(float)
    t_codes, t_idx = pd.factorize(d[target])
    r_codes, r_idx = pd.factorize(d[rater])
    mu = y.mean()
    r_eff = np.zeros(len(r_idx))
    n_t = np.bincount(t_codes)
    n_r = np.bincount(r_codes)
    for _ in range(n_iter):
        t_eff = np.bincount(t_codes, weights=y - mu - r_eff[r_codes]) / n_t
        r_new = np.bincount(r_codes, weights=y - mu - t_eff[t_codes]) / n_r
        r_new -= r_new.mean()
        if np.max(np.abs(r_new - r_eff)) < 1e-10:
            r_eff = r_new
            break
        r_eff = r_new
    t_eff = np.bincount(t_codes, weights=y - mu - r_eff[r_codes]) / n_t
    resid = y - mu - t_eff[t_codes] - r_eff[r_codes]
    dof = len(y) - len(t_idx) - len(r_idx) + 1
    return (pd.Series(mu + t_eff, index=t_idx), pd.Series(r_eff, index=r_idx), float((resid ** 2).sum() / dof))


def variance_decomposition(d: pd.DataFrame, value: str, rater_adjust: bool = True,
                           target: str = "segment_id", unit: str = "unit", rater: str = "user") -> dict:
    """구간 점수의 분산을 '악보 단위 간'과 '같은 단위 안 연주 간'으로 나누는 모멘트 추정.

    1) 구간 효과를 추정합니다 (rater_adjust=True 면 평가자 관대도를 함께 적합해 제거).
    2) 관측된 구간 효과 분산에서 측정오차 σ²_e/k_i 를 빼서 참 분산을 추정합니다.
    3) 단위 안 중심화한 구간 효과에서 같은 방식으로 단위 내 참 분산을 구합니다.
    """
    dd = d[[target, unit, rater, value]].dropna()
    if rater_adjust:
        seg, rat_eff, s2e = two_way_additive(dd, target, rater, value)
    else:
        g = dd.groupby(target)[value]
        seg = g.mean()
        s2e = float(((dd[value] - g.transform("mean")) ** 2).sum() / (len(dd) - seg.size))
        rat_eff = None
    k = dd.groupby(target).size().reindex(seg.index).to_numpy(float)
    u = dd.groupby(target)[unit].first().reindex(seg.index)
    err = s2e / k
    s2_seg_total = max(seg.var(ddof=1) - err.mean(), 0.0)
    dev = seg - seg.groupby(u).transform("mean")
    n_u = u.map(u.value_counts()).to_numpy(float)
    # 단위 안 중심화 후 기대 분산: σ²_w (n-1)/n + 평균 오차 × (n-1)/n
    shrink = (n_u - 1) / n_u
    s2_within = max((dev ** 2).sum() / shrink.sum() - err.mean(), 0.0)
    s2_unit = max(s2_seg_total - s2_within, 0.0)
    return {
        "s2_unit": s2_unit,
        "s2_within": s2_within,
        "s2_error": s2e,
        "s2_rater": float(rat_eff.var()) if rat_eff is not None else np.nan,
        "share_unit": s2_unit / (s2_unit + s2_within) if (s2_unit + s2_within) > 0 else np.nan,
        "icc1": s2_seg_total / (s2_seg_total + s2e),
        "rel_mean": float(np.mean(s2_seg_total / (s2_seg_total + err))),
        "rel_within_mean": float(np.mean(s2_within / (s2_within + err))) if s2_within > 0 else 0.0,
        "k_mean": float(k.mean()),
        "n_segments": int(seg.size),
    }


def split_half_within_unit(d: pd.DataFrame, value: str, n_iter: int = 100, seed: int = 0,
                           target: str = "segment_id", unit: str = "unit", rater: str = "user") -> dict:
    """모형 없는 점검: 구간마다 평가자를 반분 → 두 반쪽의 '단위 평균 대비 편차' 상관 → Spearman–Brown."""
    rng = np.random.default_rng(seed)
    dd = d[[target, unit, rater, value]].dropna()
    groups = [(t, u, g[value].to_numpy()) for (t, u), g in dd.groupby([target, unit]) if len(g) >= 4]
    units = np.array([u for _, u, _ in groups])
    rs_all, rs_within = [], []
    for _ in range(n_iter):
        a, b = np.empty(len(groups)), np.empty(len(groups))
        for i, (_, _, v) in enumerate(groups):
            p = rng.permutation(len(v))
            h = len(v) // 2
            a[i], b[i] = v[p[:h]].mean(), v[p[h:]].mean()
        rs_all.append(np.corrcoef(a, b)[0, 1])
        sa = pd.Series(a).groupby(units).transform("mean").to_numpy()
        sb = pd.Series(b).groupby(units).transform("mean").to_numpy()
        rs_within.append(np.corrcoef(a - sa, b - sb)[0, 1])
    r_all, r_w = float(np.mean(rs_all)), float(np.mean(rs_within))
    return {"r_half_all": r_all, "sb_all": spearman_brown(r_all, 2),
            "r_half_within": r_w, "sb_within": spearman_brown(r_w, 2) if r_w > -1 / 1 else np.nan,
            "n_segments": len(groups)}
