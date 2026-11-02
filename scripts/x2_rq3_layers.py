"""X2 · RQ3 교차 검정: 개인층 잔차를 관습 방향 성분 α 와 직교 성분 ρ 로 나눠, 공유 축 라벨은 α 로, 설득력 개인차는 ρ 로 읽히는지.
(설계·코드만, 미실행 2026-10-25)   사전등록: docs/x1x2_prereg_2026-10-25.md (H3a, H3b, H3c)

분해 (단위 u, 전역 z-점수 특징 공간)
  letter_u = Score 렌더 특징,  x̄_u = 사람 평균,  c_u = (x̄_u − letter_u)/‖·‖  (관습 방향)
  r_j = x_j − x̄_u^(−j)  (LOO 잔차),  α_j = r_j·c_u,  ρ_j = r_j − α_j c_u
H3a  같은 단위 사람 쌍: |Δ합의라벨| vs |Δα| / ‖Δρ‖ / ‖Δr‖  (단위 내 순위 풀링 Spearman, 라벨 셔플 순열)
H3b  설득력 개인 편차 e_ij ~ ρ_j, 평가자 유형(k=2) × ρ 상호작용 (유형 라벨 순열)
H3c  ‖ρ_j‖ vs 설득력 합의 평균(−), vs 평가자 간 SD(+)   (Repp 1997 역설의 기계론)
H3d  D = ρ(‖ρ‖, SD) − ρ(|α|, SD): 의견 갈림이 직교 성분에 특이한가 (최소 거리 가설 vs 층 가설, 2026-11-01 추가)
대조  c_u 를 무작위 방향으로 100회 → H3a 의 ρ(|Δα|) 분포, H3d 의 D 분포

입력  data/interim/pp_ratings_clean.parquet (없으면 원자료 정제), data/interim/pp_features.parquet (Score 세그먼트 포함)
출력  reports/x2/rq3_summary.json, rq3_pairs.csv, alpha_rho_coords.csv, rq3_layers.png
실행  .venv/bin/python -I scripts/x2_rq3_layers.py [--whiten] [--perm 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eda import paths, percepiano_io as pp  # noqa: E402

L, S = pp.LABELS19, pp.SHORT19
SHORT = dict(zip(L, S))
EXEC = ["Timing_Stable_Unstable", "Articulation_Long_Short", "Articulation_Soft_cushioned_Hard_solid",
        "Pedal_Sparse/dry_Saturated/wet", "Pedal_Clean_Blurred", "Timbre_Shallow_Rich",
        "Dynamic_Little_dynamic_range_Large_dynamic_range", "Music_Making_Fast_paced_Slow_paced"]
CONV = "Interpretation_Unsatisfactory/doubtful_Convincing"
FX = ["log_tempo", "tempo_logstd", "timing_std_ms", "articulation_median", "vel_std", "vel_range", "pedal_down_ratio"]
MIN_SEG, MIN_HUM = 40, 4
OUT = ROOT / "reports" / "x2"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- 입력
def load_ratings() -> pd.DataFrame:
    p = paths.INTERIM / "pp_ratings_clean.parquet"
    d = pd.read_parquet(p) if p.exists() else pp.clean(pp.load_raw())
    d = d[d.piece.str.startswith("Schubert_D960")].copy()
    cnt = d[~d.is_score].groupby("user").size()
    return d[d.user.isin(cnt[cnt >= MIN_SEG].index)].reset_index(drop=True)


def load_features() -> pd.DataFrame:
    f = pd.read_parquet(paths.INTERIM / "pp_features.parquet")
    if "align_error" in f.columns:
        f = f[f.align_error.isna()]
    if {"match", "score_notes"} <= set(f.columns):
        f = f[(f.match / f.score_notes) >= 0.8]
    if "log_tempo" not in f.columns and "tempo_median_bpm" in f.columns:
        f = f.assign(log_tempo=np.log2(f.tempo_median_bpm))
    return f.dropna(subset=FX).drop_duplicates("segment_id").set_index("segment_id")[FX].astype(float)


# ----------------------------------------------------------------------------- 분해
def decompose(seg: pd.DataFrame, Xz: pd.DataFrame, whiten: bool, rng=None, random_dir=False):
    """seg: segment_id, unit, is_score.  반환: 사람 세그먼트의 r, α, ρ 좌표 DataFrame."""
    W = np.eye(len(FX))
    if whiten:  # 풀 사람 잔차 공분산으로 백색화
        R = []
        for u, g in seg[~seg.is_score].groupby("unit"):
            X = Xz.loc[g.segment_id].to_numpy(); R.append(X - X.mean(0))
        C = np.cov(np.vstack(R).T) + 1e-6 * np.eye(len(FX))
        ev, V = np.linalg.eigh(C); W = V @ np.diag(ev ** -0.5) @ V.T
    rows = []
    for u, g in seg.groupby("unit"):
        hum, sc = g[~g.is_score], g[g.is_score]
        if len(hum) < MIN_HUM or len(sc) == 0:
            continue
        Xh = Xz.loc[hum.segment_id].to_numpy() @ W; Xs = Xz.loc[sc.segment_id].to_numpy().mean(0) @ W
        mean_all = Xh.mean(0)
        c = (mean_all - Xs)
        if random_dir:
            c = rng.normal(size=len(FX))
        c = c / (np.linalg.norm(c) + 1e-12)
        for i, sid in enumerate(hum.segment_id):
            loo = (Xh.sum(0) - Xh[i]) / (len(Xh) - 1)
            r = Xh[i] - loo; a = float(r @ c); rho = r - a * c
            rows.append(dict(segment_id=sid, unit=u, alpha=a, rho_norm=float(np.linalg.norm(rho)), r_norm=float(np.linalg.norm(r)),
                             conv_dist=float(np.linalg.norm(Xh[i] - Xs)),
                             **{f"r_{k}": v for k, v in zip(FX, r)}, **{f"rho_{k}": v for k, v in zip(FX, rho)}))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 통계 도구
def pooled_rank_spearman(x, y, units):
    """단위 내 순위로 바꾼 뒤 풀링한 Spearman (단위 간 차이를 제거)."""
    rx, ry = np.empty_like(x), np.empty_like(y)
    for u in np.unique(units):
        m = units == u; rx[m] = rankdata(x[m]); ry[m] = rankdata(y[m])
    return float(spearmanr(rx, ry).correlation)


def perm_p(x, y, units, stat, n_perm, rng):
    """단위 내 y 셔플 순열 p (한쪽 꼬리: |통계| 기준 양쪽)."""
    obs = stat(x, y, units); cnt = 0
    for _ in range(n_perm):
        yp = y.copy()
        for u in np.unique(units):
            m = np.where(units == u)[0]; yp[m] = y[rng.permutation(m)]
        if abs(stat(x, yp, units)) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ----------------------------------------------------------------------------- H3a
def h3a(coords: pd.DataFrame, segmean: pd.DataFrame, n_perm, rng):
    C = coords.set_index("segment_id"); rho_cols = [f"rho_{k}" for k in FX]; r_cols = [f"r_{k}" for k in FX]
    pairs = []
    for u, g in C.groupby("unit"):
        ids = list(g.index)
        for a, b in combinations(ids, 2):
            pairs.append(dict(unit=u, a=a, b=b, d_alpha=abs(C.at[a, "alpha"] - C.at[b, "alpha"]),
                              d_rho=float(np.linalg.norm(C.loc[a, rho_cols].to_numpy() - C.loc[b, rho_cols].to_numpy())),
                              d_r=float(np.linalg.norm(C.loc[a, r_cols].to_numpy() - C.loc[b, r_cols].to_numpy()))))
    P = pd.DataFrame(pairs); units = P.unit.to_numpy()
    rows = []
    for dim in EXEC:
        lab = segmean.set_index("segment_id")[dim]
        dl = np.abs(lab.reindex(P.a).to_numpy() - lab.reindex(P.b).to_numpy())
        ok = ~np.isnan(dl)
        rec = dict(dim=SHORT[dim], n_pairs=int(ok.sum()))
        for name in ("d_alpha", "d_rho", "d_r"):
            obs, p = perm_p(P[name].to_numpy()[ok], dl[ok], units[ok], pooled_rank_spearman, n_perm, rng)
            rec[f"rho_{name}"] = round(obs, 3); rec[f"p_{name}"] = p
        rows.append(rec)
    T = pd.DataFrame(rows)
    for name in ("d_alpha", "d_rho", "d_r"):
        T[f"q_{name}"] = bh(T[f"p_{name}"])
    T["supports_H3a"] = (T.rho_d_alpha >= T.rho_d_r - 0.03) & (T.rho_d_alpha > T.rho_d_rho + 0.05)
    return T, P


# ----------------------------------------------------------------------------- H3b
def h3b(coords: pd.DataFrame, d: pd.DataFrame, n_perm, rng, lam=1.0, k=2):
    """e_ij = conv_ij − LOO 단위 합의(평가자 i 제외) ~ ρ_j.  유형 × ρ 상호작용의 보류 R² 이득을 유형 라벨 순열로 검정."""
    rho_cols = [f"rho_{k_}" for k_ in FX]
    C = coords.set_index("segment_id")
    hum = d[~d.is_score & d.segment_id.isin(C.index)]
    m = hum.pivot_table(index="segment_id", columns="user", values=CONV)
    s, n = m.sum(1), m.notna().sum(1)
    loo = (s.to_numpy()[:, None] - np.nan_to_num(m.to_numpy())) / np.maximum(n.to_numpy()[:, None] - 1, 1)
    E = pd.DataFrame(m.to_numpy() - loo, index=m.index, columns=m.columns).where(m.notna())
    Rho = C.loc[m.index, rho_cols].to_numpy(); Rho = (Rho - Rho.mean(0)) / (Rho.std(0) + 1e-9)
    # 평가자별 β^ρ
    B, users = [], []
    for u in E.columns:
        y = E[u].to_numpy(); ok = ~np.isnan(y)
        if ok.sum() < 30:
            continue
        X = Rho[ok]; beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y[ok]); B.append(beta); users.append(u)
    B = np.stack(B)
    # 유형
    from scipy.cluster.vq import kmeans2
    Bz = (B - B.mean(0)) / (B.std(0) + 1e-9)
    _, lab = kmeans2(Bz, k, seed=0, minit="++")
    cos = float(B[lab == 0].mean(0) @ B[lab == 1].mean(0) / (np.linalg.norm(B[lab == 0].mean(0)) * np.linalg.norm(B[lab == 1].mean(0)) + 1e-12)) if k == 2 else np.nan

    def heldout_gain(lab, folds=5):
        """세그먼트 5겹: 풀 β vs 유형별 β 의 보류 R² 차이."""
        segs = np.array(m.index); f = rng.permutation(len(segs)) % folds
        sse_pool = sse_type = sst = 0.0
        for kf in range(folds):
            tr, te = f != kf, f == kf
            Ytr = E.loc[segs[tr], users]; Yte = E.loc[segs[te], users]
            rows_tr = [(Rho[tr][np.flatnonzero(~np.isnan(Ytr[u].to_numpy()))], Ytr[u].dropna().to_numpy(), lab[ui]) for ui, u in enumerate(users)]
            Xp = np.vstack([x for x, _, _ in rows_tr]); yp = np.concatenate([y for _, y, _ in rows_tr])
            bp = np.linalg.solve(Xp.T @ Xp + lam * np.eye(Xp.shape[1]), Xp.T @ yp)
            bt = {}
            for t in np.unique(lab):
                Xt = np.vstack([x for x, _, l in rows_tr if l == t]); yt = np.concatenate([y for _, y, l in rows_tr if l == t])
                bt[t] = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1]), Xt.T @ yt)
            for ui, u in enumerate(users):
                y = Yte[u].to_numpy(); ok = ~np.isnan(y)
                if not ok.any():
                    continue
                X = Rho[te][ok]
                sse_pool += ((y[ok] - X @ bp) ** 2).sum(); sse_type += ((y[ok] - X @ bt[lab[ui]]) ** 2).sum(); sst += ((y[ok] - y[ok].mean()) ** 2).sum()
        return (sse_pool - sse_type) / sst

    obs = heldout_gain(lab)
    cnt = sum(heldout_gain(rng.permutation(lab)) >= obs for _ in range(min(n_perm, 300)))
    p = (cnt + 1) / (min(n_perm, 300) + 1)
    prof = pd.DataFrame(B, index=users, columns=FX).assign(type=lab)
    return dict(n_raters=len(users), r2_gain_type_vs_pool=round(float(obs), 4), perm_p=p, cos_between_types=round(cos, 3),
                supports_H3b=bool(p < .05 and cos < 0)), prof


# ----------------------------------------------------------------------------- H3c
def h3c(coords: pd.DataFrame, d: pd.DataFrame, n_perm, rng):
    C = coords.set_index("segment_id")
    hum = d[~d.is_score & d.segment_id.isin(C.index)]
    g = hum.groupby("segment_id")[CONV]
    stats = pd.DataFrame({"conv_mean": g.mean(), "conv_sd": g.std(), "n": g.size()}).reindex(C.index)
    ok = (stats.n >= 4).to_numpy()
    units = C.unit.to_numpy()[ok]
    out = {}
    for xname in ("rho_norm", "r_norm", "alpha_abs"):
        x = (C.alpha.abs() if xname == "alpha_abs" else C[xname]).to_numpy()[ok]
        for yname in ("conv_mean", "conv_sd"):
            y = stats[yname].to_numpy()[ok]; m = ~np.isnan(y)
            r, p = perm_p(x[m], y[m], units[m], pooled_rank_spearman, n_perm, rng)
            out[f"{xname}__{yname}"] = dict(rho=round(r, 3), p=p, n=int(m.sum()))
    q = bh([v["p"] for v in out.values()])
    for (k_, v), qq in zip(out.items(), q):
        v["q"] = round(float(qq), 4)
    a, b = out["rho_norm__conv_mean"], out["rho_norm__conv_sd"]
    out["supports_H3c"] = bool(a["rho"] <= -0.10 and a["q"] < .05 and b["rho"] >= 0.10 and b["q"] < .05)
    out["partial_mean_only"] = bool(a["rho"] <= -0.10 and a["q"] < .05 and not out["supports_H3c"])
    return out


# ----------------------------------------------------------------------------- H3d
def conv_sd(coords: pd.DataFrame, d: pd.DataFrame) -> pd.Series:
    """구간별 설득력 평가자 간 SD (평가 4개 이상). H3c 와 같은 정의."""
    hum = d[~d.is_score & d.segment_id.isin(coords.segment_id)]
    g = hum.groupby("segment_id")[CONV]
    return g.std()[g.size() >= 4].dropna()


def contrast(C: pd.DataFrame, sd: pd.Series):
    """C: segment_id 인덱스 좌표. 반환 (s_ρ, s_α, D = s_ρ − s_α)."""
    C = C.loc[sd.index]; y = sd.to_numpy(); u = C.unit.to_numpy()
    s_rho = pooled_rank_spearman(C.rho_norm.to_numpy(), y, u)
    s_alpha = pooled_rank_spearman(C.alpha.abs().to_numpy(), y, u)
    return s_rho, s_alpha, s_rho - s_alpha


def h3d(coords: pd.DataFrame, d: pd.DataFrame, h3c_out: dict, rnd_d, n_boot, rng):
    """최소 거리 가설 vs 층 가설. 판정 규칙은 docs/x1x2_prereg_2026-10-25.md 의 사후 변경 기록(2026-11-01)."""
    sd = conv_sd(coords, d); C = coords.set_index("segment_id").loc[sd.index]
    s_rho, s_alpha, D = contrast(C, sd)
    unit = C.unit.to_numpy(); units = np.unique(unit); idx = {u: np.flatnonzero(unit == u) for u in units}
    x_r, x_a, y = C.rho_norm.to_numpy(), C.alpha.abs().to_numpy(), sd.to_numpy()
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(units, len(units))
        ii = np.concatenate([idx[u] for u in pick]); uu = np.concatenate([np.full(len(idx[u]), i) for i, u in enumerate(pick)])
        boot.append(pooled_rank_spearman(x_r[ii], y[ii], uu) - pooled_rank_spearman(x_a[ii], y[ii], uu))
    lo, hi = np.percentile(boot, [2.5, 97.5]); p95 = float(np.percentile(rnd_d, 95))
    perf = C.index.map(lambda s: pp.parse_name(s)["performer"]).to_numpy()
    lopo = [contrast(C[perf != p], sd[perf != p])[2] for p in np.unique(perf)]  # 판정에는 쓰지 않는 민감도
    sig_r = s_rho >= 0.10 and h3c_out["rho_norm__conv_sd"]["q"] < .05
    sig_a = s_alpha >= 0.10 and h3c_out["alpha_abs__conv_sd"]["q"] < .05
    if sig_r and D >= 0.10 and lo > 0 and D > p95:
        verdict = "layer"
    elif sig_r and sig_a and abs(D) < 0.10:
        verdict = "min_distance"
    elif sig_a and s_alpha >= s_rho + 0.10:
        verdict = "reject_layer"
    elif not sig_r and not sig_a:
        verdict = "neither"
    else:
        verdict = "inconclusive"
    return dict(s_rho=round(s_rho, 3), s_alpha=round(s_alpha, 3), D=round(D, 3), ci95=[round(float(lo), 3), round(float(hi), 3)],
                random_dir_p95=round(p95, 3), lopo_min=round(float(min(lopo)), 3), lopo_max=round(float(max(lopo)), 3), verdict=verdict)


# ----------------------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whiten", action="store_true"); ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d, F = load_ratings(), load_features()
    seg = d.drop_duplicates("segment_id")[["segment_id", "unit", "is_score"]]
    seg = seg[seg.segment_id.isin(F.index)].reset_index(drop=True)
    Xz = (F.loc[seg.segment_id] - F.loc[seg[~seg.is_score].segment_id].mean()) / (F.loc[seg[~seg.is_score].segment_id].std() + 1e-9)
    coords = decompose(seg, Xz, a.whiten)
    coords.to_csv(OUT / "alpha_rho_coords.csv", index=False)
    print(f"단위 {coords.unit.nunique()} · 사람 세그먼트 {len(coords)} · 평가자 {d.user.nunique()}  (whiten={a.whiten})")
    print("관습 방향 양 α: 평균 %.3f, SD %.3f · ‖ρ‖ 중앙값 %.3f · ‖r‖ 중앙값 %.3f" % (coords.alpha.mean(), coords.alpha.std(), coords.rho_norm.median(), coords.r_norm.median()))

    segmean = pp.segment_means(d[~d.is_score])
    T, P = h3a(coords, segmean, a.perm, rng)
    P.to_csv(OUT / "rq3_pairs.csv", index=False)
    print("\n== H3a: |Δ합의라벨| vs |Δα| / ‖Δρ‖ / ‖Δr‖ (단위 내 순위 Spearman) =="); print(T.to_string(index=False))
    # 대조: 무작위 방향
    rnd, rnd_d = [], []
    sd = conv_sd(coords, d)
    for _ in range(100):
        Cr = decompose(seg, Xz, a.whiten, rng, random_dir=True).set_index("segment_id")  # 순열 없이 ρ(|Δα|)만
        rnd_d.append(contrast(Cr, sd)[2])
        vals = []
        for dim in EXEC:
            lab = segmean.set_index("segment_id")[dim]
            da = np.abs(Cr.alpha.reindex(P.a).to_numpy() - Cr.alpha.reindex(P.b).to_numpy())
            dl = np.abs(lab.reindex(P.a).to_numpy() - lab.reindex(P.b).to_numpy()); ok = ~np.isnan(dl) & ~np.isnan(da)
            vals.append(pooled_rank_spearman(da[ok], dl[ok], P.unit.to_numpy()[ok]))
        rnd.append(vals)
    rnd = np.array(rnd); T["rho_d_alpha_random_p95"] = np.percentile(rnd, 95, axis=0).round(3)
    T["beats_random"] = T.rho_d_alpha > T.rho_d_alpha_random_p95

    B, prof = h3b(coords, d, a.perm, rng); prof.to_csv(OUT / "h3b_rater_beta_rho.csv")
    print("\n== H3b: 설득력 개인 편차 ~ ρ, 유형 × ρ =="); print(json.dumps(B, ensure_ascii=False, indent=1))
    Cc = h3c(coords, d, a.perm, rng)
    print("\n== H3c: ‖ρ‖ vs 설득력 합의 평균 / 평가자 간 SD =="); print(json.dumps(Cc, ensure_ascii=False, indent=1))
    Dd = h3d(coords, d, Cc, rnd_d, 2000, rng)
    print("\n== H3d: 최소 거리 가설 vs 층 가설 =="); print(json.dumps(Dd, ensure_ascii=False, indent=1))

    summary = dict(whiten=a.whiten, n_units=int(coords.unit.nunique()), n_segments=len(coords), n_raters=int(d.user.nunique()),
                   H3a=T.to_dict("records"), H3a_supported=bool(T.supports_H3a.sum() >= 4), H3a_dims_supporting=int(T.supports_H3a.sum()),
                   H3b=B, H3c=Cc, H3d=Dd)
    json.dump(summary, open(OUT / "rq3_summary.json", "w"), ensure_ascii=False, indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(T)); wd = 0.27
    for i, (nm, c) in enumerate((("rho_d_alpha", "#9b2335"), ("rho_d_r", "gray"), ("rho_d_rho", "#b08d57"))):
        ax[0].bar(x + (i - 1) * wd, T[nm], wd, color=c, label=nm.replace("rho_", ""))
    ax[0].set_xticks(x); ax[0].set_xticklabels(T.dim, rotation=35, ha="right"); ax[0].legend(); ax[0].set_title("H3a: which component explains shared-axis label differences")
    st = d[~d.is_score].groupby("segment_id")[CONV].agg(["mean", "std"]).reindex(coords.segment_id)
    ax[1].scatter(coords.rho_norm, st["mean"], s=10, alpha=.5); ax[1].set_xlabel("||rho|| (individual, orthogonal)"); ax[1].set_ylabel("convincing mean"); ax[1].set_title("H3c: Repp paradox (mean)")
    ax[2].scatter(coords.rho_norm, st["std"], s=10, alpha=.5, color="#b08d57"); ax[2].set_xlabel("||rho||"); ax[2].set_ylabel("convincing SD across raters"); ax[2].set_title("H3c: disagreement")
    fig.tight_layout(); fig.savefig(OUT / "rq3_layers.png", dpi=130)
    print("saved", OUT)


if __name__ == "__main__":
    main()
