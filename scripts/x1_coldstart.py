"""X1 · 엔진 A: 두 층 선호 모델과 cold-start 곡선. (설계·코드만, 미실행 2026-10-25)

사전등록: docs/x1x2_prereg_2026-10-25.md (X1-1 … X1-5). 판정 기준은 결과를 보기 전에 고정했다.

모델 (차원 d, 평가자 i, 연주 j ∈ 단위 u, x̃_j = 단위 내 LOO 중심화 z-특징)
  층 1 (실행 차원):  r̃_ijd = w_id · (g_d · x̃_j) + ε     g_d 공유 축, w_id 개인 감도
  층 2 (설득력):     r̃_ij  = β_i · x̃_j + ε               β_i 개인 선호, 유형 = k-means(β)
  쌍 형태(Bradley–Terry 선형 근사):  Δr = β · Δx̃  — 관측·예측 모두 같은 단위 쌍으로 한다.
모델 3종: S(합의 β/g) · T(유형 배정) · P(개인 릿지, 사전 평균 = T, 관측이 적을수록 수축)

cold-start: 보류 평가자 LOO → 단위 50:50 분할(20회) → 관측-풀에서 n∈{0,5,10,20,40} 쌍(무작위/능동) → 시험-풀 쌍 예측.
지표: 쌍 정확도(방향), 쌍 차이 MAE(크기; 층 1 의 w 효과).  대조: 관측 평가를 단위 내에서 섞음(X1-3).

입력  data/interim/pp_ratings_clean.parquet (없으면 원자료 정제), data/interim/pp_features.parquet,
      (선택) data/interim/pp_aria_emb.npz + data/interim/feas/normalization.npz
출력  reports/x1/coldstart_curves.csv, coldstart_summary.json, coldstart.png, type_profiles.csv
실행  .venv/bin/python -I scripts/x1_coldstart.py [--k 2] [--with-aria] [--reps 20] [--lam 2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

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
NS = [0, 5, 10, 20, 40]
OUT = ROOT / "reports" / "x1"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- 입력
def load_ratings() -> pd.DataFrame:
    p = paths.INTERIM / "pp_ratings_clean.parquet"
    d = pd.read_parquet(p) if p.exists() else pp.clean(pp.load_raw())
    d = d[d.piece.str.startswith("Schubert_D960") & (~d.is_score)].copy()
    cnt = d.groupby("user").size()
    d = d[d.user.isin(cnt[cnt >= MIN_SEG].index)]
    return d.reset_index(drop=True)


def load_features() -> pd.DataFrame:
    f = pd.read_parquet(paths.INTERIM / "pp_features.parquet")
    if "align_error" in f.columns:
        f = f[f.align_error.isna()]
    if {"match", "score_notes"} <= set(f.columns):
        f = f[(f.match / f.score_notes) >= 0.8]
    if "log_tempo" not in f.columns and "tempo_median_bpm" in f.columns:
        f = f.assign(log_tempo=np.log2(f.tempo_median_bpm))
    f = f.dropna(subset=FX).drop_duplicates("segment_id").set_index("segment_id")
    return f[FX].astype(float)


def loo_unit_center(X: np.ndarray, units: np.ndarray) -> np.ndarray:
    """같은 단위의 다른 연주 평균을 뺀다 (자기 자신 제외)."""
    Xc = np.empty_like(X)
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        s = X[idx].sum(0)
        Xc[idx] = X[idx] - (s - X[idx]) / max(len(idx) - 1, 1)
    return Xc


def aria_residual_pcs(seg: pd.DataFrame, k: int = 10) -> pd.DataFrame | None:
    """선택 입력: Aria 임베딩의 단위 내 잔차 PCA 상위 k 성분 (H1c 보조)."""
    p = paths.INTERIM / "pp_aria_emb.npz"
    if not p.exists():
        return None
    z = np.load(p)
    emb = dict(zip([str(s) for s in z["segment_id"]], z["emb"].astype(np.float64)))
    ids = [s for s in seg.segment_id if s in emb]
    E = np.stack([emb[s] for s in ids])
    n = paths.INTERIM / "feas" / "normalization.npz"
    if n.exists():
        nz = np.load(n)
        E = (E - nz["mean"]) / nz["std"]
    units = seg.set_index("segment_id").loc[ids, "unit"].to_numpy()
    R = loo_unit_center(E, units)
    U, s, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
    P = (R - R.mean(0)) @ Vt[:k].T
    P = P / (P.std(0) + 1e-9)
    return pd.DataFrame(P, index=ids, columns=[f"aria_pc{i+1}" for i in range(k)])


# ----------------------------------------------------------------------------- 쌍 구성
def pairs_for_rater(ri: pd.DataFrame, dim: str, units_keep) -> list[tuple[str, str, float]]:
    """평가자 i 가 같은 단위에서 평가한 세그먼트 쌍 (j, k, r_j − r_k). 동점은 방향 지표에서 제외(차이 0 유지)."""
    out = []
    for u, g in ri.groupby("unit"):
        if u not in units_keep:
            continue
        g = g.dropna(subset=[dim])
        for a, b in combinations(range(len(g)), 2):
            out.append((g.segment_id.iat[a], g.segment_id.iat[b], float(g[dim].iat[a] - g[dim].iat[b])))
    return out


def design(pairs, Xc: dict) -> tuple[np.ndarray, np.ndarray]:
    D = np.stack([Xc[j] - Xc[k] for j, k, _ in pairs]) if pairs else np.zeros((0, len(FX)))
    y = np.array([d for _, _, d in pairs], float)
    return D, y


def ridge_to_prior(D, y, prior, lam):
    """β = prior + (DᵀD + λI)⁻¹ Dᵀ (y − D·prior)."""
    if len(y) == 0:
        return prior.copy()
    A = D.T @ D + lam * np.eye(D.shape[1])
    return prior + np.linalg.solve(A, D.T @ (y - D @ prior))


def acc(D, y, beta) -> float:
    m = y != 0
    if m.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(D[m] @ beta) == np.sign(y[m])))


def mae(D, y, beta, w=1.0) -> float:
    return float(np.mean(np.abs(y - w * (D @ beta)))) if len(y) else np.nan


# ----------------------------------------------------------------------------- 학습 평가자 풀 → S, T
def fit_pool(train: pd.DataFrame, dim: str, Xc: dict, lam: float, k: int, rng):
    """풀 β_S (모든 학습 평가자의 쌍 차이), 평가자별 β_i, 유형 중심 β_type."""
    Ds, ys, per = [], [], {}
    for u, ri in train.groupby("user"):
        pr = pairs_for_rater(ri, dim, set(ri.unit))
        D, y = design(pr, Xc)
        if len(y) < 15:
            continue
        Ds.append(D); ys.append(y)
        per[u] = ridge_to_prior(D, y, np.zeros(D.shape[1]), lam)
    D, y = np.vstack(Ds), np.concatenate(ys)
    beta_S = ridge_to_prior(D, y, np.zeros(D.shape[1]), lam)
    B = np.stack(list(per.values()))
    types = kmeans(B, k, rng) if k > 1 and len(B) >= 2 * k else np.array([0] * len(B))
    beta_T = np.stack([B[types == t].mean(0) for t in range(types.max() + 1)])
    return beta_S, beta_T, pd.DataFrame(B, index=list(per.keys()), columns=Xc["__cols__"]).assign(type=types)


def kmeans(B, k, rng, iters=50):
    Bz = (B - B.mean(0)) / (B.std(0) + 1e-9)
    best, best_in = None, np.inf
    for _ in range(10):
        C = Bz[rng.choice(len(Bz), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((Bz[:, None, :] - C[None]) ** 2).sum(-1), 1)
            C = np.stack([Bz[lab == t].mean(0) if (lab == t).any() else C[t] for t in range(k)])
        inertia = ((Bz - C[lab]) ** 2).sum()
        if inertia < best_in:
            best, best_in = lab, inertia
    return best


def assign_type(D, y, beta_T, beta_S):
    if len(y) == 0 or len(beta_T) == 1:
        return beta_S if len(y) == 0 else beta_T[0]
    sc = [acc(D, y, b) for b in beta_T]
    return beta_T[int(np.nanargmax(sc))] if not np.all(np.isnan(sc)) else beta_S


def select_pairs(pool, Xc, n, mode, beta_fn, rng):
    """무작위 또는 능동(|Δx̃·β| 작은 것부터, 5개 배치)으로 n 쌍 선택."""
    if n == 0 or not pool:
        return []
    idx = list(range(len(pool)))
    if mode == "random":
        return [pool[i] for i in rng.choice(idx, min(n, len(idx)), replace=False)]
    chosen = list(rng.choice(idx, min(5, len(idx)), replace=False))
    while len(chosen) < min(n, len(idx)):
        beta = beta_fn([pool[i] for i in chosen])
        rest = [i for i in idx if i not in chosen]
        Dr, _ = design([pool[i] for i in rest], Xc)
        order = np.argsort(np.abs(Dr @ beta))
        chosen += [rest[o] for o in order[:5]]
    return [pool[i] for i in chosen[:n]]


# ----------------------------------------------------------------------------- 메인 루프
def run(d, F, lam, k, reps, variant, control, rng):
    Xz = (F - F.mean()) / (F.std() + 1e-9)
    seg = d.drop_duplicates("segment_id")[["segment_id", "unit"]].set_index("segment_id").loc[Xz.index]
    Xc_arr = loo_unit_center(Xz.to_numpy(), seg.unit.to_numpy())
    Xc = dict(zip(Xz.index, Xc_arr)); Xc["__cols__"] = list(Xz.columns)
    rows, profiles = [], None
    dims = [CONV] + EXEC
    for i, (user, ri) in enumerate(d.groupby("user")):
        train = d[d.user != user]
        for dim in dims:
            beta_S, beta_T, prof = fit_pool(train, dim, Xc, lam, k, rng)
            if dim == CONV and profiles is None:
                profiles = prof
            units = ri.unit.unique()
            for rep in range(reps):
                perm = rng.permutation(units); h = len(perm) // 2
                obs_u, test_u = set(perm[:h]), set(perm[h:])
                ri_obs = ri.copy()
                if control == "shuffle":  # X1-3: 관측 평가를 단위 내에서 섞어 개인 신호를 지운다
                    ri_obs[dim] = ri_obs.groupby("unit")[dim].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).to_numpy())
                pool = pairs_for_rater(ri_obs, dim, obs_u)
                Dt, yt = design(pairs_for_rater(ri, dim, test_u), Xc)
                if len(yt) < 10 or len(pool) < 5:
                    continue
                for mode in ("random", "active"):
                    for n in NS:
                        bf = lambda sel, _bS=beta_S, _bT=beta_T: ridge_to_prior(*design(sel, Xc), assign_type(*design(sel, Xc), _bT, _bS), lam * max(1, 20 / max(len(sel), 1)))
                        sel = select_pairs(pool, Xc, n, mode, bf, rng) if mode == "active" else select_pairs(pool, Xc, n, "random", None, rng)
                        Do, yo = design(sel, Xc)
                        b_T = assign_type(Do, yo, beta_T, beta_S)
                        b_P = ridge_to_prior(Do, yo, b_T, lam * max(1, 20 / max(n, 1)))
                        # 층 1 크기: 개인 감도 w (관측 쌍에서 Δr ~ w·(g·Δx̃)), 공유 g = beta_S
                        s_o = Do @ beta_S
                        w = float((s_o @ yo) / (s_o @ s_o)) if n > 0 and (s_o @ s_o) > 0 else 1.0
                        for name, b in (("S", beta_S), ("T", b_T), ("P", b_P)):
                            rows.append(dict(user=user, dim=SHORT[dim], layer="conv" if dim == CONV else "exec", n=n, mode=mode, rep=rep,
                                             model=name, variant=variant, control=control,
                                             acc=acc(Dt, yt, b), mae=mae(Dt, yt, b), mae_w=mae(Dt, yt, beta_S, w) if name == "S" else np.nan,
                                             w=w if name == "S" else np.nan, n_test=int((yt != 0).sum())))
        print(f"  평가자 {i+1}/{d.user.nunique()} 완료", flush=True)
    return pd.DataFrame(rows), profiles


def verdict(R: pd.DataFrame) -> dict:
    g = R[(R.n == 20) & (R.mode == "random") & (R.control == "none") & (R.variant == "fx")]
    med = g.groupby(["layer", "dim", "model"]).acc.median().unstack("model")
    out = {}
    conv = med.loc["conv"].iloc[0] if "conv" in med.index.get_level_values(0) else None
    if conv is not None:
        out["X1-1_T_minus_S_pt"] = round(100 * (conv["T"] - conv["S"]), 1)
        out["X1-1_P_minus_S_pt"] = round(100 * (conv["P"] - conv["S"]), 1)
        out["X1-1_types_supported"] = bool(out["X1-1_T_minus_S_pt"] >= 3)
        out["X1-1_personal_supported"] = bool(out["X1-1_P_minus_S_pt"] >= 5)
    ex = med.loc["exec"] if "exec" in med.index.get_level_values(0) else None
    if ex is not None:
        gain = 100 * (ex["P"] - ex["S"])
        gS = g[(g.layer == "exec") & (g.model == "S")].groupby("dim")[["mae", "mae_w"]].median()
        mae_gain = 1 - gS.mae_w / gS.mae
        ok = ((gain < 3) & (mae_gain.reindex(gain.index) >= 0.05))
        out["X1-2_exec_dims_supporting"] = int(ok.sum()); out["X1-2_supported"] = bool(ok.sum() >= 5)
        out["X1-2_detail"] = {d: {"acc_gain_pt": round(float(gain[d]), 1), "mae_gain_frac": round(float(mae_gain.get(d, np.nan)), 3)} for d in gain.index}
    sh = R[(R.n == 20) & (R.mode == "random") & (R.control == "shuffle") & (R.variant == "fx") & (R.layer == "conv")]
    if len(sh):
        m = sh.groupby("model").acc.median()
        out["X1-3_shuffle_P_minus_S_pt"] = round(100 * (m["P"] - m["S"]), 1)
        out["X1-3_control_clean"] = bool(abs(out["X1-3_shuffle_P_minus_S_pt"]) < 1)
    ac = R[(R.n == 20) & (R.control == "none") & (R.variant == "fx") & (R.layer == "conv") & (R.model == "P")].groupby("mode").acc.median()
    if {"active", "random"} <= set(ac.index):
        out["X1-4_active_minus_random_pt"] = round(100 * (ac["active"] - ac["random"]), 1)
        out["X1-4_use_active"] = bool(out["X1-4_active_minus_random_pt"] >= 2)
    va = R[(R.n == 20) & (R.mode == "random") & (R.control == "none") & (R.layer == "conv") & (R.model == "P")].groupby("variant").acc.median()
    if {"fx", "fx+aria"} <= set(va.index):
        out["X1-5_aria_gain_pt"] = round(100 * (va["fx+aria"] - va["fx"]), 1)
        out["X1-5_drop_embedding"] = bool(out["X1-5_aria_gain_pt"] < 1)
    return out


def plot(R):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    g = R[(R.mode == "random") & (R.control == "none") & (R.variant == "fx")]
    for layer, a in zip(("conv", "exec"), ax):
        for m, c in (("S", "gray"), ("T", "#b08d57"), ("P", "#9b2335")):
            q = g[(g.layer == layer) & (g.model == m)].groupby("n").acc.quantile([.25, .5, .75]).unstack()
            a.plot(q.index, q[.5], marker="o", color=c, label=m); a.fill_between(q.index, q[.25], q[.75], color=c, alpha=.15)
        a.set_title(f"cold-start · {'convincing (layer 2)' if layer=='conv' else 'execution dims pooled (layer 1)'}")
        a.set_xlabel("observed pairs n"); a.set_ylabel("held-out pair accuracy"); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "coldstart.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2); ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--lam", type=float, default=2.0); ap.add_argument("--with-aria", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    d, F = load_ratings(), load_features()
    d = d[d.segment_id.isin(F.index)]
    hum = d.drop_duplicates("segment_id").groupby("unit").size()
    d = d[d.unit.isin(hum[hum >= MIN_HUM].index)].reset_index(drop=True)
    F = F.loc[d.segment_id.unique()]
    print(f"D960 사람 세그먼트 {len(F)} · 단위 {d.unit.nunique()} · 평가자 {d.user.nunique()} · 평가 {len(d)}")
    parts = []
    R, prof = run(d, F, a.lam, a.k, a.reps, "fx", "none", rng); parts.append(R)
    prof.to_csv(OUT / "type_profiles.csv")
    Rs, _ = run(d, F, a.lam, a.k, max(5, a.reps // 4), "fx", "shuffle", rng); parts.append(Rs)
    if a.with_aria:
        A = aria_residual_pcs(d.drop_duplicates("segment_id"))
        if A is not None:
            FA = F.join(A, how="inner")
            global FX
            FX_backup, FX = FX, list(FA.columns)
            Ra, _ = run(d[d.segment_id.isin(FA.index)].reset_index(drop=True), FA, a.lam, a.k, a.reps, "fx+aria", "none", rng); parts.append(Ra)
            FX = FX_backup
    R = pd.concat(parts, ignore_index=True)
    R.to_csv(OUT / "coldstart_curves.csv", index=False)
    v = verdict(R); v["n_raters"] = int(d.user.nunique()); v["n_units"] = int(d.unit.nunique())
    json.dump(v, open(OUT / "coldstart_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(v, ensure_ascii=False, indent=1))
    plot(R)


if __name__ == "__main__":
    main()
