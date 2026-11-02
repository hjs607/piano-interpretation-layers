"""Q 점검 공통 로더 (docs/q_audit_prereg_2026-11-01.md). 실행: .venv/bin/python -B scripts/q0X_*.py"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eda import paths, percepiano_io as pp, stats_utils as su  # noqa: E402

OUT = ROOT / "reports" / "q"; OUT.mkdir(parents=True, exist_ok=True)
L, S = pp.LABELS19, pp.SHORT19; SHORT = dict(zip(L, S))
EXEC = ["Timing_Stable_Unstable", "Articulation_Long_Short", "Articulation_Soft_cushioned_Hard_solid",
        "Pedal_Sparse/dry_Saturated/wet", "Pedal_Clean_Blurred", "Timbre_Shallow_Rich",
        "Dynamic_Little_dynamic_range_Large_dynamic_range", "Music_Making_Fast_paced_Slow_paced"]
CONV = "Interpretation_Unsatisfactory/doubtful_Convincing"
NINE = EXEC + [CONV]
MIN_SEG = 40


def ratings(repeats="mean"):
    """정제 평점. repeats: mean | first | keep."""
    if repeats == "mean":
        p = paths.INTERIM / "pp_ratings_clean.parquet"
        if p.exists(): return pd.read_parquet(p)
    return pp.clean(pp.load_raw(), repeats=repeats)[0]


def d960(df, human_only=False, min_seg=MIN_SEG):
    d = df[df.piece.str.startswith("Schubert_D960")]
    cnt = d[~d.is_score].groupby("user").size()
    d = d[d.user.isin(cnt[cnt >= min_seg].index)]
    return d[~d.is_score] if human_only else d


def additive_residuals(d, value):
    """y = μ + 구간 + 평가자 + e (교대 평균). 반환: d 에 맞춘 e, 구간효과, 평가자효과."""
    x = d[["segment_id", "user", value]].dropna()
    y = x[value].to_numpy(float); seg = x.segment_id.to_numpy(); usr = x.user.to_numpy()
    mu = y.mean(); a = pd.Series(0.0, index=np.unique(seg)); b = pd.Series(0.0, index=np.unique(usr))
    for _ in range(50):
        a = pd.Series(y - mu - b.reindex(usr).to_numpy()).groupby(seg).mean()
        b = pd.Series(y - mu - a.reindex(seg).to_numpy()).groupby(usr).mean()
        mu = mu + a.mean() + b.mean(); a -= a.mean(); b -= b.mean()
    e = y - mu - a.reindex(seg).to_numpy() - b.reindex(usr).to_numpy()
    return pd.Series(e, index=x.index), a, b


def save(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float))
    print("saved", OUT / f"{name}.json")


def within_unit_split_half(df, value, n_iter=100, seed=0, min_raters=4):
    """구간마다 평가자를 반분 → 두 반쪽 평균의 단위 내 편차 상관 → Spearman–Brown. (rater_layers.py 와 같은 정의)"""
    g0 = np.random.default_rng(seed); x = df[["segment_id", "unit", value]].dropna()
    groups = [(u, v[value].to_numpy()) for (s, u), v in x.groupby(["segment_id", "unit"]) if len(v) >= min_raters]
    units = np.array([u for u, _ in groups]); rs = []
    for _ in range(n_iter):
        a = np.empty(len(groups)); b = np.empty(len(groups))
        for i, (_, v) in enumerate(groups):
            pm = g0.permutation(len(v)); h = len(v) // 2; a[i], b[i] = v[pm[:h]].mean(), v[pm[h:]].mean()
        sa = pd.Series(a).groupby(units).transform("mean").to_numpy(); sb = pd.Series(b).groupby(units).transform("mean").to_numpy()
        rs.append(np.corrcoef(a - sa, b - sb)[0, 1])
    rh = float(np.mean(rs)); return 2 * rh / (1 + rh)


def rater_adjusted(df, cols):
    """Q4 결정: 차원별 가법 모형의 평가자 주효과를 뺀 평점. (D960 사람 연주 기준으로 적합)"""
    out = df.copy()
    for c in cols:
        _, _, b = additive_residuals(df, c); out[c] = df[c] - b.reindex(df.user).to_numpy()
    return out


def loo_consensus(df, c):
    """세그먼트 × 평가자 행렬과 LOO 합의 (평가자 자신 제외 평균)."""
    m = df.pivot_table(index="segment_id", columns="user", values=c); s = m.sum(1); n = m.notna().sum(1)
    loo = (s.to_numpy()[:, None] - np.nan_to_num(m.to_numpy())) / np.maximum(n.to_numpy()[:, None] - 1, 1)
    return m, pd.DataFrame(loo, index=m.index, columns=m.columns).where(m.notna())
