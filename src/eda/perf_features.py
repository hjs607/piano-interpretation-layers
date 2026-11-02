"""정렬된 음 단위 파라미터(align.note_params) → 구간 단위 요약 특징.

PercePiano 구간에 씁니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def onset_tempo(p: pd.DataFrame) -> pd.DataFrame:
    """악보 onset 묶음(화음)마다 beat_period 하나 → 템포 곡선."""
    g = p.groupby("onset_group").agg(onset_beat=("onset_beat", "min"), beat_period=("beat_period", "median"))
    g = g.sort_values("onset_beat")
    g["tempo_bpm"] = 60.0 / g["beat_period"]
    return g


def robust_mask(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    """log 척도에서 중앙값 ± k·IQR 밖을 이상치로 봅니다."""
    lx = np.log2(x)
    ok = np.isfinite(lx)
    q1, med, q3 = np.nanpercentile(lx[ok], [25, 50, 75]) if ok.any() else (np.nan,) * 3
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return ok
    return ok & (np.abs(lx - med) <= k * max(iqr, 1e-6))


def summarize(p: pd.DataFrame) -> dict:
    tc = onset_tempo(p)
    t = tc["tempo_bpm"].to_numpy()
    m = robust_mask(t)
    lt = np.log2(t[m])
    timing = p["timing"].to_numpy()
    timing = timing[np.isfinite(timing) & (np.abs(timing) <= 0.25)]
    art = p["articulation_log"].to_numpy()
    art = art[np.isfinite(art)]
    vel = p["velocity_norm"].to_numpy() * 127
    return {
        "n_matched": len(p),
        "n_onsets": len(tc),
        "tempo_outlier_frac": 1 - m.mean() if len(m) else np.nan,
        "tempo_median_bpm": float(np.median(t[m])) if m.any() else np.nan,
        "tempo_logstd": float(np.std(lt)) if m.sum() > 2 else np.nan,
        "tempo_range_bpm": float(np.subtract(*np.percentile(t[m], [95, 5]))) if m.sum() > 2 else np.nan,
        "tempo_volatility": float(np.std(np.diff(lt))) if m.sum() > 3 else np.nan,
        "tempo_std_naive": float(np.nanstd(t[np.isfinite(t)])),  # RenCon expressive_vs_score_metrics 와 같은 방식
        "timing_std_ms": float(np.std(timing) * 1000) if len(timing) > 2 else np.nan,
        "articulation_median": float(np.median(art)) if len(art) else np.nan,
        "articulation_iqr": float(np.subtract(*np.percentile(art, [75, 25]))) if len(art) > 3 else np.nan,
        "vel_mean": float(vel.mean()),
        "vel_std": float(vel.std()),
        "vel_range": float(np.subtract(*np.percentile(vel, [95, 5]))),
    }


def summarize_by_bars(p: pd.DataFrame, bar_ranges: dict[str, tuple[int, int]], beats_per_bar: float) -> pd.DataFrame:
    """onset_beat 기준으로 마디 범위(1부터, 양끝 포함)를 잘라 각각 요약."""
    rows = []
    for name, (lo, hi) in bar_ranges.items():
        sub = p[(p.onset_beat >= (lo - 1) * beats_per_bar) & (p.onset_beat < hi * beats_per_bar)]
        if len(sub) > 5:
            rows.append({"section": name, **summarize(sub)})
    return pd.DataFrame(rows)


WOO80_SECTIONS = {"thema": (1, 8), "var1": (9, 16), "var2": (17, 24), "var3": (25, 32), "var4": (33, 40), "var5": (41, 48)}


def bar_curve(p: pd.DataFrame, beats_per_bar: float) -> pd.DataFrame:
    """마디 단위 템포(마디 안 onset 묶음 beat_period 중앙값)와 평균 벨로시티."""
    tc = onset_tempo(p)
    tc["bar"] = (tc.onset_beat // beats_per_bar).astype(int) + 1
    v = p.assign(bar=(p.onset_beat // beats_per_bar).astype(int) + 1).groupby("bar").velocity_norm.mean() * 127
    out = tc.groupby("bar").beat_period.median().rename("beat_period").to_frame()
    out["tempo_bpm"] = 60 / out.beat_period
    out["vel_mean"] = v
    return out.reset_index()


def pedal_ratio_window(midi_path, t0: float, t1: float, threshold: int = 64) -> float:
    """[t0, t1] 구간에서 서스테인 페달이 밟힌 시간 비율."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    ev = sorted((c.time, c.value) for inst in pm.instruments for c in inst.control_changes if c.number == 64)
    if not ev or t1 <= t0:
        return 0.0
    down, state, last = 0.0, False, t0
    for t, v in ev:
        if state:
            down += max(min(t, t1) - max(last, t0), 0)
        state, last = v >= threshold, t
    if state:
        down += max(t1 - max(last, t0), 0)
    return down / (t1 - t0)
