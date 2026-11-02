"""조건(열화) 함수: 음 표(int64 배열 dict)를 받아 (새 음 표, 새 beats_ms)를 돌려줍니다.

- 음 표 키: onset_ms, off_key_ms, off_ped_ms, pitch, velocity (모두 int64)
- 공통 인자: notes, beats_ms, db_ms(연주 p 의 downbeat), region(r_lo, r_hi, t0, t1)
  region.t0/t1 = [db[r_lo]-30, db[r_hi]-30). onset 이 이 안인 음만 바꿉니다.
- 출력 음 표는 onset 기준 안정 정렬. 구간 밖 음은 배열 5개 모두 원본과 같습니다.

해석 메모
- tempo: 매듭 = (t0,t0), (구간 안 박 b_j -> (1-a)b_j + a*g_j), (t1,t1), (db[r_hi], db[r_hi]), 그 밖은 항등.
  g_j 는 b_0..b_M(구간 시작·끝 downbeat 사이 박)을 등간격으로 편 위치. t0/t1 을 고정 매듭으로 넣은 것은
  구간 안 음이 구간 밖(=다른 창)으로 새지 않게 하기 위함이며, 마지막 박 하나의 기울기만 아주 조금 바뀝니다.
  beats_ms: 구간 안 박만 변환(양끝 박 고정), 나머지 원본.
- swap/deadpan: 대체할 음은 원천(q 또는 악보)의 같은 구간 시간 [src_db[r_lo]-30, src_db[r_hi]-30) 의 음입니다.
  시간 변환은 마디별 선형(원천 downbeat k -> p 의 downbeat k). 구간 끝 너머(음 길이)는 마지막 마디의 시간 비율로,
  구간 시작 앞(30ms 허용 구간)은 첫 마디의 비율로 외삽합니다. 변환된 onset 이 [t0, t1) 을 벗어나면
  [t0, t1-1] 로 자르고(off 도 같은 양만큼 이동) 그 수를 info['n_clipped'] 에 적습니다.
  beats_ms: p 의 [db[r_lo], db[r_hi]) 박을 지우고 원천의 같은 범위 박을 마디별 선형으로 옮겨 넣습니다.
- jit_time: 화음 묶음(asap_windows.onset_clusters) 단위 가우스 shift, 음 길이 유지.
  묶음 안 모든 음이 자기 칸(창 경계 격자)에 남도록 shift 를 자릅니다(0 은 항상 허용).
  자르면 크기가 줄어드니 같은 난수 z 로 sigma 를 이분 탐색해 음 단위 평균 |Δonset| 이 목표와 맞게 합니다.
- jit_vel: 음 단위 가우스, v' = clip(round(v + d), 1, 127). 같은 방식으로 sigma 를 맞춥니다.
"""
from typing import NamedTuple

import numpy as np

from .asap_windows import onset_clusters

KEYS = ("onset_ms", "off_key_ms", "off_ped_ms", "pitch", "velocity")
BIG = 1e7


class Region(NamedTuple):
    r_lo: int
    r_hi: int
    t0: int  # db[r_lo] - 30
    t1: int  # db[r_hi] - 30


def make_region(db_ms, r_lo, r_hi) -> Region:
    return Region(r_lo, r_hi, int(db_ms[r_lo]) - 30, int(db_ms[r_hi]) - 30)


def _copy(notes):
    return {k: np.asarray(notes[k], dtype=np.int64).copy() for k in KEYS}


def _sort(notes):
    o = np.argsort(notes["onset_ms"], kind="stable")
    return {k: notes[k][o] for k in KEYS}


def in_region(notes, region: Region) -> np.ndarray:
    o = notes["onset_ms"]
    return (o >= region.t0) & (o < region.t1)


def mean_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float)))) if len(a) else 0.0


# ------------------------------------------------------------------ clean / vel / pedal
def clean(notes, beats_ms, db_ms=None, region=None, **_):
    return _sort(_copy(notes)), np.asarray(beats_ms, dtype=np.int64).copy()


def vel(notes, beats_ms, db_ms, region: Region, alpha: float, **_):
    """v' = v + alpha (구간 중앙값 - v), 반올림, [1,127]."""
    out = _copy(notes)
    m = in_region(out, region)
    if m.any():
        v = out["velocity"][m].astype(float)
        med = np.median(v)
        out["velocity"][m] = np.clip(np.rint(v + alpha * (med - v)), 1, 127).astype(np.int64)
    return _sort(out), np.asarray(beats_ms, dtype=np.int64).copy()


def pedal(notes, beats_ms, db_ms, region: Region, **_):
    """구간 안 음의 off_ped 를 off_key 로 되돌립니다."""
    out = _copy(notes)
    m = in_region(out, region)
    out["off_ped_ms"][m] = out["off_key_ms"][m]
    return _sort(out), np.asarray(beats_ms, dtype=np.int64).copy()


# ------------------------------------------------------------------ tempo
def _tempo_knots(beats_ms, db_ms, region: Region, alpha: float):
    b0, bM = float(db_ms[region.r_lo]), float(db_ms[region.r_hi])
    beats = np.asarray(beats_ms, dtype=float)
    i0 = int(np.argmin(np.abs(beats - b0)))
    i1 = int(np.argmin(np.abs(beats - bM)))
    assert abs(beats[i0] - b0) <= 1 and abs(beats[i1] - bM) <= 1, "구간 끝 downbeat 가 beats 에 없음"
    bb = beats[i0:i1 + 1]
    M = i1 - i0
    assert M >= 2 and np.all(np.diff(bb) > 0)
    g = bb[0] + np.arange(M + 1) * (bb[-1] - bb[0]) / M
    tgt = bb + alpha * (g - bb)
    t0, t1 = float(region.t0), float(region.t1)
    assert bb[M - 1] < t1 < bb[M] and t0 < bb[0]
    xs = np.concatenate([[t0 - BIG, t0], bb[:M], [t1], [bb[M]], [bM + BIG]])
    ys = np.concatenate([[t0 - BIG, t0], tgt[:M], [t1], [bb[M]], [bM + BIG]])
    return xs, ys, (i0, i1, tgt)


def tempo(notes, beats_ms, db_ms, region: Region, alpha: float, **_):
    """구간 안 박을 등간격으로 펴는 구간별 선형 시간 왜곡 (alpha=1: 완전 등간격, 양끝 고정)."""
    xs, ys, (i0, i1, tgt) = _tempo_knots(beats_ms, db_ms, region, alpha)
    out = _copy(notes)
    m = in_region(out, region)
    for k in ("onset_ms", "off_key_ms", "off_ped_ms"):
        out[k][m] = np.rint(np.interp(out[k][m].astype(float), xs, ys)).astype(np.int64)
    out["onset_ms"][m] = np.clip(out["onset_ms"][m], region.t0, region.t1 - 1)
    out["off_key_ms"][m] = np.maximum(out["off_key_ms"][m], out["onset_ms"][m])
    out["off_ped_ms"][m] = np.maximum(out["off_ped_ms"][m], out["off_key_ms"][m])
    nb = np.asarray(beats_ms, dtype=np.int64).copy()
    nb[i0:i1 + 1] = np.rint(tgt).astype(np.int64)
    return _sort(out), nb


# ------------------------------------------------------------------ swap / deadpan
def _bar_map(src_db, db_ms, region: Region):
    xs = np.asarray(src_db[region.r_lo:region.r_hi + 1], dtype=float)
    ys = np.asarray(db_ms[region.r_lo:region.r_hi + 1], dtype=float)
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)
    r_first = (ys[1] - ys[0]) / (xs[1] - xs[0])
    r_last = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])

    def f(t):
        t = np.asarray(t, dtype=float)
        v = np.interp(t, xs, ys)
        v = np.where(t < xs[0], ys[0] + (t - xs[0]) * r_first, v)
        v = np.where(t > xs[-1], ys[-1] + (t - xs[-1]) * r_last, v)
        return v
    return f


def replace_region(notes, beats_ms, db_ms, region: Region, src_notes, src_beats_ms, src_db_ms, info=None):
    """p 의 구간을 원천(q 또는 악보)의 같은 마디 음으로 교체. swap/deadpan 의 공통 구현."""
    f = _bar_map(src_db_ms, db_ms, region)
    out = _copy(notes)
    src = _copy(src_notes)
    s0 = int(src_db_ms[region.r_lo]) - 30
    s1 = int(src_db_ms[region.r_hi]) - 30
    sm = (src["onset_ms"] >= s0) & (src["onset_ms"] < s1)
    ins = {k: src[k][sm] for k in KEYS}
    on = np.rint(f(ins["onset_ms"])).astype(np.int64)
    off_k = np.rint(f(ins["off_key_ms"])).astype(np.int64)
    off_p = np.rint(f(ins["off_ped_ms"])).astype(np.int64)
    on_c = np.clip(on, region.t0, region.t1 - 1)
    d = on_c - on
    ins["onset_ms"] = on_c
    ins["off_key_ms"] = np.maximum(off_k + d, on_c)
    ins["off_ped_ms"] = np.maximum(off_p + d, ins["off_key_ms"])
    if info is not None:
        info["n_clipped"] = int((d != 0).sum())
        info["n_before"] = int(in_region(out, region).sum())
        info["n_after"] = int(len(on_c))
    keep = ~in_region(out, region)
    merged = {k: np.concatenate([out[k][keep], ins[k]]) for k in KEYS}
    # 박: p 의 [db[lo], db[hi]) 를 지우고 원천 박을 옮겨 넣음
    pb = np.asarray(beats_ms, dtype=np.int64)
    sb = np.asarray(src_beats_ms, dtype=np.int64)
    pk = ~((pb >= db_ms[region.r_lo]) & (pb < db_ms[region.r_hi]))
    smb = (sb >= src_db_ms[region.r_lo]) & (sb < src_db_ms[region.r_hi])
    nb = np.sort(np.concatenate([pb[pk], np.rint(f(sb[smb])).astype(np.int64)]), kind="stable")
    return _sort(merged), nb


def swap(notes, beats_ms, db_ms, region, q_notes, q_beats_ms, q_db_ms, info=None, **_):
    return replace_region(notes, beats_ms, db_ms, region, q_notes, q_beats_ms, q_db_ms, info)


def deadpan(notes, beats_ms, db_ms, region, score_notes, score_beats_ms, score_db_ms, info=None, **_):
    return replace_region(notes, beats_ms, db_ms, region, score_notes, score_beats_ms, score_db_ms, info)


# ------------------------------------------------------------------ jitter
def _bisect_sigma(ach_fn, target: float, iters: int = 60):
    """ach_fn(sigma) 가 sigma 에 대해 (거의) 단조 증가라고 보고 target 에 가장 가까운 sigma."""
    lo, hi = 0.0, max(target, 1.0)
    for _ in range(60):
        if ach_fn(hi) >= target:
            break
        hi *= 2
    best = (abs(ach_fn(hi) - target), hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = ach_fn(mid)
        best = min(best, (abs(a - target), mid))
        if a < target:
            lo = mid
        else:
            hi = mid
    return best[1]


def jit_time(notes, beats_ms, db_ms, region: Region, target_mean_abs: float, seed: int, edges: np.ndarray,
             info=None, **_):
    """화음 묶음 단위 가우스 shift. target_mean_abs = tempo α=1 의 음 단위 평균 |Δonset|(ms)."""
    out = _sort(_copy(notes))
    idx = np.flatnonzero(in_region(out, region))
    nb = np.asarray(beats_ms, dtype=np.int64).copy()
    if len(idx) == 0:
        return out, nb
    o = out["onset_ms"][idx]
    cid = onset_clusters(o)
    nc = int(cid.max()) + 1
    cell = np.searchsorted(edges, o.astype(float), side="right") - 1
    lo_n = edges[cell] - o          # 이 음이 자기 칸 안에 남는 shift 범위
    hi_n = edges[cell + 1] - 1 - o
    lo = np.full(nc, -np.inf)
    hi = np.full(nc, np.inf)
    np.maximum.at(lo, cid, lo_n)
    np.minimum.at(hi, cid, hi_n)
    lo = np.minimum(lo, 0.0)
    hi = np.maximum(hi, 0.0)
    z = np.random.default_rng(np.random.SeedSequence([seed, 0])).standard_normal(nc)

    def shifts(sig):
        return np.rint(np.clip(sig * z, lo, hi))

    def ach(sig):
        return float(np.mean(np.abs(shifts(sig)[cid])))

    sig = _bisect_sigma(ach, target_mean_abs)
    d = shifts(sig)[cid].astype(np.int64)
    for k in ("onset_ms", "off_key_ms", "off_ped_ms"):
        out[k][idx] = out[k][idx] + d
    if info is not None:
        info.update(sigma=float(sig), achieved=float(np.mean(np.abs(d))), target=float(target_mean_abs),
                    n_clusters=nc)
    return _sort(out), nb


def jit_vel(notes, beats_ms, db_ms, region: Region, target_mean_abs: float, seed: int, info=None, **_):
    """음 단위 가우스 벨로시티 흔들기. target_mean_abs = vel α=1 의 평균 |Δv|."""
    out = _sort(_copy(notes))
    idx = np.flatnonzero(in_region(out, region))
    nb = np.asarray(beats_ms, dtype=np.int64).copy()
    if len(idx) == 0:
        return out, nb
    v = out["velocity"][idx]
    z = np.random.default_rng(np.random.SeedSequence([seed, 1])).standard_normal(len(idx))

    def newv(sig):
        return np.clip(np.rint(v + sig * z), 1, 127).astype(np.int64)

    sig = _bisect_sigma(lambda s: float(np.mean(np.abs(newv(s) - v))), target_mean_abs)
    nv = newv(sig)
    out["velocity"][idx] = nv
    if info is not None:
        info.update(sigma=float(sig), achieved=float(np.mean(np.abs(nv - v))), target=float(target_mean_abs))
    return _sort(out), nb
