"""가능성 실험 조건 음 표와 manifest 생성 + 자체 검사.

입력: data/interim/feas/notes/{piece}/{perf}.npz, midi_score.npz (feas_export_notes.py 가 만듦)
출력: data/interim/feas/cond/{piece}/*.npz, data/interim/feas/manifest_{piece}.csv
사용: .venv/bin/python -I scripts/feas_make_conditions.py [piece ...] [--check]   (--check: 생성 없이 검사만)
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.eda.paths import FEAS, FEAS_PIECES
from src.feas import asap_windows as aw
from src.feas import degrade as dg

NOTES = FEAS / "notes"
COND = FEAS / "cond"
HAND = ["vel_mean", "vel_std", "vel_p95_p5", "ped_ext", "beat_ioi_logstd", "dur_logmean", "onset_density"]
COLS = ["row_id", "piece", "perf", "cond", "alpha", "region", "win_idx", "k_lo", "k_hi", "t0_ms", "t1_ms",
        "n_notes", "overlap", "clean_row_id", "needs_embed", "ref_exclude", "seed", "notes_file"] + HAND
CONDS = [("vel", 0.5), ("vel", 1.0), ("tempo", 0.5), ("tempo", 1.0), ("pedal", 1.0), ("deadpan", 1.0),
         ("swap", 1.0), ("jit_time", 1.0), ("jit_vel", 1.0)]


def seed_for(piece, perf, region) -> int:
    h = hashlib.sha256(f"{piece}|{perf}|{region}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - 1)


def load_notes(piece, perf):
    z = np.load(NOTES / piece / f"{perf}.npz")
    return {k: z[k].astype(np.int64) for k in dg.KEYS}


def cond_name(perf, cond, alpha=1.0, region=-1):
    return f"{perf}__clean" if cond == "clean" else f"{perf}__{cond}__a{alpha:.1f}__r{region}"


def save_cond(piece, name, notes, beats):
    d = COND / piece
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{name}.npz", beats_ms=np.asarray(beats, dtype=np.int64), **{k: notes[k] for k in dg.KEYS})
    return f"data/interim/feas/cond/{piece}/{name}.npz"


def load_cond(piece, name):
    z = np.load(COND / piece / f"{name}.npz")
    return {k: z[k] for k in dg.KEYS}, z["beats_ms"]


def hand_features(notes, beats, t0, t1):
    nan = dict.fromkeys(HAND, np.nan)
    o = notes["onset_ms"]
    m = (o >= t0) & (o < t1)
    if not m.any():
        return nan
    v = notes["velocity"][m].astype(float)
    on = o[m]
    dk = np.maximum(notes["off_key_ms"][m] - on, 10).astype(float)
    dp = np.maximum(notes["off_ped_ms"][m] - on, 10).astype(float)
    b = beats[(beats >= t0) & (beats < t1)]
    ioi = np.diff(b)
    ioi = ioi[ioi > 0]
    ncl = int(aw.onset_clusters(on).max()) + 1
    return dict(vel_mean=v.mean(), vel_std=v.std(), vel_p95_p5=np.percentile(v, 95) - np.percentile(v, 5),
                ped_ext=float(np.median(np.log(dp / dk))), dur_logmean=float(np.mean(np.log(dp))),
                beat_ioi_logstd=float(np.std(np.log(ioi))) if len(ioi) >= 2 else np.nan,
                onset_density=ncl / ((t1 - t0) / 1000.0))


def build(piece):
    cfg = aw.PIECE_CFG[piece]
    L, hop, regions = cfg["L"], cfg["hop"], cfg["regions"]
    perfs = aw.performers(piece)
    assert (NOTES / piece / "meta.json").exists()
    clean = {p: load_notes(piece, p) for p in perfs + [aw.SCORE]}
    timing = {p: aw.timing(piece, p) for p in perfs + [aw.SCORE]}
    wins = {p: aw.window_grid(timing[p][0], L, hop, int(clean[p]["onset_ms"].max())) for p in clean}
    rows = []
    jit_info = []
    clean_rid = {}  # (perf, win_idx) -> row_id

    def add(perf, cond, alpha, region, notes, beats, fpath, ref_exclude="", seed=-1):
        r_lo, r_hi = (regions[region] if region >= 0 else (None, None))
        for w in wins[perf]:
            rid = len(rows)
            if cond == "clean":
                ov, crid, ne = "na", rid, 1
                clean_rid[(perf, w["win_idx"])] = rid
            else:
                ov = aw.overlap_label(w["k_lo"], w["k_hi"], r_lo, r_hi)
                crid = clean_rid[(perf, w["win_idx"])]
                ne = int(ov != "out")
            n = int(aw.select_notes(notes["onset_ms"], w["t0_ms"], w["t1_ms"]).sum())
            rows.append(dict(row_id=rid, piece=piece, perf=perf, cond=cond, alpha=alpha, region=region, **w,
                             n_notes=n, overlap=ov, clean_row_id=crid, needs_embed=ne,
                             ref_exclude=ref_exclude, seed=seed, notes_file=fpath,
                             **hand_features(notes, beats, w["t0_ms"], w["t1_ms"])))

    for p in perfs + [aw.SCORE]:
        n, b = dg.clean(clean[p], timing[p][1])
        add(p, "clean", 1.0, -1, n, b, save_cond(piece, cond_name(p, "clean"), n, b))
    edges = aw.cell_edges
    for i, p in enumerate(perfs):
        db, beats = timing[p]
        q = perfs[(i + 1) % len(perfs)]
        for r, (r_lo, r_hi) in enumerate(regions):
            reg = dg.make_region(db, r_lo, r_hi)
            seed = seed_for(piece, p, r)
            nt, _ = dg.tempo(clean[p], beats, db, reg, 1.0)
            nv, _ = dg.vel(clean[p], beats, db, reg, 1.0)
            m = dg.in_region(clean[p], reg)
            m_time = dg.mean_abs_delta(nt["onset_ms"][m], clean[p]["onset_ms"][m])
            m_vel = dg.mean_abs_delta(nv["velocity"][m], clean[p]["velocity"][m])
            for cond, alpha in CONDS:
                info = {}
                kw = dict(alpha=alpha, info=info)
                if cond == "swap":
                    kw.update(q_notes=clean[q], q_beats_ms=timing[q][1], q_db_ms=timing[q][0])
                elif cond == "deadpan":
                    kw.update(score_notes=clean[aw.SCORE], score_beats_ms=timing[aw.SCORE][1],
                              score_db_ms=timing[aw.SCORE][0])
                elif cond == "jit_time":
                    kw.update(target_mean_abs=m_time, seed=seed, edges=edges(db, hop))
                elif cond == "jit_vel":
                    kw.update(target_mean_abs=m_vel, seed=seed)
                n, b = getattr(dg, cond)(clean[p], beats, db, reg, **kw)
                fpath = save_cond(piece, cond_name(p, cond, alpha, r), n, b)
                add(p, cond, alpha, r, n, b, fpath, ref_exclude=q if cond == "swap" else "",
                    seed=seed if cond.startswith("jit") else -1)
    df = pd.DataFrame(rows)[COLS]
    df.to_csv(FEAS / f"manifest_{piece}.csv", index=False)
    return df


# ====================================================================== 자체 검사
def aria_bin(v):
    """ariautils AbsTokenizer._quantize_velocity 와 같은 규칙 (step 10, 0..120, 0 -> 10)."""
    q = list(range(0, 121, 10))

    def closest(n):  # _base._find_closest_int 의 이분 탐색 그대로
        left, right, c = 0, len(q) - 1, float("inf")
        while left <= right:
            mid = (left + right) // 2
            if abs(q[mid] - n) < abs(c - n):
                c = q[mid]
            if q[mid] < n:
                left = mid + 1
            else:
                right = mid - 1
        return c

    out = np.empty(len(v), dtype=np.int64)
    for i, x in enumerate(v):
        c = closest(int(x))
        out[i] = 10 if (c == 0 and x != 0) else c
    return out


def run_checks(piece):
    cfg = aw.PIECE_CFG[piece]
    L, hop, regions = cfg["L"], cfg["hop"], cfg["regions"]
    perfs = aw.performers(piece)
    man = pd.read_csv(FEAS / f"manifest_{piece}.csv", keep_default_na=False)
    clean = {p: load_cond(piece, cond_name(p, "clean")) for p in perfs + [aw.SCORE]}
    res = {}  # name -> (ok, detail)
    notes_ok = {}

    # 1 구간 밖 동일
    bad = []
    for p in perfs:
        db = aw.timing(piece, p)[0]
        for cond, alpha in CONDS:
            for r, (lo, hi) in enumerate(regions):
                n, _ = load_cond(piece, cond_name(p, cond, alpha, r))
                t0, t1 = aw.region_span(db, lo, hi)
                c = clean[p][0]
                oc = ~aw.select_notes(c["onset_ms"], t0, t1)
                on = ~aw.select_notes(n["onset_ms"], t0, t1)
                if oc.sum() != on.sum() or any(not np.array_equal(c[k][oc], n[k][on]) for k in dg.KEYS):
                    bad.append(f"{p}/{cond}/{alpha}/r{r}")
    res["1 구간 밖 음 동일(5배열)"] = (not bad, "위반 " + (", ".join(bad[:5]) or "0") + f" (파일 {len(perfs)*2*len(CONDS)}개)")

    # 2 tempo 양끝 보존, 등간격
    bad, mx = [], 0
    for p in perfs:
        db, beats = aw.timing(piece, p)
        for r, (lo, hi) in enumerate(regions):
            for a in (0.5, 1.0):
                n, nb = load_cond(piece, cond_name(p, "tempo", a, r))
                i0 = int(np.argmin(abs(beats - db[lo])))
                i1 = int(np.argmin(abs(beats - db[hi])))
                if nb[i0] != beats[i0] or nb[i1] != beats[i1] or len(nb) != len(beats) or not np.array_equal(
                        nb[:i0], beats[:i0]) or not np.array_equal(nb[i1:], beats[i1:]):
                    bad.append(f"{p}/r{r}/a{a} endpoints")
                if a == 1.0:
                    sp = np.diff(nb[i0:i1 + 1])
                    mx = max(mx, int(sp.max() - sp.min()))
                    if sp.max() - sp.min() > 2:
                        bad.append(f"{p}/r{r} spacing {sp.max()-sp.min()}")
    res["2 tempo 양끝 고정·α=1 등간격"] = (not bad, f"위반 {bad[:3] or 0}; 간격 max-min 최대 {mx} ms")

    # 3 vel 토큰 bin 변화율
    fr = {0.5: [], 1.0: []}
    for p in perfs:
        db = aw.timing(piece, p)[0]
        for r, (lo, hi) in enumerate(regions):
            t0, t1 = aw.region_span(db, lo, hi)
            c = clean[p][0]
            m = aw.select_notes(c["onset_ms"], t0, t1)
            for a in (0.5, 1.0):
                n, _ = load_cond(piece, cond_name(p, "vel", a, r))
                mm = aw.select_notes(n["onset_ms"], t0, t1)
                assert np.array_equal(n["onset_ms"][mm], c["onset_ms"][m])
                fr[a].append(float(np.mean(aria_bin(c["velocity"][m]) != aria_bin(n["velocity"][mm]))))
    res["3 vel bin 변화율(보고)"] = (True, "α=0.5: mean %.3f (min %.3f); α=1: mean %.3f (min %.3f)" % (
        np.mean(fr[0.5]), np.min(fr[0.5]), np.mean(fr[1.0]), np.min(fr[1.0])))

    # 4 jitter 크기, 창 경계 비교차
    worst_t, worst_v, bad = 0.0, 0.0, []
    for p in perfs:
        db = aw.timing(piece, p)[0]
        cw = man[(man.perf == p) & (man.cond == "clean")].set_index("win_idx")["n_notes"]
        for r, (lo, hi) in enumerate(regions):
            t0, t1 = aw.region_span(db, lo, hi)
            c = clean[p][0]
            m = aw.select_notes(c["onset_ms"], t0, t1)
            tn, _ = load_cond(piece, cond_name(p, "tempo", 1.0, r))
            vn, _ = load_cond(piece, cond_name(p, "vel", 1.0, r))
            m_time = dg.mean_abs_delta(tn["onset_ms"][m], c["onset_ms"][m])
            m_vel = dg.mean_abs_delta(vn["velocity"][m], c["velocity"][m])
            jt, _ = load_cond(piece, cond_name(p, "jit_time", 1.0, r))
            jv, _ = load_cond(piece, cond_name(p, "jit_vel", 1.0, r))
            # 짝짓기: 지터가 바꾸지 않는 열로 정렬 (pitch, 길이, 벨로시티)
            def key(n):
                return np.lexsort((n["onset_ms"], n["velocity"], n["off_ped_ms"] - n["onset_ms"],
                                   n["off_key_ms"] - n["onset_ms"], n["pitch"]))
            ci, ti = key(c), key(jt)
            mc = m[ci]
            dt = np.abs(jt["onset_ms"][ti] - c["onset_ms"][ci])[mc]
            ach_t = float(dt.mean())
            ach_v = dg.mean_abs_delta(jv["velocity"][aw.select_notes(jv["onset_ms"], t0, t1)],
                                      c["velocity"][m][np.argsort(c["onset_ms"][m], kind="stable")]) \
                if np.array_equal(np.sort(jv["onset_ms"][aw.select_notes(jv["onset_ms"], t0, t1)]),
                                  np.sort(c["onset_ms"][m])) else np.nan
            rt, rv = ach_t / m_time, ach_v / m_vel
            worst_t = max(worst_t, abs(rt - 1))
            worst_v = max(worst_v, abs(rv - 1))
            if abs(rt - 1) > 0.2 or not abs(rv - 1) <= 0.2:
                bad.append(f"{p}/r{r} t={rt:.2f} v={rv:.2f}")
            for cond in ("jit_time", "jit_vel", "vel", "pedal"):  # tempo 는 창 안 음 분배가 달라지는 게 정상
                for a in ((0.5, 1.0) if cond == "vel" else (1.0,)):
                    sub = man[(man.perf == p) & (man.cond == cond) & (man.region == r) & (man.alpha == a)]
                    if not np.array_equal(sub.set_index("win_idx")["n_notes"].reindex(cw.index).values, cw.values):
                        bad.append(f"{p}/r{r}/{cond}/{a} window count")
    res["4 jitter 크기 ±20%·창 경계 비교차"] = (not bad, f"위반 {bad[:4] or 0}; 최악 편차 time {worst_t:.3f}, vel {worst_v:.3f}")

    # 5 swap/deadpan
    bad, tot = [], {"swap": [], "deadpan": []}
    minbar = {"swap": 10**9, "deadpan": 10**9}
    for p in perfs:
        db = aw.timing(piece, p)[0]
        for r, (lo, hi) in enumerate(regions):
            t0, t1 = aw.region_span(db, lo, hi)
            c = clean[p][0]
            nb_ = int(aw.select_notes(c["onset_ms"], t0, t1).sum())
            for cond in ("swap", "deadpan"):
                n, nbeats = load_cond(piece, cond_name(p, cond, 1.0, r))
                na = int(aw.select_notes(n["onset_ms"], t0, t1).sum())
                outside = len(n["onset_ms"]) - na
                if outside != len(c["onset_ms"]) - nb_:
                    bad.append(f"{p}/{cond}/r{r} 구간 밖으로 샘")
                tot[cond].append((nb_, na))
                bar = np.histogram(n["onset_ms"], bins=(db[lo:hi + 1] - 30))[0]
                minbar[cond] = min(minbar[cond], int(bar.min()))
                if bar.min() == 0:
                    bad.append(f"{p}/{cond}/r{r} 빈 마디")
                inb = nbeats[(nbeats >= db[lo]) & (nbeats < db[hi])]
                if not np.all(np.isin(db[lo:hi], inb)):
                    bad.append(f"{p}/{cond}/r{r} downbeat 불일치")
    det = "; ".join(f"{c}: 구간 안 음 수 전 {np.mean([a for a,_ in v]):.0f} -> 후 {np.mean([b for _,b in v]):.0f} "
                    f"(후/전 비 min {min(b/a for a,b in v):.2f} max {max(b/a for a,b in v):.2f}, 마디당 최소 {minbar[c]})"
                    for c, v in tot.items())
    res["5 swap/deadpan 구간 안 삽입"] = (not bad, f"위반 {bad[:3] or 0}; {det}")

    # 6 창 격자
    bad, det = [], []
    for p in [aw.SCORE] + perfs:
        db = aw.timing(piece, p)[0]
        w = aw.window_grid(db, L, hop, int(clean[p][0]["onset_ms"].max()))
        for lo, hi in regions:
            cnt = sum(aw.overlap_label(x["k_lo"], x["k_hi"], lo, hi) == "in" for x in w)
            if cnt != 3:
                bad.append(f"{p} in={cnt}")
        if p == perfs[0]:
            det.append(f"n_db={len(db)} n_win={len(w)}")
        if len(w) != len(aw.window_grid(aw.timing(piece, aw.SCORE)[0], L, hop, 10**9)):
            bad.append(f"{p} n_win mismatch")
    if piece == "schubert_d899_3" and (len(db) != 86 or len(w) != 42):
        bad.append("schubert 86/42 아님")
    res["6 창 격자(구간당 in 3개)"] = (not bad, f"위반 {bad[:3] or 0}; {det[0]}; L={L} hop={hop}")

    # 7 downbeat-onset 거리
    meds = {}
    for p in perfs + [aw.SCORE]:
        db = aw.timing(piece, p)[0]
        o = np.sort(clean[p][0]["onset_ms"])
        j = np.clip(np.searchsorted(o, db), 1, len(o) - 1)
        d = np.minimum(abs(o[j] - db), abs(o[j - 1] - db))
        meds[p] = float(np.median(d))
    res["7 downbeat~onset 중앙값 <50ms"] = (max(meds[p] for p in perfs) < 50,
                                            "performers max %.1f ms; " % max(meds[p] for p in perfs) +
                                            ", ".join(f"{p}={v:.0f}" for p, v in meds.items()))
    print(f"\n=== {piece} 자체 검사 ===")
    for k, (ok, d) in res.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {k}: {d}")
    return res


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pieces = args or list(FEAS_PIECES)
    for pc in pieces:
        if "--check" not in sys.argv:
            df = build(pc)
            print(f"{pc}: rows={len(df)} needs_embed={int(df.needs_embed.sum())} files={df.notes_file.nunique()}")
        run_checks(pc)
