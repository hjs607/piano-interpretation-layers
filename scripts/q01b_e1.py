"""Q1b③ · E1 재현: 증강형 변환(전역 템포 ×0.8/×1.2, 벨로시티 ±10) 과 증강에 없는 변환의 임베딩 이동 ÷ 사람끼리 거리.
IdioMap venv 로 실행:  ~/Desktop/IdioMap/Aria/aria/bin/python -I scripts/q01b_e1.py
정의는 reports/exp_2026-10-24.md 부록(04 §8 sensitivity) 을 따른다. 표준화 코사인."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from feas_embed import embed_seq, get_device, load_model, table_to_middict, tokens_of  # noqa: E402
FEAS = ROOT / "data" / "interim" / "feas"; OUT = ROOT / "reports" / "q"; PIECE = "schubert_d899_3"
man = pd.read_csv(FEAS / f"manifest_{PIECE}.csv", keep_default_na=False); z = np.load(FEAS / f"emb_{PIECE}.npz"); norm = np.load(FEAS / "normalization.npz")
E = {int(r): (e - norm["mean"]) / norm["std"] for r, e in zip(z["row_id"], z["emb"].astype(np.float64))}
cos = lambda a, b: 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
clean = man[man.cond == "clean"]; humans = sorted(set(clean.perf) - {"midi_score"})
in_wins = sorted(set(man[(man.cond == "tempo") & (man.overlap == "in")].win_idx))
hh = {}
for w in in_wins:
    rows = clean[(clean.win_idx == w) & clean.perf.isin(humans)].row_id.tolist(); V = [E[r] for r in rows]
    hh[w] = np.median([cos(V[i], V[j]) for i in range(len(V)) for j in range(i + 1, len(V))])
dev = get_device(); model = load_model(dev)
def embed_notes(t):
    return (embed_seq(model, tokens_of(table_to_middict(t)), dev).astype(np.float64) - norm["mean"]) / norm["std"]
def window_notes(perf, row):
    n = np.load(ROOT / row.notes_file); m = (n["onset_ms"] >= row.t0_ms) & (n["onset_ms"] < row.t1_ms)
    t = {k: n[k][m].copy() for k in ("onset_ms", "off_key_ms", "off_ped_ms", "pitch", "velocity")}
    t0 = t["onset_ms"].min()
    for k in ("onset_ms", "off_key_ms", "off_ped_ms"): t[k] = t[k] - t0     # 창 시작을 0 으로 (토크나이저와 동일)
    return t
def transform(t, kind):
    t = {k: v.copy() for k, v in t.items()}
    if kind.startswith("tempo_scale"):
        f = float(kind.split("_")[-1])                                           # 템포 ×f → 시간 ×1/f
        for k in ("onset_ms", "off_key_ms", "off_ped_ms"): t[k] = np.round(t[k] / f).astype(np.int64)
    elif kind.startswith("vel_offset"):
        t["velocity"] = np.clip(t["velocity"] + int(kind.split("_")[-1]), 1, 127).astype(np.int64)
    return t
res = []
for perf in humans:
    for w in in_wins:
        row = clean[(clean.perf == perf) & (clean.win_idx == w)].iloc[0]; base = E[int(row.row_id)]; t = window_notes(perf, row)
        chk = embed_notes(t); res.append(dict(perf=perf, win=w, cond="clean_recompute", ratio=cos(base, chk) / hh[w]))
        for kind in ("tempo_scale_0.8", "tempo_scale_1.2", "vel_offset_+10", "vel_offset_-10"):
            e = embed_notes(transform(t, kind)); res.append(dict(perf=perf, win=w, cond=kind, ratio=cos(base, e) / hh[w]))
# 증강에 없는 변환 (manifest 의 cond 행, in 창): tempo α=1, pedal, vel α=1, jit_time
for cond in ("tempo", "pedal", "vel", "jit_time"):
    sub = man[(man.cond == cond) & (man.overlap == "in") & (man.alpha == 1.0) & man.perf.isin(humans)]
    for r in sub.itertuples():
        if int(r.row_id) in E: res.append(dict(perf=r.perf, win=r.win_idx, cond=f"{cond}_a1_from_feas", ratio=cos(E[int(r.clean_row_id)], E[int(r.row_id)]) / hh[r.win_idx]))
R = pd.DataFrame(res); T = R.groupby("cond").ratio.agg(n="size", median="median", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75)).round(3)
print(T.to_string()); R.to_csv(OUT / "q01b_e1_rows.csv", index=False); T.to_csv(OUT / "q01b_e1_table.csv")
ref = {"tempo_scale_0.8": 0.129, "tempo_scale_1.2": 0.058, "vel_offset_+10": 0.209, "vel_offset_-10": 0.146, "tempo_a1_from_feas": 0.103, "pedal_a1_from_feas": 0.881, "vel_a1_from_feas": 2.067, "jit_time_a1_from_feas": 1.769}
cmp = {k: dict(reported=v, now=float(T.loc[k, "median"]) if k in T.index else None) for k, v in ref.items()}
ok = all(c["now"] is not None and abs(c["now"] - c["reported"]) <= 0.02 for c in cmp.values()); print("\n보고값 대비 ±0.02:", cmp, "\n판정:", "통과" if ok else "불일치")
json.dump(dict(table=T.to_dict("index"), compare=cmp, pass_=ok), open(OUT / "q01b_e1.json", "w"), ensure_ascii=False, indent=1)
