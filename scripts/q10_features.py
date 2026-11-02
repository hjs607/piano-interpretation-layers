"""Q10 · 손 특징 측정 품질: 주입 실험 / 겹침 재측정 / 탈락 편향 / 중복."""
import warnings, numpy as np, pandas as pd, pretty_midi
from joblib import Parallel, delayed
from q_common import *
from src.eda import align, midi_stats, perf_features
warnings.filterwarnings("ignore"); rng = np.random.default_rng(0)
FX = ["log_tempo", "tempo_logstd", "timing_std_ms", "articulation_median", "vel_std", "vel_range", "pedal_down_ratio"]
f = pd.read_parquet(paths.INTERIM / "pp_features.parquet"); f["log_tempo"] = np.log2(f.tempo_median_bpm); f["mr"] = f.match / f.score_notes
TMP = Path("/private/tmp/claude-501/-Users-jaeseongheo-Desktop-Dataset------/6b4df605-a5a0-4a51-88ad-4d9c9cb23b30/scratchpad/q10"); TMP.mkdir(parents=True, exist_ok=True)

def measure(mid_path, sid):
    score = align.load_score(pp.score_xml_path(sid)); sna = align.score_note_array(score); perf = align.load_performance(mid_path)
    al = align.align(sna, perf.note_array()); p = align.note_params(score, perf, al); row = perf_features.summarize(p); row.update(midi_stats.midi_stats(Path(mid_path)))
    row["log_tempo"] = np.log2(row["tempo_median_bpm"]); return {k: row[k] for k in FX + ["vel_mean"]}
def modify(sid, kind):
    pm = pretty_midi.PrettyMIDI(str(pp.midi_path(sid)))
    for ins in pm.instruments:
        if kind == "tempo_x1.1":
            for n in ins.notes: n.start /= 1.1; n.end /= 1.1
            for c in ins.control_changes: c.time /= 1.1
        elif kind == "vel_+10":
            for n in ins.notes: n.velocity = min(127, n.velocity + 10)
        elif kind == "dur_x1.2":
            for n in ins.notes: n.end = n.start + 1.2 * (n.end - n.start)
        elif kind == "no_pedal":
            ins.control_changes = [c for c in ins.control_changes if c.number != 64]
    out = TMP / f"{sid}__{kind}.mid"; pm.write(str(out)); return out
# ① 주입: D960 사람 구간 30개 (매치율 ≥0.8, 벨로시티 상한 여유 있는 것)
pool = f[~f.is_score & f.piece.str.startswith("Schubert_D960") & (f.mr >= 0.8) & (f.vel_p95 <= 110)]
ids = list(rng.choice(pool.segment_id.to_numpy(), 30, replace=False)); kinds = ["tempo_x1.1", "vel_+10", "dur_x1.2", "no_pedal"]
def one(sid):
    base = measure(pp.midi_path(sid), sid); out = {"segment_id": sid, "base": base}
    for k in kinds: out[k] = measure(modify(sid, k), sid)
    return out
res = Parallel(n_jobs=8)(delayed(one)(s) for s in ids)
# 기대값 (음 수준 변환 → 요약 특징)
exp = {"tempo_x1.1": {"log_tempo": lambda b: b + np.log2(1.1), "timing_std_ms(/1.1)": lambda b: b / 1.1, "timing_std_ms(unchanged)": lambda b: b, "tempo_logstd": lambda b: b, "vel_std": lambda b: b, "pedal_down_ratio": lambda b: b},
       "vel_+10": {"vel_mean": lambda b: b + 10, "vel_std": lambda b: b, "vel_range": lambda b: b, "log_tempo": lambda b: b},
       "dur_x1.2": {"articulation_median": lambda b: b + np.log(1.2), "log_tempo": lambda b: b, "vel_std": lambda b: b},
       "no_pedal": {"pedal_down_ratio": lambda b: 0.0, "log_tempo": lambda b: b, "articulation_median": lambda b: b}}
rows = []
for k, feats in exp.items():
    for ft, fn in feats.items():
        ft0 = ft.split("(")[0]; got = np.array([r[k][ft0] for r in res]); want = np.array([fn(r["base"][ft0]) for r in res]); scale = np.std([r["base"][ft0] for r in res]) + 1e-9
        err = np.abs(got - want); tol = np.where(np.abs(want) > 1e-9, np.abs(want) * 0.05, 0.05 * scale)
        rows.append(dict(cond=k, feature=ft, median_abs_err=round(float(np.median(err)), 4), frac_within_5pct=round(float(np.mean(err <= tol)), 2), expected_change=("yes" if fn(1.0) != 1.0 or ft0 == "pedal_down_ratio" else "no")))
INJ = pd.DataFrame(rows); print("① 주입 실험 (30구간): 기대값 대비 |오차|, 5% 허용 내 비율\n", INJ.to_string(index=False))
# ② 겹침 재측정: 단위 내 잔차의 16↔8 상관
d9 = f[~f.is_score & f.piece.str.startswith("Schubert_D960") & (f.mr >= 0.8)].dropna(subset=FX).copy()
W = d9[FX] - d9.groupby("unit")[FX].transform("mean"); W["segment_id"] = d9.segment_id; W = W.set_index("segment_id")
info = d9.set_index("segment_id"); info["segment"] = [pp.parse_name(x)["segment"] for x in info.index]; pairs = []
for sid, r in info[info.n_bars == 16].iterrows():
    stem = sid.rsplit("_", 2)[0].replace("16bars", "8bars"); kids = [f"{stem}_{r.performer}_{k:02d}" for k in (2 * r.segment - 1, 2 * r.segment)]
    if all(k in W.index for k in kids): pairs.append(pd.concat([W.loc[sid].add_suffix("_16"), W.loc[kids].mean().add_suffix("_8")]))
P = pd.DataFrame(pairs); OV = {ft: round(float(P[f"{ft}_16"].corr(P[f"{ft}_8"], method="spearman")), 3) for ft in FX}
print("\n② 겹침 재측정 (단위 내 잔차, 16마디 vs 8마디 2개 평균, n=%d):" % len(P), OV)
# ③ 탈락 편향
drop = f[~f.is_score & (f.mr < 0.8)]; print("\n③ 매치율 <0.8 탈락: %d구간 → 곡별 %s · 연주자별 %s" % (len(drop), drop.piece.value_counts().to_dict(), drop.performer.value_counts().head(6).to_dict()))
mv3 = f[~f.is_score & (f.piece == "Schubert_D960_mv3")]; print("   mv3 연주 쪽 매치율 중앙 %.2f (트레몰로); mv3 vs mv2 timing_std_ms 중앙 %.1f vs %.1f, articulation_median %.2f vs %.2f" % (
    (mv3.match / mv3.perf_notes_pt).median(), mv3.timing_std_ms.median(), f[~f.is_score & (f.piece == "Schubert_D960_mv2")].timing_std_ms.median(), mv3.articulation_median.median(), f[~f.is_score & (f.piece == "Schubert_D960_mv2")].articulation_median.median()))
# ④ 중복: 단위 내 잔차 상관
Cm = W[FX].corr(method="spearman"); hi = [(a, b, round(Cm.loc[a, b], 2)) for i, a in enumerate(FX) for b in FX[i + 1:] if abs(Cm.loc[a, b]) >= 0.5]
print("\n④ 단위 내 잔차 상관 |ρ|≥0.5 쌍:", hi)
grade = {ft: ("사용" if OV[ft] >= 0.7 else "주의" if OV[ft] >= 0.5 else "제외") for ft in FX}
inj_fail = INJ[(INJ.expected_change == "yes") & (INJ.frac_within_5pct < 0.8)][["cond", "feature", "frac_within_5pct"]].to_dict("records")
verdict = dict(grade=grade, n_usable=sum(g != "제외" for g in grade.values()), injection_failures=inj_fail, redundant_pairs=hi, redesign_needed=bool(sum(g != "제외" for g in grade.values()) < 4))
print("\n판정:", verdict); save("q10_features", dict(injection=INJ.to_dict("records"), overlap=OV, verdict=verdict))
