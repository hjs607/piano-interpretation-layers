"""Q13 · ASAP 마디 단위 사용 가능성: 표본·중복·주석·정렬 시범(30연주). ASAP 은 읽기만."""
import json, hashlib, warnings, numpy as np, pandas as pd, mido
from joblib import Parallel, delayed
from q_common import *
from src.eda import align
warnings.filterwarnings("ignore"); rng = np.random.default_rng(0); R = paths.ASAP
meta = pd.read_csv(R / "metadata.csv"); ann = json.load(open(R / "asap_annotations.json"))
meta["folder"] = meta.folder.str.rstrip("/"); meta["aligned"] = meta.midi_performance.map(lambda k: ann[k]["score_and_performance_aligned"])
# ① 표본: 같은 midi_score + aligned + 6개 이상
g = meta[meta.aligned].groupby("folder"); pool = g.filter(lambda x: len(x) >= 6); print("① 시작 표본: %d곡 %d연주" % (pool.folder.nunique(), len(pool)))
# ② 중복: 파일 해시, 음열 해시, 같은 연주자 이름(stem 의 알파벳 부분) 반복
def pitch_hash(p):
    m = mido.MidiFile(str(p)); ev = []
    for tr in m.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0: ev.append((t, msg.note))
    ev.sort(); return hashlib.md5(bytes(x for _, x in ev[:2000])).hexdigest(), len(ev)
pool = pool.copy(); pool["md5"] = pool.midi_performance.map(lambda k: hashlib.md5((R / k).read_bytes()).hexdigest())
ph = pool.midi_performance.map(lambda k: pitch_hash(R / k)); pool["pitch_md5"] = [a for a, _ in ph]; pool["n_notes"] = [b for _, b in ph]
pool["performer_stem"] = pool.midi_performance.str.split("/").str[-1].str.replace(r"\d+M?\.mid$", "", regex=True)
dup_file = pool[pool.duplicated("md5", keep=False)]; dup_pitch = pool[pool.duplicated(["folder", "pitch_md5"], keep=False)]
same_perf = pool.groupby(["folder", "performer_stem"]).size(); same_perf = same_perf[same_perf > 1]
print("② 중복: 바이트 동일 %d, 같은 곡 음열(앞 2000음) 동일 %d, 같은 곡 같은 연주자 이름 여러 해 %d쌍" % (len(dup_file), len(dup_pitch), len(same_perf)))
# ③ 주석: downbeats_score_map 일치, 마디 길이 이상치, bR 비율
def ann_stats(k):
    a = ann[k]; db = np.array(a["performance_downbeats"]); L_ = np.diff(db); lo = np.log(L_[L_ > 0]); med = np.median(lo); mad = np.median(np.abs(lo - med)) + 1e-9
    out = np.mean(np.abs(lo - med) > 3 * 1.4826 * mad); bR = np.mean([v == "bR" for v in a["performance_beats_type"].values()])
    return dict(n_db=len(db), bar_outlier_frac=float(out), bR_frac=float(bR), map_hash=hashlib.md5(json.dumps(a["downbeats_score_map"]).encode()).hexdigest())
A = pd.DataFrame({k: ann_stats(k) for k in pool.midi_performance}).T; pool = pool.join(A, on="midi_performance")
map_consistent = pool.groupby("folder").map_hash.nunique(); print("③ 주석: downbeats_score_map 이 곡 안에서 다른 곡 %d/%d · 마디 길이 이상치 비율 중앙 %.3f (>2%%: %d연주) · bR 비율 중앙 %.3f" % (
    (map_consistent > 1).sum(), len(map_consistent), pool.bar_outlier_frac.median(), (pool.bar_outlier_frac > 0.02).sum(), pool.bR_frac.median()))
# ④ 정렬 시범: 무작위 30연주 (곡 다양하게)
samp = pool.groupby("folder").sample(1, random_state=0).sample(min(30, pool.folder.nunique()), random_state=0)
def trial(row):
    try:
        score = align.load_score(R / row.xml_score); sna = align.score_note_array(score); perf = align.load_performance(R / row.midi_performance)
        al = align.align(sna, perf.note_array()); c = align.alignment_counts(al); return dict(midi=row.midi_performance, score_notes=len(sna), perf_notes=len(perf.note_array()), **c, mr_score=c["match"] / len(sna))
    except Exception as e: return dict(midi=row.midi_performance, err=repr(e)[:120])
T = pd.DataFrame(Parallel(n_jobs=6)(delayed(trial)(r) for r in samp.itertuples()))
ok = T[T.get("err", pd.Series(dtype=object)).isna()] if "err" in T else T
print("④ 정렬 시범 %d연주: 오류 %d · 악보 쪽 매치율 중앙 %.3f, 최소 %.3f" % (len(T), len(T) - len(ok), ok.mr_score.median(), ok.mr_score.min()))
print(T.sort_values("mr_score").head(5).to_string(index=False) if "mr_score" in T else T.head())
# ⑤ midi_score 가 '문자'인가: 벨로시티 종류·템포 이벤트
def score_info(folder):
    m = mido.MidiFile(str(R / folder / "midi_score.mid")); vel = set(); tempi = set(); cc64 = 0
    for tr in m.tracks:
        for msg in tr:
            if msg.type == "note_on" and msg.velocity > 0: vel.add(msg.velocity)
            if msg.type == "set_tempo": tempi.add(msg.tempo)
            if msg.type == "control_change" and msg.control == 64: cc64 += 1
    return dict(vel_levels=len(vel), tempo_events=len(tempi), cc64=cc64)
SI = pd.DataFrame({fo: score_info(fo) for fo in pool.folder.unique()}).T; print("⑤ midi_score: 벨로시티 종류 중앙 %.0f (1 이면 기호 미반영), 템포 이벤트 중앙 %.0f, 페달 CC 있는 곡 %d/%d" % (SI.vel_levels.median(), SI.tempo_events.median(), (SI.cc64 > 0).sum(), len(SI)))
# ⑥ MAESTRO 연결·연도
pool["maestro"] = pool.maestro_midi_performance.notna(); yrs = pool[pool.maestro].maestro_midi_performance.str.extract(r"^(\d{4})")[0]
print("⑥ MAESTRO 연결 %d/%d · 연도 분포 %s" % (pool.maestro.sum(), len(pool), yrs.value_counts().sort_index().to_dict()))
# 품질 필터 후 곡 수
good = pool[(pool.bar_outlier_frac <= 0.02)]; good = good[~good.duplicated(["folder", "pitch_md5"])]; gc = good.groupby("folder").size(); n_good = int((gc >= 6).sum())
verdict = dict(start_pieces=int(pool.folder.nunique()), start_perfs=len(pool), align_mr_median=float(ok.mr_score.median()), pass_align=bool(ok.mr_score.median() >= 0.9),
               bar_outlier_fail=int((pool.bar_outlier_frac > 0.02).sum()), pieces_ge6_after_filter=n_good, perfs_after_filter=int(gc[gc >= 6].sum()), pass_pieces=bool(n_good >= 40),
               midi_score_vel_levels_median=float(SI.vel_levels.median()))
print("\n판정:", verdict); pool.to_csv(OUT / "q13_asap_pool.csv", index=False); save("q13_asap", dict(verdict=verdict, align_trial=T.astype(str).to_dict("records")))
