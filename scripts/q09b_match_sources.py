"""Q9 ②③ · PercePiano 연주 ↔ MAESTRO / ASAP 원본 대조 (음높이 열 부분 일치). 외부 자료는 읽기만."""
import sys, re, numpy as np, pandas as pd, mido
from q_common import *
M = Path(sys.argv[1]) / "maestro-v3.0.0"; ASAP = paths.ASAP
meta = pd.read_csv(M / "maestro-v3.0.0.csv")
def pitch_seq(path, max_notes=None):
    m = mido.MidiFile(str(path)); ev = []
    for tr in m.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0: ev.append((t, msg.note))
    ev.sort(); s = [p for _, p in ev]; return s if max_notes is None else s[:max_notes]
def onset_groups(seq_with_t):
    pass
def to_str(seq): return "".join(chr(0x100 + p) for p in seq)
def chord_sorted_seq(path):
    """같은 tick 의 음은 음높이 순으로 정렬해 화음 내 순서 차이를 없앤다."""
    m = mido.MidiFile(str(path)); ev = []
    for tr in m.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0: ev.append((t, msg.note))
    ev.sort(); return [p for _, p in ev]
# 후보 원본: MAESTRO 의 D960 / D935 no.3, ASAP op142/3
cand = meta[meta.canonical_composer.str.contains("Schubert")]
print("MAESTRO 후보 %d개 (연도별 %s)" % (len(cand), cand.year.value_counts().sort_index().to_dict()))
cand_seq = {r.midi_filename: to_str(chord_sorted_seq(M / r.midi_filename)) for r in cand.itertuples()}
asap = {str(p.relative_to(ASAP)): to_str(chord_sorted_seq(p)) for p in (ASAP / "Schubert/Impromptu_op142/3").glob("*.mid") if p.name != "midi_score.mid"}
f = pd.read_parquet(paths.INTERIM / "pp_features.parquet"); f = f[~f.is_score & (f.piece != "Beethoven_WoO80")]
rows = []
for (piece, perf), g in f.groupby(["piece", "performer"]):
    # 연주자별로 8마디 구간 3개(가장 음이 많은 것) 의 음열을 원본 안에서 찾는다
    segs = g[g.n_bars.isin([4, 8])].nlargest(3, "n_notes")
    hits = {}
    for sid in segs.segment_id:
        q = to_str(chord_sorted_seq(pp.midi_path(sid)))
        for k in range(0, max(1, len(q) - 40), 20):   # 40음 조각을 여러 위치에서 시도 (화음 순서 차이 완화)
            sub = q[k:k + 40]
            for name, s in list(cand_seq.items()) + list(asap.items()):
                if sub in s: hits[name] = hits.get(name, 0) + 1
    rows.append(dict(piece=piece, performer=perf, n_probe_segments=len(segs), hits={k: v for k, v in sorted(hits.items(), key=lambda kv: -kv[1])[:3]}))
R = pd.DataFrame(rows)
def year_of(name):
    r = meta[meta.midi_filename == name]; return int(r.year.iloc[0]) if len(r) else ("ASAP" if isinstance(name, str) and name.startswith("Schubert") else None)
R["best"] = R.hits.map(lambda h: next(iter(h)) if h else None); R["best_year"] = R.best.map(lambda n: year_of(n) if n else None); R["best_hits"] = R.hits.map(lambda h: next(iter(h.values())) if h else 0)
pd.set_option("display.width", 250); print(R[["piece", "performer", "n_probe_segments", "best", "best_year", "best_hits"]].to_string(index=False))
# 같은 원본 파일에 두 곡의 연주자가 붙으면 → 같은 사람(같은 녹음)
same = R.dropna(subset=["best"]).groupby("best").apply(lambda g: list(zip(g.piece, g.performer))); print("\n같은 원본 파일을 가리키는 연주자:", {k: v for k, v in same.items() if len(v) > 1})
R.to_csv(OUT / "q09b_source_match.csv", index=False); save("q09b_match_sources", dict(matched=int(R.best.notna().sum()), total=len(R), years=R.best_year.astype(str).value_counts().to_dict(), rows=R.astype(str).to_dict("records")))
