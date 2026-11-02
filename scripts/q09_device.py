"""Q9 ① · 연주자별 MIDI 기기 지문 (D960 + 전체). ②③(MAESTRO·ASAP 대조)은 q09b."""
import numpy as np, pandas as pd
from q_common import *
import mido
f = pd.read_parquet(paths.INTERIM / "pp_features.parquet")
def fingerprint(sid):
    m = mido.MidiFile(str(pp.midi_path(sid))); cc64 = set(); vel = set(); times = []; progs = set(); ntrk = len(m.tracks); t = 0
    for tr in m.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "control_change" and msg.control == 64: cc64.add(msg.value)
            if msg.type == "note_on" and msg.velocity > 0: vel.add(msg.velocity); times.append(t)
            if msg.type == "program_change": progs.add(msg.program)
    dt = np.diff(sorted(set(times))); gcd = int(np.gcd.reduce(dt[dt > 0])) if (dt > 0).any() else 0
    return dict(ppq=m.ticks_per_beat, n_tracks=ntrk, type=m.type, cc64_levels=len(cc64), cc64_binary=bool(cc64 <= {0, 127} or cc64 <= {0, 64, 127}), vel_levels=len(vel), tick_gcd=gcd, programs=sorted(progs))
rows = []
for sid in f.segment_id:
    try: rows.append(dict(segment_id=sid, **fingerprint(sid)))
    except Exception as e: rows.append(dict(segment_id=sid, err=repr(e)))
F = pd.DataFrame(rows).merge(f[["segment_id", "piece", "performer", "is_score", "n_tempo_events", "tempo_first_bpm", "vel_n_unique", "cc66", "cc67", "poly_aftertouch", "chan_aftertouch"]], on="segment_id")
agg = F.groupby(["piece", "performer"]).agg(n=("segment_id", "size"), ppq=("ppq", lambda s: sorted(set(s))), tracks=("n_tracks", lambda s: sorted(set(s))),
      cc64_levels_med=("cc64_levels", "median"), cc64_binary_frac=("cc64_binary", "mean"), vel_levels_med=("vel_levels", "median"), tick_gcd=("tick_gcd", lambda s: sorted(set(s))[:4]),
      tempo_bpm=("tempo_first_bpm", lambda s: sorted(set(np.round(s, 1)))[:4]), cc66=("cc66", "sum"), cc67=("cc67", "sum"), aftertouch=("poly_aftertouch", "sum"))
pd.set_option("display.width", 250); print(agg.to_string())
# 기기 지문 → 집단
F["dev"] = F.ppq.astype(str) + "/" + F.cc64_binary.map({True: "pedal2", False: "pedalN"}) + "/" + np.where(F.tick_gcd > 1, "gcd" + F.tick_gcd.astype(str), "gcd1")
dev = F[~F.is_score].groupby(["piece", "performer"]).dev.agg(lambda s: s.value_counts().index[0])
print("\n기기 지문 종류(사람 연주):", dev.value_counts().to_dict())
F.to_csv(OUT / "q09_fingerprint.csv", index=False); save("q09_device", dict(by_performer=agg.reset_index().astype(str).to_dict("records"), device_classes=dev.value_counts().to_dict()))
