"""MIDI 파일 단위 통계 (악보 정렬 없이 계산)."""
from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pretty_midi


def midi_stats(path: Path) -> dict:
    mf = mido.MidiFile(str(path))
    n_tempo, tempos, cc, n_poly_at, n_chan_at = 0, [], {}, 0, 0
    for track in mf.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                n_tempo += 1
                tempos.append(mido.tempo2bpm(msg.tempo))
            elif msg.type == "control_change":
                cc[msg.control] = cc.get(msg.control, 0) + 1
            elif msg.type == "polytouch":
                n_poly_at += 1
            elif msg.type == "aftertouch":
                n_chan_at += 1
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    on = np.array([n.start for n in notes])
    off = np.array([n.end for n in notes])
    vel = np.array([n.velocity for n in notes], float)
    t0, t1 = (on.min(), off.max()) if len(notes) else (0.0, 0.0)
    dur = max(t1 - t0, 1e-9)
    # 서스테인 페달(CC64 ≥ 64) 밟힌 시간 비율과 밟는 횟수
    ped = sorted(((c.time, c.value) for inst in pm.instruments for c in inst.control_changes if c.number == 64))
    down_time, n_press, state, last_t = 0.0, 0, False, t0
    for t, v in ped:
        now = v >= 64
        if state:
            down_time += max(min(t, t1) - max(last_t, t0), 0)
        if now and not state:
            n_press += 1
        state, last_t = now, t
    if state:
        down_time += max(t1 - max(last_t, t0), 0)
    # 평균 동시 발음 수 (발음 시간 가중)
    if len(notes):
        ev = np.concatenate([np.c_[on, np.ones_like(on)], np.c_[off, -np.ones_like(off)]])
        ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
        level = np.cumsum(ev[:, 1])
        seg = np.diff(ev[:, 0])
        sounding = seg[level[:-1] > 0].sum()
        poly = (level[:-1] * seg).sum() / sounding if sounding > 0 else np.nan
    else:
        poly = np.nan
    return {
        "ppq": mf.ticks_per_beat,
        "n_tempo_events": n_tempo,
        "tempo_first_bpm": tempos[0] if tempos else np.nan,
        "n_notes": len(notes),
        "duration_s": dur,
        "notes_per_s": len(notes) / dur,
        "vel_mean": vel.mean() if len(vel) else np.nan,
        "vel_std": vel.std() if len(vel) else np.nan,
        "vel_p5": np.percentile(vel, 5) if len(vel) else np.nan,
        "vel_p95": np.percentile(vel, 95) if len(vel) else np.nan,
        "vel_n_unique": len(np.unique(vel)),
        "cc64": cc.get(64, 0),
        "cc66": cc.get(66, 0),
        "cc67": cc.get(67, 0),
        "cc_other": sum(v for k, v in cc.items() if k not in (64, 66, 67)),
        "poly_aftertouch": n_poly_at,
        "chan_aftertouch": n_chan_at,
        "pedal_down_ratio": down_time / dur,
        "pedal_presses_per_s": n_press / dur,
        "polyphony": poly,
    }
