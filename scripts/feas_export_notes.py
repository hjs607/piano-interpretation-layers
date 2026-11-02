"""ASAP MIDI -> 음 표 (.npz) 내보내기 (계약: docs/feas_interface.md §1).

IdioMap venv(ariautils)로 실행:
    <IdioMap>/aria/bin/python -I scripts/feas_export_notes.py [--piece KEY ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

ROOT = Path(__file__).resolve().parents[1]
IDIOMAP = Path.home() / "Desktop" / "IdioMap" / "Aria"
PIECES = {
    "schubert_d899_3": "Schubert/Impromptu_op.90_D.899/3",
}


def midi_to_note_table(path: Path, apply_pedal: bool = True) -> dict[str, np.ndarray]:
    """note_msgs 순서대로 (onset, off_key, off_ped, pitch, velocity). ms 는 tokenizer 와 같은 tick_to_ms."""
    md = MidiDict.from_midi(path)
    # AbsTokenizer 와 같은 전처리: 무시 악기 제거 후 pedal interval 구축
    md.remove_instruments(AbsTokenizer().config["ignore_instruments"])
    intervals = md._build_pedal_intervals() if apply_pedal else {}
    on, off_k, off_p, pit, vel = [], [], [], [], []
    n_ext = 0
    for m in md.note_msgs:
        s, e = m["data"]["start"], m["data"]["end"]
        e_ped = e
        for ps, pe in intervals.get(m["channel"], []):
            if ps < e < pe:  # absolute.py 와 동일 (첫 일치 구간에서 break)
                e_ped = pe
                break
        on.append(md.tick_to_ms(s))
        off_k.append(md.tick_to_ms(e))
        off_p.append(md.tick_to_ms(e_ped))
        pit.append(m["data"]["pitch"])
        vel.append(m["data"]["velocity"])
        n_ext += e_ped != e
    arr = {k: np.asarray(v, dtype=np.int64) for k, v in
           dict(onset_ms=on, off_key_ms=off_k, off_ped_ms=off_p, pitch=pit, velocity=vel).items()}
    assert (arr["off_ped_ms"] >= arr["off_key_ms"]).all()
    return arr


def export_piece(piece: str, asap: Path, out_root: Path) -> None:
    rel = PIECES[piece]
    ann = json.loads((asap / "asap_annotations.json").read_text())
    perfs = sorted(
        Path(k).stem for k, v in ann.items()
        if k.startswith(rel + "/") and Path(k).parent.as_posix() == rel
        and Path(k).name != "midi_score.mid" and v.get("score_and_performance_aligned") is True
    )
    out = out_root / "notes" / piece
    out.mkdir(parents=True, exist_ok=True)
    meta = {"piece": piece, "piece_dir": rel, "performances": perfs, "files": {}, "n_notes": {}}
    for name in perfs + ["midi_score"]:
        src = asap / rel / f"{name}.mid"
        is_score = name == "midi_score"
        arr = midi_to_note_table(src, apply_pedal=not is_score)
        if is_score:
            arr["off_ped_ms"] = arr["off_key_ms"].copy()
        np.savez(out / f"{name}.npz", **arr)
        n = len(arr["pitch"])
        frac = float((arr["off_ped_ms"] != arr["off_key_ms"]).mean())
        meta["files"][name] = str(src.relative_to(asap))
        meta["n_notes"][name] = n
        print(f"{piece:16s} {name:14s} n={n:5d} ped_ext_frac={frac:.3f} "
              f"off_ped>=off_key={bool((arr['off_ped_ms'] >= arr['off_key_ms']).all())}")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", nargs="*", default=list(PIECES))
    ap.add_argument("--asap", type=Path, default=IDIOMAP / "asap-dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "interim" / "feas")
    a = ap.parse_args()
    for p in a.piece:
        export_piece(p, a.asap, a.out)


if __name__ == "__main__":
    main()
