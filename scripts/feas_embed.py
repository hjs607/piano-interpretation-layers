"""Aria 임베딩 (가능성 실험, 계약: docs/feas_interface.md §3).

IdioMap venv 로 실행 (-I):
    PY=~/Desktop/IdioMap/Aria/aria/bin/python
    $PY -I scripts/feas_embed.py --replicate
    $PY -I scripts/feas_embed.py --check-tokens --piece schubert_d899_3
    $PY -I scripts/feas_embed.py --piece schubert_d899_3 [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from aria.config import load_model_config
from aria.embedding import MAX_EMBEDDING_SEQ_LEN, get_embedding_from_seq
from aria.model import ModelConfig, TransformerEMB
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
IDIOMAP = Path.home() / "Desktop" / "IdioMap" / "Aria"
FEAS = ROOT / "data" / "interim" / "feas"
CKPT = IDIOMAP / "models" / "aria-embedding" / "model.safetensors"
CACHE = IDIOMAP / "embeddings" / "aria_slice_embeddings.pkl"

TOK = AbsTokenizer()


# ---------------------------------------------------------------- note tables
def midi_to_note_table(path: Path) -> dict[str, np.ndarray]:
    """feas_export_notes.py 와 같은 로직 (복제 검증용 사본)."""
    md = MidiDict.from_midi(path)
    md.remove_instruments(TOK.config["ignore_instruments"])
    intervals = md._build_pedal_intervals()
    cols = {k: [] for k in ("onset_ms", "off_key_ms", "off_ped_ms", "pitch", "velocity")}
    for m in md.note_msgs:
        s, e = m["data"]["start"], m["data"]["end"]
        e_ped = e
        for ps, pe in intervals.get(m["channel"], []):
            if ps < e < pe:
                e_ped = pe
                break
        cols["onset_ms"].append(md.tick_to_ms(s))
        cols["off_key_ms"].append(md.tick_to_ms(e))
        cols["off_ped_ms"].append(md.tick_to_ms(e_ped))
        cols["pitch"].append(m["data"]["pitch"])
        cols["velocity"].append(m["data"]["velocity"])
    return {k: np.asarray(v, dtype=np.int64) for k, v in cols.items()}


def table_to_middict(t: dict[str, np.ndarray], idx=None) -> MidiDict:
    """1 tick = 1 ms (500 tpb, 500000 us/beat), 음 끝 = off_ped_ms, 페달 없음, metadata {}."""
    idx = np.arange(len(t["pitch"])) if idx is None else idx
    notes = [
        {"type": "note", "tick": int(t["onset_ms"][i]), "channel": 0,
         "data": {"pitch": int(t["pitch"][i]), "velocity": int(t["velocity"][i]),
                  "start": int(t["onset_ms"][i]), "end": int(t["off_ped_ms"][i])}}
        for i in idx
    ]
    return MidiDict(
        meta_msgs=[], pedal_msgs=[], note_msgs=notes, ticks_per_beat=500, metadata={},
        tempo_msgs=[{"type": "tempo", "data": 500000, "tick": 0}],
        instrument_msgs=[{"type": "instrument", "data": 0, "tick": 0, "channel": 0}],
    )


def tokens_of(md: MidiDict) -> list:
    seq = TOK.tokenize(md, add_dim_tok=False)[:MAX_EMBEDDING_SEQ_LEN]
    if seq[-1] != TOK.eos_tok:
        seq[-1] = TOK.eos_tok
    return seq


# ---------------------------------------------------------------- model
def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(device: torch.device) -> TransformerEMB:
    cfg = ModelConfig(**load_model_config(name="medium-emb"))
    cfg.set_vocab_size(TOK.vocab_size)
    model = TransformerEMB(cfg)
    model.load_state_dict(load_file(str(CKPT)), strict=True)
    model.eval()
    return model.to(device)


def embed_seq(model, seq, device) -> np.ndarray:
    with torch.no_grad():
        return get_embedding_from_seq(model=model, seq=seq, device=str(device)).float().cpu().numpy()


# ---------------------------------------------------------------- log
def update_log(**kv) -> None:
    p = FEAS / "embed_log.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(kv)
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- replicate
def replicate(args) -> None:
    rows = pickle.load(open(CACHE, "rb"))
    by_perf: dict[str, list] = {}
    for r in rows:
        by_perf.setdefault(r["midi_path"], []).append(r)
    # ASAP 에서 슬라이스가 여러 개인 연주를 곡 다양하게 (Schubert/Chopin 우선) 고른다
    prio = sorted(by_perf, key=lambda p: (not any(k in p for k in ("Impromptu_op.90_D.899/3", "Etudes_op_10/5")),
                                          -len(by_perf[p]), p))
    chosen = [p for p in prio if (IDIOMAP / p).exists() and len(by_perf[p]) >= 2][: args.n_perf]
    dev = get_device()
    model = load_model(dev)
    results, cos_all = [], []
    for p in chosen:
        t = midi_to_note_table(IDIOMAP / p)
        n = len(t["pitch"])
        for r in sorted(by_perf[p], key=lambda r: r["raw_slice_index"])[: args.n_win]:
            assert r["n_notes_total"] == n, (p, r["n_notes_total"], n)
            md = table_to_middict(t, np.arange(r["note_start"], r["note_end"]))
            seq = tokens_of(md)
            e = embed_seq(model, seq, dev)
            ref = r["embedding"].float().numpy()
            cos = float(e @ ref / (np.linalg.norm(e) * np.linalg.norm(ref)))
            cos_all.append(cos)
            results.append({"perf": p, "slice": r["raw_slice_index"], "cos": cos, "len_tokens": len(seq)})
            print(f"{p} slice {r['raw_slice_index']} cos={cos:.6f}")
    mn = min(cos_all)
    ok = mn >= 0.999 and len(chosen) >= 3
    print(f"min cosine {mn:.6f} over {len(cos_all)} windows, {len(chosen)} performances -> {'PASS' if ok else 'FAIL'}")
    update_log(replicate={"min_cos": mn, "n_windows": len(cos_all), "n_perfs": len(chosen),
                          "pass": bool(ok), "windows": results})
    if not ok:
        raise SystemExit(1)


# ---------------------------------------------------------------- manifest helpers
def read_manifest(piece: str) -> list[dict]:
    with open(FEAS / f"manifest_{piece}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_notes_cache: dict[str, dict] = {}


def load_notes(rel: str) -> dict:
    if rel not in _notes_cache:
        if len(_notes_cache) > 200:
            _notes_cache.clear()
        z = np.load(ROOT / rel)
        _notes_cache[rel] = {k: z[k] for k in z.files}
    return _notes_cache[rel]


def row_midi(row: dict) -> MidiDict:
    t = load_notes(row["notes_file"])
    sel = np.nonzero((t["onset_ms"] >= int(row["t0_ms"])) & (t["onset_ms"] < int(row["t1_ms"])))[0]
    assert len(sel) == int(row["n_notes"]), (row["row_id"], len(sel), row["n_notes"])
    # 계약: 조건 파일은 onset 오름차순 안정 정렬이어야 한다
    assert (np.diff(t["onset_ms"]) >= 0).all(), f"not onset-sorted: {row['notes_file']}"
    return table_to_middict(t, sel)


# ---------------------------------------------------------------- check tokens
def check_tokens(piece: str, seed: int = 0) -> None:
    rows = read_manifest(piece)
    by_id = {int(r["row_id"]): r for r in rows}
    per_file: dict[str, list] = {}
    for r in rows:
        if r["overlap"] == "out":
            per_file.setdefault(r["notes_file"], []).append(r)
    rng = random.Random(seed)
    n_checked, fails = 0, []
    for nf, rs in sorted(per_file.items()):
        for r in rng.sample(rs, min(3, len(rs))):
            ref = by_id[int(r["clean_row_id"])]
            same = tokens_of(row_midi(r)) == tokens_of(row_midi(ref))
            n_checked += 1
            if not same:
                fails.append({"row_id": int(r["row_id"]), "clean_row_id": int(ref["row_id"]), "notes_file": nf})
    print(f"{piece}: {len(per_file)} files, {n_checked} out-windows checked, {len(fails)} mismatches")
    update_log(**{f"token_check_{piece}": {"n_files": len(per_file), "n_checked": n_checked,
                                           "n_mismatch": len(fails), "mismatches": fails[:50]}})
    if fails:
        raise SystemExit(f"token mismatch: {fails[:5]}")


# ---------------------------------------------------------------- main embed
def embed_piece(piece: str, limit: int | None, save_every: int = 50) -> None:
    rows = [r for r in read_manifest(piece) if r["needs_embed"] == "1"]
    out = FEAS / f"emb_{piece}.npz"
    if out.exists():
        z = np.load(out)
        ids, embs = list(z["row_id"]), list(z["emb"])
    else:
        ids, embs = [], []
    done = set(int(i) for i in ids)
    todo = [r for r in rows if int(r["row_id"]) not in done]
    if limit is not None:
        todo = todo[:limit]
    dev = get_device()
    print(f"{piece}: {len(rows)} needed, {len(done)} done, {len(todo)} to run on {dev}")
    model = load_model(dev)

    def save():
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, row_id=np.asarray(ids, dtype=np.int64), emb=np.asarray(embs, dtype=np.float32).reshape(-1, 512))
        os.replace(tmp, out)

    t_start = time.time()
    for k, r in enumerate(todo, 1):
        seq = tokens_of(row_midi(r))
        embs.append(embed_seq(model, seq, dev))
        ids.append(int(r["row_id"]))
        if k % save_every == 0:
            save()
            el = time.time() - t_start
            print(f"  {k}/{len(todo)}  {el / k:.3f} s/window  eta {el / k * (len(todo) - k) / 60:.1f} min", flush=True)
    if todo:
        save()
        el = time.time() - t_start
        print(f"done {len(todo)} windows in {el:.1f}s ({el / len(todo):.3f} s/window)")
        update_log(**{f"embed_{piece}": {"device": str(dev), "n_new": len(todo), "seconds": el,
                                         "s_per_window": el / len(todo), "n_total": len(ids),
                                         "limited": limit is not None}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", action="store_true")
    ap.add_argument("--check-tokens", action="store_true")
    ap.add_argument("--piece")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-perf", type=int, default=4)
    ap.add_argument("--n-win", type=int, default=3)
    a = ap.parse_args()
    if a.replicate:
        replicate(a)
    elif a.check_tokens:
        assert a.piece, "--piece required"
        check_tokens(a.piece)
    else:
        assert a.piece, "--piece required"
        embed_piece(a.piece, a.limit)


if __name__ == "__main__":
    main()
