"""PercePiano 구간 MIDI → Aria 임베딩 (초안, 2026-10-24, 아직 실행하지 않음).

목적: 같은 악보 단위의 사람 연주 쌍에서 |라벨 차|와 임베딩 거리를 비교하기 위한 입력
      (notebooks_src/05_pp_aria_vs_labels.py).
IdioMap venv(ariautils, torch)로 실행:
    PY=~/Desktop/IdioMap/Aria/aria/bin/python
    $PY -I scripts/pp_aria_embed.py --prefix Schubert_D960 [--limit 20]
출력: data/interim/pp_aria_emb.npz  (segment_id: 문자열 배열, emb: float32 N×512). 이어서 실행 가능.

feas_embed.py 의 함수를 그대로 재사용한다 — 복제 검증(코사인 ≥ 0.999)을 통과한 경로:
음 표(페달 반영 off_ped) → MidiDict(1 tick = 1 ms) → AbsTokenizer → EOS 임베딩.
주의: 16마디 구간은 2048 토큰을 넘을 수 있고 그 경우 앞부분만 임베딩된다(tokens_of 가 자름). n_tokens 를 함께 저장한다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from feas_embed import embed_seq, get_device, load_model, midi_to_note_table, table_to_middict, tokens_of  # noqa: E402

MIDI_DIR = ROOT / "data" / "raw" / "PercePiano" / "virtuoso" / "data" / "all_2rounds"
OUT = ROOT / "data" / "interim" / "pp_aria_emb.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs="*", default=["Schubert_D960"], help="segment_id 접두어 (곡)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    files = sorted(p for p in MIDI_DIR.glob("*.mid") if any(p.name.strip().startswith(x) for x in a.prefix))
    ids, embs, ntok = [], [], []
    if OUT.exists():
        z = np.load(OUT)
        ids, embs, ntok = list(z["segment_id"]), list(z["emb"]), list(z["n_tokens"])
    done = set(ids)
    todo = [p for p in files if p.name[:-4].strip() not in done]
    if a.limit:
        todo = todo[: a.limit]
    dev = get_device()
    print(f"{len(files)} files, {len(done)} done, {len(todo)} to run on {dev}", flush=True)
    if not todo:
        return
    model = load_model(dev)

    def save():
        tmp = OUT.with_name(OUT.stem + ".tmp.npz")
        np.savez(tmp, segment_id=np.asarray(ids), emb=np.asarray(embs, dtype=np.float32).reshape(-1, 512),
                 n_tokens=np.asarray(ntok, dtype=np.int64))
        os.replace(tmp, OUT)

    t0 = time.time()
    for k, p in enumerate(todo, 1):
        seq = tokens_of(table_to_middict(midi_to_note_table(p)))
        embs.append(embed_seq(model, seq, dev))
        ids.append(p.name[:-4].strip())
        ntok.append(len(seq))
        if k % 50 == 0 or k == len(todo):
            save()
            el = time.time() - t0
            print(f"  {k}/{len(todo)}  {el / k:.2f} s/seg  eta {el / k * (len(todo) - k) / 60:.1f} min", flush=True)
    print(f"done {len(todo)} in {time.time() - t0:.0f}s; truncated (==2048 tokens): {sum(n >= 2048 for n in ntok)}")


if __name__ == "__main__":
    main()
