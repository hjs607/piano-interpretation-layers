"""PercePiano 1,202 구간 정렬·표현 파라미터·MIDI 통계 캐시 생성 (M2).

각 연주 MIDI 를 자기 악보 구간 `score_xml/{stem}_Score_{seg}.musicxml` 에 정렬합니다.
출력: data/interim/pp_alignment_summary.parquet, pp_note_params.parquet, pp_midi_stats.parquet, pp_features.parquet
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.eda import align, midi_stats, paths, perf_features, percepiano_io as pp


def one(segment_id: str):
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    info = pp.parse_name(segment_id)
    mp = pp.midi_path(segment_id)
    sp = pp.score_xml_path(segment_id)
    row = {"segment_id": segment_id, **{k: info[k] for k in ("piece", "unit", "performer", "is_score", "n_bars")}}
    try:
        row.update(midi_stats.midi_stats(mp))
    except Exception as e:  # noqa: BLE001
        row["midi_error"] = repr(e)
    try:
        score = align.load_score(sp)
        sna = align.score_note_array(score)
        perf = align.load_performance(mp)
        al = align.align(sna, perf.note_array())
        row.update({"score_notes": len(sna), "perf_notes_pt": len(perf.note_array()), **align.alignment_counts(al)})
        p = align.note_params(score, perf, al)
        p.insert(0, "segment_id", segment_id)
        row.update(perf_features.summarize(p))
        return row, p
    except Exception as e:  # noqa: BLE001
        row["align_error"] = repr(e)
        return row, None


def main():
    ids = sorted({m.name[:-4].strip() for m in paths.PP_MIDI.glob("*.mid")})
    res = Parallel(n_jobs=8, verbose=5)(delayed(one)(s) for s in ids)
    rows = pd.DataFrame([r for r, _ in res])
    params = pd.concat([p for _, p in res if p is not None], ignore_index=True)
    rows.to_parquet(paths.INTERIM / "pp_features.parquet")
    params.to_parquet(paths.INTERIM / "pp_note_params.parquet")
    print("segments", len(rows), "align errors", rows.get("align_error", pd.Series(dtype=str)).notna().sum(),
          "midi errors", rows.get("midi_error", pd.Series(dtype=str)).notna().sum())


if __name__ == "__main__":
    main()
