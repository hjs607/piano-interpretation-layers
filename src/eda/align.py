"""악보–연주 정렬과 표현 파라미터.

RenCon `analysis/align_midis.py`·`compute_performance_params.py` 의 호출 방식을 따릅니다
(실행하지 않고 같은 API 호출을 다시 씁니다):
partitura.load_musicxml + assign_note_ids → parangonar.AutomaticNoteMatcher()(score_na, perf_na)
→ partitura encode_performance(score, performance, alignment, return_u_onset_idx=True)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt
from parangonar import AutomaticNoteMatcher
from partitura.musicanalysis.performance_codec import encode_performance

warnings.filterwarnings("ignore")


def load_score(path: Path):
    score = pt.load_musicxml(str(path)) if str(path).endswith(("xml", "musicxml", "mxl")) else pt.load_score_midi(str(path))
    pt.score.assign_note_ids(score.parts)
    return score


def score_note_array(score, measure_range: tuple[int, int] | None = None):
    """measure_range=(첫 마디, 끝 마디), 1부터 세는 마디 번호(악보의 measure.number) 기준으로 자릅니다."""
    na = score.note_array(include_metrical_position=True)
    if measure_range is None:
        return na
    part = score.parts[0] if hasattr(score, "parts") else score
    lo, hi = measure_range
    starts, ends = [], []
    for m in part.measures:
        if m.number is not None and lo <= int(m.number) <= hi:
            starts.append(m.start.t)
            ends.append(m.end.t)
    t0, t1 = min(starts), max(ends)
    return na[(na["onset_div"] >= t0) & (na["onset_div"] < t1)]


def load_performance(path: Path):
    return pt.load_performance_midi(str(path))


def align(score_na, perf_na) -> list[dict]:
    return AutomaticNoteMatcher()(score_na, perf_na)


def alignment_counts(alignment: list[dict]) -> dict:
    lab = pd.Series([a["label"] for a in alignment])
    return {"match": int((lab == "match").sum()), "deletion": int((lab == "deletion").sum()),
            "insertion": int((lab == "insertion").sum())}


def note_params(score, performance, alignment) -> pd.DataFrame:
    """음 단위 표현 파라미터 (score onset 순서). RenCon note_params.csv 와 같은 열 구성."""
    params, snote_ids, u_onset_idx = encode_performance(score, performance, alignment, return_u_onset_idx=True)
    na = score.note_array()
    info = pd.DataFrame({"score_id": na["id"], "onset_beat": na["onset_beat"], "duration_beat": na["duration_beat"],
                         "pitch": na["pitch"], "voice": na["voice"]}).set_index("score_id")
    p = pd.DataFrame({k: params[k] for k in params.dtype.names})
    p["score_id"] = snote_ids
    p = p.join(info, on="score_id")
    p["tempo_bpm"] = 60.0 / p["beat_period"]
    p = p.rename(columns={"velocity": "velocity_norm"})
    match = {a["score_id"]: a["performance_id"] for a in alignment if a["label"] == "match"}
    p["performance_id"] = p["score_id"].map(match)
    onset_group = np.empty(len(p), dtype=int)
    for g, idx in enumerate(u_onset_idx):
        onset_group[np.asarray(idx, int)] = g
    p["onset_group"] = onset_group
    return p
