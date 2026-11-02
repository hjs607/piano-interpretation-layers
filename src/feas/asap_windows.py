"""ASAP 주석에서 마디(downbeat)·박(beat)을 ms 로 읽고, 창 격자·구간·겹침 라벨·음 선택을 제공합니다.

규칙의 출처는 docs/feas_interface.md 입니다. 시간은 모두 정수 ms 입니다.
"""
import json
from functools import lru_cache

import numpy as np

from src.eda.paths import ASAP, ASAP_ANNOTATIONS, FEAS_PIECES

TOL_MS = 30  # 모든 조건에 똑같이 적용하는 창 허용치
CLUSTER_MS = 30  # 화음 묶음: 첫 onset 에서 30ms 이내

# 곡 설정: 창 길이 L(마디), hop(마디), 구간(downbeat 인덱스 [lo, hi))
PIECE_CFG = {
    "schubert_d899_3": dict(L=4, hop=2, regions=[(24, 32), (56, 64)]),
}

SCORE = "midi_score"


@lru_cache(maxsize=1)
def annotations() -> dict:
    with open(ASAP_ANNOTATIONS) as f:
        return json.load(f)


def piece_dir(piece: str) -> str:
    return FEAS_PIECES[piece]


def performers(piece: str) -> list[str]:
    """`midi_score` 를 뺀 .mid 중 score_and_performance_aligned == True 인 것, 이름순."""
    ann = annotations()
    d = ASAP / piece_dir(piece)
    out = []
    for p in sorted(d.glob("*.mid")):
        if p.stem == SCORE:
            continue
        a = ann.get(f"{piece_dir(piece)}/{p.name}")
        if a is not None and a.get("score_and_performance_aligned") is True:
            out.append(p.stem)
    return out


def _ms(x) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=float) * 1000.0).astype(np.int64)


def timing(piece: str, perf: str) -> tuple[np.ndarray, np.ndarray]:
    """(downbeats_ms, beats_ms). perf == 'midi_score' 이면 악보 쪽 주석을 씁니다 (곡의 첫 연주 항목에서)."""
    ann = annotations()
    if perf == SCORE:
        key = f"{piece_dir(piece)}/{performers(piece)[0]}.mid"
        a = ann[key]
        return _ms(a["midi_score_downbeats"]), _ms(a["midi_score_beats"])
    a = ann[f"{piece_dir(piece)}/{perf}.mid"]
    return _ms(a["performance_downbeats"]), _ms(a["performance_beats"])


def window_grid(db_ms: np.ndarray, L: int, hop: int, last_onset_ms: int) -> list[dict]:
    """창 i: k_lo = hop*i, k_hi = k_lo + L (k_hi <= n_db).

    t0 = db[k_lo] - 30, t1 = db[k_hi] - 30. 단 k_hi == n_db 이면 t1 = 마지막 onset + 1.
    """
    n_db = len(db_ms)
    wins = []
    i = 0
    while hop * i + L <= n_db:
        k_lo, k_hi = hop * i, hop * i + L
        t0 = int(db_ms[k_lo]) - TOL_MS
        t1 = int(last_onset_ms) + 1 if k_hi == n_db else int(db_ms[k_hi]) - TOL_MS
        wins.append(dict(win_idx=i, k_lo=k_lo, k_hi=k_hi, t0_ms=t0, t1_ms=t1))
        i += 1
    return wins


def region_span(db_ms: np.ndarray, r_lo: int, r_hi: int) -> tuple[int, int]:
    """구간 시간 [db[r_lo] - 30, db[r_hi] - 30)."""
    return int(db_ms[r_lo]) - TOL_MS, int(db_ms[r_hi]) - TOL_MS


def overlap_label(k_lo: int, k_hi: int, r_lo: int, r_hi: int) -> str:
    if k_lo >= r_lo and k_hi <= r_hi:
        return "in"
    if k_hi <= r_lo or k_lo >= r_hi:
        return "out"
    return "partial"


def cell_edges(db_ms: np.ndarray, hop: int) -> np.ndarray:
    """창 경계 격자: 창 경계는 항상 hop 의 배수 downbeat - 30ms. 칸 j = [edges[j], edges[j+1]).

    (L 이 hop 의 배수이므로 어떤 창의 시작·끝도 이 경계 위에 있습니다.)
    """
    e = [-np.inf] + [float(db_ms[k]) - TOL_MS for k in range(hop, len(db_ms), hop)] + [np.inf]
    return np.asarray(e)


def select_notes(onset_ms: np.ndarray, t0_ms: int, t1_ms: int) -> np.ndarray:
    """t0 <= onset < t1 인 음의 boolean 마스크."""
    return (onset_ms >= t0_ms) & (onset_ms < t1_ms)


def onset_clusters(onsets_ms: np.ndarray) -> np.ndarray:
    """정렬된 onset 에 대한 화음 묶음 번호. 묶음의 첫 onset 에서 30ms 넘게 벗어나면 새 묶음."""
    o = np.asarray(onsets_ms)
    ids = np.zeros(len(o), dtype=np.int64)
    c, start = 0, None
    for i, t in enumerate(o):
        if start is None:
            start = t
        elif t - start > CLUSTER_MS:
            c += 1
            start = t
        ids[i] = c
    return ids
