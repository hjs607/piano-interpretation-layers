"""PercePiano 라벨 로드·정제·원 절차 재현.

원 절차: labels/map_midi_to_label.py (실행하지 않고 로직만 다시 구현).
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

from . import paths

# virtuoso/virtuoso/train_m2pf.py 의 LABEL_LIST19 순서 (CSV 문항 열 순서와 위치로 대응)
LABELS19 = [
    "Timing_Stable_Unstable",
    "Articulation_Long_Short",
    "Articulation_Soft_cushioned_Hard_solid",
    "Pedal_Sparse/dry_Saturated/wet",
    "Pedal_Clean_Blurred",
    "Timbre_Even_Colorful",
    "Timbre_Shallow_Rich",
    "Timbre_Bright_Dark",
    "Timbre_Soft_Loud",
    "Dynamic_Sophisticated/mellow_Raw/crude",
    "Dynamic_Little_dynamic_range_Large_dynamic_range",
    "Music_Making_Fast_paced_Slow_paced",
    "Music_Making_Flat_Spacious",
    "Music_Making_Disproportioned_Balanced",
    "Music_Making_Pure_Dramatic/expressive",
    "Emotion_&_Mood_Optimistic/pleasant_Dark",
    "Emotion_&_Mood_Low_Energy_High_Energy",
    "Emotion_&_Mood_Honest_Imaginative",
    "Interpretation_Unsatisfactory/doubtful_Convincing",
]
# 그림용 짧은 이름 (1점 쪽 → 7점 쪽, 코드 표기 기준)
SHORT19 = [
    "stable→unstable", "long→short*", "cushioned→solid", "dry→wet", "clean→blurred",
    "even→colorful", "shallow→rich", "bright→dark", "soft→loud", "mellow→raw",
    "little→large range", "fast→slow", "flat→spacious", "disprop.→balanced*", "pure→dramatic",
    "pleasant→dark", "low→high energy", "honest→imaginative", "unconvincing→convincing",
]
# 범주: 코드 이름 접두어 기준. 논문 Table 2는 soft/loud(8)를 Dynamic으로 분류해 다릅니다.
CATEGORY_CODE = (["Timing"] + ["Articulation"] * 2 + ["Pedal"] * 2 + ["Timbre"] * 4
                 + ["Dynamic"] * 2 + ["Music making"] * 4 + ["Emotion"] * 3 + ["Interpretation"])
CATEGORY_PAPER = (["Timing"] + ["Articulation"] * 2 + ["Pedal"] * 2 + ["Timbre"] * 3
                  + ["Dynamic"] * 3 + ["Music making"] * 4 + ["Emotion"] * 3 + ["Interpretation"])
# 논문 Table 2 의 차원별 "I don't know" 개수와 곡별 평균
PAPER_IDK = [5, 64, 67, 76, 115, 23, 20, 38, 89, 46, 32, 30, 51, 28, 54, 46, 36, 69, 32]
PAPER_ANNOTATIONS_PER_PIECE = {"Schubert_D960_mv2": 4076, "Schubert_D960_mv3": 6708,
                               "Schubert_D935_no.3": 624, "Beethoven_WoO80": 1244}

IDK_CODES = (0.0, 8.0, 9.0)
NAME_RE = re.compile(
    r"^(?P<piece>Schubert_D960_mv2|Schubert_D960_mv3|Schubert_D935_no\.3|Beethoven_WoO80)"
    r"(?:_(?P<section>thema|var\d+))?_(?P<bars>\d+)bars_(?P<performer>\d+|Score2?)_(?P<segment>\d+)$"
)
PIECES = ["Schubert_D960_mv2", "Schubert_D960_mv3", "Schubert_D935_no.3", "Beethoven_WoO80"]


def rater_block(user: int) -> str:
    if user < 70:
        return "B1(0-28)"
    if user < 200:
        return "B2(70-121)"
    return "B3(201-211)"


def parse_name(stem: str) -> dict:
    """`{곡}[_{thema|varN}]_{N}bars_{연주자}_{구간}` 파싱. README의 순서 설명은 반대입니다."""
    m = NAME_RE.match(stem.strip())
    if m is None:
        raise ValueError(f"unparsable name: {stem!r}")
    g = m.groupdict()
    unit = "_".join(x for x in [g["piece"], g["section"], f"{g['bars']}bars", g["segment"]] if x)
    perf = g["performer"]
    return {
        "segment_id": stem.strip(),
        "piece": g["piece"],
        "section": g["section"],
        "n_bars": int(g["bars"]),
        "performer": perf,
        "is_score": perf.startswith("Score"),
        "segment": int(g["segment"]),
        "unit": unit,
    }


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(paths.PP_LABELS_CSV)
    df.insert(0, "row", np.arange(len(df)))
    df["segment_id"] = df["filename"].str.replace(r"\.wav$", "", regex=True)
    return df


def question_cols(df: pd.DataFrame) -> list[str]:
    q = [c for c in df.columns if c.startswith("Question_")]
    assert len(q) == 20
    return q[:19]  # Question_9_2_1 은 원 스크립트처럼 제외


def clean(df_raw: pd.DataFrame, repeats: str = "mean") -> tuple[pd.DataFrame, pd.DataFrame]:
    """정제 버전.

    1) 완전 중복행 제거 2) 19문항 모두 빈 행 제거 3) 0·8·9 → NaN("모르겠음")
    4) 같은 평가자·같은 구간 반복 평가 처리: "mean" | "first" | "keep"
    반환: (평가 단위 wide DataFrame, 단계별 행 수 표)
    """
    q = question_cols(df_raw)
    steps = [("raw", len(df_raw))]
    d = df_raw.drop_duplicates(subset=[c for c in df_raw.columns if c != "row"])
    steps.append(("drop exact duplicate rows", len(d)))
    d = d[~d[q].isna().all(axis=1)]
    steps.append(("drop rows with all 19 empty", len(d)))
    d = d.copy()
    d[q] = d[q].mask(d[q].isin(IDK_CODES))
    if repeats == "first":
        d = d.sort_values("row").drop_duplicates(["user", "segment_id"], keep="first")
    elif repeats == "mean":
        meta = d.groupby(["user", "segment_id"], as_index=False).agg(row=("row", "min"), n_repeat=("row", "size"))
        vals = d.groupby(["user", "segment_id"], as_index=False)[q].mean()
        d = meta.merge(vals, on=["user", "segment_id"])
    steps.append((f"collapse repeats ({repeats})", len(d)))
    d = d.rename(columns=dict(zip(q, LABELS19)))
    meta = pd.DataFrame([parse_name(s) for s in d["segment_id"]], index=d.index)
    d = pd.concat([d[["row", "user", "segment_id"] + LABELS19], meta.drop(columns="segment_id")], axis=1)
    d["block"] = d["user"].map(rater_block)
    return d.reset_index(drop=True), pd.DataFrame(steps, columns=["step", "n_rows"])


def reproduce_original(df_raw: pd.DataFrame, remap: bool) -> tuple[dict, dict]:
    """map_midi_to_label.py 재현. remap=True 면 Score→Score2 재명명까지 그대로 따릅니다."""
    q = question_cols(df_raw)
    groups: dict[str, list] = defaultdict(list)
    for fname, row in zip(df_raw["filename"], df_raw[q].to_numpy(dtype=float)):
        name = ".".join(fname.split(".")[:-1]).replace("_score", "_Score")
        seg = int(name.split("_")[-1])
        if remap and (("Beethoven_WoO80" in name and "Score" in name and seg in range(5, 17))
                      or ("Schubert_" in name and "_no.3_4bars" in name and "_Score_" in name and seg == 1)):
            name = name.replace("_Score_", "_Score2_")
        row = np.nan_to_num(row, nan=0.0)
        if (row > 7.1).any():
            continue
        groups[name].append(row)
    mean, std, pianist = {}, {}, OrderedDict()
    for key, rows in groups.items():
        a = np.array(rows)
        cols = []
        for col in a.T:
            col = col[col != 0]
            if len(col) == 0:
                break
            cols.append(col.mean() / 7)
        if len(cols) == 19:
            mean[key] = cols
        std[key] = np.std(a / 7, axis=0).tolist()
    for key in mean:
        pianist.setdefault(key.split("_")[-2], len(pianist))
    for key in mean:
        mean[key] = mean[key] + [pianist[key.split("_")[-2]]]
    return mean, std


def load_published(which: str = "root") -> tuple[dict, dict]:
    if which == "root":
        return json.loads(paths.PP_JSON_ROOT.read_text()), json.loads(paths.PP_STD_ROOT.read_text())
    return json.loads(paths.PP_JSON_LABELS_DIR.read_text()), json.loads(paths.PP_STD_LABELS_DIR.read_text())


def segment_means(ratings: pd.DataFrame) -> pd.DataFrame:
    """평가 단위 → 구간 평균(원 척도 1–7) + 평가 수."""
    meta_cols = ["piece", "section", "n_bars", "performer", "is_score", "segment", "unit"]
    g = ratings.groupby("segment_id")
    m = g[LABELS19].mean()
    m["n_ratings"] = g.size()
    meta = g[meta_cols].first()
    return meta.join(m).reset_index()


def midi_path(segment_id: str):
    p = paths.PP_MIDI / f"{segment_id}.mid"
    if not p.exists():
        p = paths.PP_MIDI / f" {segment_id}.mid"  # 공백으로 시작하는 파일 1개
    return p


def score_xml_path(segment_id: str):
    stem, _performer, seg = segment_id.strip().rsplit("_", 2)  # D960 은 구간 번호가 0으로 채워져 있음 ("01")
    return paths.PP_SCORE_XML / f"{stem}_Score_{seg}.musicxml"
