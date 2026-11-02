"""matplotlib 공통 스타일 (dataviz 스킬의 기본 팔레트, 라이트 모드)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import paths

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
# 범주 슬롯 순서 고정 (순환 금지). 산점도처럼 모든 쌍이 맞닿는 그림은 앞 3개만 씁니다.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_NEG, DIV_POS = "#2a78d6", "#e34948"  # 파랑 ↔ 빨강, 가운데 회색

# 막대·선(인접 쌍)용 앞 4슬롯. 4색이 모두 맞닿는 산점도는 패싯이나 마커 모양을 함께 씁니다
# (validate_palette.js --pairs all: 4번째 슬롯은 orange 와 구분 하한 미달).
PIECE_COLORS = {
    "Schubert_D960_mv2": CAT[0],
    "Schubert_D960_mv3": CAT[1],
    "Schubert_D935_no.3": CAT[2],
    "Beethoven_WoO80": CAT[3],
}
PIECE_MARKERS = {"Schubert_D960_mv2": "o", "Schubert_D960_mv3": "s",
                 "Schubert_D935_no.3": "^", "Beethoven_WoO80": "D"}
PIECE_SHORT = {
    "Schubert_D960_mv2": "D960 2악장",
    "Schubert_D960_mv3": "D960 3악장",
    "Schubert_D935_no.3": "D935 No.3",
    "Beethoven_WoO80": "WoO 80",
}


def setup():
    mpl.rcParams.update({
        "font.family": ["Apple SD Gothic Neo", "AppleGothic", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def diverging_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("div", [DIV_NEG, NEUTRAL, DIV_POS])


def seq_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ)


def save(fig, name: str):
    out = paths.FIGURES / f"{name}.png"
    fig.savefig(out)
    return out
