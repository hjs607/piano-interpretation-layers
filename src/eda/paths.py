"""프로젝트 경로 상수."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FIGURES = ROOT / "reports" / "figures"

PP = RAW / "PercePiano"
PP_LABELS_CSV = PP / "labels" / "total_2rounds.csv"
PP_JSON_LABELS_DIR = PP / "labels" / "label_2round_mean_reg_19_with0_rm_highstd0.json"
PP_JSON_ROOT = PP / "label_2round_mean_reg_19_with0_rm_highstd0.json"
PP_STD_LABELS_DIR = PP / "labels" / "label_2round_std_reg_19_with0_rm_highstd0.json"
PP_STD_ROOT = PP / "label_2round_std_reg_19_with0_rm_highstd0.json"
PP_MIDI = PP / "virtuoso" / "data" / "all_2rounds"
PP_SCORE_XML = PP / "virtuoso" / "data" / "score_xml"
PP_SCORE_MIDI = PP / "virtuoso" / "data" / "score_midi"

# ASAP·Aria: IdioMap repo 는 읽기만 한다
IDIOMAP = Path.home() / "Desktop" / "IdioMap" / "Aria"
IDIOMAP_PY = IDIOMAP / "aria" / "bin" / "python"
ASAP = IDIOMAP / "asap-dataset"
ASAP_ANNOTATIONS = ASAP / "asap_annotations.json"
FEAS = INTERIM / "feas"
FEAS_PIECES = {
    "schubert_d899_3": "Schubert/Impromptu_op.90_D.899/3",
}

for _d in (INTERIM, FIGURES, FEAS):
    _d.mkdir(parents=True, exist_ok=True)
