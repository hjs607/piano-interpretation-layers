# 같은 악보, 다른 연주: 청취자가 함께 듣는 층과 각자 다르게 듣는 층

연구 질문: 같은 악보에서 나온 여러 연주의 차이 가운데, 청취자들이 함께 듣는 부분과 각자 다르게 듣는 부분은 연주의 어느 층(문자 / 관습 / 개인)에 있는가.
확정 주제 문서: `seminar/10_25.pdf`. 최신 미팅 자료: `seminar/11_02.pdf`.

- **RQ1** 연주 층을 세 층으로 나눌 수 있는가 (ASAP)
- **RQ2** 청취자 층이 공유 축 + 개인 감도로 나뉘는가 (PercePiano D960)
- **RQ3** 어느 층에서 합의하고 어느 층에서 갈라지는가

## 폴더

| 경로 | 내용 |
|---|---|
| `docs/q_audit_prereg_2026-11-01.md` | 자료 점검(Q1–Q13) 사전등록. 판정 기준과 사후 변경 기록 |
| `docs/x1x2_prereg_2026-10-25.md` | RQ2(X1)·RQ3(X2) 본 실험 사전등록. 점검 결과에 따른 수정은 사후 변경 기록에 |
| `docs/feas_interface.md` | ASAP 창·조건 파일 형식 계약 |
| `scripts/q0*_*.py`, `q1*_*.py` | 자료 점검 코드. `q_common.py` 가 공통 로더 |
| `scripts/x1_coldstart.py`, `x2_rq3_layers.py` | 본 실험 (미실행) |
| `scripts/rater_layers.py` | RQ2 두 층 추정 (10/25 문서의 R5) |
| `scripts/align_percepiano.py`, `variance_components.py` | PercePiano 정렬·손 특징, 분산 성분 캐시 생성 |
| `scripts/feas_*.py`, `pp_aria_embed.py`, `q01b_e1.py` | Aria 임베딩 관련 (IdioMap venv 필요) |
| `src/eda/` | 라벨 로딩·정제, 정렬, 특징, 통계 유틸 |
| `src/feas/` | ASAP 창 격자·손상 조건 |
| `notebooks_src/01_percepiano.py` → `notebooks/01_percepiano.ipynb` | PercePiano EDA (percent 형식 원본 → `scripts/py2nb.py`) |
| `notebooks_src/05_pp_aria_vs_labels.py` | 같은 단위 연주 쌍: 라벨 차 vs Aria 거리 vs 손 특징 (E3) |
| `reports/q/` | 점검 결과. `audit_summary.md` 가 요약, `usage_table_*.csv` 가 사용 허가표 |
| `reports/exp_2026-10-24.md` | Aria 민감도 실험(E1–E4)과 재현 기록 |
| `reports/rater_layers_2026-10-25.md` | RQ2 첫 추정 (점검 후 수치 일부 수정 대상) |
| `reports/figures/` | EDA 그림 |
| `seminar/` | 미팅 자료 |

## 자료 준비 (저장소에 포함되지 않음)

```bash
uv venv .venv && uv pip install -r requirements.txt
git clone --depth 1 https://github.com/JonghoKimSNU/PercePiano data/raw/PercePiano   # 코드는 실행하지 않고 읽기만
.venv/bin/python -B scripts/align_percepiano.py        # data/interim/pp_features.parquet 등
.venv/bin/python -B notebooks_src/01_percepiano.py     # pp_ratings_clean.parquet 등 (노트북 01 과 같음)
.venv/bin/python -B scripts/variance_components.py
```

- ASAP: `~/Desktop/IdioMap/Aria/asap-dataset` 경로를 `src/eda/paths.py` 에서 가리킴 (Foscarin et al. 2020, 읽기만).
- Aria 임베딩: `loubb/aria-medium-embedding` 가중치와 `aria`·`ariautils` 가 설치된 별도 venv (`src/eda/paths.py` 의 `IDIOMAP_PY`). `scripts/pp_aria_embed.py` 가 `data/interim/pp_aria_emb.npz` 를 만든다.
- MAESTRO v3 MIDI (출처 대조용, Q9): `scripts/q09b_match_sources.py <maestro 폴더>`.

## 점검 실행

```bash
for q in 02 03 04 05 06 06b 07 08 09 10 11 12 13 13b; do .venv/bin/python -B scripts/q${q}_*.py; done
.venv/bin/python -B scripts/q99_usage_table.py
```

## 규칙
- 판정 기준은 결과를 보기 전에 사전등록 문서에 적고, 변경은 "사후 변경 기록"에만 적는다.
- 내려받은 자료의 코드는 실행하지 않는다. 자료를 읽는 스크립트는 `-B` 또는 `-I` 로 실행한다.
- 평가자는 합의와 다르다는 이유로 제외하지 않는다 (그 차이가 연구 대상).
