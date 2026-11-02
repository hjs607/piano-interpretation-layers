# 가능성 실험 인터페이스 (feas-data ↔ feas-embed ↔ 분석)

계획 원본은 `~/.claude/plans/eda-wobbly-moon.md`의 ② 절입니다. 이 문서는 파일 형식과 규칙의 **계약**이므로, 바꿀 때는 메인과 합의합니다.

## 공통
- **곡 키** (`src/eda/paths.py`의 `FEAS_PIECES`)

  | 곡 키 | L | hop | 구간 0 | 구간 1 |
  |---|---|---|---|---|
  | `schubert_d899_3` | 4 | 2 | k ∈ [24,32) | k ∈ [56,64) |

- **연주 목록**: `ASAP/<piece_dir>/*.mid` 중 `midi_score.mid`를 뺀 것입니다. `asap_annotations.json`의 `score_and_performance_aligned`가 True인 것만 쓰고, 이름순으로 정렬합니다. 연주 이름은 파일 stem입니다 (예: `Hou06M`).
- 시간은 모두 **정수 ms**입니다.
- `.npz`에는 int64 배열만 넣습니다 (object 배열 금지). 두 venv의 numpy 버전이 다르기 때문입니다.
- 내려받은 데이터(ASAP)를 읽는 Python은 `-I`로 실행합니다. IdioMap repo는 **읽기만** 합니다.

## 1. 음 표 내보내기 (feas-embed, IdioMap venv)
- 출력 위치
  - 연주: `data/interim/feas/notes/{piece}/{perf}.npz`
  - 악보: `data/interim/feas/notes/{piece}/midi_score.npz`
- **배열** (길이 N, ariautils `MidiDict.note_msgs` 순서 = 시작 tick 오름차순, 순서 보존)

  | 배열 | 뜻 |
  |---|---|
  | `onset_ms` | 음 시작 |
  | `off_key_ms` | 건반을 뗀 시점 (원래 note-off) |
  | `off_ped_ms` | 페달을 반영한 끝 시점. ariautils `AbsTokenizer`의 규칙(CC64 ≥ 64, `_build_pedal_intervals`, `absolute.py`의 note-off 연장)을 **그대로** 재현합니다. 악보는 `off_key_ms`와 같습니다 |
  | `pitch`, `velocity` | 음높이, 세기 |

- ms 변환은 ariautils가 토큰화할 때 쓰는 tick→ms 변환과 같아야 합니다.
- `data/interim/feas/notes/{piece}/meta.json`에 연주 목록, 원 파일 경로, 음 수를 적습니다.

## 2. 조건 음 표와 manifest (feas-data, 프로젝트 venv)
- **downbeat**: `asap_annotations.json`에서 가져와 ×1000 후 반올림합니다.
  - 연주는 `performance_downbeats`, 악보는 `midi_score_downbeats`입니다.
  - 박은 `performance_beats` / `midi_score_beats`입니다.
- **창 i**: `k_lo = hop·i`, `k_hi = k_lo + L`이고, `k_hi ≤ n_db`인 동안 이어집니다.
  - `t0_ms = db[k_lo] − 30`
  - `t1_ms = db[k_hi] − 30`. 단 `k_hi == n_db`이면 마지막 onset + 1입니다.
  - 창에 드는 음: `t0_ms ≤ onset_ms < t1_ms`
- **구간 r** = [r_lo, r_hi)이고, 구간 시간은 [db[r_lo] − 30, db[r_hi] − 30)입니다. onset이 이 범위 안인 음만 바꿉니다.
- **겹침 판정**
  - `in`: k_lo ≥ r_lo 이고 k_hi ≤ r_hi
  - `out`: k_hi ≤ r_lo 이거나 k_lo ≥ r_hi
  - 나머지는 `partial`. 깨끗한 연주는 `na`
- **조건 파일**: `data/interim/feas/cond/{piece}/{perf}__{cond}__a{alpha}__r{region}.npz`
  - 깨끗한 연주는 `{perf}__clean.npz`
  - 배열은 음 표와 같고, 이 조건의 박 시각 `beats_ms`를 추가합니다.
  - 음은 onset 오름차순으로 정렬합니다 (안정 정렬).
- **조건 정의**

  | 조건 | 정의 |
  |---|---|
  | `vel` | `pedal`, `tempo`, `deadpan`, `swap`, `jit_time`, `jit_vel`과 함께 계획서 ② "조건" 표를 따릅니다 |
  | `swap` | q는 정렬된 연주 목록에서 p 다음 연주자입니다 (마지막이면 처음으로). `ref_exclude=q` |
  | `jit_*` | `seed`는 (곡, 연주, 구간)으로 정해 manifest에 적습니다 |

- **`data/interim/feas/manifest_{piece}.csv`**: 조건 파일 × 창마다 1행입니다.

  | 열 | 내용 |
  |---|---|
  | `row_id` | int, 곡 안에서 유일 |
  | `piece, perf` | 악보는 `perf = "midi_score"` |
  | `cond` | clean / vel / tempo / pedal / deadpan / swap / jit_time / jit_vel |
  | `alpha` | 기본 1.0 |
  | `region` | 0 또는 1. clean은 −1 |
  | `win_idx, k_lo, k_hi, t0_ms, t1_ms, n_notes` | 창 정보 |
  | `overlap` | in / partial / out / na |
  | `clean_row_id` | 같은 연주 clean 창의 row_id. clean이면 자기 자신 |
  | `needs_embed` | clean과 in/partial은 1, out은 0 |
  | `ref_exclude` | 기준에서 뺄 연주자. 없으면 빈 문자열 |
  | `seed` | 무작위 조건의 seed |
  | `notes_file` | 프로젝트 루트 기준 상대 경로 |
  | 손 특징 | `vel_mean, vel_std, vel_p95_p5, ped_ext, beat_ioi_logstd, dur_logmean, onset_density` |

- **손 특징 정의**
  - `ped_ext` = median log(max(off_ped − onset, 10) / max(off_key − onset, 10))
  - `dur_logmean` = mean log(max(off_ped − onset, 10))
  - `beat_ioi_logstd` = 창 안의 `beats_ms` 간격의 log 표준편차
  - `onset_density` = 30ms 묶음 수 / 창 길이(초)

## 3. 임베딩 (feas-embed, IdioMap venv)
- **입력**: manifest에서 `needs_embed == 1`인 행
- **창 만들기**
  1. `notes_file`에서 `t0_ms ≤ onset < t1_ms`인 음을 고릅니다. 개수가 `n_notes`와 **같아야 합니다** (assert).
  2. 1틱 = 1ms인 `MidiDict`를 만듭니다. 음 끝은 `off_ped_ms`이고, 페달 메시지는 없고, `metadata = {}`입니다.
  3. `AbsTokenizer.tokenize(add_dim_tok=False)`로 토큰화하고 2048에서 자릅니다. EOS를 보정하고 `get_embedding_from_seq`를 호출합니다.
- **출력**: `data/interim/feas/emb_{piece}.npz`. 배열은 `row_id` (int64)와 `emb` (float32, N×512)입니다.
  - 이어서 돌릴 수 있어야 합니다 (이미 계산한 row_id는 건너뜀).
  - 장치는 MPS입니다.
- **토큰 동일성 검사**: 조건 파일마다 `out` 창을 최대 3개 골라, 토큰 열이 `clean_row_id` 창과 같은지 assert합니다. 결과는 `data/interim/feas/embed_log.json`에 적습니다.
- **복제 검증**
  - 내보낸 clean 음 표로 IdioMap 캐시(`embeddings/aria_slice_embeddings.pkl`)와 같은 300음 창(`note_start:note_end`)을 만들어 임베딩합니다.
  - 연주 3개 이상에서 코사인 ≥ 0.999여야 합니다. 결과는 같은 로그에 적습니다.
