"""Q11 · 문자 기준점(Score 렌더)과 관습 방향 c_u 의 기하."""
import numpy as np, pandas as pd
from q_common import *
import mido
rng = np.random.default_rng(0)
f = pd.read_parquet(paths.INTERIM / "pp_features.parquet"); f = f[f.align_error.isna()] if "align_error" in f else f
f = f[(f.match / f.score_notes) >= 0.8].copy(); f["log_tempo"] = np.log2(f.tempo_median_bpm)
FX = ["log_tempo", "tempo_logstd", "timing_std_ms", "articulation_median", "vel_std", "vel_range", "pedal_down_ratio"]
d9 = f[f.piece.str.startswith("Schubert_D960")].dropna(subset=FX)
# ① Score MIDI 직접 확인
sc = d9[d9.is_score]; print("① D960 Score 렌더 %d개: vel_std 중앙 %.1f (사람 %.1f) · vel_range %.1f (사람 %.1f) · tempo_logstd %.3f (사람 %.3f) · pedal_down_ratio %.2f (사람 %.2f) · cc64 이벤트 합 %d · n_tempo_events 중앙 %.0f · tempo_first 종류 %s" % (
    len(sc), sc.vel_std.median(), d9[~d9.is_score].vel_std.median(), sc.vel_range.median(), d9[~d9.is_score].vel_range.median(), sc.tempo_logstd.median(), d9[~d9.is_score].tempo_logstd.median(),
    sc.pedal_down_ratio.median(), d9[~d9.is_score].pedal_down_ratio.median(), int(sc.cc64.sum()), sc.n_tempo_events.median(), sorted(set(np.round(sc.tempo_first_bpm, 0)))[:8]))
print("   articulation_median: Score %.2f vs 사람 %.2f · timing_std_ms: Score %.1f vs 사람 %.1f" % (sc.articulation_median.median(), d9[~d9.is_score].articulation_median.median(), sc.timing_std_ms.median(), d9[~d9.is_score].timing_std_ms.median()))
# ② c_u 구성과 단위 간 코사인
hum = d9[~d9.is_score]; Z = (d9[FX] - hum[FX].mean()) / (hum[FX].std() + 1e-9); Z.index = d9.segment_id
def cvecs(use_score=True, feats=FX):
    out = {}
    for u, g in d9.groupby("unit"):
        h, s = g[~g.is_score], g[g.is_score]
        if len(h) < 4 or (use_score and len(s) == 0): continue
        ref = Z.loc[s.segment_id, feats].mean(0).to_numpy() if use_score else np.zeros(len(feats))
        c = Z.loc[h.segment_id, feats].mean(0).to_numpy() - ref; out[u] = c / (np.linalg.norm(c) + 1e-12)
    return pd.DataFrame(out, index=feats).T
C = cvecs(); mean_dir = C.mean(0); mean_dir /= np.linalg.norm(mean_dir)
cos = C.to_numpy() @ C.to_numpy().T; iu = np.triu_indices(len(C), 1); mean_cos = float(cos[iu].mean())
print("\n② c_u (단위 %d): 평균 방향 성분" % len(C), dict(zip(FX, mean_dir.round(2))), "\n   단위 간 평균 코사인 = %.2f (판정 기준 0.8) · ‖사람평균 − Score‖ 중앙값 = %.2f (z 단위)" % (
    mean_cos, np.median([np.linalg.norm(Z.loc[g[~g.is_score].segment_id].mean(0) - Z.loc[g[g.is_score].segment_id].mean(0)) for u, g in d9.groupby("unit") if (g.is_score.any() and (~g.is_score).sum() >= 4)])))
# ③ 구간 고유 성분의 재현성: 연주자 6+6 으로 나눠 c_u − 전체 평균 방향 의 코사인
perfs = sorted(hum.performer.unique()); rep = []
for _ in range(50):
    pm = rng.permutation(perfs); A, B = set(pm[:6]), set(pm[6:])
    def cu(sub):
        out = {}
        for u, g in d9.groupby("unit"):
            h, s = g[~g.is_score & g.performer.isin(sub)], g[g.is_score]
            if len(h) < 3 or len(s) == 0: continue
            c = Z.loc[h.segment_id].mean(0).to_numpy() - Z.loc[s.segment_id].mean(0).to_numpy(); out[u] = c / (np.linalg.norm(c) + 1e-12)
        D = pd.DataFrame(out, index=FX).T; return D - D.mean(0)
    a, b = cu(A), cu(B); j = a.index.intersection(b.index)
    rep.append(np.mean([a.loc[u].to_numpy() @ b.loc[u].to_numpy() / (np.linalg.norm(a.loc[u]) * np.linalg.norm(b.loc[u]) + 1e-12) for u in j]))
print("③ 구간 고유 성분(전체 평균 방향 제거) 반분 코사인 평균 = %.2f (기준 0.5)" % np.mean(rep))
# ④ 대안 정의 민감도
for name, kw in (("Score 없이 (사람 평균 자체 = 원점에서의 방향)", dict(use_score=False)), ("log_tempo 제외", dict(feats=[x for x in FX if x != "log_tempo"]))):
    Cx = cvecs(**kw); cx = Cx.to_numpy() @ Cx.to_numpy().T; print("④ %s: 단위 간 평균 코사인 = %.2f" % (name, cx[np.triu_indices(len(Cx), 1)].mean()))
verdict = dict(mean_cos_between_units=round(mean_cos, 3), alpha_is_expression_amount=bool(mean_cos >= 0.8), unique_component_reproducible=bool(np.mean(rep) >= 0.5), unique_component_cos=round(float(np.mean(rep)), 3),
               mean_direction={k: round(float(v), 3) for k, v in mean_dir.items()})
print("\n판정:", verdict); save("q11_letter_convention", verdict)
