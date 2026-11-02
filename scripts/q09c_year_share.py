"""Q9 판정: 녹음 연도(기기 세대)가 연주자 간 벨로시티 특징 차이를 얼마나 설명하는가 (D960 사람 연주, 단위 내 잔차)."""
import numpy as np, pandas as pd
from q_common import *
m = pd.read_csv(OUT / "q09b_source_match.csv"); m = m[m.piece.str.startswith("Schubert_D960")].dropna(subset=["best_year"])
year = m.drop_duplicates("performer").set_index("performer").best_year.astype(int); year.index = year.index.astype(str)
f = pd.read_parquet(paths.INTERIM / "pp_features.parquet"); f = f[~f.is_score & f.piece.str.startswith("Schubert_D960") & ((f.match / f.score_notes) >= 0.8)].copy()
f["log_tempo"] = np.log2(f.tempo_median_bpm); FX = ["vel_mean", "vel_std", "vel_range", "pedal_down_ratio", "articulation_median", "timing_std_ms", "log_tempo", "tempo_logstd"]
W = f[FX] - f.groupby("unit")[FX].transform("mean"); W["performer"] = f.performer.astype(str).to_numpy(); W["year"] = W.performer.map(year)
W = W.dropna(subset=["year"]); print("연주자 → 연도:", year.sort_index().to_dict(), "\n연도별 연주자 수:", year.value_counts().sort_index().to_dict())
rows = {}
for c in FX:
    pm = W.groupby("performer")[c].mean(); ym = W.groupby("performer").year.first()
    ss_perf = float(((pm - pm.mean()) ** 2).sum()); ss_year = float(((pm.groupby(ym).transform("mean") - pm.mean()) ** 2).sum())
    rows[c] = dict(performer_sd=round(float(pm.std()), 3), year_share_of_performer_var=round(ss_year / ss_perf, 2) if ss_perf > 0 else None)
T = pd.DataFrame(rows).T; print("\n연주자 평균(단위 내 잔차)의 분산 중 연도 몫:\n", T.to_string())
# 참고: 연도 수가 적어(5개, 연주자 10명) 우연 기대치도 큼 → 연도 라벨 순열
rng = np.random.default_rng(0); null = {c: [] for c in FX}
for _ in range(2000):
    yp = pd.Series(rng.permutation(year.reindex(W.performer.unique()).to_numpy()), index=W.performer.unique())
    for c in FX:
        pm = W.groupby("performer")[c].mean(); ss_perf = ((pm - pm.mean()) ** 2).sum(); null[c].append(float(((pm.groupby(yp.reindex(pm.index)).transform("mean") - pm.mean()) ** 2).sum() / ss_perf))
T["null_mean"] = [round(float(np.mean(null[c])), 2) for c in FX]; T["perm_p"] = [round(float((np.sum(np.array(null[c]) >= T.loc[c, "year_share_of_performer_var"]) + 1) / 2001), 3) for c in FX]
print("\n순열 기준:\n", T.to_string())
vel_share = T.loc[["vel_mean", "vel_std", "vel_range"], "year_share_of_performer_var"].max()
verdict = dict(matched_performers=int(len(year)), years=sorted(set(year)), max_velocity_year_share=float(vel_share), any_vel_perm_sig=bool((T.loc[["vel_mean", "vel_std", "vel_range"], "perm_p"] < .05).any()),
               action="연도 안 표준화 검토" if (vel_share >= 0.30 and (T.loc[["vel_mean", "vel_std", "vel_range"], "perm_p"] < .05).any()) else "조치 없음(연도 몫이 우연 기대와 구분되지 않음)")
print("\n판정:", verdict); save("q09c_year_share", dict(table=T.to_dict("index"), verdict=verdict))
