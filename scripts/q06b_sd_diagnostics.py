"""Q6 진단(판정에 쓰지 않음): ① 구간 간 '갈림' 이질성이 존재하는가(순열) ② k≈10 설계의 검출력(모의)."""
import numpy as np, pandas as pd
from q_common import *
rng = np.random.default_rng(0)
d = rater_adjusted(d960(ratings("mean"), human_only=True), NINE)
rows = {}
for c in NINE:
    e, a, b = additive_residuals(d, c); x = d.loc[e.index, ["segment_id", "unit"]].assign(e=e.to_numpy())
    def stat(df):  # 구간별 SD 의 단위 내 분산
        s = df.groupby("segment_id").e.std(); u = x.drop_duplicates("segment_id").set_index("segment_id").unit.reindex(s.index)
        return float((s - s.groupby(u).transform("mean")).var())
    obs = stat(x); null = []
    for _ in range(500):
        xp = x.copy(); xp["e"] = xp.groupby("unit").e.transform(lambda v: rng.permutation(v.to_numpy())); null.append(stat(xp))
    rows[SHORT[c]] = dict(obs_var_sd=round(obs, 4), null_mean=round(float(np.mean(null)), 4), ratio=round(obs / np.mean(null), 2), p=round(float((np.sum(np.array(null) >= obs) + 1) / 501), 3))
T = pd.DataFrame(rows).T; print("① 구간별 SD 의 단위 내 분산: 관측 vs 단위 안 잔차 섞기 순열\n", T.to_string())
# ② 검출력: 구간별 참 SD 가 평균 1.0, 구간 간 CV = c 로 달라질 때, k=10 반분 신뢰도
k = 10; nseg = 778
def sim(cv, n_iter=50):
    rs = []
    for _ in range(n_iter):
        sd_true = np.clip(rng.normal(1.0, cv, nseg), 0.2, None)
        A, B = np.empty(nseg), np.empty(nseg)
        for i in range(nseg):
            v = rng.normal(0, sd_true[i], k); A[i], B[i] = v[:5].std(ddof=1), v[5:].std(ddof=1)
        rs.append(np.corrcoef(A, B)[0, 1])
    r = float(np.mean(rs)); return 2 * r / (1 + r)
power = {cv: round(sim(cv), 3) for cv in (0.0, 0.1, 0.2, 0.3, 0.5)}
print("\n② 모의: 구간별 참 SD 의 변동계수 → k=10 반분 신뢰도", power)
save("q06b_sd_diagnostics", dict(heterogeneity=rows, power=power))
