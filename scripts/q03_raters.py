"""Q3 · 평가자 응답 품질과 순서 효과 (D960, 사람 연주, 평가자 ≥40구간)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from q_common import *

keep = ratings("keep"); first = ratings("first"); mean = ratings("mean")
d9k = d960(keep, human_only=True); d9f = d960(first, human_only=True); d9m = d960(mean, human_only=True)
raw = pp.load_raw()
raw_d = raw[raw.filename.str.startswith("Schubert_D960")].copy(); raw_d["segment_id"] = raw_d.filename.str.replace(r"\.wav$", "", regex=True)

rows = []
for u, g in d9k.groupby("user"):
    X = g[L]
    flat = float((X.nunique(axis=1) == 1).mean())                     # 19문항 전부 같은 값
    sd_med = float(X.std().median())                                 # 문항별 SD 의 중앙값
    idk = float(X.isna().mean().mean())                              # 모르겠음 비율
    # 재검사: 같은 구간 반복
    rep = g.groupby("segment_id").filter(lambda x: len(x) > 1).sort_values("row")
    npair = rep.segment_id.nunique(); rt = np.nan
    if npair >= 20:
        a = rep.drop_duplicates("segment_id", keep="first").set_index("segment_id")[L]
        b = rep[rep.duplicated("segment_id", keep="first")].drop_duplicates("segment_id").set_index("segment_id")[L].loc[a.index]
        rt = float(np.nanmean([a[c].corr(b[c]) for c in L if a[c].notna().sum() >= 8]))
    msgs = raw_d[raw_d.user == u].message.dropna(); top = msgs.value_counts().iloc[0] if len(msgs) else 0
    rows.append(dict(user=int(u), block=pp.rater_block(u), n=len(g), flat_rate=round(flat, 3), sd_median=round(sd_med, 2), idk_rate=round(idk, 3),
                     n_repeat_pairs=int(npair), retest_r=None if np.isnan(rt) else round(rt, 2), msg_n=int(len(msgs)), msg_top_repeat=int(top)))
R = pd.DataFrame(rows)
R["flag_flat"] = R.flat_rate >= 0.10; R["flag_sd"] = R.sd_median < 0.5; R["flag_idk"] = R.idk_rate >= 0.30; R["flag_retest"] = R.retest_r.fillna(1) <= 0.2
R["caution"] = R[["flag_flat", "flag_sd", "flag_idk", "flag_retest"]].any(axis=1)
# 보고용: 합의와의 상관 (제외 기준에 쓰지 않음)
agree = {}
for c in NINE:
    m = d9m.pivot_table(index="segment_id", columns="user", values=c); s = m.sum(1); n = m.notna().sum(1)
    loo = pd.DataFrame((s.to_numpy()[:, None] - np.nan_to_num(m.to_numpy())) / np.maximum(n.to_numpy()[:, None] - 1, 1), index=m.index, columns=m.columns).where(m.notna())
    agree[c] = {u: m[u].corr(loo[u]) for u in m.columns if m[u].notna().sum() >= 15}
R["agree_r_mean9"] = R.user.map(lambda u: np.nanmean([agree[c].get(u, np.nan) for c in NINE])).round(2)
print("== 평가자별 지표 (주의 표시만) =="); print(R[R.caution].to_string(index=False) if R.caution.any() else "주의 평가자 없음")
print("\n분포: flat_rate 최대 %.3f · sd_median 최소 %.2f · idk 최대 %.3f · retest_r 범위 %s · 합의 상관 평균9 범위 %.2f–%.2f" % (
    R.flat_rate.max(), R.sd_median.min(), R.idk_rate.max(), [R.retest_r.min(), R.retest_r.max()], R.agree_r_mean9.min(), R.agree_r_mean9.max()))
print("합의 상관이 낮은 평가자 5명(보고만):\n", R.nsmallest(5, "agree_r_mean9")[["user", "block", "n", "agree_r_mean9", "retest_r", "caution"]].to_string(index=False))

# 주의 평가자 제외 전후 단위 내 신뢰도
def rel(df): return {SHORT[c]: round(within_unit_split_half(df, c), 3) for c in NINE}
def _shw(df, value, n_iter=100, seed=0):
    g0 = np.random.default_rng(seed); x = df[["segment_id", "unit", value]].dropna()
    groups = [(u, v[value].to_numpy()) for (s, u), v in x.groupby(["segment_id", "unit"]) if len(v) >= 4]
    units = np.array([u for u, _ in groups]); rs = []
    for _ in range(n_iter):
        a = np.empty(len(groups)); b = np.empty(len(groups))
        for i, (_, v) in enumerate(groups):
            pm = g0.permutation(len(v)); h = len(v) // 2; a[i], b[i] = v[pm[:h]].mean(), v[pm[h:]].mean()
        sa = pd.Series(a).groupby(units).transform("mean").to_numpy(); sb = pd.Series(b).groupby(units).transform("mean").to_numpy()
        rs.append(np.corrcoef(a - sa, b - sb)[0, 1])
    rh = float(np.mean(rs)); return 2 * rh / (1 + rh)
r_all = rel(d9m); r_ex = rel(d9m[~d9m.user.isin(R[R.caution].user)]) if R.caution.any() else r_all
delta = {k: round(r_ex[k] - r_all[k], 3) for k in r_all}
print("\n== 주의 평가자 제외 전후 단위 내 신뢰도 차이 ==", delta)

# 순서 효과: dataID 가 제시 순서인가
raw_d = raw_d.sort_values(["user", "dataID"]); pos_rows = []
dd = raw.drop_duplicates(subset=[c for c in raw.columns if c != "row"])
gap = dd.sort_values("dataID").groupby("user").dataID.apply(lambda s: np.diff(s.to_numpy()))
print("\n== dataID: 평가자 내 간격 중앙값(전체) =", np.median(np.concatenate(gap.values)), "· 간격 1 비율 =", round(np.mean(np.concatenate(gap.values) == 1), 2))
# 위치 → 평점 변화: 평가자 내 순번(0–1) 을 공변량으로, 가법 잔차에 대한 설명 분산
first_h = d9f.copy()
order = raw_d.drop_duplicates(["user", "segment_id"], keep="first").set_index(["user", "segment_id"]).dataID
first_h["pos"] = first_h.set_index(["user", "segment_id"]).index.map(order.to_dict())
first_h["pos"] = first_h.groupby("user").pos.rank(pct=True)
pos_eff = {}
for c in NINE:
    e, _, _ = additive_residuals(first_h, c); x = first_h.loc[e.index]
    # 평가자 내 기울기: 잔차 ~ pos, 평가자 간 평균 기울기와 설명 분산
    slopes = x.assign(e=e).groupby("user").apply(lambda g: np.polyfit(g.pos, g.e, 1)[0] if len(g) >= 15 and g.pos.std() > 0 else np.nan).dropna()
    pred = x.assign(e=e).groupby("user").apply(lambda g: pd.Series(np.polyval(np.polyfit(g.pos, g.e, 1), g.pos), index=g.index) if len(g) >= 15 and g.pos.std() > 0 else pd.Series(0.0, index=g.index))
    pred = pred.reset_index(level=0, drop=True).reindex(e.index).fillna(0)
    r2 = 1 - ((e - pred) ** 2).sum() / ((e - e.mean()) ** 2).sum()
    # 변동폭: 평가자 내 |잔차| ~ pos
    sl_abs = x.assign(ae=e.abs()).groupby("user").apply(lambda g: np.polyfit(g.pos, g.ae, 1)[0] if len(g) >= 15 and g.pos.std() > 0 else np.nan).dropna()
    pos_eff[SHORT[c]] = dict(mean_slope=round(float(slopes.mean()), 3), slope_sd=round(float(slopes.std()), 3), r2_position=round(float(r2), 4), mean_slope_abs_resid=round(float(sl_abs.mean()), 3))
print("\n== 순서 효과 (잔차 ~ 평가자 내 제시 순번 0–1) =="); print(pd.DataFrame(pos_eff).T.to_string())
verdict = dict(n_caution=int(R.caution.sum()), keep_all=bool(max(abs(v) for v in delta.values()) <= 0.05),
               position_covariate_dims=[k for k, v in pos_eff.items() if v["r2_position"] >= 0.05])
print("\n판정:", verdict)
R.to_csv(OUT / "q03_raters.csv", index=False)
save("q03_raters", dict(verdict=verdict, reliability_all=r_all, reliability_excl=r_ex, delta=delta, position=pos_eff, raters=R.to_dict("records")))
