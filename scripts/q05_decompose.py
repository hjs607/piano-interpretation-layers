"""Q5 · 평점 분산 분해: 합의 / 안정된 개인차 / 잡음 (D960 사람 연주, 1차 평점 + 반복 쌍)."""
import numpy as np, pandas as pd
from q_common import *
rng = np.random.default_rng(0)
first = d960(ratings("first"), human_only=True); keep = d960(ratings("keep"), human_only=True)
raw = pp.load_raw(); dd = raw.drop_duplicates(subset=[c for c in raw.columns if c != "row"])
dd = dd[~dd[pp.question_cols(raw)].isna().all(axis=1)]; dd["segment_id"] = dd.filename.str.replace(r"\.wav$", "", regex=True)
gap = dd.groupby(["user", "segment_id"]).dataID.agg(lambda s: s.max() - s.min())

def pairs(df, c):
    """같은 (평가자, 구간) 의 1차·2차 평점 쌍."""
    x = df.sort_values("row"); x = x.assign(k=x.groupby(["user", "segment_id"]).cumcount())
    a = x[x.k == 0].set_index(["user", "segment_id"])[c]; b = x[x.k == 1].set_index(["user", "segment_id"])[c]
    idx = a.index.intersection(b.index)
    return pd.DataFrame({"y1": a.loc[idx].to_numpy(), "y2": b.loc[idx].to_numpy()}, index=idx).dropna()
def decomp(c, first_df, keep_df, pair_mask=None):
    e, a, b = additive_residuals(first_df, c); x = first_df.loc[e.index]
    s2e = float(e.var())
    P = pairs(keep_df, c)
    if pair_mask is not None: P = P[pair_mask.reindex(P.index).fillna(False).to_numpy()]
    s2eps = float(0.5 * ((P.y1 - P.y2) ** 2).mean())
    # 합의(단위 내 구간 효과): 구간 효과의 단위 내 분산 − 표집 잡음
    unit = x.drop_duplicates("segment_id").set_index("segment_id").unit; k = x.groupby("segment_id").size()
    aw = a - a.groupby(unit.reindex(a.index)).transform("mean"); s2w = float(aw.var() - (s2e / k.reindex(a.index)).mean())
    return dict(s2_consensus=s2w, s2_resid=s2e, s2_noise=s2eps, s2_stable_indiv=s2e - s2eps, n_pairs=len(P), raters=x.user.unique(), e=e, P=P, x=x)
keep = keep.assign(row=keep.row)  # keep 은 원 row 순서 보존
rows = {}; detail = {}
top4 = [71, 121, 97, 70]
for c in NINE:
    r = decomp(c, first, keep)
    # 평가자 재표집 2,000회 (가법 효과 고정, 잔차·쌍만 재표집)
    users = r["raters"]; e_by = {u: r["e"][r["x"].user == u].to_numpy() for u in users}; P = r["P"]; p_by = {u: P.loc[u] if u in P.index.get_level_values(0) else None for u in users}
    boot = []
    for _ in range(2000):
        pick = rng.choice(users, len(users)); ee = np.concatenate([e_by[u] for u in pick]); pp_ = [p_by[u] for u in pick if p_by[u] is not None]
        if not pp_: continue
        pz = pd.concat(pp_); boot.append(ee.var() - 0.5 * ((pz.y1 - pz.y2) ** 2).mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # 민감도
    g = gap.reindex(P.index); q = pd.qcut(g, 3, labels=["short", "mid", "long"])
    sens = {f"gap_{lab}": round(float(0.5 * ((P[q == lab].y1 - P[q == lab].y2) ** 2).mean()), 3) for lab in ["short", "mid", "long"]}
    P4 = P[~P.index.get_level_values(0).isin(top4)]; sens["noise_excl_top4"] = round(float(0.5 * ((P4.y1 - P4.y2) ** 2).mean()), 3) if len(P4) else None
    loro = [0.5 * ((P[P.index.get_level_values(0) != u].y1 - P[P.index.get_level_values(0) != u].y2) ** 2).mean() for u in P.index.get_level_values(0).unique()]
    sens["noise_loro_range"] = [round(float(min(loro)), 3), round(float(max(loro)), 3)]
    rows[SHORT[c]] = dict(consensus=round(r["s2_consensus"], 3), resid=round(r["s2_resid"], 3), noise=round(r["s2_noise"], 3), stable_indiv=round(r["s2_stable_indiv"], 3),
                          ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3), n_pairs=r["n_pairs"], stable_share_of_resid=round(r["s2_stable_indiv"] / r["s2_resid"], 2), **sens)
T = pd.DataFrame(rows).T
print("== 한 평점의 분산 (평가자 주효과 제거 후): 합의(단위 내 구간) / 잔차 / 잡음(재검사) / 안정된 개인차 = 잔차 − 잡음 ==")
print(T[["consensus", "resid", "noise", "stable_indiv", "ci_lo", "ci_hi", "stable_share_of_resid", "n_pairs"]].to_string())
print("\n민감도 (잡음 추정):\n", T[["gap_short", "gap_mid", "gap_long", "noise_excl_top4", "noise_loro_range"]].to_string())
passed = {k: bool(v["ci_lo"] > 0) for k, v in rows.items()}
print("\n판정 (안정된 개인차 CI 하한 > 0):", passed, "\nconvincing:", "개인 층 실재" if passed["unconvincing→convincing"] else "개인 선호 층 주장 철회")
save("q05_decompose", dict(table=rows, passed=passed))
