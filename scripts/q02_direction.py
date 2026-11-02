"""Q2 · 문항 방향. docs/q_audit_prereg_2026-11-01.md Q2 + 사후 변경(③ 원값 상관)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from q_common import *

raw = pp.load_raw(); q = pp.question_cols(raw)
d = raw.drop_duplicates(subset=[c for c in raw.columns if c != "row"]); d = d[~d[q].isna().all(axis=1)].copy()
d[q] = d[q].mask(d[q].isin(pp.IDK_CODES)).astype(float); d = d.rename(columns=dict(zip(q, L)))
d["piece"] = d.filename.str.extract(r"^(Schubert_D960_mv2|Schubert_D960_mv3|Schubert_D935_no\.3|Beethoven_WoO80)")[0]

# ① ② 논문 Table 2 평균 (M) — 곡별. 1점=Option1 로 표기된 값.
T2 = {"Schubert_D960_mv2": [2.96,3.33,3.74,2.88,4.26,3.66,3.41,3.82,3.75,3.48,4.13,4.23,4.20,3.36,3.98,3.99,3.96,3.81,3.69],
      "Schubert_D960_mv3": [3.35,4.48,4.13,3.99,3.59,3.58,4.10,3.13,4.48,4.02,4.35,3.24,3.68,3.92,3.90,3.06,3.66,4.44,4.22],
      "Schubert_D935_no.3":[3.94,3.90,3.68,3.87,3.42,3.84,4.03,3.13,3.50,3.58,3.83,3.40,3.90,3.98,4.32,3.13,4.25,3.84,3.64],
      "Beethoven_WoO80":   [3.31,3.63,3.92,3.22,3.09,3.59,3.91,3.99,3.73,3.54,3.63,3.62,3.87,4.16,4.05,4.10,4.02,3.69,3.77]}
rows = []
for i, c in enumerate(L):
    r = {"item": i + 1, "dim": S[i]}
    for pc, vals in T2.items():
        m = d[d.piece == pc][c].mean(); t = vals[i]
        r[f"{pc}_csv"] = round(m, 2); r[f"{pc}_T2"] = t; r[f"{pc}_err_same"] = round(abs(m - t), 2); r[f"{pc}_err_flip"] = round(abs(8 - m - t), 2)
    def call(pcs):
        es = max(r[f"{p}_err_same"] for p in pcs); ef = max(r[f"{p}_err_flip"] for p in pcs)
        if min(es, ef) > 0.3: return "판별불가"
        return "그대로" if es < ef else "8-x" if ef < es - 0.15 else "판별불가"
    r["D960_vs_T2"] = call(["Schubert_D960_mv2", "Schubert_D960_mv3"]); r["round2_vs_T2"] = call(["Schubert_D935_no.3", "Beethoven_WoO80"])
    rows.append(r)
T = pd.DataFrame(rows)
print("== ①② CSV 평균 vs 논문 Table 2 (그대로 / 8-x) ==")
print(T[["item", "dim", "Schubert_D960_mv2_csv", "Schubert_D960_mv2_T2", "Schubert_D960_mv3_csv", "Schubert_D960_mv3_T2", "D960_vs_T2",
         "Beethoven_WoO80_csv", "Beethoven_WoO80_T2", "Schubert_D935_no.3_csv", "Schubert_D935_no.3_T2", "round2_vs_T2"]].to_string(index=False))

# ③ D960: B1 vs B2 구간 평균의 원값 Spearman (같은 구간을 두 집단이 평가)
d["block"] = d.user.map(pp.rater_block); d["segment_id"] = d.filename.str.replace(r"\.wav$", "", regex=True)
d9 = d[d.piece.str.startswith("Schubert_D960")]
sm = d9.groupby(["block", "segment_id"])[L].mean()
b1, b2 = sm.loc["B1(0-28)"], sm.loc["B2(70-121)"]; common = b1.index.intersection(b2.index)
cnt = d9[d9.block == "B1(0-28)"].groupby("segment_id").size().reindex(common)
rows3 = []
for i, c in enumerate(L):
    x, y = b1.loc[common, c], b2.loc[common, c]; ok = x.notna() & y.notna()
    rho = spearmanr(x[ok], y[ok]).correlation
    rows3.append({"dim": S[i], "n_seg": int(ok.sum()), "rho_B1_B2": round(rho, 2), "pass": bool(rho > 0 and (rho >= 0.3 or ok.sum() < 20))})
T3 = pd.DataFrame(rows3); print(f"\n== ③ D960 B1 vs B2 구간 평균 원값 Spearman (공통 구간 {len(common)}) =="); print(T3.to_string(index=False))

# ④ 평가자 73 의 message 로 설득력 방향
m = raw[(raw.user == 73) & raw.message.isin(["설득력이 있다", "애매하다"])].copy()
m["conv"] = pd.to_numeric(m[q[-1]], errors="coerce").mask(lambda s: s.isin(pp.IDK_CODES))
g = m.groupby("message").conv.agg(["mean", "median", "count"]); print("\n== ④ 평가자 73: message 별 설득력 점수 =="); print(g.round(2).to_string())
u = mannwhitneyu(m[m.message == "설득력이 있다"].conv.dropna(), m[m.message == "애매하다"].conv.dropna(), alternative="greater")
print("Mann-Whitney(설득력이 있다 > 애매하다) p =", f"{u.pvalue:.2e}")

# B3 문항 1·9·17: 설득력과의 부호 (노트북 1.4) 를 기록용으로 재계산
d["blk"] = d.block
sign = {}
for blk, g_ in d.groupby("blk"):
    x = g_[L] - g_.groupby("user")[L].transform("mean"); cc = x.corr()[CONV]
    sign[blk] = {S[L.index(c)]: round(cc[c], 2) for c in ["Timing_Stable_Unstable", "Timbre_Soft_Loud", "Emotion_&_Mood_Low_Energy_High_Energy"]}
print("\n== 참고: 설득력과의 평가자 내 상관, 문항 1·9·17 ==", sign)

# 종합 판정
final = []
for i, c in enumerate(L):
    r = T.iloc[i]; t3 = T3.iloc[i]
    verdict = "확정" if (r.D960_vs_T2 != "판별불가" or r.round2_vs_T2 != "판별불가") and t3["pass"] else "미확정"
    final.append({"item": i + 1, "dim": S[i], "csv_vs_T2_D960": r.D960_vs_T2, "csv_vs_T2_round2": r.round2_vs_T2, "rho_B1_B2": t3.rho_B1_B2, "verdict": verdict})
F = pd.DataFrame(final); print("\n== 종합 ==\n", F.to_string(index=False))
save("q02_direction", {"table2_compare": T.to_dict("records"), "B1_B2": T3.to_dict("records"), "rater73_message": g.to_dict("index"),
                       "rater73_p": float(u.pvalue), "sign_1_9_17": sign, "final": F.to_dict("records")})
