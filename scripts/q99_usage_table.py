"""Q 점검 종합: reports/q/*.json → usage_table.csv + audit_summary.md 의 표 부분."""
import json, pandas as pd, numpy as np
from q_common import *
J = {p.stem: json.loads(p.read_text()) for p in OUT.glob("q*.json")}
q2 = {r["dim"]: r for r in J["q02_direction"]["final"]}; q3 = J["q03_raters"]; q4 = J["q04_assignment"]; q5 = J["q05_decompose"]["table"]; q6 = J["q06_sd_reliability"]["table"]
q6b = J["q06b_sd_diagnostics"]["heterogeneity"]; q7 = J["q07_w_beta"]; q10 = J["q10_features"]
d9 = d960(ratings("mean"), human_only=True); d9a = rater_adjusted(d9, L)
rel_raw = {SHORT[c]: round(within_unit_split_half(d9, c), 3) for c in L}; rel_adj = {SHORT[c]: round(within_unit_split_half(d9a, c), 3) for c in L}
# 19차원
axes = {r["dim"]: r for r in pd.read_csv(OUT.parent / "rater_layers" / "layer1_axes.csv").to_dict("records")} if (OUT.parent / "rater_layers" / "layer1_axes.csv").exists() else {}
rows = []
for s in S:
    w = q7["w"][s]; r = dict(dim=s, direction=q2[s]["verdict"], within_unit_rel_raw=rel_raw[s], within_unit_rel_adj=rel_adj.get(s))
    if s in q5: r.update(consensus=q5[s]["consensus"], stable_indiv=q5[s]["stable_indiv"], noise=q5[s]["noise"], stable_indiv_ci_lo=q5[s]["ci_lo"], sd_reliability=q6[s]["adj_mean"], sd_heterogeneity_p=q6b[s]["p"])
    r.update(w_splithalf_std=w["splithalf_std"], w_trait=("특성" if w["splithalf_std"] >= 0.5 else "약함" if w["splithalf_std"] >= 0.3 else "불안정"))
    rel = r["within_unit_rel_adj"] if r["within_unit_rel_adj"] is not None else r["within_unit_rel_raw"]
    r["rq2_axis"] = "공유 축" if rel is not None and rel >= 0.5 else "약한 공유" if rel is not None and rel >= 0.3 else "축 아님"
    r["rq3_use"] = ("H3a 종속변수" if s in [SHORT[c] for c in EXEC] and rel >= 0.5 else "H3b 종속변수(개인 편차)" if s == "unconvincing→convincing" else "제외/탐색")
    rows.append(r)
DIM = pd.DataFrame(rows); DIM.to_csv(OUT / "usage_table_dims.csv", index=False)
# 평가자
R = pd.read_csv(OUT / "q03_raters.csv"); R["decision"] = np.where(R.caution, "유지(주의 표시)", "유지"); R[["user", "block", "n", "retest_r", "caution", "agree_r_mean9", "decision"]].to_csv(OUT / "usage_table_raters.csv", index=False)
# 특징
FT = pd.DataFrame({"feature": list(q10["overlap"]), "overlap_retest": list(q10["overlap"].values()), "grade": [q10["verdict"]["grade"][k] for k in q10["overlap"]]})
FT["note"] = FT.feature.map({"vel_range": "vel_std 와 0.90 → X2 에서 제외", "articulation_median": "페달과 0.72, 주입: 페달 제거에 반응 → 백색화 필요", "timing_std_ms": "재측정 0.67 → 주의"}).fillna("")
FT.to_csv(OUT / "usage_table_features.csv", index=False)
# ASAP
asap = J.get("q13_asap", {}).get("verdict", {})
print(DIM.to_string(index=False)); print(); print(FT.to_string(index=False)); print("\nASAP:", asap)
