"""Q8 · 설득력의 구성 타당도: 18차원 합의로 설명하고 남은 잔차의 단위 내 신뢰도, 1주성분과의 상관, message 분류."""
import numpy as np, pandas as pd, re
from q_common import *
d = rater_adjusted(d960(ratings("mean"), human_only=True), L)
others = [c for c in L if c != CONV]
unit_of = d.drop_duplicates("segment_id").set_index("segment_id").unit
# 합의(전원) 세그먼트 평균, 단위 내 중심화 → b 적합
sm = d.groupby("segment_id")[L].mean(); u = unit_of.reindex(sm.index); smw = sm - sm.groupby(u).transform("mean")
A = np.c_[np.ones(len(smw)), smw[others].to_numpy()]; b = np.linalg.lstsq(A, smw[CONV].to_numpy(), rcond=None)[0]
r2 = 1 - ((smw[CONV] - A @ b) ** 2).sum() / ((smw[CONV] - smw[CONV].mean()) ** 2).sum()
print("단위 내 합의 설득력 ~ 18 합의: R² = %.2f" % r2)
# ① 반분 신뢰도: 구간마다 평가자를 반분, 각 반쪽에서 (설득력 평균 − 그 반쪽의 18차원 평균·b) 를 구해 단위 내 상관.
#    예측 변수를 같은 반쪽에서 구해야 "a−b" 형태의 음의 인공 상관이 생기지 않는다 (전원 합의를 쓰면 −1 근처가 됨: 1차 시도에서 확인).
rng = np.random.default_rng(0)
x = d[["segment_id", "unit", "user"] + L].dropna(subset=[CONV])
groups = [(uu, v) for (sid, uu), v in x.groupby(["segment_id", "unit"]) if len(v) >= 6]
units = np.array([uu for uu, _ in groups]); um18 = smw[others].groupby(u).transform("mean")  # (0 by construction) 자리표시
unit_mean18 = sm[others].groupby(u).transform("mean")
rs_res, rs_raw = [], []
for _ in range(100):
    A = np.empty(len(groups)); B = np.empty(len(groups)); Ar = np.empty(len(groups)); Br = np.empty(len(groups))
    for i, (uu, v) in enumerate(groups):
        pm = rng.permutation(len(v)); h = len(v) // 2; va, vb = v.iloc[pm[:h]], v.iloc[pm[h:]]
        sid = v.segment_id.iloc[0]; c18 = unit_mean18.loc[sid].to_numpy()
        for arr, arr_raw, part in ((A, Ar, va), (B, Br, vb)):
            m18 = part[others].mean().to_numpy() - c18; arr_raw[i] = part[CONV].mean(); arr[i] = arr_raw[i] - (b[0] + m18 @ b[1:])
    def wc(a_, b_):
        a_ = a_ - pd.Series(a_).groupby(units).transform("mean").to_numpy(); b_ = b_ - pd.Series(b_).groupby(units).transform("mean").to_numpy(); return np.corrcoef(a_, b_)[0, 1]
    rs_res.append(wc(A, B)); rs_raw.append(wc(Ar, Br))
sb = lambda r: 2 * r / (1 + r)
rel_conv, rel_resid = sb(float(np.mean(rs_raw))), sb(float(np.mean(rs_res)))
print("단위 내 신뢰도(구간 ≥6 평가자): 설득력 원점수(보정) %.3f → 같은 반쪽 18차원 제거 잔차 %.3f" % (rel_conv, rel_resid))
# ② 1주성분과의 단위 내 상관 (사람 세그먼트 평균)
Z = (smw - smw.mean()) / (smw.std() + 1e-9); U, s, Vt = np.linalg.svd(Z[others].to_numpy(), full_matrices=False); pc1 = U[:, 0] * s[0]
print("1주성분(18차원) 분산 몫 %.2f · corr(설득력 합의, PC1) 단위 내 = %.2f" % (s[0] ** 2 / (s ** 2).sum(), abs(np.corrcoef(pc1, smw[CONV])[0, 1])))
# ③ message 거친 분류 (키워드; 사람 검수 필요)
raw = pp.load_raw(); m = raw[raw.message.notna() & raw.filename.str.startswith("Schubert_D960")].copy(); m["msg"] = m.message.astype(str)
pat = {"평가": r"설득|애매|좋|별로|아쉽|훌륭|무난|부족|잘 |못 |어색|기계적|딱딱|지루|인상적|매력|감동|탁월",
       "선호": r"취향|개인적|선호|내가|저는|제가|나는|싫|좋아|원하|했으면|주세요|했으면",
       "서술": r"페달|템포|빠|느리|소리|음색|루바토|프레이|다이나믹|셈여림|아티큘|레가토|스타카토|박자|리듬|강약|음량|터치|멜로디|베이스|화음|왼손|오른손"}
def cls(t):
    hit = [k for k, p in pat.items() if re.search(p, t)]
    return "+".join(hit) if hit else "기타"
m["cls"] = m.msg.map(cls); dist = m.cls.value_counts(); print("\nmessage 분류(키워드, D960 %d건):" % len(m)); print(dist.head(10).to_string())
per = m.groupby("user").cls.apply(lambda s: s.str.contains("선호").mean()).sort_values(ascending=False)
print("평가자별 '선호' 표현 비율 상위:", per.head(5).round(2).to_dict())
print("예시(선호):", m[m.cls.str.contains("선호")].msg.drop_duplicates().head(6).tolist())
verdict = dict(rel_conv_adj=round(rel_conv, 3), rel_resid=round(rel_resid, 3), independent=bool(rel_resid >= 0.20), r2_consensus=round(float(r2), 3))
print("\n판정:", verdict)
save("q08_convincing", dict(verdict=verdict, pc1_share=float(s[0] ** 2 / (s ** 2).sum()), msg_classes=dist.to_dict()))
