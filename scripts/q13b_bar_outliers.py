"""Q13 ③ 보정: 마디 길이 이상치를 '같은 곡 다른 연주들의 같은 마디 길이 중앙값' 기준으로 다시 센다 (사후 변경 기록 참조)."""
import json, numpy as np, pandas as pd
from q_common import *
R = paths.ASAP; ann = json.load(open(R / "asap_annotations.json")); pool = pd.read_csv(OUT / "q13_asap_pool.csv")
rows = []
for folder, g in pool.groupby("folder"):
    L = {}
    for k in g.midi_performance:
        db = np.array(ann[k]["performance_downbeats"]); L[k] = np.diff(db)
    n = min(len(v) for v in L.values()); M = np.array([v[:n] for v in L.values()]); med = np.median(M, 0)
    for k, v in L.items():
        r = v[:n] / med; bad = (r > 2) | (r < 0.5); rows.append(dict(folder=folder, midi=k, n_bars=n, frac_bad=float(bad.mean()), n_bars_mismatch=int(len(v) != n)))
T = pd.DataFrame(rows); print("마디 길이가 같은 곡 중앙값의 2배/절반을 벗어나는 비율: 중앙 %.4f · >2%%: %d/%d 연주 · 마디 수 불일치 연주 %d" % (T.frac_bad.median(), (T.frac_bad > .02).sum(), len(T), T.n_bars_mismatch.sum()))
print(T[T.frac_bad > .02].sort_values("frac_bad", ascending=False).head(8).to_string(index=False))
good = T[(T.frac_bad <= .02) & (T.n_bars_mismatch == 0)]; gc = good.groupby("folder").size(); print("필터 후 6연주 이상 곡 %d, 연주 %d" % ((gc >= 6).sum(), gc[gc >= 6].sum()))
save("q13b_bar_outliers", dict(median_frac_bad=float(T.frac_bad.median()), n_fail=int((T.frac_bad > .02).sum()), pieces_ge6=int((gc >= 6).sum()), perfs=int(gc[gc >= 6].sum()), pass_pieces=bool((gc >= 6).sum() >= 40)))
T.to_csv(OUT / "q13b_bar_outliers.csv", index=False)
