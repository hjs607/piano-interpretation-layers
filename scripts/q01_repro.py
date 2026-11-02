"""Q1① · 원자료에서 재생성한 pp_* 캐시와 현재 캐시 대조. 인자: 재생성 캐시 폴더."""
import sys, numpy as np, pandas as pd
from q_common import *
new = Path(sys.argv[1]); res = {}
for f in sorted((paths.INTERIM).glob("pp_*.parquet")):
    a = pd.read_parquet(f); g = new / f.name
    if not g.exists(): res[f.name] = "재생성본 없음"; continue
    b = pd.read_parquet(g)
    key = [c for c in ("segment_id", "user", "variant", "group", "dim") if c in a.columns]
    if a.shape != b.shape: res[f.name] = f"shape {a.shape} vs {b.shape}"; continue
    a2, b2 = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    num = [c for c in a2.columns if pd.api.types.is_numeric_dtype(a2[c])]
    diff = (a2[num].astype(float) - b2[num].astype(float)).abs().max().max(); same_obj = all((a2[c].astype(object).where(a2[c].notna(), "∅").astype(str) == b2[c].astype(object).where(b2[c].notna(), "∅").astype(str)).all() for c in a2.columns if c not in num)
    res[f.name] = {"shape": list(a.shape), "max_abs_diff": float(diff), "non_numeric_equal": bool(same_obj)}
for k, v in res.items(): print(k, v)
ok = all(isinstance(v, dict) and v["max_abs_diff"] < 1e-6 and v["non_numeric_equal"] for v in res.values())
print("Q1① 판정:", "통과" if ok else "불일치"); save("q01_repro", {"files": res, "pass": ok})
