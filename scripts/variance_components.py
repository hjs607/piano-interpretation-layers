"""PercePiano 분산성분·신뢰도 캐시 생성 (M1).

평가자 집단과 곡이 완전히 겹치므로 두 집단을 따로 추정합니다.
- 모멘트 추정(평가자 보정 전·후), 모형 없는 분할-반 점검
- statsmodels MixedLM (단위·구간·평가자 교차 무선효과, REML) 검증
- 반복 평가 처리 민감도(평균 vs 유지)
출력: data/interim/pp_variance_components.parquet
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.eda import paths, percepiano_io as pp, stats_utils as su

warnings.filterwarnings("ignore")

GROUPS = {"D960": lambda d: d[d.block != "B3(201-211)"], "WoO80+D935": lambda d: d[d.block == "B3(201-211)"]}


def mixedlm(d: pd.DataFrame, c: str) -> dict:
    x = d[["user", "segment_id", "unit", c]].dropna().rename(columns={c: "y"}).copy()
    x["g"] = 1
    m = smf.mixedlm("y ~ 1", x, groups="g",
                    vc_formula={"unit": "0 + C(unit)", "seg": "0 + C(segment_id)", "rater": "0 + C(user)"})
    r = m.fit(method="lbfgs", reml=True)
    vc = dict(zip(r.model.exog_vc.names, r.vcomp))
    return {"mlm_unit": vc["unit"], "mlm_within": vc["seg"], "mlm_rater": vc["rater"], "mlm_error": r.scale,
            "mlm_converged": bool(r.converged)}


def main():
    variants = {"mean": pd.read_parquet(paths.INTERIM / "pp_ratings_clean.parquet"),
                "keep": pd.read_parquet(paths.INTERIM / "pp_ratings_keep.parquet")}
    rows = []
    for vname, rat in variants.items():
        for gname, sel in GROUPS.items():
            d = sel(rat)
            for i, c in enumerate(pp.LABELS19):
                row = {"variant": vname, "group": gname, "dim_idx": i, "dim": pp.SHORT19[i]}
                for adj in (True, False):
                    a = su.variance_decomposition(d, c, rater_adjust=adj)
                    tag = "adj" if adj else "raw"
                    row.update({f"{k}_{tag}": v for k, v in a.items()})
                if vname == "mean":
                    sh = su.split_half_within_unit(d, c, n_iter=200, seed=i)
                    row.update({"sh_all": sh["sb_all"], "sh_within": sh["sb_within"]})
                    row.update(mixedlm(d, c))
                rows.append(row)
                print(vname, gname, pp.SHORT19[i], flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(paths.INTERIM / "pp_variance_components.parquet")
    print("saved", out.shape)


if __name__ == "__main__":
    main()
