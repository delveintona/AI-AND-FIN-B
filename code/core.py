"""
core.py — 패널 만들기와 walk-forward 평가의 핵심 로직 (스크립트와 웹 앱이 같이 쓴다).

  build_panel(premium, mom_extra_lag)   구 × 월 패널 (정답 y + 모든 변수, 공표 시차 적용). 같은 인자면 캐시.
  default_spec()                        config.py의 기본 변수 구성 → {그룹: {"vars": [...], "pca": n}}
  run(panel, name, spec, years, C)      walk-forward 평가 → (연도별 결과 표, 부가 정보)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import *  # noqa

_panels = {}


def _read_wide(path):
    df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df


# ------------------------------------------------------------------ 패널
def build_panel(premium=PREMIUM, mom_extra_lag=MOM_EXTRA_LAG):
    key = (float(premium), int(mom_extra_lag))
    if key in _panels:
        return _panels[key]

    price = _read_wide(SEED / "rone_apt_index_wide.csv")
    price = price.loc[price.index <= pd.Period(LAST_PRICE_MONTH, "M")]
    ecos = _read_wide(SEED / "ecos_monthly.csv")
    ctl = pd.read_csv(SEED / "controls_panel.csv", encoding="utf-8-sig")
    ctl["month"] = pd.PeriodIndex(ctl["month"], freq="M")

    # 정답 y: 향후 12개월 수익률 >= 기준금리 + premium
    fwd = (price.shift(-HORIZON) / price - 1) * 100
    long = fwd.stack().rename("fwd_ret_12m").rename_axis(["month", "gu"]).reset_index()
    long = long.merge(ecos[["base_rate"]].rename(columns={"base_rate": "hurdle_base"}), left_on="month", right_index=True)
    long["hurdle"] = long["hurdle_base"] + premium
    long["y"] = (long["fwd_ret_12m"] >= long["hurdle"]).astype(float)
    long.loc[long["fwd_ret_12m"].isna(), "y"] = np.nan

    # 금리 (서울 공통)
    macro = ecos.copy()
    macro["mortgage_spread"] = macro["mortgage_rate"] - macro["base_rate"]
    macro["term_spread"] = macro["ktb_10y"] - macro["ktb_3y"]

    # 공급 (서울 공통): 최근 12개월 합계
    sup = _read_wide(SEED / "supply_monthly.csv")
    sup = sup.reindex(pd.period_range(sup.index.min(), sup.index.max(), freq="M"))
    roll = lambda s: s.rolling(12, min_periods=12).sum()
    supply = pd.DataFrame({
        "permits_12m": roll(sup["permits_hh"]),
        "permits_apt_12m": roll(sup["permits_apt"]),
        "starts_12m": roll(sup["starts"]),
        "completions_12m": roll(sup["completions"]),
        "unsold": sup["unsold"],
    })
    supply.loc[supply.index < pd.Period("2006-12", "M"), "unsold"] = np.nan   # 그 이전은 12월 값뿐
    macro = macro.join(supply, how="outer").shift(PUB_LAG)                    # 공표 시차

    # 구별 변수
    mom = pd.concat({"mom_3m": (price / price.shift(3) - 1) * 100,
                     "mom_12m": (price / price.shift(12) - 1) * 100}, axis=1).shift(PUB_LAG + int(mom_extra_lag))
    mom_long = mom.stack(level=1, future_stack=True).rename_axis(["month", "gu"]).reset_index()
    val = ctl[["month", "gu", "jeonse_ratio", "jeonse_12m"]].copy()
    for c in ["jeonse_ratio", "jeonse_12m"]:
        val[c] = val.groupby("gu")[c].shift(PUB_LAG)

    panel = (long.merge(macro.drop(columns=["m2"], errors="ignore"), left_on="month", right_index=True, how="left")
                 .merge(mom_long, on=["month", "gu"], how="left")
                 .merge(val, on=["month", "gu"], how="left"))
    panel = panel.sort_values(["gu", "month"]).reset_index(drop=True)
    panel["month"] = panel["month"].astype(str)
    _panels[key] = panel
    return panel


# ------------------------------------------------------------------ 모델
def default_spec():
    return {g: {"vars": list(FEATURE_GROUPS[g]), "pca": int(PCA_GROUPS.get(g, 0))} for g in ACTIVE_GROUPS}


def spec_vars(spec):
    return [c for s in spec.values() for c in s["vars"]]


def _n_pca(s):
    return min(int(s.get("pca", 0)), len(s["vars"])) if len(s["vars"]) > 1 else 0


def make_model(spec, C_=C):
    parts = []
    for g, s in spec.items():
        n = _n_pca(s)
        if n >= 1:
            parts.append((g, Pipeline([("s", StandardScaler()), ("p", PCA(n))]), s["vars"]))
        else:
            parts.append((g, StandardScaler(), s["vars"]))
    return Pipeline([("x", ColumnTransformer(parts)), ("m", LogisticRegression(C=C_, max_iter=1000))])


def describe_fit(model, spec):
    """학습된 모델의 표준화 계수와 PCA 적재값."""
    coef = model.named_steps["m"].coef_[0]
    out, i = [], 0
    for g, s in spec.items():
        n = _n_pca(s)
        if n >= 1:
            p = model.named_steps["x"].named_transformers_[g].named_steps["p"]
            for k in range(n):
                out.append(dict(name=f"{g} · 성분{k + 1}", group=g, coef=float(coef[i]), pca=True,
                                explained=float(p.explained_variance_ratio_[k]),
                                loadings={v: float(w) for v, w in zip(s["vars"], p.components_[k])}))
                i += 1
        else:
            for v in s["vars"]:
                out.append(dict(name=v, group=g, coef=float(coef[i]), pca=False))
                i += 1
    return out


def run(panel, name, spec, years, C_=C):
    """walk-forward. spec이 비어 있으면 학습구간 평균(prior) 기준선."""
    cols = spec_vars(spec)
    df = panel.dropna(subset=["y"] + cols)
    mm = df["month"].map(pd.Period)
    rows, last = [], None
    for Y in years:
        test_start = pd.Period(f"{Y}-01", "M")
        train_end = test_start - HORIZON - 1
        tr = df[(mm >= pd.Period(TRAIN_START, "M")) & (mm <= train_end)]
        te = df[(mm >= test_start) & (mm <= test_start + 11)]
        base = dict(model=name, year=Y, n_train=len(tr), n_test=len(te))
        if len(tr) < 200 or te.y.nunique() < 2:
            reason = "학습 표본 부족" if len(tr) < 200 else "그해 정답이 한쪽뿐(전부 0 또는 전부 1)"
            rows.append(dict(base, auc=np.nan, brier=np.nan, test_buy_rate=float(te.y.mean()) if len(te) else np.nan, note=reason))
            continue
        if cols:
            m = make_model(spec, C_).fit(tr[cols], tr.y)
            p = m.predict_proba(te[cols])[:, 1]
            last = (Y, m)
        else:
            p = np.full(len(te), tr.y.mean())
        rows.append(dict(base, auc=roc_auc_score(te.y, p) if np.ptp(p) > 0 else 0.5,
                         brier=brier_score_loss(te.y, p), test_buy_rate=float(te.y.mean()), note=""))
    info = dict(first_month=str(df["month"].min()) if len(df) else None, n_rows=int(len(df)),
                fit=describe_fit(last[1], spec) if last else None, fit_year=last[0] if last else None)
    return pd.DataFrame(rows), info
