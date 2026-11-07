"""
03_supply.py — 공급 기초 자료 (서울, 월별) 받기. KOSIS 국토교통부 통계.

  permits_hh   주택건설 인허가 (가구수, 전 유형)      DT_MLTM_1948  2007-01~  ← 월별 "누계"라 월 값으로 풀어 쓴다
  permits_apt  주택건설 인허가 (가구수, 아파트)       DT_MLTM_1948  2007-01~
  starts       주택건설 착공 (호)                     DT_MLTM_5386  2011-01~
  completions  주택건설 준공(사용검사) (호)           DT_MLTM_5372  2010-01~
  unsold       미분양 주택 재고 (호, 월말)            DT_MLTM_2082  2000-01~  ← 구별 표의 서울 합계

출력  data/seed/supply_monthly.csv   (월 × 5열, 원자료 그대로. 가공은 04_features/panel 단계)
실행  python code/03_supply.py
"""

import pandas as pd

from config import SEED
from kosis import fetch, meta_ids

ORG = "116"
START, END = "200001", "202512"


def pick(ids, obj, name):
    return ids[obj][name][0]


def permits():
    ids = meta_ids(ORG, "DT_MLTM_1948")
    o = [k for k in ids if k != "ITEM"]            # A 시도, B 대분류, C 중분류, D 소분류
    seoul = pick(ids, o[0], "서울")
    out = {}
    for nm, key in {"permits_hh": "합계(가구수기준)", "permits_apt": "아파트"}.items():
        if key == "합계(가구수기준)":
            sel = [pick(ids, o[1], key), pick(ids, o[2], key), pick(ids, o[3], key)]
        else:
            sel = [pick(ids, o[1], "아파트"), pick(ids, o[2], "아파트"), pick(ids, o[3], "아파트")]
        df = fetch(ORG, "DT_MLTM_1948", {"objL1": seoul, "objL2": sel[0], "objL3": sel[1], "objL4": sel[2]}, "200701", END)
        s = pd.Series(df["DT"].values, index=pd.PeriodIndex(df["PRD_DE"], freq="M")).sort_index()
        # 누계 → 월 값: 같은 해 안에서 차분 (1월은 그대로)
        month_val = s.groupby(s.index.year).diff().fillna(s)
        out[nm] = month_val
    return out


def sido_monthly(tbl, name):
    ids = meta_ids(ORG, tbl)
    o = [k for k in ids if k != "ITEM"]            # A 구분, B 부문, C 시도
    df = fetch(ORG, tbl, {"objL1": pick(ids, o[0], "총계"), "objL2": pick(ids, o[1], "총계"),
                          "objL3": pick(ids, o[2], "서울")}, "200901", END)
    return {name: pd.Series(df["DT"].values, index=pd.PeriodIndex(df["PRD_DE"], freq="M")).sort_index()}


def unsold():
    ids = meta_ids(ORG, "DT_MLTM_2082")
    o = [k for k in ids if k != "ITEM"]
    df = fetch(ORG, "DT_MLTM_2082", {"objL1": pick(ids, o[0], "서울"), "objL2": pick(ids, o[1], "계")}, START, END)
    return {"unsold": pd.Series(df["DT"].values, index=pd.PeriodIndex(df["PRD_DE"], freq="M")).sort_index()}


def main():
    parts = {}
    for label, fn in [("인허가", permits), ("착공", lambda: sido_monthly("DT_MLTM_5386", "starts")),
                      ("준공", lambda: sido_monthly("DT_MLTM_5372", "completions")), ("미분양", unsold)]:
        print(f"받는 중: {label}")
        parts.update(fn())
    wide = pd.concat(parts, axis=1).sort_index()
    wide.index.name = "month"
    wide.to_csv(SEED / "supply_monthly.csv", encoding="utf-8-sig")

    print("\n=== 공급 자료 리포트 (서울) ===")
    for c in wide:
        s = wide[c].dropna()
        full = pd.period_range(s.index.min(), s.index.max(), freq="M")
        print(f"  {c:<12} {s.index.min()} ~ {s.index.max()}  {len(s)}개월 (빠진 달 {len(full.difference(s.index))})"
              f"  월평균 {s.mean():,.0f}  최소 {s.min():,.0f}  최대 {s.max():,.0f}")
    print(f"\n저장: {SEED / 'supply_monthly.csv'}")


if __name__ == "__main__":
    main()
