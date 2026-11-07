"""
04_pca.py — 그룹별 PCA 탐색 (금리 / 공급): 어떤 변수를 묶을 수 있는가?

보는 것
  1) 변수 간 상관 (높으면 한 축으로 묶을 수 있다)
  2) 주성분별 설명 비율, 누적 설명 비율
  3) 주성분 적재값 (PC1이 "무엇"을 뜻하는지)
  4) 변수 조합마다 쓸 수 있는 표본 시작 시점

주의  이 스크립트는 전체 기간으로 계산한 "탐색용" 결과다. 실제 모델에서는 PCA를 각 walk-forward 학습 구간에서만 맞춘다
      (02_walkforward.py의 make_model). 미래 정보가 섞이지 않게 하기 위해서다.
실행  python code/04_pca.py   →  report/B_pca.md
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import *  # noqa

# 탐색할 조합 {이름: 변수들}
SETS = {
    "금리 전체": ["base_rate", "mortgage_rate", "mortgage_spread", "ktb_3y", "term_spread", "m2_yoy"],
    "금리 수준만": ["base_rate", "mortgage_rate", "ktb_3y"],
    "공급 전체 (2012~)": ["permits_12m", "permits_apt_12m", "starts_12m", "completions_12m", "unsold"],
    "공급 인허가+미분양 (2008~)": ["permits_12m", "permits_apt_12m", "unsold"],
    "공급 흐름 (인허가·착공·준공, 2012~)": ["permits_12m", "starts_12m", "completions_12m"],
}


def analyze(name, cols, macro):
    d = macro[cols].dropna()
    z = StandardScaler().fit_transform(d)
    p = PCA().fit(z)
    ev = p.explained_variance_ratio_
    md = [f"### {name}\n", f"- 표본: {d.index.min()} ~ {d.index.max()} ({len(d)}개월), 변수 {len(cols)}개\n"]
    md.append("**상관**\n\n" + d.corr().round(2).to_markdown() + "\n")
    md.append("**설명 비율**\n\n" + pd.DataFrame({"설명": ev.round(3), "누적": ev.cumsum().round(3)},
                                                index=[f"PC{i + 1}" for i in range(len(ev))]).to_markdown() + "\n")
    k = int(np.searchsorted(ev.cumsum(), 0.8) + 1)
    md.append(f"- 누적 80%에 필요한 성분 수: **{k}개**\n")
    load = pd.DataFrame(p.components_[:min(3, len(cols))].T, index=cols,
                        columns=[f"PC{i + 1}" for i in range(min(3, len(cols)))]).round(2)
    md.append("**적재값** (절댓값이 큰 변수가 그 성분의 뜻)\n\n" + load.to_markdown() + "\n")
    print(f"\n[{name}] {d.index.min()}~{d.index.max()} ({len(d)}개월)  PC1 {ev[0]:.0%}  PC1+2 {ev[:2].sum():.0%}  → 80%까지 {k}개")
    print(load.to_string())
    return "\n".join(md)


def main():
    panel = pd.read_csv(PROC / "panel.csv", encoding="utf-8-sig")
    # 금리·공급은 모든 구가 같은 값 → 구 하나만 골라 월별 표로
    macro = panel[panel.gu == panel.gu.iloc[0]].set_index(pd.PeriodIndex(panel[panel.gu == panel.gu.iloc[0]].month, freq="M"))
    out = ["# B 모델 PCA 탐색 (금리 · 공급)\n",
           "전체 기간 기준 **탐색용**. 모델에서는 PCA를 walk-forward 학습 구간 안에서만 맞춘다.\n"]
    for name, cols in SETS.items():
        out.append(analyze(name, cols, macro))
    REPORT.mkdir(exist_ok=True)
    (REPORT / "B_pca.md").write_text("\n".join(out), encoding="utf-8")
    print(f"\n저장: {REPORT / 'B_pca.md'}")


if __name__ == "__main__":
    main()
