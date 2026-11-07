"""
02_walkforward.py — 1년씩 늘려가는 walk-forward 평가 (명령줄 버전. 화면으로 하려면 python code/app.py)

각 테스트 연도 Y마다
  학습  = TRAIN_START ~ (Y년 1월 - HORIZON개월 - 1)  ← 12개월 선행 라벨이 확정된 달만 (purge)
  평가  = Y년 1~12월 (구 × 월)
모델  B (config.ACTIVE_GROUPS)  /  모멘텀만 기준선  /  학습구간 평균(prior) 기준선

실행  python code/02_walkforward.py            (VALID_YEARS 만 → 모델 조정용)
      python code/02_walkforward.py --test     (TEST_YEARS 도 포함 → 최종 1회만)
"""

import argparse

import pandas as pd

from config import *  # noqa
from core import build_panel, default_spec, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="TEST_YEARS 포함 (최종 평가, 1회만)")
    a = ap.parse_args()
    years = VALID_YEARS + (TEST_YEARS if a.test else [])

    panel = build_panel()
    spec = default_spec()
    runs = [run(panel, f"B({'+'.join(spec)})", spec, years)[0]]
    if BASELINE_MOMENTUM:
        runs.append(run(panel, "기준선: 모멘텀만", {"momentum": {"vars": FEATURE_GROUPS["momentum"], "pca": 0}}, years)[0])
    runs.append(run(panel, "기준선: 학습평균(prior)", {}, years)[0])
    res = pd.concat(runs, ignore_index=True)

    auc = res.pivot(index="year", columns="model", values="auc")
    summary = pd.DataFrame({"AUC 평균": auc.mean(), "AUC 최저": auc.min(), "AUC 표준편차": auc.std(),
                            "Brier 평균": res.groupby("model").brier.mean()})
    print(f"\n=== walk-forward AUC (연도 {years[0]}~{years[-1]}, {'테스트 포함' if a.test else '검증만'}) ===")
    print(auc.round(3).to_string())
    print("\n" + summary.round(3).to_string())

    REPORT.mkdir(exist_ok=True)
    out = REPORT / ("B_walkforward_test.md" if a.test else "B_walkforward_valid.md")
    out.write_text(
        f"# B 모델 walk-forward 결과 ({'검증+테스트' if a.test else '검증'})\n\n"
        f"- 변수 그룹: {list(spec)} · PCA: {PCA_GROUPS or '없음'} · y: {HORIZON}개월 수익률 ≥ 기준금리+{PREMIUM}%p\n"
        f"- 모멘텀 시차: 공표 {PUB_LAG}개월 + 추가 {MOM_EXTRA_LAG}개월\n\n"
        f"## 연도별 AUC\n\n{auc.round(3).to_markdown()}\n\n## 요약\n\n{summary.round(3).to_markdown()}\n",
        encoding="utf-8")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
