"""
01_build_panel.py — 구 × 월 학습 패널 만들기 (정답 y + 변수, 공표 시차 적용). 로직은 core.build_panel.

입력  data/seed/*.csv   (A 모델 데이터 + 03_supply.py)
출력  data/processed/panel.csv
실행  python code/01_build_panel.py
"""

from config import *  # noqa
from core import build_panel


def main():
    panel = build_panel()
    PROC.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROC / "panel.csv", index=False, encoding="utf-8-sig")

    lab = panel.dropna(subset=["y"])
    print(f"패널 {len(panel):,}행 · 라벨 있는 구간 {lab.month.min()} ~ {lab.month.max()} · 매수(y=1) 비율 {lab.y.mean() * 100:.1f}%")
    print("\n변수별 첫 값이 생기는 달 / 결측 비율 (그룹)")
    for g, cols in FEATURE_GROUPS.items():
        for c in cols:
            first = panel.loc[panel[c].notna(), "month"].min()
            print(f"  [{g:<11}] {c:<16} 시작 {first}  결측 {panel[c].isna().mean() * 100:4.1f}%")
    print(f"\n저장: {PROC / 'panel.csv'}")


if __name__ == "__main__":
    main()
