"""
catalog.py — 변수·자료 설명(화면에 표시) 과 결과 해석 문장 생성.
"""

import numpy as np

GROUP_LABEL = {
    "rates_level": "금리 수준",
    "rates_other": "금리 기타 (스프레드·유동성)",
    "supply": "공급 (핵심)",
    "supply_ext": "공급 (확장: 착공·준공)",
    "momentum": "모멘텀 (가격 추세)",
    "valuation": "가치평가 (전세가율)",
}

# 변수 → 설명. scope: 서울 전체 공통이면 "seoul", 구마다 다르면 "gu"
VARS = {
    "base_rate": dict(label="기준금리", unit="%", scope="seoul", source="한국은행 ECOS 722Y001",
                      desc="한국은행 기준금리. 대출·예금 금리의 출발점이라 주택 구매 비용의 바탕이 된다."),
    "mortgage_rate": dict(label="주택담보대출 금리", unit="%", scope="seoul", source="한국은행 ECOS 121Y006",
                          desc="은행 신규 주택담보대출 평균 금리. 집을 살 때 실제로 내는 이자 수준."),
    "ktb_3y": dict(label="국고채 3년", unit="%", scope="seoul", source="한국은행 ECOS 721Y001",
                   desc="3년 만기 국채 금리. 시장이 예상하는 향후 몇 년의 금리 방향을 반영."),
    "mortgage_spread": dict(label="주담대 스프레드", unit="%p", scope="seoul", source="계산: 주담대금리 − 기준금리",
                            desc="은행이 기준금리에 얹는 가산 폭. 클수록 대출이 까다롭거나 비싸다는 신호(신용 여건)."),
    "term_spread": dict(label="장단기 금리차", unit="%p", scope="seoul", source="계산: 국고채10년 − 국고채3년",
                        desc="장기와 단기 금리 차이. 경기 기대와 금리 방향에 대한 시장 전망."),
    "m2_yoy": dict(label="M2 증가율", unit="% (전년동월비)", scope="seoul", source="한국은행 ECOS 161Y006 (2004-09 이전은 옛 통계 연결)",
                   desc="시중 통화량 증가율. 돈이 풀리는 정도(유동성)."),
    "permits_12m": dict(label="인허가 (12개월 합)", unit="가구", scope="seoul", source="KOSIS 국토교통부 DT_MLTM_1948",
                        desc="최근 12개월간 서울 주택건설 인허가 가구 수. 1~3년 뒤 공급을 미리 보여주는 지표."),
    "permits_apt_12m": dict(label="인허가·아파트 (12개월 합)", unit="가구", scope="seoul", source="KOSIS 국토교통부 DT_MLTM_1948",
                            desc="인허가 중 아파트만. 전체 인허가와 매우 비슷하게 움직인다."),
    "unsold": dict(label="미분양 재고", unit="호", scope="seoul", source="KOSIS 국토교통부 DT_MLTM_2082 (구별 합계)",
                   desc="팔리지 않고 남은 신축 주택 수. 많을수록 수요가 약하다는 신호. 월별 자료는 2006-12부터."),
    "starts_12m": dict(label="착공 (12개월 합)", unit="호", scope="seoul", source="KOSIS 국토교통부 DT_MLTM_5386",
                       desc="최근 12개월 착공 호수. 자료가 2011년부터라 같이 쓰면 학습 표본이 줄어든다."),
    "completions_12m": dict(label="준공 (12개월 합)", unit="호", scope="seoul", source="KOSIS 국토교통부 DT_MLTM_5372",
                            desc="최근 12개월 사용검사(입주) 호수. 자료가 2010-07부터."),
    "mom_3m": dict(label="모멘텀 3개월", unit="%", scope="gu", source="계산: 한국부동산원 R-ONE 구별 매매가격지수",
                   desc="최근 3개월 가격 상승률. 가격 자체가 정보라서 '오르니까 오른다'에 가깝다."),
    "mom_12m": dict(label="모멘텀 12개월", unit="%", scope="gu", source="계산: 한국부동산원 R-ONE 구별 매매가격지수",
                    desc="최근 12개월 가격 상승률. 팀 논의에서 통제변수로 쓰였고, 메인 축으로 쓰면 설명이 약하다는 지적이 있었다."),
    "jeonse_ratio": dict(label="전세가율", unit="%", scope="gu", source="한국부동산원 R-ONE (A 모델 데이터)",
                         desc="매매가 대비 전세가 비율. 높을수록 사는 비용과 빌리는 비용 차이가 작아 매수 수요가 늘어난다. 2012-01 이전은 지수로 역산한 값."),
    "jeonse_12m": dict(label="전세 상승률 12개월", unit="%", scope="gu", source="한국부동산원 R-ONE (A 모델 데이터)",
                       desc="최근 12개월 전세가격 상승률. 실수요 압력의 신호."),
}

DATASETS = [
    dict(file="rone_apt_index_wide.csv", name="서울 25개 구 아파트 매매가격지수", source="한국부동산원 R-ONE (A 저장소에서 가져옴)",
         use="정답 y(12개월 수익률)와 모멘텀 계산"),
    dict(file="ecos_monthly.csv", name="금리·통화량 월별", source="한국은행 ECOS (A 저장소에서 가져옴)",
         use="금리 수준, 금리 기타, 매수 기준선(기준금리 + 프리미엄)"),
    dict(file="supply_monthly.csv", name="서울 공급 월별 (인허가·착공·준공·미분양)", source="KOSIS 국토교통부 (03_supply.py로 직접 수집)",
         use="공급 그룹"),
    dict(file="controls_panel.csv", name="전세가율·전세 상승률 등", source="한국부동산원 R-ONE (A 저장소에서 가져옴)",
         use="가치평가 그룹 (거래량·준공은 사용 안 함)"),
]


# ------------------------------------------------------------------ 해석
def _tier(m):
    if m >= 0.65:
        return "꽤 쓸 만한 수준"
    if m >= 0.55:
        return "약간의 신호가 있는 수준"
    if m >= 0.45:
        return "동전 던지기와 거의 같은 수준"
    return "오히려 방향이 반대로 맞는 수준(예측으로는 쓸 수 없음)"


def interpret(b, mom, prior, info, spec, years, include_test, premium, mom_lag):
    """b, mom: {auc: {year: v}, mean, std, min, max}. 문단 리스트를 돌려준다. 각 문단은 (제목, 본문)."""
    out = []
    n_ok = int(sum(v is not None for v in b["auc"].values()))
    nvars = sum(len(s["vars"]) for s in spec.values())

    # 1 한 줄 결론
    out.append(("한 줄 요약",
                f"선택한 변수 {nvars}개(축 {len(spec)}개)로 평가한 평균 AUC는 {b['mean']:.2f}로, {_tier(b['mean'])}입니다. "
                f"AUC 0.5는 무작위, 1.0은 완벽한 예측을 뜻합니다. 평가한 {n_ok}개 연도의 평균이며 연도 범위는 {years[0]}~{years[-1]}입니다."))

    # 2 기준선 비교
    if mom is not None and mom["mean"] is not None:
        d = b["mean"] - mom["mean"]
        if abs(d) < 0.05:
            txt = f"모멘텀만 쓴 기준선({mom['mean']:.2f})과 차이가 {abs(d):.2f}로 작아, 이 설정이 모멘텀보다 낫다고 말하기 어렵습니다."
        elif d > 0:
            txt = f"모멘텀만 쓴 기준선({mom['mean']:.2f})보다 {d:.2f} 높습니다. 다만 평가 연도가 적어 우연일 수 있으니 연도별 표를 같이 보세요."
        else:
            txt = f"모멘텀만 쓴 기준선({mom['mean']:.2f})보다 {abs(d):.2f} 낮습니다. 즉 이 변수 구성은 '가격이 오르던 추세'만 쓰는 것보다 못합니다."
        out.append(("기준선과 비교", txt + " 학습평균(prior) 기준선은 항상 0.5이고, 이것보다 높아야 의미가 있습니다."))

    # 3 안정성
    yrs = [(y, v) for y, v in b["auc"].items() if v is not None]
    below = [y for y, v in yrs if v < 0.5]
    if yrs:
        best = max(yrs, key=lambda t: t[1])
        worst = min(yrs, key=lambda t: t[1])
        txt = f"연도별 AUC는 최저 {worst[1]:.2f}({worst[0]}년)에서 최고 {best[1]:.2f}({best[0]}년)까지 움직였고, 0.5 미만인 해는 {len(below)}개({', '.join(map(str, below)) or '없음'})입니다."
        if b["std"] is not None and b["std"] > 0.15:
            txt += " 해마다 성능이 크게 뒤집히므로 평균 하나만 믿으면 안 됩니다. 금리·공급 사이클이 바뀌면 같은 변수가 반대로 작동하는 모습입니다."
        elif b["std"] is not None:
            txt += " 연도 간 변동이 비교적 작아 안정적인 편입니다."
        out.append(("연도별 안정성", txt))

    # 4 구 구분
    gu_vars = {"mom_3m", "mom_12m", "jeonse_ratio", "jeonse_12m"}
    used = {v for s in spec.values() for v in s["vars"]}
    if not (used & gu_vars):
        out.append(("한계: 구를 구분하지 못함",
                    "선택한 변수가 모두 서울 전체 공통 값이라 25개 구가 같은 점수를 받습니다. 이 모델은 '서울이 오를 시기'만 맞힐 수 있고 '어느 구가 오르는지'는 구분하지 못합니다. "
                    "연도별 AUC가 극단적으로 튀는 이유 중 하나입니다. 구별로 다른 변수(모멘텀, 전세가율, 구별 공급)가 필요합니다."))

    # 5 변수 방향
    fit = info.get("fit")
    if fit:
        top = sorted(fit, key=lambda f: -abs(f["coef"]))[:3]
        parts = []
        for f in top:
            direction = "높을수록 매수 확률을 올리는" if f["coef"] > 0 else "높을수록 매수 확률을 낮추는"
            label = VARS[f["name"]]["label"] if f["name"] in VARS else f["name"]
            parts.append(f"{label}({f['coef']:+.2f}, {direction} 방향)")
        txt = (f"{info['fit_year']}년 평가에 쓴 마지막 학습 결과에서 영향이 큰 순서는 " + ", ".join(parts) +
               " 입니다. 계수는 표준화된 값이라 크기를 서로 비교할 수 있지만, 학습 기간이 달라지면 부호가 바뀔 수 있고 인과관계를 뜻하지 않습니다.")
        # 같은 그룹 안에서 큰 계수가 반대 부호 → 변수끼리 너무 비슷해서 생기는 현상(다중공선성)일 수 있다
        for g in spec:
            fs = [f for f in fit if f["group"] == g and not f["pca"]]
            if len(fs) >= 2 and max(f["coef"] for f in fs) > 0.5 and min(f["coef"] for f in fs) < -0.5:
                txt += (f" 주의: '{GROUP_LABEL.get(g, g)}' 안에서 큰 계수가 서로 반대 부호입니다. 비슷하게 움직이는 변수들이 서로의 효과를 상쇄하며 "
                        "계수가 부풀려지는 현상(다중공선성)일 수 있으니, 이 계수의 방향은 해석하지 말고 변수를 하나로 줄이거나 PCA로 묶어 보세요.")
        out.append(("영향이 큰 변수", txt))

    # 6 표본 주의
    notes = [f"학습 표본은 {info['first_month']}부터 시작합니다(선택한 변수 중 가장 늦게 생기는 값 기준)." if info.get("first_month") else ""]
    nan_years = [y for y, v in b["auc"].items() if v is None]
    if nan_years:
        notes.append(f"{', '.join(map(str, nan_years))}년은 계산하지 못했습니다(표 아래 사유 참고).")
    notes.append("정답이 12개월 앞을 보기 때문에 인접한 달의 라벨이 거의 같아, 행 수가 많아 보여도 실제로 독립적인 표본은 적습니다.")
    out.append(("표본 주의", " ".join(n for n in notes if n)))

    # 7 설정 관련 경고
    warn = []
    if include_test:
        warn.append("테스트 연도가 포함된 결과입니다. 이 결과를 보고 변수를 고르면 테스트 데이터로 모델을 튜닝한 셈이 되므로, 최종 보고용 1회로만 쓰세요.")
    else:
        warn.append("검증 연도 결과입니다. 설정을 바꿔가며 가장 좋은 값을 고르면 검증 연도에 과적합되므로, 우열이 작은 차이는 신호로 보지 마세요.")
    if premium != 3.0:
        warn.append(f"매수 기준이 기준금리 + {premium:g}%p로 A 모델(+3%p)과 달라 직접 비교가 어렵습니다.")
    if mom_lag:
        warn.append(f"모멘텀에 공표 시차 외 {mom_lag}개월을 더 적용했습니다.")
    out.append(("사용상 주의", " ".join(warn)))
    return out
