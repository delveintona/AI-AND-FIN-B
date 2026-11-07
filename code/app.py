"""
app.py — 로컬 실험 화면. 변수를 넣고 빼고, PCA를 켜고 끄고, 모델을 돌려 결과와 해석을 본다.

실행  python code/app.py            → 브라우저에서 http://127.0.0.1:8765
옵션  python code/app.py --port 9000
외부 패키지 없이 파이썬 기본 서버만 쓴다 (내 컴퓨터에서만 접속 가능).
"""

import argparse
import json
import math
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from catalog import DATASETS, GROUP_LABEL, VARS, interpret
from config import *  # noqa
from core import build_panel, default_spec, run

WEB = ROOT / "web"
LOCK = threading.Lock()   # sklearn 실행은 한 번에 하나씩


def clean(o):
    """NaN/inf → None, numpy → 파이썬 기본형 (JSON용)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def first_month(panel, col):
    s = panel.loc[panel[col].notna(), "month"]
    return s.min() if len(s) else None


def state():
    panel = build_panel()
    lab = panel.dropna(subset=["y"])
    groups = []
    for g, cols in FEATURE_GROUPS.items():
        vs = []
        for c in cols:
            vs.append(dict(id=c, first_month=first_month(panel, c), missing=float(panel[c].isna().mean()), **VARS[c]))
        groups.append(dict(id=g, label=GROUP_LABEL.get(g, g), vars=vs))
    datasets = []
    for d in DATASETS:
        p = SEED / d["file"]
        df = pd.read_csv(p, encoding="utf-8-sig")
        datasets.append(dict(d, rows=int(len(df)), columns=[c for c in df.columns[:12]],
                             period=f"{df.iloc[:, 0].min()} ~ {df.iloc[:, 0].max()}"))
    return dict(groups=groups, datasets=datasets, default_spec=default_spec(),
                settings=dict(premium=PREMIUM, C=C, mom_lag=MOM_EXTRA_LAG, pub_lag=PUB_LAG, horizon=HORIZON,
                              valid_years=VALID_YEARS, test_years=TEST_YEARS, train_start=TRAIN_START),
                label=dict(first=lab.month.min(), last=lab.month.max(), buy_rate=float(lab.y.mean()), rows=int(len(lab)),
                           n_gu=int(panel.gu.nunique())))


def series(name, premium, mom_lag):
    panel = build_panel(premium, mom_lag)
    if name not in panel.columns:
        raise ValueError("알 수 없는 변수")
    s = panel.groupby("month")[name].mean().dropna()
    return dict(name=name, months=list(s.index), values=[round(float(v), 4) for v in s.values],
                note="구별 변수는 25개 구 평균. 공표 시차 1개월이 적용된 값입니다." if VARS[name]["scope"] == "gu"
                else "서울 공통 값. 공표 시차 1개월이 적용된 값입니다.")


def summarize(df):
    auc = {int(r.year): (None if pd.isna(r.auc) else float(r.auc)) for r in df.itertuples()}
    v = pd.Series([x for x in auc.values() if x is not None], dtype=float)
    return dict(auc=auc, mean=float(v.mean()) if len(v) else None, std=float(v.std()) if len(v) > 1 else None,
                min=float(v.min()) if len(v) else None, max=float(v.max()) if len(v) else None,
                brier=float(df.brier.mean()) if df.brier.notna().any() else None)


def do_run(req):
    spec = {}
    for g, s in req["spec"].items():
        vars_ = [v for v in s["vars"] if v in VARS]
        if vars_:
            spec[g] = dict(vars=vars_, pca=int(s.get("pca", 0)))
    if not spec:
        raise ValueError("변수를 하나 이상 선택하세요.")
    premium = float(req.get("premium", PREMIUM))
    C_ = float(req.get("C", C))
    mom_lag = int(req.get("mom_lag", MOM_EXTRA_LAG))
    include_test = bool(req.get("include_test", False))
    years = VALID_YEARS + (TEST_YEARS if include_test else [])

    panel = build_panel(premium, mom_lag)
    name = "B 모델 (선택한 구성)"
    res_b, info = run(panel, name, spec, years, C_)
    models = {name: summarize(res_b)}
    mom_spec = {"momentum": {"vars": FEATURE_GROUPS["momentum"], "pca": 0}}
    mom_name = "기준선: 모멘텀만"
    if req.get("baseline", True):
        models[mom_name] = summarize(run(panel, mom_name, mom_spec, years, C_)[0])
    prior_name = "기준선: 학습평균(0.5)"
    models[prior_name] = summarize(run(panel, prior_name, {}, years, C_)[0])

    folds = [dict(year=int(r.year), n_train=int(r.n_train), n_test=int(r.n_test), buy_rate=r.test_buy_rate, brier=r.brier, note=r.note)
             for r in res_b.itertuples()]
    lab = panel.dropna(subset=["y"])
    text = interpret(models[name], models.get(mom_name), models[prior_name], info, spec, years, include_test, premium, mom_lag)
    return dict(years=years, models=models, folds=folds, info=info, interpretation=text, spec=spec,
                settings=dict(premium=premium, C=C_, mom_lag=mom_lag, include_test=include_test),
                label=dict(buy_rate=float(lab.y.mean()), first=lab.month.min(), last=lab.month.max()))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(clean(obj), ensure_ascii=False))

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                with LOCK:
                    return self._json(state())
            if u.path == "/api/series":
                q = parse_qs(u.query)
                with LOCK:
                    return self._json(series(q["name"][0], float(q.get("premium", [PREMIUM])[0]), int(q.get("mom_lag", [MOM_EXTRA_LAG])[0])))
            self._send(404, "not found", "text/plain; charset=utf-8")
        except Exception as e:
            traceback.print_exc()
            self._json(dict(error=str(e)), 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            return self._send(404, "not found", "text/plain; charset=utf-8")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
            with LOCK:
                self._json(do_run(req))
        except ValueError as e:
            self._json(dict(error=str(e)), 400)
        except Exception as e:
            traceback.print_exc()
            self._json(dict(error=f"실행 중 오류: {e}"), 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"B 모델 실험 화면: http://127.0.0.1:{a.port}   (종료: Ctrl+C)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
