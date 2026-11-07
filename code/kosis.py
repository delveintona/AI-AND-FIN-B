"""
kosis.py — KOSIS(국가통계포털) Open API 공용 함수.

    from kosis import meta_ids, fetch
    ids = meta_ids("116", "DT_MLTM_1948")            # {분류ID: {이름: 항목ID}}
    df  = fetch("116", "DT_MLTM_1948", {...}, "200701", "202512")

탐색:  python code/kosis.py search 주택 인허가     |   python code/kosis.py meta 116 DT_MLTM_1948
"""

import json
import sys
import urllib.parse
import urllib.request

import pandas as pd

from env import get_key

BASE = "https://kosis.kr/openapi"


def call(path, **params):
    q = {k: v for k, v in params.items() if v not in (None, "")}
    q.update(format="json", jsonVD="Y", apiKey=get_key("KOSIS_API_KEY"))
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    if isinstance(data, dict) and ("err" in data or "errMsg" in data):
        raise RuntimeError(f"KOSIS 오류 {data.get('err')}: {data.get('errMsg')}")
    return data


def meta_ids(org, tbl):
    """분류별 {이름: 항목ID}. 같은 이름이 여러 개면 리스트로."""
    df = pd.DataFrame(call("statisticsData.do", method="getMeta", type="ITM", orgId=org, tblId=tbl))
    out = {}
    for obj, g in df.groupby("OBJ_ID", sort=False):
        d = {}
        for r in g.itertuples():
            d.setdefault(r.ITM_NM, []).append(r.ITM_ID)
        out[obj] = d
    return out


def fetch(org, tbl, objs, start, end, prd="M"):
    """objs = {"objL1": 항목ID 또는 'ALL', ...}. 40,000셀 제한이 있어 분류를 좁혀 요청한다."""
    r = call("Param/statisticsParameterData.do", method="getList", orgId=org, tblId=tbl, itmId="ALL",
             prdSe=prd, startPrdDe=start, endPrdDe=end, **objs)
    df = pd.DataFrame(r)
    df["DT"] = pd.to_numeric(df["DT"], errors="coerce")
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "search":
        rows = call("statisticsSearch.do", method="getList", searchNm=" ".join(sys.argv[2:]),
                    sort="RANK", startCount=1, resultCount=40)
        for r in rows:
            print(f"ORG={r['ORG_ID']:<4} TBL={r['TBL_ID']:<16} {r.get('STRT_PRD_DE')}~{r.get('END_PRD_DE')}  {r['TBL_NM']} [{r.get('ORG_NM')}]")
    elif cmd == "meta":
        for obj, d in meta_ids(sys.argv[2], sys.argv[3]).items():
            print(obj, {k: v for k, v in list(d.items())[:25]})
    else:
        print(__doc__)
