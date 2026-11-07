"""
env.py — .env에서 API 키를 읽는 공용 함수 (외부 패키지 없이).

사용
    from env import get_key
    key = get_key("ECOS_API_KEY")      # 환경변수 → .env 순으로 찾고, 없으면 안내 후 종료

점검
    python code/env.py                  # 키 3개가 채워졌는지(값은 숨김) 확인
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["ECOS_API_KEY", "KOSIS_API_KEY", "RONE_API_KEY"]


def _from_file():
    f = ROOT / ".env"
    out = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def get_key(name, required=True):
    val = os.environ.get(name) or _from_file().get(name, "")
    if not val and required:
        sys.exit(f"{name}가 비어 있습니다. 저장소 루트의 .env 파일에 {name}=키 를 넣어주세요.")
    return val


if __name__ == "__main__":
    for k in KEYS:
        v = get_key(k, required=False)
        print(f"{k:<15} {'설정됨 (' + str(len(v)) + '자)' if v else '비어 있음'}")
