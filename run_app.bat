@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv (
  echo 처음 실행: 가상환경을 만들고 패키지를 설치합니다...
  py -3.11 -m venv .venv
  .venv\Scripts\python -m pip install -r requirements.txt
)
start "" http://127.0.0.1:8765
.venv\Scripts\python code\app.py
