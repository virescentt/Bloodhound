#!/usr/bin/env bash
# Ślad Fejka — uruchomienie (macOS / Linux / Git Bash na Windows).
# Pierwsze uruchomienie: tworzy .venv (Python 3.13), instaluje zależności i pobiera modele twarzy.
#   ./start.sh                 → interfejs graficzny http://localhost:8501
#   ./start.sh demo            → zbiór syntetyczny → przetwarzanie offline → ewaluacja
#   ./start.sh <komenda> ...   → dowolna komenda CLI, np. ./start.sh run data/nask   (lista: ./start.sh --help)
set -euo pipefail
cd "$(dirname "$0")"

ok_version() { "$@" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)' >/dev/null 2>&1; }

# Python w .venv: bin/ (macOS/Linux) albo Scripts/ (Windows, Git Bash)
venv_python() {
  if [ -x .venv/bin/python ]; then echo .venv/bin/python
  elif [ -f .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe
  fi
}

VPY="$(venv_python)"
if [ -n "$VPY" ] && ! ok_version "$VPY"; then
  echo "⚠ .venv ma Pythona starszego niż 3.13 — tworzę środowisko od nowa"
  rm -rf .venv
  VPY=""
fi

if [ -z "$VPY" ]; then
  PY=()
  for cand in python3.13 python3 python python3.14; do
    if command -v "$cand" >/dev/null 2>&1 && ok_version "$cand"; then PY=("$cand"); break; fi
  done
  if [ ${#PY[@]} -eq 0 ] && command -v py >/dev/null 2>&1 && ok_version py -3.13; then PY=(py -3.13); fi
  if [ ${#PY[@]} -eq 0 ]; then
    echo "❌ Potrzebny Python 3.13 (lub nowszy) — nie znaleziono go w PATH."
    echo "   macOS:   brew install python@3.13"
    echo "   Linux:   sudo apt install python3.13 python3.13-venv   (Ubuntu: ppa:deadsnakes/ppa) albo pyenv"
    echo "   Windows: https://www.python.org/ftp/python/3.13.17/python-3.13.17-amd64.exe (zaznacz „Add python.exe to PATH”)"
    echo "   Wszystkie systemy: https://www.python.org/downloads/   ·   albo Docker: docker compose up --build app"
    exit 1
  fi
  [ -d .venv ] && { echo "⚠ Folder .venv jest niepełny — tworzę od nowa"; rm -rf .venv; }
  echo "▶ Tworzę środowisko .venv ($("${PY[@]}" --version 2>&1))"
  "${PY[@]}" -m venv .venv
  VPY="$(venv_python)"
  if [ -z "$VPY" ]; then
    echo "❌ Nie udało się utworzyć .venv (Linux: zainstaluj pakiet python3.13-venv)"
    exit 1
  fi
fi

# Zależności: przy pierwszym uruchomieniu i po każdej zmianie requirements.txt
STAMP=.venv/.slad-requirements.txt
if ! cmp -s requirements.txt "$STAMP"; then
  echo "▶ Instaluję zależności (pierwsze uruchomienie: kilka minut)…"
  "$VPY" -m pip install --upgrade pip -q
  if ! "$VPY" -m pip install -r requirements.txt; then
    echo "❌ Instalacja zależności nie powiodła się — wyślij zespołowi cały log powyżej"
    exit 1
  fi
  cp requirements.txt "$STAMP"
fi

if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi

# Modele twarzy (YuNet + SFace, ≈40 MB) — bez nich działa wszystko poza rozpoznawaniem twarzy
if [ ! -f models/face_detection_yunet_2023mar.onnx ] || [ ! -f models/face_recognition_sface_2021dec.onnx ]; then
  echo "▶ Pobieram modele twarzy…"
  "$VPY" -m slad models || echo "⚠ Nie udało się pobrać modeli twarzy — rozpoznawanie twarzy wyłączone (ponów: ./start.sh models)"
fi

if [ $# -eq 0 ]; then
  exec "$VPY" -m slad ui
fi
exec "$VPY" -m slad "$@"
