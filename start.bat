@echo off
setlocal
REM Slad Fejka - uruchomienie (Windows: cmd / PowerShell; w Git Bash uzyj ./start.sh).
REM Pierwsze uruchomienie: tworzy .venv (Python 3.13), instaluje zaleznosci i pobiera modele twarzy.
REM   start.bat               -> interfejs graficzny http://localhost:8501
REM   start.bat demo          -> zbior syntetyczny -> przetwarzanie offline -> ewaluacja
REM   start.bat <komenda> ... -> dowolna komenda CLI, np. start.bat run data\nask   (lista: start.bat --help)
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "VPY=.venv\Scripts\python.exe"

REM .venv bez Scripts\python.exe = niepelny (przerwana instalacja) albo z innego systemu (Git Bash, macOS)
if exist ".venv" if not exist "%VPY%" (
    echo [!] Folder .venv jest niepelny - tworze od nowa
    rmdir /s /q .venv
)
if exist "%VPY%" "%VPY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul || (
    echo [!] .venv ma Pythona starszego niz 3.13 - tworze srodowisko od nowa
    rmdir /s /q .venv
)

REM Python 3.13: launcher py albo python z PATH
set "PY="
if not exist "%VPY%" (
    where py >nul 2>nul && py -3.13 -c "import sys" >nul 2>nul && set "PY=py -3.13"
)
if not exist "%VPY%" if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)" >nul 2>nul && set "PY=python"
)
if not exist "%VPY%" if not defined PY (
    echo [X] Potrzebny Python 3.13: https://www.python.org/ftp/python/3.13.17/python-3.13.17-amd64.exe
    echo     Przy instalacji zaznacz "Add python.exe to PATH", potem otworz nowy terminal.
    echo     Inne wersje: https://www.python.org/downloads/windows/   Albo Docker: docker compose up --build app
    exit /b 1
)
if not exist "%VPY%" (
    echo [i] Tworze srodowisko .venv ...
    %PY% -m venv .venv
)
if not exist "%VPY%" (
    echo [X] Nie udalo sie utworzyc .venv - uzyj Pythona z python.org, nie z Git Bash/MSYS/Cygwin.
    exit /b 1
)

REM Zaleznosci: przy pierwszym uruchomieniu i po kazdej zmianie requirements.txt
fc /b "requirements.txt" ".venv\.slad-requirements.txt" >nul 2>nul || (
    echo [i] Instaluje zaleznosci ^(pierwsze uruchomienie: kilka minut^)...
    "%VPY%" -m pip install --upgrade pip -q
    "%VPY%" -m pip install -r requirements.txt && copy /y "requirements.txt" ".venv\.slad-requirements.txt" >nul
)
fc /b "requirements.txt" ".venv\.slad-requirements.txt" >nul 2>nul || (
    echo [X] Instalacja zaleznosci nie powiodla sie - wyslij zespolowi caly log powyzej.
    exit /b 1
)

if not exist ".env" if exist ".env.example" copy ".env.example" ".env" >nul

REM Modele twarzy (YuNet + SFace, ok. 40 MB) - bez nich dziala wszystko poza rozpoznawaniem twarzy
set "NEED_MODELS="
if not exist "models\face_detection_yunet_2023mar.onnx" set "NEED_MODELS=1"
if not exist "models\face_recognition_sface_2021dec.onnx" set "NEED_MODELS=1"
if defined NEED_MODELS (
    echo [i] Pobieram modele twarzy...
    "%VPY%" -m slad models || echo [!] Nie udalo sie pobrac modeli twarzy - rozpoznawanie twarzy wylaczone ^(ponow: start.bat models^)
)

if "%~1"=="" (
    "%VPY%" -m slad ui
) else (
    "%VPY%" -m slad %*
)
exit /b %errorlevel%
