@echo off
setlocal EnableExtensions

set "APP_DIR=%~dp0"
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"

:: Tasinabilir ortam degiskenlerini Python baslamadan once proje icine kilitle
set "TEMP=%APP_DIR%\_temp"
set "TMP=%APP_DIR%\_temp"
set "TMPDIR=%APP_DIR%\_temp"
set "HF_HOME=%APP_DIR%\_modeller\huggingface"
set "TORCH_HOME=%APP_DIR%\_modeller\torch"
set "TTS_HOME=%APP_DIR%\_modeller\coqui_tts"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "COQUI_TOS_AGREED=1"
set "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"

if not exist "%APP_DIR%\_temp" mkdir "%APP_DIR%\_temp"
if not exist "%APP_DIR%\_modeller" mkdir "%APP_DIR%\_modeller"
if not exist "%APP_DIR%\_yedekler" mkdir "%APP_DIR%\_yedekler"
if not exist "%APP_DIR%\_veri" mkdir "%APP_DIR%\_veri"

if not exist "%APP_DIR%\.venv\Scripts\python.exe" (
    echo [HATA] Yerel sanal ortam ^(.venv^) bulunamadi!
    echo Kurulum icin proje klasorunde su komutlari calistirin:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo SesAI Dublaj Studyosu baslatiliyor...
cd /d "%APP_DIR%"
"%APP_DIR%\.venv\Scripts\python.exe" "%APP_DIR%\main.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [HATA] Uygulama beklenmedik bir hata ile kapandi ^(Kod: %EXIT_CODE%^).
    echo Yukaridaki hata mesajini inceleyebilirsiniz.
    echo.
    pause
)

endlocal & exit /b %EXIT_CODE%
