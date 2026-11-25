@echo off
setlocal EnableExtensions

set "APP_DIR=%~dp0"
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"

:: Tum gecici ve pip onbellek dosyalarini sadece proje icine yonlendir
if not exist "%APP_DIR%\_temp\pip_cache" mkdir "%APP_DIR%\_temp\pip_cache"
set "TEMP=%APP_DIR%\_temp"
set "TMP=%APP_DIR%\_temp"
set "PIP_CACHE_DIR=%APP_DIR%\_temp\pip_cache"
set "HF_HOME=%APP_DIR%\_modeller\huggingface"
set "TORCH_HOME=%APP_DIR%\_modeller\torch"
set "TTS_HOME=%APP_DIR%\_modeller\coqui_tts"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"

if not exist "%APP_DIR%\.venv\Scripts\python.exe" (
    echo [HATA] .venv bulunamadi!
    pause
    exit /b 1
)

echo ============================================================================
echo  SesAI Dublaj Studyosu - Yerel NVIDIA CUDA GPU Modu Kurulumu ^(RTX 4070 12GB^)
echo  Kurulanlar: PyTorch CUDA + faster-whisper + VoxCPM2 ^(48 kHz^) + Trendyol-TTS
echo              + Chatterbox Multilingual + Coqui XTTS v2 + Meta HTDemucs
echo  NOT: Bu kurulum opsiyoneldir. Varsayilan Colab Pro Bulut Modu veya
echo  ElevenLabs API icin bu kurulum gerekmez.
echo ============================================================================
echo.

echo [1/3] CUDA 12.4 destekli PyTorch + torchaudio kuruluyor...
"%APP_DIR%\.venv\Scripts\python.exe" -m pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [HATA] CUDA PyTorch kurulumu basarisiz oldu.
    goto :cleanup
)

echo.
echo [2/3] faster-whisper, Coqui XTTS v2, HTDemucs, VoxCPM2 ve Chatterbox bagimliliklari kuruluyor...
"%APP_DIR%\.venv\Scripts\python.exe" -m pip install --no-cache-dir -r "%APP_DIR%\requirements-local-gpu.txt"
if errorlevel 1 (
    echo [HATA] Yerel GPU ses paketleri kurulamadi.
    goto :cleanup
)

echo.
echo [3/3] VoxCPM2 ^(48 kHz^), Seed-VC ve Chatterbox Multilingual kuruluyor ^(--no-deps^)...
"%APP_DIR%\.venv\Scripts\python.exe" -m pip install --no-cache-dir --no-deps chatterbox-tts voxcpm descript-audio-codec
if errorlevel 1 (
    echo [UYARI] chatterbox-tts veya voxcpm kurulamadi; XTTS v2 ile devam edebilirsiniz.
)

echo.
echo [BASARILI] Yerel CUDA GPU modu hazir!

:cleanup
echo Gecici kurulum dosyalari temizleniyor...
rmdir /s /q "%APP_DIR%\_temp\pip_cache" 2>nul
pause
endlocal
