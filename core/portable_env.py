"""%100 Taşınabilir Ortam ve Geçici Dosya Yönetim Modülü.

BU MODÜL diğer tüm üçüncü parti kütüphanelerden (PySide6, torch, onnxruntime,
faster-whisper, TTS, google.genai vb.) ÖNCE çalıştırılmalıdır.

Kurallar:
- Windows %APPDATA%, %LOCALAPPDATA%, Kayıt Defteri (Registry) veya sistem Temp
  klasörlerine asla dosya yazılmaz.
- TEMP, TMP, TMPDIR, HF_HOME, TORCH_HOME, TTS_HOME, XDG_* ve diğer önbellek
  değişkenleri uygulamanın kendi dizinindeki ./_temp, ./_modeller ve ./_veri
  klasörlerine yönlendirilir.
- İşlem bittiğinde, iptal edildiğinde, hata oluştuğunda veya pencere kapatıldığında
  ./_temp içeriği otomatik ve eksiksiz olarak temizlenir.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

APP_ROOT: Path = Path(__file__).resolve().parent.parent
TEMP_DIR: Path = APP_ROOT / "_temp"
MODELS_DIR: Path = APP_ROOT / "_modeller"
BACKUPS_DIR: Path = APP_ROOT / "_yedekler"
DATA_DIR: Path = APP_ROOT / "_veri"
TOOLS_DIR: Path = APP_ROOT / "_araclar"

_BOOTSTRAPPED = False
_CLEANUP_LOCK = threading.Lock()
_PRE_CLEANUP_HOOKS: list[Callable[[], None]] = []


def ensure_portable_dirs() -> None:
    """Uygulamanın tüm taşınabilir dizinlerini kendi kökü altında oluşturur."""
    for folder in (
        TEMP_DIR,
        MODELS_DIR,
        MODELS_DIR / "vad",
        MODELS_DIR / "huggingface",
        MODELS_DIR / "huggingface" / "hub",
        MODELS_DIR / "torch",
        MODELS_DIR / "coqui_tts",
        MODELS_DIR / "whisper",
        BACKUPS_DIR,
        DATA_DIR,
        TOOLS_DIR,
        TOOLS_DIR / "ffmpeg",
    ):
        folder.mkdir(parents=True, exist_ok=True)


def register_pre_cleanup_hook(callback: Callable[[], None]) -> None:
    """_temp temizlenmeden hemen önce çağrılacak serbest bırakma kancası ekler.

    Özellikle Windows'ta ses oynatıcının (QMediaPlayer) açık tuttuğu dosya
    tanıtıcılarını serbest bırakmak için kullanılır.
    """
    if callback not in _PRE_CLEANUP_HOOKS:
        _PRE_CLEANUP_HOOKS.append(callback)


def unregister_pre_cleanup_hook(callback: Callable[[], None]) -> None:
    """Daha önce kaydedilen temizlik öncesi kancayı güvenle kaldırır."""
    try:
        _PRE_CLEANUP_HOOKS.remove(callback)
    except ValueError:
        pass


def _handle_remove_readonly(func, path, _exc_info) -> None:
    """Salt-okunur dosya özniteliğini kaldırıp silmeyi tekrar dener."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def cleanup_temp_dir(keep_root: bool = True, retries: int = 3) -> tuple[int, list[str]]:
    """./_temp klasöründeki tüm geçici dosya ve alt dizinleri temizler.

    Returns:
        (silinen_oge_sayisi, silinemeyen_yollar)
    """
    with _CLEANUP_LOCK:
        for hook in list(_PRE_CLEANUP_HOOKS):
            try:
                hook()
            except Exception:
                pass

        if not TEMP_DIR.exists():
            if keep_root:
                TEMP_DIR.mkdir(parents=True, exist_ok=True)
            return 0, []

        deleted = 0
        failed: list[str] = []

        # Alt öğeleri listele ve sil
        entries = list(TEMP_DIR.iterdir())
        for entry in entries:
            removed = False
            for attempt in range(retries):
                try:
                    if entry.is_dir() and not entry.is_symlink():
                        shutil.rmtree(entry, onerror=_handle_remove_readonly)
                    else:
                        try:
                            os.chmod(entry, stat.S_IWRITE)
                        except OSError:
                            pass
                        entry.unlink(missing_ok=True)
                    removed = True
                    deleted += 1
                    break
                except OSError:
                    if attempt < retries - 1:
                        time.sleep(0.15)
            if not removed and entry.exists():
                failed.append(str(entry))

        if keep_root:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)

        return deleted, failed


def make_temp_subdir(prefix: str = "oturum") -> Path:
    """./_temp altında izole bir çalışma alt klasörü oluşturur."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in prefix)
    subdir = TEMP_DIR / f"{safe_prefix}_{uuid.uuid4().hex[:8]}"
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def _atexit_cleanup() -> None:
    """Program çıkışında _temp klasörünü sessizce temizler."""
    try:
        cleanup_temp_dir(keep_root=True, retries=2)
    except Exception:
        pass


def _signal_handler(signum, _frame) -> None:
    """Sonlandırma sinyallerinde _temp klasörünü temizler."""
    try:
        cleanup_temp_dir(keep_root=True, retries=1)
    finally:
        raise SystemExit(128 + signum)


def bootstrap_portable_environment() -> dict[str, str]:
    """Tüm ortam değişkenlerini uygulamanın kendi klasörüne yönlendirir.

    Returns:
        Ayarlanan taşınabilir ortam değişkenleri sözlüğü.
    """
    global _BOOTSTRAPPED

    ensure_portable_dirs()

    portable_vars: dict[str, str] = {
        "TEMP": str(TEMP_DIR),
        "TMP": str(TEMP_DIR),
        "TMPDIR": str(TEMP_DIR),
        "HF_HOME": str(MODELS_DIR / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(MODELS_DIR / "huggingface" / "hub"),
        "TORCH_HOME": str(MODELS_DIR / "torch"),
        "TTS_HOME": str(MODELS_DIR / "coqui_tts"),
        "XDG_CACHE_HOME": str(TEMP_DIR / "xdg_cache"),
        "XDG_CONFIG_HOME": str(DATA_DIR / "xdg_config"),
        "XDG_DATA_HOME": str(MODELS_DIR / "xdg_data"),
        "NUMBA_CACHE_DIR": str(TEMP_DIR / "numba_cache"),
        "MPLCONFIGDIR": str(TEMP_DIR / "matplotlib"),
        "PIP_CACHE_DIR": str(TEMP_DIR / "pip_cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "COQUI_TOS_AGREED": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }

    for key, value in portable_vars.items():
        os.environ[key] = value

    # Python tempfile modülünün önbelleklenmiş dizinini de zorla güncelle
    tempfile.tempdir = str(TEMP_DIR)

    # huggingface_hub veya torch daha önce import edildiyse modül içi sabitlerini de güncelle
    try:
        import sys as _sys

        if "huggingface_hub.constants" in _sys.modules:
            _hfc = _sys.modules["huggingface_hub.constants"]
            _hfc.HF_HOME = portable_vars["HF_HOME"]
            _hfc.HF_HUB_CACHE = portable_vars["HUGGINGFACE_HUB_CACHE"]
            _hfc.HUGGINGFACE_HUB_CACHE = portable_vars["HUGGINGFACE_HUB_CACHE"]
        if "torch.hub" in _sys.modules:
            _sys.modules["torch.hub"].set_dir(str(MODELS_DIR / "torch" / "hub"))
    except Exception:
        pass

    if not _BOOTSTRAPPED:
        atexit.register(_atexit_cleanup)
        # Yalnızca ana iş parçacığında sinyal yakalayıcı kaydet
        if threading.current_thread() is threading.main_thread():
            for sig_name in ("SIGINT", "SIGTERM"):
                sig = getattr(signal, sig_name, None)
                if sig is not None:
                    try:
                        signal.signal(sig, _signal_handler)
                    except (ValueError, OSError):
                        pass
        _BOOTSTRAPPED = True

    return portable_vars


def get_temp_size_bytes() -> int:
    """./_temp klasörünün toplam boyutunu bayt cinsinden döndürür."""
    if not TEMP_DIR.exists():
        return 0
    total = 0
    for path in TEMP_DIR.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def verify_portable_isolation() -> dict[str, bool]:
    """Taşınabilir ortam değişkenlerinin proje kökü içinde olduğunu doğrular."""
    root_resolved = APP_ROOT.resolve()
    checked_keys = ("TEMP", "TMP", "TMPDIR", "HF_HOME", "TORCH_HOME", "TTS_HOME")
    results: dict[str, bool] = {}
    for key in checked_keys:
        raw = os.environ.get(key, "")
        if not raw:
            results[key] = False
            continue
        try:
            Path(raw).resolve().relative_to(root_resolved)
            results[key] = True
        except ValueError:
            results[key] = False
    try:
        Path(tempfile.gettempdir()).resolve().relative_to(root_resolved)
        results["tempfile.gettempdir"] = True
    except ValueError:
        results["tempfile.gettempdir"] = False
    return results
