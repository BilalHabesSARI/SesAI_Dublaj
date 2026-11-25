"""Taşınabilir ortam değişkenleri ve otomatik ./_temp temizliği testleri."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from core.portable_env import (
    APP_ROOT,
    MODELS_DIR,
    TEMP_DIR,
    bootstrap_portable_environment,
    cleanup_temp_dir,
    make_temp_subdir,
    register_pre_cleanup_hook,
    verify_portable_isolation,
)


def test_portable_env_isolation() -> None:
    bootstrap_portable_environment()
    checks = verify_portable_isolation()
    assert all(checks.values()), f"İzolasyon başarısız: {checks}"

    assert Path(os.environ["TEMP"]).resolve() == TEMP_DIR.resolve()
    assert Path(os.environ["TMP"]).resolve() == TEMP_DIR.resolve()
    assert Path(os.environ["HF_HOME"]).resolve().is_relative_to(MODELS_DIR.resolve())
    assert Path(os.environ["TORCH_HOME"]).resolve().is_relative_to(MODELS_DIR.resolve())
    assert Path(os.environ["TTS_HOME"]).resolve().is_relative_to(MODELS_DIR.resolve())
    assert Path(tempfile.gettempdir()).resolve() == TEMP_DIR.resolve()


def test_cleanup_temp_dir_and_pre_hook() -> None:
    bootstrap_portable_environment()
    sub = make_temp_subdir("test_temizlik")
    sample = sub / "ara_dosya.wav"
    sample.write_bytes(b"RIFF0000WAVE")
    assert sample.is_file()

    hook_called = {"flag": False}
    register_pre_cleanup_hook(lambda: hook_called.__setitem__("flag", True))

    deleted, failed = cleanup_temp_dir(keep_root=True)
    assert hook_called["flag"] is True
    assert deleted >= 1
    assert failed == []
    assert TEMP_DIR.is_dir()
    assert list(TEMP_DIR.iterdir()) == []
    assert APP_ROOT.is_dir()
