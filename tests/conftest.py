"""Pytest başlangıcında diğer tüm modüllerden önce taşınabilir ortamı kilitler."""

from __future__ import annotations

import shutil

from core.portable_env import APP_ROOT, bootstrap_portable_environment, cleanup_temp_dir

bootstrap_portable_environment()

_PYTEST_BASETEMP = APP_ROOT / "tests" / "_pytest_tmp"


def pytest_configure(config) -> None:
    if not getattr(config.option, "basetemp", None):
        config.option.basetemp = str(_PYTEST_BASETEMP)


def pytest_sessionfinish(session, exitstatus) -> None:
    shutil.rmtree(_PYTEST_BASETEMP, ignore_errors=True)
    cleanup_temp_dir(keep_root=True)
