"""SesAI Dublaj Stüdyosu — Taşınabilir Ana Giriş Noktası.

DİKKAT: `bootstrap_portable_environment()` diğer tüm kütüphanelerden (PySide6,
onnxruntime, soundfile vb.) ÖNCE çalıştırılarak %APPDATA%, %LOCALAPPDATA% ve
sistem Temp dizinlerine tek bir dosya bile yazılmaması garanti altına alınır.
"""

from __future__ import annotations

import sys

# 1. İLK ADIM: Tüm ortam değişkenlerini ./_temp, ./_modeller ve ./_veri'ye kilitle
from core.portable_env import bootstrap_portable_environment, cleanup_temp_dir

bootstrap_portable_environment()

# 2. Yalnızca taşınabilir ortam kilitlendikten sonra PySide6 ve diğer modülleri yükle
from PySide6.QtWidgets import QApplication

from engines.colab_notebook_generator import DEFAULT_NOTEBOOK_PATH, generate_colab_notebook
from ui.main_window import MainWindow
from ui.theme import DARK_THEME_QSS


def main() -> int:
    # Uygulama kökünde colab_motoru.ipynb yoksa ilk açılışta otomatik üret
    if not DEFAULT_NOTEBOOK_PATH.is_file():
        try:
            generate_colab_notebook(DEFAULT_NOTEBOOK_PATH)
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("SesAI Dublaj Stüdyosu")
    app.setStyleSheet(DARK_THEME_QSS)

    window = MainWindow()
    window.show()

    try:
        return app.exec()
    finally:
        cleanup_temp_dir(keep_root=True)


if __name__ == "__main__":
    raise SystemExit(main())
