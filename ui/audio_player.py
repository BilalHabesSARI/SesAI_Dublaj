"""Güvenli Önizleme Ses Oynatıcısı (Windows Dosya Kilidi Korumalı).

PySide6 QMediaPlayer kullanır ve ./_temp temizlenmeden hemen önce açık olan
dosya tanıtıcısını (source URL) serbest bırakarak Windows'ta dosya kilidi
hatası oluşmasını önler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from shiboken6 import isValid

from core.portable_env import register_pre_cleanup_hook, unregister_pre_cleanup_hook


class AudioPreviewPlayer(QObject):
    """Orijinal ve Türkçe dublaj ses dosyaları için önizleme oynatıcısı."""

    status_changed = Signal(str)
    playback_stopped = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._audio_output = QAudioOutput(self)
        self._audio_output.setVolume(0.95)

        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.errorOccurred.connect(self._on_error)

        self._current_label: str = ""
        self._closed = False
        register_pre_cleanup_hook(self.release)
        self.destroyed.connect(self._on_destroyed)

    def _on_destroyed(self, *_args) -> None:
        self._closed = True
        unregister_pre_cleanup_hook(self.release)

    def shutdown(self) -> None:
        """Pencere kapanırken oynatıcıyı ve atexit kancasını güvenle kapatır."""
        self.release()
        self._closed = True
        unregister_pre_cleanup_hook(self.release)

    def play_file(self, path: Path, label: str = "") -> bool:
        """Belirtilen ses dosyasını çalar."""
        if self._closed or not isValid(self._player):
            return False
        if not path or not path.is_file():
            self.status_changed.emit("Ses dosyası bulunamadı.")
            return False

        self.release()
        self._current_label = label or path.name
        self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self._player.play()
        self.status_changed.emit(f"Oynatılıyor: {self._current_label}")
        return True

    def stop(self) -> None:
        self.release()
        if not self._closed and isValid(self):
            self.status_changed.emit("Oynatma durduruldu.")

    def release(self) -> None:
        """Oynatmayı durdurur ve Windows dosya kilidini hemen serbest bırakır."""
        if self._closed:
            return
        try:
            if isValid(self._player):
                self._player.stop()
                self._player.setSource(QUrl())
        except Exception:
            pass

    def _on_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.playback_stopped.emit()
            if self._current_label:
                self.status_changed.emit("Oynatma hazır.")

    def _on_error(self, _error, error_string: str) -> None:
        if error_string:
            self.status_changed.emit(f"Oynatma hatası: {error_string}")
