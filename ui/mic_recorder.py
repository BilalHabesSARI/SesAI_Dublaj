"""Taşınabilir 48 kHz Mono Mikrofon Oyunculuk Kaydedicisi (./_temp/ içine yazar).

Özellikler:
- Kullanıcının mikrofondan yaptığı gerçek oyunculuk kaydını (fısıltı, vurgu,
  öfke, alay, nefes ritmi) 48000 Hz Mono 16-bit PCM WAV olarak doğrudan
  `./_temp/mikrofon_kayit/` klasörüne kaydeder.
- Çift motorlu donanım desteği:
  1. `sounddevice` (PortAudio / Windows WASAPI-MME)
  2. `PySide6.QtMultimedia` (`QMediaDevices`, `QAudioSource`, `QAudioFormat`)
- Bilgisayarda bağlı/etkin mikrofon yoksa çökmez; kullanıcıya net Türkçe uyarı döndürür.
"""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import QByteArray, QBuffer, QIODevice, QObject, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

from core.portable_env import make_temp_subdir


class MicrophoneNotAvailableError(RuntimeError):
    """Bilgisayarda kayıt yapılabilir bir mikrofon bulunamadığında fırlatılır."""


class PortableMicRecorder(QObject):
    """48 kHz Mono WAV mikrofon oyunculuk kaydedicisi."""

    status_changed = Signal(str)

    SAMPLE_RATE: int = 48000
    CHANNELS: int = 1

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._is_recording = False
        self._backend_used = ""
        self._start_time = 0.0
        self._output_wav_path: Optional[Path] = None

        # sounddevice durumu
        self._sd_stream = None
        self._sd_chunks: list[np.ndarray] = []
        self._sd_lock = threading.Lock()

        # QtMultimedia durumu
        self._qt_audio_source: Optional[QAudioSource] = None
        self._qt_buffer: Optional[QBuffer] = None
        self._qt_byte_array: Optional[QByteArray] = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @staticmethod
    def detect_microphone() -> tuple[bool, str]:
        """Sistemde kullanılabilir bir mikrofon olup olmadığını tespit eder.

        Returns:
            (mikrofon_var_mi, cihaz_adi_veya_uyari_mesaji)
        """
        # 1. sounddevice (PortAudio) kontrolü
        try:
            import sounddevice as sd  # type: ignore

            devs = sd.query_devices()
            default_in = sd.default.device[0] if sd.default.device else None
            if default_in is not None and int(default_in) >= 0:
                info = sd.query_devices(default_in, "input")
                if int(info.get("max_input_channels", 0)) > 0:
                    return True, str(info.get("name") or "Varsayılan Mikrofon")
            for d in devs:
                if isinstance(d, dict) and int(d.get("max_input_channels", 0)) > 0:
                    return True, str(d.get("name") or "Ses Giriş Cihazı")
        except Exception:
            pass

        # 2. PySide6 QtMultimedia kontrolü
        try:
            qt_inputs = QMediaDevices.audioInputs()
            if qt_inputs:
                default_dev = QMediaDevices.defaultAudioInput()
                desc = (
                    default_dev.description()
                    if not default_dev.isNull()
                    else qt_inputs[0].description()
                )
                return True, str(desc or "Qt Mikrofon")
        except Exception:
            pass

        return (
            False,
            "Bilgisayarda bağlı veya etkin bir mikrofon algılanamadı!\n\n"
            "• Lütfen mikrofonunuzun takılı olduğundan ve Windows Ses Ayarları'nda "
            "mikrofon erişimine izin verildiğinden emin olun.\n"
            "• Alternatif olarak telefonunuzdan veya başka bir cihazdan kaydettiğiniz "
            "sesi '📂 Ses Yükle (STS)' butonuyla doğrudan yükleyebilirsiniz.",
        )

    def start_recording(self) -> Path:
        """./_temp/mikrofon_kayit/ altında 48 kHz mono oyunculuk kaydını başlatır."""
        if self._is_recording:
            raise RuntimeError("Mikrofon kaydı zaten devam ediyor.")

        has_mic, mic_info = self.detect_microphone()
        if not has_mic:
            raise MicrophoneNotAvailableError(mic_info)

        rec_dir = make_temp_subdir("mikrofon_kayit")
        self._output_wav_path = (
            rec_dir / f"oyunculuk_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        )
        self._start_time = time.monotonic()

        # Önce sounddevice (48 kHz float32) dene
        try:
            import sounddevice as sd  # type: ignore

            with self._sd_lock:
                self._sd_chunks.clear()

            def _sd_callback(indata, _frames, _time_info, _status):
                with self._sd_lock:
                    self._sd_chunks.append(indata.copy())

            self._sd_stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype="float32",
                callback=_sd_callback,
            )
            self._sd_stream.start()
            self._is_recording = True
            self._backend_used = "sounddevice"
            self.status_changed.emit(f"🔴 Mikrofon kaydı başladı ({mic_info}, 48 kHz Mono)...")
            return self._output_wav_path
        except Exception:
            self._sd_stream = None

        # Yedek: PySide6 QtMultimedia QAudioSource (48 kHz Int16 Mono)
        qt_inputs = QMediaDevices.audioInputs()
        if not qt_inputs:
            raise MicrophoneNotAvailableError(mic_info)

        dev = QMediaDevices.defaultAudioInput()
        if dev.isNull():
            dev = qt_inputs[0]

        fmt = QAudioFormat()
        fmt.setSampleRate(self.SAMPLE_RATE)
        fmt.setChannelCount(self.CHANNELS)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        self._qt_byte_array = QByteArray()
        self._qt_buffer = QBuffer(self._qt_byte_array, self)
        self._qt_buffer.open(QIODevice.OpenModeFlag.WriteOnly)

        self._qt_audio_source = QAudioSource(dev, fmt, self)
        self._qt_audio_source.start(self._qt_buffer)
        self._is_recording = True
        self._backend_used = "qt_multimedia"
        self.status_changed.emit(
            f"🔴 Mikrofon kaydı başladı ({dev.description()}, 48 kHz Mono)..."
        )
        return self._output_wav_path

    def stop_recording(self) -> Path:
        """Aktif mikrofon kaydını durdurur ve 48 kHz mono WAV dosyasının yolunu döndürür."""
        if not self._is_recording or self._output_wav_path is None:
            raise RuntimeError("Durdurulacak aktif bir mikrofon kaydı yok.")

        out_path = self._output_wav_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._is_recording = False

        try:
            if self._backend_used == "sounddevice" and self._sd_stream is not None:
                try:
                    self._sd_stream.stop()
                    self._sd_stream.close()
                finally:
                    self._sd_stream = None

                with self._sd_lock:
                    if self._sd_chunks:
                        audio_np = np.concatenate(self._sd_chunks, axis=0)
                    else:
                        audio_np = np.zeros((0, 1), dtype=np.float32)
                    self._sd_chunks.clear()

                if audio_np.ndim > 1:
                    audio_np = audio_np.squeeze(-1)
                if audio_np.size < int(self.SAMPLE_RATE * 0.15):
                    raise RuntimeError(
                        "Mikrofon kaydı çok kısa (<0.15 sn) veya ses verisi alınamadı."
                    )
                sf.write(str(out_path), audio_np, self.SAMPLE_RATE, subtype="PCM_16")
                self.status_changed.emit(
                    f"✅ Mikrofon kaydı tamamlandı ({len(audio_np) / self.SAMPLE_RATE:.2f} sn)."
                )
                return out_path

            if self._backend_used == "qt_multimedia" and self._qt_audio_source is not None:
                try:
                    self._qt_audio_source.stop()
                    if self._qt_buffer is not None:
                        self._qt_buffer.close()
                finally:
                    self._qt_audio_source = None

                raw_bytes = bytes(self._qt_byte_array.data()) if self._qt_byte_array else b""
                self._qt_byte_array = None
                self._qt_buffer = None

                pcm_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
                if pcm_i16.size < int(self.SAMPLE_RATE * 0.15):
                    raise RuntimeError(
                        "Mikrofon kaydı çok kısa (<0.15 sn) veya ses verisi alınamadı."
                    )
                audio_f32 = (pcm_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
                sf.write(str(out_path), audio_f32, self.SAMPLE_RATE, subtype="PCM_16")
                self.status_changed.emit(
                    f"✅ Mikrofon kaydı tamamlandı ({len(audio_f32) / self.SAMPLE_RATE:.2f} sn)."
                )
                return out_path

            raise RuntimeError("Mikrofon kayıt arka ucu başlatılamadı.")
        finally:
            self._backend_used = ""
