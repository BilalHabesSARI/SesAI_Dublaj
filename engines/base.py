"""Ses Deşifre (Whisper), Klonlama (Chatterbox / XTTS / ElevenLabs) ve Vokal Ayırma Soyut Arayüzü."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.models import EngineHealth, EngineMode, TTSGenerationOptions


@dataclass
class TranscriptionResult:
    """Whisper / ASR deşifre çıktısı."""

    text: str
    detected_language: str = "en"


class SpeechEngine(ABC):
    """Colab Bulut, Yerel GPU ve ElevenLabs API motorları için ortak sözleşme."""

    mode: EngineMode

    @abstractmethod
    def health_check(self) -> EngineHealth:
        """Motorun erişilebilirliğini ve GPU/model/kota durumunu test eder."""

    @abstractmethod
    def transcribe(
        self,
        audio_wav_path: Path,
        language: Optional[str] = "en",
        cancel_event: Optional[threading.Event] = None,
    ) -> TranscriptionResult:
        """Konuşma dosyasını metne döker."""

    @abstractmethod
    def synthesize_clone(
        self,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        language: str = "tr",
        cancel_event: Optional[threading.Event] = None,
        options: Optional[TTSGenerationOptions] = None,
    ) -> Path:
        """Referans sesi (veya seçili ses profilini) kullanarak dublaj WAV üretir."""

    def convert_speech_to_speech(
        self,
        source_acting_wav: Path,
        target_speaker_wav: Path,
        output_wav_path: Path,
        diffusion_steps: int = 25,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        cancel_event: Optional[threading.Event] = None,
        options: Optional[TTSGenerationOptions] = None,
    ) -> Path:
        """Kullanıcının mikrofon/dosya oyunculuk sesini hedef karakter sesine dönüştürür (Seed-VC / STS)."""
        raise NotImplementedError("Bu motor üzerinde doğrudan Speech-to-Speech (STS) desteklenmiyor.")

    def separate_vocals(
        self,
        audio_path: Path,
        vocals_out: Path,
        background_out: Path,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """Motor sunucu tarafında vokal ayırma destekliyorsa (örn. Colab Demucs) uygular."""
        return False

    def unload_tts(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Yalnızca aktif TTS modelini VRAM'den boşaltır (çoklu aday üretimi sonrası Whisper hakemi öncesi)."""
        return None

    def unload_whisper(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Yalnızca aktif Whisper deşifre modelini VRAM'den boşaltır."""
        return None

    def unload(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Varsa yüklü GPU modellerini veya geçici klonlanmış API seslerini serbest bırakır."""
        return None
