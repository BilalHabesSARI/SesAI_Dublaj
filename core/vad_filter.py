"""Yerel CPU Tabanlı Silero VAD (Voice Activity Detection) Konuşma Filtresi.

Müzik, ayak sesi, silah efekti, arayüz tıklaması ve ortam seslerini insan
konuşmasından ayırır; yalnızca insan konuşması içeren dosyaları dublaj
hattına geçirir.

- Silero VAD v5 ONNX modelini tamamen yerel ./_modeller/vad/silero_vad.onnx
  üzerinden onnxruntime (CPUExecutionProvider) ile çok düşük kaynak tüketimiyle
  çalıştırır (PyTorch gerektirmez).
- Model dosyası yoksa yalnızca bir kez ./_modeller/vad/ dizinine indirir.
- İnternet yoksa ve model eksikse güvenli spektral/enerji yedek filtresine düşer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import soundfile as sf

from core.ffmpeg_manager import FFmpegManager
from core.portable_env import MODELS_DIR, TEMP_DIR

SILERO_VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
)
DEFAULT_VAD_MODEL_PATH: Path = MODELS_DIR / "vad" / "silero_vad.onnx"


@dataclass
class VADResult:
    """VAD analiz sonucu."""

    is_speech: bool
    speech_ratio: float
    max_speech_prob: float
    speech_duration_ms: int
    total_duration_ms: int
    engine_used: str


class SileroVADFilter:
    """CPU üzerinde son derece hafif çalışan Silero VAD v5 ONNX konuşma filtresi."""

    SAMPLE_RATE = 16000
    WINDOW_SAMPLES = 512  # 32 ms @ 16kHz
    CONTEXT_SAMPLES = 64  # Silero VAD v5 context penceresi

    def __init__(
        self,
        model_path: Path = DEFAULT_VAD_MODEL_PATH,
        threshold: float = 0.45,
        min_speech_ms: int = 220,
        ffmpeg: Optional[FFmpegManager] = None,
    ) -> None:
        self.model_path = model_path
        self.threshold = float(threshold)
        self.min_speech_ms = int(min_speech_ms)
        self.ffmpeg = ffmpeg or FFmpegManager()
        self._session = None
        self._backend_name = "silero_onnx"

    def ensure_model(self) -> bool:
        """Silero VAD ONNX modelinin ./_modeller/vad/ içinde hazır olmasını sağlar."""
        if self.model_path.is_file() and self.model_path.stat().st_size > 100_000:
            return True
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(SILERO_VAD_ONNX_URL)
                resp.raise_for_status()
                tmp = self.model_path.with_suffix(".onnx.tmp")
                tmp.write_bytes(resp.content)
                tmp.replace(self.model_path)
            return True
        except Exception:
            return False

    def _load_session(self):
        if self._session is not None:
            return self._session
        if not self.ensure_model():
            self._backend_name = "spectral_fallback"
            return None
        try:
            import onnxruntime as ort  # type: ignore

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._backend_name = "silero_onnx"
            return self._session
        except Exception:
            self._backend_name = "spectral_fallback"
            return None

    def analyze_file(
        self,
        audio_path: Path,
        work_dir: Optional[Path] = None,
    ) -> VADResult:
        """Herhangi bir ses dosyasını 16kHz mono WAV'a çevirip konuşma analizi yapar."""
        tmp_dir = work_dir or TEMP_DIR
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_wav = tmp_dir / f"vad_{abs(hash(str(audio_path))) % 10_000_000}.wav"
        try:
            self.ffmpeg.convert_to_wav(
                audio_path,
                tmp_wav,
                sample_rate=self.SAMPLE_RATE,
                channels=1,
            )
            samples, sr = sf.read(str(tmp_wav), dtype="float32")
            if samples.ndim > 1:
                samples = np.mean(samples, axis=1).astype(np.float32)
            return self.analyze_samples(samples, sr)
        finally:
            tmp_wav.unlink(missing_ok=True)

    def analyze_samples(self, samples: np.ndarray, sample_rate: int = 16000) -> VADResult:
        """16kHz float32 mono ses dizisi üzerinde konuşma tespiti yapar."""
        if samples.size == 0:
            return VADResult(False, 0.0, 0.0, 0, 0, self._backend_name)

        if sample_rate != self.SAMPLE_RATE:
            # Basit doğrusal yeniden örnekleme yedeği (normalde ffmpeg 16kHz verir)
            ratio = self.SAMPLE_RATE / float(sample_rate)
            new_len = max(1, int(round(len(samples) * ratio)))
            x_old = np.linspace(0.0, 1.0, len(samples), endpoint=False)
            x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
            samples = np.interp(x_new, x_old, samples).astype(np.float32)

        total_ms = int(round((len(samples) / float(self.SAMPLE_RATE)) * 1000.0))

        # Tamamen sessiz dosyayı anında pas geç
        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms < 1e-4:
            return VADResult(False, 0.0, 0.0, 0, total_ms, self._backend_name)

        session = self._load_session()
        if session is None:
            return self._spectral_fallback(samples, total_ms)

        # Silero VAD v5 ONNX akış çıkarımı (64 örnek context + 512 örnek pencere)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, self.CONTEXT_SAMPLES), dtype=np.float32)
        sr_arr = np.array(self.SAMPLE_RATE, dtype=np.int64)

        num_chunks = math.ceil(len(samples) / self.WINDOW_SAMPLES)
        padded_len = num_chunks * self.WINDOW_SAMPLES
        if padded_len > len(samples):
            samples = np.pad(samples, (0, padded_len - len(samples)), mode="constant")

        probs: list[float] = []
        for idx in range(num_chunks):
            start = idx * self.WINDOW_SAMPLES
            chunk = samples[start : start + self.WINDOW_SAMPLES].reshape(1, self.WINDOW_SAMPLES)
            model_input = np.concatenate([context, chunk], axis=1).astype(np.float32)
            out, state = session.run(
                None,
                {
                    "input": model_input,
                    "state": state,
                    "sr": sr_arr,
                },
            )
            context = model_input[:, -self.CONTEXT_SAMPLES :]
            probs.append(float(out[0][0]))

        if not probs:
            return VADResult(False, 0.0, 0.0, 0, total_ms, "silero_onnx")

        speech_frames = sum(1 for p in probs if p >= self.threshold)
        speech_ms = int(round((speech_frames * self.WINDOW_SAMPLES / float(self.SAMPLE_RATE)) * 1000.0))
        speech_ratio = speech_frames / float(len(probs))
        max_prob = max(probs)

        # Kısa seslerde (<500ms) en az 120ms, normal seslerde min_speech_ms konuşma ara
        effective_min_ms = min(self.min_speech_ms, max(96, int(total_ms * 0.25)))
        is_speech = (speech_ms >= effective_min_ms) and (max_prob >= self.threshold)

        return VADResult(
            is_speech=is_speech,
            speech_ratio=round(speech_ratio, 3),
            max_speech_prob=round(max_prob, 3),
            speech_duration_ms=speech_ms,
            total_duration_ms=total_ms,
            engine_used="silero_onnx",
        )

    def _spectral_fallback(self, samples: np.ndarray, total_ms: int) -> VADResult:
        """ONNX modeli indirilemezse insan sesi bandı (300–3400 Hz) spektral yedeği."""
        frame_len = self.WINDOW_SAMPLES
        num_frames = max(1, len(samples) // frame_len)
        speech_frames = 0
        max_score = 0.0
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / self.SAMPLE_RATE)
        voice_mask = (freqs >= 300.0) & (freqs <= 3400.0)
        high_mask = freqs > 4500.0

        for i in range(num_frames):
            frame = samples[i * frame_len : (i + 1) * frame_len]
            rms = float(np.sqrt(np.mean(frame * frame)))
            if rms < 0.01:
                continue
            spec = np.abs(np.fft.rfft(frame * np.hanning(frame_len))) ** 2
            total_e = float(np.sum(spec)) + 1e-9
            voice_ratio = float(np.sum(spec[voice_mask])) / total_e
            high_ratio = float(np.sum(spec[high_mask])) / total_e
            # İnsan konuşması 300-3400Hz bandında yoğunlaşır, aşırı yüksek gürültülü değildir
            score = max(0.0, min(1.0, voice_ratio * (1.0 - high_ratio * 0.8)))
            max_score = max(max_score, score)
            if score >= 0.55:
                speech_frames += 1

        speech_ms = int(round((speech_frames * frame_len / float(self.SAMPLE_RATE)) * 1000.0))
        ratio = speech_frames / float(num_frames)
        is_speech = speech_ms >= min(self.min_speech_ms, max(96, int(total_ms * 0.25)))
        return VADResult(
            is_speech=is_speech,
            speech_ratio=round(ratio, 3),
            max_speech_prob=round(max_score, 3),
            speech_duration_ms=speech_ms,
            total_duration_ms=total_ms,
            engine_used="spectral_fallback",
        )
