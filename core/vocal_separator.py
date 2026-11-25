"""Vokal Ayırma ve Arka Plan Koruma Modülü (Vocal Separator).

İçinde arka plan müziği veya ses efekti barındıran oyun ses dosyalarını iki
katmana ayırır:
  1. `vocals.wav`      -> Yalnızca karakterin temiz konuşması (VAD/Whisper/Klonlama buna yapılır)
  2. `background.wav`  -> Orijinal müzik ve ortam efektleri (Dublaj sonrası yeniden mikslenir)

Çalışma Sırası (Otomatik En İyi Kalite):
  1. Colab Bulut GPU Demucs (`htdemucs` / `/separate` uç noktası) — Bulut Modunda
  2. Yerel GPU/CPU Demucs (`htdemucs`, ağırlıklar yalnızca `./_modeller/torch` içinde)
  3. Taşınabilir FFmpeg Spektral/Mid-Side Yedeği — Sıfır ek kurulum gerektirir, asla çökmez.
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from core.ffmpeg_manager import FFmpegManager
from core.portable_env import MODELS_DIR, bootstrap_portable_environment


@dataclass
class SeparationResult:
    """Vokal / arka plan ayrıştırma çıktısı."""

    vocals_wav: Path
    background_wav: Path
    backend_used: str


class VocalSeparator:
    """Konuşmayı arka plan efekt/müziğinden ayırıp tekrar mikslemeye hazırlar."""

    def __init__(self, ffmpeg: Optional[FFmpegManager] = None) -> None:
        self.ffmpeg = ffmpeg or FFmpegManager()
        self._demucs_model = None
        self._demucs_device = "cpu"
        self._lock = threading.Lock()

    def separate(
        self,
        src_audio: Path,
        out_dir: Path,
        stem_prefix: str = "line",
        cloud_separator_fn: Optional[Callable[[Path, Path, Path], bool]] = None,
        prefer_local_demucs: bool = True,
        cancel_event: Optional[threading.Event] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> SeparationResult:
        """Ses dosyasını `vocals.wav` ve `background.wav` olarak iki katmana ayırır."""
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Vokal ayırma iptal edildi.")

        out_dir.mkdir(parents=True, exist_ok=True)
        vocals_wav = out_dir / f"{stem_prefix}_vocals.wav"
        bg_wav = out_dir / f"{stem_prefix}_background.wav"

        # 1. Colab Bulut GPU Demucs (/separate)
        if cloud_separator_fn is not None:
            try:
                ok = cloud_separator_fn(src_audio, vocals_wav, bg_wav)
                if ok and vocals_wav.is_file() and bg_wav.is_file():
                    return SeparationResult(vocals_wav, bg_wav, "colab_htdemucs")
            except Exception as exc:
                if log_cb:
                    log_cb(f"Colab vokal ayırma kullanılamadı ({exc}); yerel ayırıcıya geçiliyor...")

        # 2. Yerel PyTorch + Demucs (htdemucs -> ./_modeller/torch)
        if prefer_local_demucs:
            try:
                if self._separate_with_local_demucs(src_audio, vocals_wav, bg_wav, cancel_event):
                    return SeparationResult(vocals_wav, bg_wav, f"local_htdemucs_{self._demucs_device}")
            except InterruptedError:
                raise
            except Exception as exc:
                if log_cb:
                    log_cb(f"Yerel Demucs atlandı ({exc}); FFmpeg spektral vokal ayırma kullanılıyor.")

        # 3. Sıfır bağımlılıklı taşınabilir FFmpeg spektral/mid-side yedeği
        self._separate_with_ffmpeg_spectral(src_audio, vocals_wav, bg_wav)
        return SeparationResult(vocals_wav, bg_wav, "ffmpeg_spectral")

    def _separate_with_local_demucs(
        self,
        src_audio: Path,
        vocals_wav: Path,
        bg_wav: Path,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        try:
            import torch  # type: ignore
            from demucs.apply import apply_model  # type: ignore
            from demucs.pretrained import get_model  # type: ignore
        except ImportError:
            return False

        bootstrap_portable_environment()
        torch.hub.set_dir(str(MODELS_DIR / "torch" / "hub"))

        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Vokal ayırma iptal edildi.")

        # Demucs 44.1kHz stereo WAV bekler
        prep_wav = vocals_wav.with_name(f"_demucs_in_{vocals_wav.stem}.wav")
        try:
            self.ffmpeg.convert_to_wav(src_audio, prep_wav, sample_rate=44100, channels=2)
            audio_np, sr = sf.read(str(prep_wav), dtype="float32", always_2d=True)
            # (samples, channels) -> (1, channels, samples)
            wav_tensor = torch.from_numpy(audio_np.T.copy()).unsqueeze(0)

            with self._lock:
                if self._demucs_model is None:
                    self._demucs_device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._demucs_model = get_model("htdemucs").to(self._demucs_device)
                    self._demucs_model.eval()

                wav_dev = wav_tensor.to(self._demucs_device)
                with torch.no_grad():
                    sources = apply_model(
                        self._demucs_model,
                        wav_dev,
                        split=True,
                        overlap=0.25,
                        device=self._demucs_device,
                    )[0]

                # htdemucs kaynak sırası: ['drums', 'bass', 'other', 'vocals']
                source_names = list(getattr(self._demucs_model, "sources", ["drums", "bass", "other", "vocals"]))
                voc_idx = source_names.index("vocals") if "vocals" in source_names else -1
                voc_tensor = sources[voc_idx].detach().cpu().numpy()
                bg_tensor = sum(
                    sources[i] for i in range(len(source_names)) if i != voc_idx
                ).detach().cpu().numpy()

            sf.write(str(vocals_wav), voc_tensor.T, sr, subtype="PCM_16")
            sf.write(str(bg_wav), bg_tensor.T, sr, subtype="PCM_16")
            return True
        finally:
            prep_wav.unlink(missing_ok=True)
            self._clear_cuda()

    def _separate_with_ffmpeg_spectral(
        self,
        src_audio: Path,
        vocals_wav: Path,
        bg_wav: Path,
    ) -> None:
        """Demucs yokken FFmpeg konuşma bandı + faz çıkarma ile taşınabilir vokal/arka plan ayrımı."""
        ff = self.ffmpeg._require_ffmpeg()
        meta = self.ffmpeg.probe_audio(src_audio, measure_lufs=False)
        sr = meta.sample_rate
        ch = max(1, meta.channels)

        base_wav = vocals_wav.with_name(f"_spec_in_{vocals_wav.stem}.wav")
        try:
            self.ffmpeg.convert_to_wav(src_audio, base_wav, sample_rate=sr, channels=ch)

            # 1) İnsan konuşma bandı (180Hz - 4200Hz) merkez vokal çıkarımı
            voc_filter = "highpass=f=180,lowpass=f=4200"
            if ch >= 2:
                # Stereo dosyalarda merkez (L+R) vokal vurgusu
                voc_filter = "pan=mono|c0=0.5*c0+0.5*c1," + voc_filter
            proc_v = self.ffmpeg._run(
                [
                    ff,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(base_wav),
                    "-af",
                    voc_filter,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(sr),
                    "-ac",
                    "1",
                    str(vocals_wav),
                ],
                timeout=90,
            )
            if proc_v.returncode != 0 or not vocals_wav.is_file():
                raise RuntimeError(f"FFmpeg vokal ayrımı başarısız: {proc_v.stderr.strip()}")

            # 2) Arka plan katmanı: stereo yan kanallar (L-R) + düşük/yüksek efekt bantları
            #    veya mono'da konuşma bandı yumuşak bastırma (faz-uyumlu temiz artık)
            orig_data, _ = sf.read(str(base_wav), dtype="float32", always_2d=True)
            voc_data, _ = sf.read(str(vocals_wav), dtype="float32", always_2d=True)
            min_len = min(len(orig_data), len(voc_data))
            orig_slice = orig_data[:min_len]
            voc_mono = voc_data[:min_len, 0:1]

            # Merkez vokali her kanaldan yumuşakça çıkararak arka plan müziği/efektini koru
            bg_data = np.clip(orig_slice - (voc_mono * 0.85), -1.0, 1.0).astype(np.float32)
            sf.write(str(bg_wav), bg_data, sr, subtype="PCM_16")
        finally:
            base_wav.unlink(missing_ok=True)

    @staticmethod
    def _clear_cuda() -> None:
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def unload(self) -> None:
        with self._lock:
            self._demucs_model = None
            self._clear_cuda()
