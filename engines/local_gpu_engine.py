"""Yerel NVIDIA CUDA GPU Motoru (RTX 4070 12 GB VRAM Optimize).

Seçilebilir Açık Kaynak Ses Motorları:
  1. VoxCPM2 (48 kHz — Duygu & Ses Klonlama Önerilen, openbmb/VoxCPM2)
  2. Trendyol-TTS (VoxCPM2 Türkçe İnce Ayar — 48 kHz, Trendyol/Trendyol-TTS)
  3. Chatterbox Multilingual (Resemble AI — 24 kHz, language_id="tr",
     exaggeration ve cfg_weight kontrollü sinematik konuşma klonlama)
  4. Coqui XTTS v2 (Klasik 24 kHz çok dilli ses klonlama)

RTX 4070 12 GB VRAM Bellek Yönetimi:
- Aynı anda yalnızca TEK bir ağır GPU modeli (Whisper, VoxCPM2, Trendyol-TTS,
  Chatterbox veya XTTS v2) VRAM'de tutulur; aşama geçişlerinde ve motor
  değişimlerinde önceki model derhal serbest bırakılır (`gc.collect()` +
  `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` +
  `torch.cuda.ipc_collect()`). Böylece pik VRAM kullanımı ~4.5-6.0 GB düzeyinde
  kalarak 12 GB RTX 4070 üzerinde OOM (bellek taşması) riski sıfırlanır.
- Windows ve çoklu iş parçacıklı Qt ortamında Triton / CUDA Graph çökmesi
  yaşamamak için VoxCPM2 / Trendyol-TTS modelleri `load_denoiser=False,
  optimize=False` ile yüklenir.
- Tüm HuggingFace, Torch ve Coqui TTS ağırlıkları yalnızca `./_modeller/`
  dizinine indirilir.
"""

from __future__ import annotations

import gc
import math
import os
import random
import secrets
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from core.models import EngineHealth, EngineMode, TTSEngineType, TTSGenerationOptions
from core.portable_env import MODELS_DIR, bootstrap_portable_environment
from engines.base import SpeechEngine, TranscriptionResult


def _load_whisper_audio_16k(audio_wav_path: Path) -> np.ndarray:
    """WAV dosyasını PyAV (av.open) kullanmadan 16kHz mono float32 NumPy dizisine yükler."""
    audio, sr = sf.read(str(audio_wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != 16000 and audio.size > 0:
        try:
            from scipy.signal import resample_poly  # type: ignore

            g = math.gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // g, int(sr) // g).astype(np.float32)
        except Exception:
            ratio = 16000.0 / float(sr)
            new_len = max(1, int(round(len(audio) * ratio)))
            x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _patch_torchaudio_soundfile_fallback() -> None:
    """Yeni torchaudio sürümlerinde torchcodec eksikse soundfile yedeğini devreye alır."""
    try:
        import torch  # type: ignore
        import torchaudio  # type: ignore

        if not getattr(torchaudio.load, "_sesai_patched", False):
            _orig_ta_load = torchaudio.load

            def _safe_ta_load(filepath, *args, **kwargs):
                try:
                    return _orig_ta_load(filepath, *args, **kwargs)
                except Exception as _ta_err:
                    err_l = str(_ta_err).lower()
                    if "torchcodec" in err_l or "backend" in err_l:
                        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
                        return torch.from_numpy(data.T.copy()), int(sr)
                    raise

            _safe_ta_load._sesai_patched = True  # type: ignore[attr-defined]
            torchaudio.load = _safe_ta_load
    except Exception:
        pass


class LocalGPUNotReadyError(RuntimeError):
    """Yerel GPU bağımlılıkları veya CUDA hazır olmadığında fırlatılır."""


class LocalGPUEngine(SpeechEngine):
    """Yerel NVIDIA CUDA üzerinde çalışan Whisper + VoxCPM2 (48 kHz) / Trendyol-TTS / Chatterbox / XTTS v2 motoru."""

    mode = EngineMode.LOCAL_GPU

    def __init__(
        self,
        whisper_model_name: str = "large-v3-turbo",
        unload_between_stages: bool = True,
        default_tts_engine: TTSEngineType = TTSEngineType.VOXCPM2,
    ) -> None:
        bootstrap_portable_environment()
        self.whisper_model_name = whisper_model_name.strip() or "large-v3-turbo"
        self.unload_between_stages = bool(unload_between_stages)
        self.default_tts_engine = default_tts_engine

        self._whisper_model = None
        self._tts_model = None
        self._seed_vc_model = None
        self._active_tts_kind: Optional[TTSEngineType] = None
        self._lock = threading.Lock()

    @staticmethod
    def _detect_installed_backends() -> dict[str, bool]:
        status = {
            "torch": False,
            "cuda": False,
            "faster_whisper": False,
            "voxcpm": False,
            "seed_vc": False,
            "chatterbox_multilingual": False,
            "xtts_v2": False,
            "demucs": False,
        }
        try:
            import torch  # type: ignore

            status["torch"] = True
            status["cuda"] = bool(torch.cuda.is_available())
        except ImportError:
            pass
        try:
            import faster_whisper  # type: ignore  # noqa: F401

            status["faster_whisper"] = True
        except ImportError:
            pass
        try:
            from voxcpm import VoxCPM  # type: ignore  # noqa: F401

            status["voxcpm"] = True
        except ImportError:
            pass
        try:
            from engines.seed_vc.seed_vc_wrapper import SeedVCWrapper  # type: ignore  # noqa: F401

            status["seed_vc"] = True
        except ImportError:
            pass
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore  # noqa: F401

            status["chatterbox_multilingual"] = True
        except ImportError:
            pass
        try:
            from TTS.api import TTS  # type: ignore  # noqa: F401

            status["xtts_v2"] = True
        except ImportError:
            pass
        try:
            import demucs  # type: ignore  # noqa: F401

            status["demucs"] = True
        except ImportError:
            pass
        return status

    def _check_dependencies(
        self,
        required_tts: Optional[TTSEngineType] = None,
    ) -> tuple[bool, str, str, int, list[str], bool]:
        st = self._detect_installed_backends()
        avail_tts: list[str] = []
        if st["voxcpm"]:
            avail_tts.append(TTSEngineType.VOXCPM2.value)
            avail_tts.append(TTSEngineType.TRENDYOL_TTS.value)
        if st["chatterbox_multilingual"]:
            avail_tts.append(TTSEngineType.CHATTERBOX_MULTILINGUAL.value)
        if st["xtts_v2"]:
            avail_tts.append(TTSEngineType.XTTS_V2.value)

        missing: list[str] = []
        if not st["torch"]:
            missing.append("torch (CUDA)")
        if not st["faster_whisper"]:
            missing.append("faster-whisper")

        if required_tts in (TTSEngineType.VOXCPM2, TTSEngineType.TRENDYOL_TTS) and not st["voxcpm"]:
            missing.append("voxcpm (VoxCPM2 / Trendyol-TTS 48 kHz)")
        elif required_tts is TTSEngineType.CHATTERBOX_MULTILINGUAL and not st["chatterbox_multilingual"]:
            missing.append("chatterbox-tts (Chatterbox Multilingual)")
        elif required_tts is TTSEngineType.XTTS_V2 and not st["xtts_v2"]:
            missing.append("coqui-tts (XTTS v2)")
        elif required_tts is None and not avail_tts:
            missing.append("voxcpm, chatterbox-tts veya coqui-tts")

        if missing:
            return (
                False,
                "Yerel GPU paketleri eksik: "
                + ", ".join(missing)
                + ". Kurulum için proje klasöründeki 'yerel_gpu_kur.bat' dosyasını çalıştırın.",
                "",
                0,
                avail_tts,
                st["demucs"],
            )

        if not st["cuda"]:
            return (
                False,
                "PyTorch kurulu ancak CUDA destekli NVIDIA GPU algılanmadı. "
                "'yerel_gpu_kur.bat' ile CUDA destekli PyTorch sürümünü kurun.",
                "CPU (CUDA Yok)",
                0,
                avail_tts,
                st["demucs"],
            )

        import torch  # type: ignore

        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_mb = int(props.total_memory // (1024 * 1024))
        tts_summary = ", ".join(avail_tts)
        if st.get("seed_vc"):
            tts_summary = f"{tts_summary} + Seed-VC STS"
        return (
            True,
            f"Yerel CUDA hazır ({gpu_name}, {vram_mb} MB VRAM | Motorlar: {tts_summary})",
            gpu_name,
            vram_mb,
            avail_tts,
            st["demucs"],
        )

    def health_check(self) -> EngineHealth:
        ok, msg, gpu_name, vram_mb, avail_tts, has_demucs = self._check_dependencies(
            required_tts=self.default_tts_engine if self.default_tts_engine.is_open_source else None
        )
        return EngineHealth(
            ok=ok,
            mode=self.mode,
            message=msg,
            gpu_name=gpu_name,
            vram_total_mb=vram_mb,
            whisper_model=self.whisper_model_name,
            tts_model=", ".join(avail_tts) or "Yok",
            latency_ms=0,
            available_tts_engines=avail_tts,
            supports_vocal_separation=has_demucs,
        )

    def _clear_cuda_cache(self) -> None:
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _unload_whisper_only(self) -> None:
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            self._clear_cuda_cache()

    def _unload_tts_only(self) -> None:
        if self._tts_model is not None:
            del self._tts_model
            self._tts_model = None
            self._active_tts_kind = None
            self._clear_cuda_cache()

    def _unload_seed_vc_only(self) -> None:
        if self._seed_vc_model is not None:
            del self._seed_vc_model
            self._seed_vc_model = None
            self._clear_cuda_cache()

    def unload_tts(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        with self._lock:
            had_tts = self._tts_model is not None
            self._unload_tts_only()
            if had_tts and log_cb:
                log_cb("TTS modeli VRAM'den boşaltıldı (RTX 4070 12 GB VRAM -> Whisper hakem aşamasına hazır).")

    def unload_whisper(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        with self._lock:
            had_w = self._whisper_model is not None
            self._unload_whisper_only()
            if had_w and log_cb:
                log_cb("Whisper modeli VRAM'den boşaltıldı.")

    def unload(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        with self._lock:
            had_any = (
                (self._whisper_model is not None)
                or (self._tts_model is not None)
                or (self._seed_vc_model is not None)
            )
            self._whisper_model = None
            self._tts_model = None
            self._seed_vc_model = None
            self._active_tts_kind = None
            self._clear_cuda_cache()
            if had_any and log_cb:
                log_cb("Yerel GPU (RTX 4070 VRAM) bellek önbelleği tamamen boşaltıldı.")

    def _get_whisper(self):
        ok, msg, _, _, _, _ = self._check_dependencies(required_tts=None)
        if not ok:
            raise LocalGPUNotReadyError(msg)
        if self._whisper_model is not None:
            return self._whisper_model

        # 12 GB VRAM güvenliği: Whisper yüklenmeden önce aktif TTS ve Seed-VC modelini boşalt
        if self.unload_between_stages:
            if self._tts_model is not None:
                self._unload_tts_only()
            if self._seed_vc_model is not None:
                self._unload_seed_vc_only()

        from faster_whisper import WhisperModel  # type: ignore

        whisper_dir = MODELS_DIR / "whisper"
        whisper_dir.mkdir(parents=True, exist_ok=True)
        self._whisper_model = WhisperModel(
            self.whisper_model_name,
            device="cuda",
            compute_type="float16",
            download_root=str(whisper_dir),
        )
        return self._whisper_model

    def _get_tts_model(self, engine_kind: TTSEngineType):
        if not engine_kind.is_open_source:
            engine_kind = TTSEngineType.VOXCPM2

        ok, msg, _, _, _, _ = self._check_dependencies(required_tts=engine_kind)
        if not ok:
            raise LocalGPUNotReadyError(msg)

        if self._tts_model is not None and self._active_tts_kind is engine_kind:
            return self._tts_model

        # 12 GB VRAM koruması: önce Whisper, Seed-VC ve farklı TTS modelini tamamen boşalt
        if self.unload_between_stages:
            if self._whisper_model is not None:
                self._unload_whisper_only()
            if self._seed_vc_model is not None:
                self._unload_seed_vc_only()
        if self._tts_model is not None and self._active_tts_kind is not engine_kind:
            self._unload_tts_only()

        bootstrap_portable_environment()
        _patch_torchaudio_soundfile_fallback()

        if engine_kind.is_voxcpm:
            from voxcpm import VoxCPM  # type: ignore

            hf_cache_dir = MODELS_DIR / "huggingface" / "hub"
            hf_cache_dir.mkdir(parents=True, exist_ok=True)
            # Windows ve çoklu iş parçacıklı Qt QThread ortamında Triton / CUDA Graph
            # çökmesi yaşanmaması için load_denoiser=False ve optimize=False zorunludur.
            self._tts_model = VoxCPM.from_pretrained(
                hf_model_id=engine_kind.voxcpm_repo_id,
                load_denoiser=False,
                optimize=False,
                cache_dir=str(hf_cache_dir),
                device="cuda",
            )
            self._active_tts_kind = engine_kind
            return self._tts_model

        if engine_kind is TTSEngineType.CHATTERBOX_MULTILINGUAL:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore

            self._tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
            self._active_tts_kind = TTSEngineType.CHATTERBOX_MULTILINGUAL
            return self._tts_model

        # Klasik Coqui XTTS v2
        os.environ["COQUI_TOS_AGREED"] = "1"
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        os.environ["TTS_HOME"] = str(MODELS_DIR / "coqui_tts")

        from TTS.api import TTS  # type: ignore

        self._tts_model = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        ).to("cuda")
        self._active_tts_kind = TTSEngineType.XTTS_V2
        return self._tts_model

    def transcribe(
        self,
        audio_wav_path: Path,
        language: Optional[str] = "en",
        cancel_event: Optional[threading.Event] = None,
    ) -> TranscriptionResult:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Yerel Whisper deşifre iptal edildi.")

        with self._lock:
            model = self._get_whisper()
            lang_arg = None if (not language or language == "auto") else language
            audio_np = _load_whisper_audio_16k(audio_wav_path)
            segments, info = model.transcribe(
                audio_np,
                language=lang_arg,
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            parts: list[str] = []
            for seg in segments:
                if cancel_event and cancel_event.is_set():
                    raise InterruptedError("Yerel Whisper deşifre iptal edildi.")
                txt = (seg.text or "").strip()
                if txt:
                    parts.append(txt)
            detected = getattr(info, "language", None) or (lang_arg or "en")
            return TranscriptionResult(text=" ".join(parts).strip(), detected_language=detected)

    def synthesize_clone(
        self,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        language: str = "tr",
        cancel_event: Optional[threading.Event] = None,
        options: Optional[TTSGenerationOptions] = None,
    ) -> Path:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Yerel GPU seslendirme iptal edildi.")

        opts = options or TTSGenerationOptions(tts_engine=self.default_tts_engine)
        target_kind = (
            opts.tts_engine
            if opts.tts_engine.is_open_source
            else TTSEngineType.VOXCPM2
        )
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        import torch  # type: ignore

        with self._lock:
            tts = self._get_tts_model(target_kind)
            try:
                # Çoklu aday (1..4) üretiminde her aday için farklı rastgele örnekleme tohumu uygula
                seed = opts.random_seed
                if seed is None and int(opts.candidate_index) > 1:
                    seed = secrets.randbits(31) or (10007 * int(opts.candidate_index))
                if seed is not None:
                    safe_seed = int(seed) & 0x7FFFFFFF
                    random.seed(safe_seed)
                    np.random.seed(safe_seed % (2**32 - 1))
                    torch.manual_seed(safe_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(safe_seed)

                if target_kind.is_voxcpm:
                    # VoxCPM2 Controllable Voice Cloning (48 kHz):
                    # reference_wav_path + f"({acting_prompt}){tr_text}"
                    target_text = opts.format_voxcpm_text(text)
                    with torch.inference_mode():
                        wav_np = tts.generate(
                            text=target_text,
                            reference_wav_path=str(speaker_wav_path),
                            cfg_value=float(opts.voxcpm_cfg_value),
                            inference_timesteps=int(opts.voxcpm_timesteps),
                            normalize=False,
                            denoise=False,
                        )
                    wav_arr = np.asarray(wav_np, dtype=np.float32)
                    if wav_arr.ndim > 1:
                        wav_arr = wav_arr.squeeze(0)
                    sr = int(
                        getattr(
                            getattr(tts, "tts_model", None),
                            "sample_rate",
                            getattr(tts, "sample_rate", 48000),
                        )
                    )
                    sf.write(str(output_wav_path), wav_arr, sr, subtype="PCM_16")
                elif target_kind is TTSEngineType.CHATTERBOX_MULTILINGUAL:
                    with torch.inference_mode():
                        wav_tensor = tts.generate(
                            text=text,
                            language_id=language or "tr",
                            audio_prompt_path=str(speaker_wav_path),
                            exaggeration=float(opts.exaggeration),
                            cfg_weight=float(opts.cfg_weight),
                        )
                    wav_np = wav_tensor.detach().cpu().float().numpy()
                    if wav_np.ndim > 1:
                        wav_np = wav_np.squeeze(0)
                    sr = int(getattr(tts, "sr", 24000))
                    sf.write(str(output_wav_path), wav_np, sr, subtype="PCM_16")
                else:
                    tts.tts_to_file(
                        text=text,
                        speaker_wav=str(speaker_wav_path),
                        language=language or "tr",
                        file_path=str(output_wav_path),
                        split_sentences=True,
                    )
            finally:
                # Her satır sonunda parçalı CUDA bellek havuzunu rahatlat (12 GB VRAM kararlılığı)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()

        if not output_wav_path.is_file() or output_wav_path.stat().st_size <= 44:
            raise RuntimeError(f"Yerel {target_kind.label_tr} geçerli bir WAV çıktısı üretemedi.")
        return output_wav_path

    def _get_seed_vc(self):
        st = self._detect_installed_backends()
        if not st["torch"] or not st["cuda"]:
            raise LocalGPUNotReadyError(
                "Yerel GPU Ses Dönüştürme (Seed-VC) için CUDA destekli NVIDIA GPU gereklidir."
            )
        if not st["seed_vc"]:
            raise LocalGPUNotReadyError("Seed-VC modülü yüklenemedi.")
        if self._seed_vc_model is not None:
            return self._seed_vc_model

        # 12 GB RTX 4070 VRAM güvenliği: Seed-VC yüklenmeden önce Whisper ve TTS modellerini boşalt
        if self._whisper_model is not None:
            self._unload_whisper_only()
        if self._tts_model is not None:
            self._unload_tts_only()

        bootstrap_portable_environment()
        _patch_torchaudio_soundfile_fallback()
        from engines.seed_vc.seed_vc_wrapper import SeedVCWrapper

        self._seed_vc_model = SeedVCWrapper(device="cuda")
        return self._seed_vc_model

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
        """Kullanıcının mikrofon/dosya oyunculuk sesini Seed-VC (Sıfır Eğitimli DiT) ile hedef karakter sesine dönüştürür."""
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Yerel GPU ses dönüştürme (Seed-VC) iptal edildi.")

        opts = options or TTSGenerationOptions(tts_engine=self.default_tts_engine)
        steps = max(10, min(50, int(diffusion_steps or opts.sts_diffusion_steps or 25)))
        cfg_rate = float(getattr(opts, "sts_cfg_rate", 0.70) or 0.70)
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            seed_vc = self._get_seed_vc()
            try:
                seed_vc.convert_voice_to_file(
                    source_wav_path=source_acting_wav,
                    target_wav_path=target_speaker_wav,
                    output_wav_path=output_wav_path,
                    diffusion_steps=steps,
                    auto_f0_adjust=bool(auto_f0_adjust),
                    pitch_shift=int(pitch_shift),
                    f0_condition=True,
                    inference_cfg_rate=cfg_rate,
                )
            finally:
                # İşlem biter bitmez RTX 4070 VRAM'ini (synchronize + empty_cache + ipc_collect) tamamen boşalt
                self._unload_seed_vc_only()

        if not output_wav_path.is_file() or output_wav_path.stat().st_size <= 44:
            raise RuntimeError("Yerel Seed-VC (Speech-to-Speech) geçerli bir WAV çıktısı üretemedi.")
        return output_wav_path
