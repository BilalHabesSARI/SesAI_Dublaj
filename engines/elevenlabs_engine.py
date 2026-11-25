"""ElevenLabs API Sinematik Ses Motoru (Multilingual v2 TTS + Speech-to-Speech / Voice Changer).

Üst seviye sinematik oyun sahneleri için iki çalışma modu sunar:
1. Multilingual v2 TTS (`eleven_multilingual_v2`):
   - Gemini tarafından çevrilen Türkçe repliği seçilen ElevenLabs sesiyle
     veya karakterin referans WAV dosyasından anlık klonlanan (IVC) sesle üretir.
   - Oturum sonunda otomatik oluşturulan geçici IVC sesleri temizlenir.
2. Speech-to-Speech / Voice Changer (`eleven_multilingual_sts_v2`):
   - Karakterin bağırma, fısıltı, nefes, ağlama ve duygu tonlamasını birebir
     koruyarak hedef ElevenLabs sesine aktarır.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import httpx

from core.ffmpeg_manager import FFmpegManager
from core.models import EngineHealth, EngineMode, TTSEngineType, TTSGenerationOptions
from core.portable_env import TEMP_DIR
from engines.base import SpeechEngine, TranscriptionResult

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"


class ElevenLabsError(RuntimeError):
    """ElevenLabs API iletişim veya kota hatası."""


class ElevenLabsEngine(SpeechEngine):
    """ElevenLabs REST API v1 (Multilingual v2 TTS + Multilingual STS v2) istemcisi."""

    mode = EngineMode.COLAB_CLOUD

    def __init__(
        self,
        api_key: str,
        voice_id: str = "",
        tts_mode: TTSEngineType = TTSEngineType.ELEVENLABS_TTS,
        ffmpeg: Optional[FFmpegManager] = None,
        transcribe_delegate: Optional[SpeechEngine] = None,
        gemini_transcribe_fn: Optional[Callable[[Path, str, Optional[threading.Event]], str]] = None,
        base_url: str = ELEVENLABS_API_BASE,
        timeout_sec: float = 120.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.default_voice_id = voice_id.strip()
        self.tts_mode = tts_mode
        self.ffmpeg = ffmpeg or FFmpegManager()
        self.transcribe_delegate = transcribe_delegate
        self.gemini_transcribe_fn = gemini_transcribe_fn
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = max(15.0, float(timeout_sec))

        # Oturum boyunca aynı referans WAV için tekrar tekrar IVC sesi yaratmamak için önbellek
        self._ivc_cache: dict[str, str] = {}
        self._ephemeral_voice_ids: list[str] = []
        self._lock = threading.Lock()

    def _require_key(self) -> None:
        if not self.api_key:
            raise ElevenLabsError(
                "ElevenLabs API anahtarı boş! Lütfen arayüzden geçerli bir ElevenLabs API anahtarı girin."
            )

    def _headers(self, json_body: bool = False) -> dict[str, str]:
        h = {"xi-api-key": self.api_key}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _map_error(self, resp: httpx.Response, action: str) -> ElevenLabsError:
        if resp.status_code in (401, 403):
            return ElevenLabsError(
                f"ElevenLabs {action} reddedildi (HTTP {resp.status_code}): API anahtarı geçersiz veya yetkisiz."
            )
        if resp.status_code == 429:
            return ElevenLabsError(
                f"ElevenLabs {action} hız/kota sınırına takıldı (HTTP 429)."
            )
        detail = ""
        try:
            payload = resp.json()
            d = payload.get("detail")
            if isinstance(d, dict):
                detail = str(d.get("message") or d.get("status") or d)
            else:
                detail = str(d or payload)
        except Exception:
            detail = resp.text[:240]
        return ElevenLabsError(f"ElevenLabs {action} başarısız (HTTP {resp.status_code}): {detail}")

    def health_check(self) -> EngineHealth:
        try:
            self._require_key()
        except ElevenLabsError as exc:
            return EngineHealth(ok=False, mode=self.mode, message=str(exc))

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                resp = client.get(f"{self.base_url}/user", headers=self._headers())
            latency = int(round((time.perf_counter() - start) * 1000.0))
            if resp.status_code != 200:
                return EngineHealth(
                    ok=False,
                    mode=self.mode,
                    message=str(self._map_error(resp, "bağlantı testi")),
                    latency_ms=latency,
                )

            data = resp.json()
            sub = data.get("subscription") or {}
            tier = str(sub.get("tier") or "standard")
            used = int(sub.get("character_count") or 0)
            limit = int(sub.get("character_limit") or 0)
            remaining = max(0, limit - used) if limit > 0 else 0
            quota_txt = f", Kalan: {remaining:,} karakter" if limit > 0 else ""

            return EngineHealth(
                ok=True,
                mode=self.mode,
                message=f"ElevenLabs bağlantısı başarılı (Plan: {tier}{quota_txt}, {latency} ms)",
                gpu_name=f"ElevenLabs Cloud ({tier})",
                whisper_model="Colab/Yerel/Gemini",
                tts_model="eleven_multilingual_v2 + eleven_multilingual_sts_v2",
                latency_ms=latency,
                available_tts_engines=[
                    TTSEngineType.ELEVENLABS_TTS.value,
                    TTSEngineType.ELEVENLABS_STS.value,
                ],
                supports_vocal_separation=False,
            )
        except Exception as exc:
            return EngineHealth(
                ok=False,
                mode=self.mode,
                message=f"ElevenLabs bağlantı hatası: {exc}",
            )

    def list_voices(self) -> list[tuple[str, str]]:
        """Hesaptaki kullanılabilir ElevenLabs seslerini (voice_id, gorunen_ad) olarak listeler."""
        self._require_key()
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            resp = client.get(f"{self.base_url}/voices", headers=self._headers())
        if resp.status_code != 200:
            raise self._map_error(resp, "ses listesi alma")
        voices_raw = resp.json().get("voices") or []
        results: list[tuple[str, str]] = []
        for v in voices_raw:
            vid = str(v.get("voice_id") or "").strip()
            name = str(v.get("name") or vid).strip()
            cat = str(v.get("category") or "").strip()
            label = f"{name} ({cat})" if cat else name
            if vid:
                results.append((vid, label))
        return results

    def _resolve_or_clone_voice_id(
        self,
        speaker_wav_path: Path,
        options: Optional[TTSGenerationOptions],
    ) -> str:
        """Kullanıcı Voice ID seçtiyse onu kullanır; boşsa referans WAV'dan anlık klon (IVC) üretir."""
        explicit_id = (
            (options.elevenlabs_voice_id if options else "") or self.default_voice_id
        ).strip()
        if explicit_id:
            return explicit_id

        # Otomatik Anlık Ses Klonlama (Instant Voice Cloning)
        ref_bytes = speaker_wav_path.read_bytes()
        ref_hash = hashlib.sha256(ref_bytes).hexdigest()[:16]
        with self._lock:
            if ref_hash in self._ivc_cache:
                return self._ivc_cache[ref_hash]

        voice_name = f"SesAI_Temp_{ref_hash[:8]}"
        with httpx.Client(timeout=self.timeout_sec, follow_redirects=True) as client:
            resp = client.post(
                f"{self.base_url}/voices/add",
                headers=self._headers(),
                data={"name": voice_name, "description": "SesAI Dublaj gecici karakter klonu"},
                files={"files": (speaker_wav_path.name, ref_bytes, "audio/wav")},
            )
        if resp.status_code != 200:
            raise self._map_error(
                resp,
                "otomatik ses klonlama (IVC — planınız IVC desteklemiyorsa bir Voice ID seçin)",
            )
        new_id = str(resp.json().get("voice_id") or "").strip()
        if not new_id:
            raise ElevenLabsError("ElevenLabs IVC geçerli bir voice_id döndürmedi.")

        with self._lock:
            self._ivc_cache[ref_hash] = new_id
            self._ephemeral_voice_ids.append(new_id)
        return new_id

    def transcribe(
        self,
        audio_wav_path: Path,
        language: Optional[str] = "en",
        cancel_event: Optional[threading.Event] = None,
    ) -> TranscriptionResult:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Deşifre iptal edildi.")

        # 1. Öncelik: Colab veya Yerel Whisper motoru hazırsa onu kullan
        if self.transcribe_delegate is not None:
            return self.transcribe_delegate.transcribe(
                audio_wav_path, language=language, cancel_event=cancel_event
            )

        # 2. Saf ElevenLabs + Gemini modu: Gemini multimodal ses deşifresi yedeği
        if self.gemini_transcribe_fn is not None:
            txt = self.gemini_transcribe_fn(
                audio_wav_path, language or "en", cancel_event
            )
            return TranscriptionResult(text=txt.strip(), detected_language=language or "en")

        raise ElevenLabsError(
            "Deşifre için aktif bir Colab/Yerel Whisper veya Gemini API yapılandırması bulunamadı."
        )

    def separate_vocals(
        self,
        audio_path: Path,
        vocals_out: Path,
        background_out: Path,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        if self.transcribe_delegate is not None:
            return self.transcribe_delegate.separate_vocals(
                audio_path, vocals_out, background_out, cancel_event=cancel_event
            )
        return False

    def _save_audio_bytes_as_wav(self, audio_bytes: bytes, output_wav_path: Path) -> Path:
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        # Eğer zaten RIFF WAV döndüyse doğrudan yaz, MP3 döndüyse FFmpeg ile temiz WAV'a çevir
        if audio_bytes[:4] == b"RIFF":
            output_wav_path.write_bytes(audio_bytes)
            return output_wav_path

        tmp_mp3 = TEMP_DIR / f"_el_raw_{uuid.uuid4().hex[:8]}.mp3"
        try:
            tmp_mp3.write_bytes(audio_bytes)
            self.ffmpeg.convert_to_wav(tmp_mp3, output_wav_path, sample_rate=44100, channels=1)
            return output_wav_path
        finally:
            tmp_mp3.unlink(missing_ok=True)

    def synthesize_tts(
        self,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        options: Optional[TTSGenerationOptions] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """ElevenLabs Multilingual v2 (`eleven_multilingual_v2`) ile Türkçe dublaj üretir."""
        self._require_key()
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("ElevenLabs TTS iptal edildi.")

        opts = options or TTSGenerationOptions(tts_engine=TTSEngineType.ELEVENLABS_TTS)
        voice_id = self._resolve_or_clone_voice_id(speaker_wav_path, opts)

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": max(0.0, min(1.0, float(opts.elevenlabs_stability))),
                "similarity_boost": max(0.0, min(1.0, float(opts.elevenlabs_similarity))),
                "style": max(0.0, min(1.0, float(opts.elevenlabs_style))),
                "use_speaker_boost": True,
            },
        }

        with httpx.Client(timeout=self.timeout_sec, follow_redirects=True) as client:
            resp = client.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers=self._headers(json_body=True),
                json=payload,
            )

        if resp.status_code != 200 or len(resp.content) < 64:
            raise self._map_error(resp, "Multilingual v2 TTS")

        return self._save_audio_bytes_as_wav(resp.content, output_wav_path)

    def speech_to_speech(
        self,
        source_audio_path: Path,
        speaker_wav_path: Path,
        output_wav_path: Path,
        options: Optional[TTSGenerationOptions] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """ElevenLabs Speech-to-Speech (`eleven_multilingual_sts_v2`) ile duygu/bağırmayı birebir aktarır."""
        self._require_key()
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("ElevenLabs Speech-to-Speech iptal edildi.")

        opts = options or TTSGenerationOptions(tts_engine=TTSEngineType.ELEVENLABS_STS)
        voice_id = self._resolve_or_clone_voice_id(speaker_wav_path, opts)

        voice_settings_json = json.dumps(
            {
                "stability": max(0.0, min(1.0, float(opts.elevenlabs_stability))),
                "similarity_boost": max(0.0, min(1.0, float(opts.elevenlabs_similarity))),
                "style": max(0.0, min(1.0, float(opts.elevenlabs_style))),
                "use_speaker_boost": True,
            }
        )
        form_data = {
            "model_id": "eleven_multilingual_sts_v2",
            "voice_settings": voice_settings_json,
            "remove_background_noise": "true" if opts.elevenlabs_remove_bg_noise else "false",
        }

        src_bytes = source_audio_path.read_bytes()
        with httpx.Client(timeout=self.timeout_sec, follow_redirects=True) as client:
            resp = client.post(
                f"{self.base_url}/speech-to-speech/{voice_id}",
                headers=self._headers(json_body=False),
                data=form_data,
                files={"audio": (source_audio_path.name, src_bytes, "audio/wav")},
            )

        if resp.status_code != 200 or len(resp.content) < 64:
            raise self._map_error(resp, "Speech-to-Speech (Voice Changer)")

        return self._save_audio_bytes_as_wav(resp.content, output_wav_path)

    def synthesize_clone(
        self,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        language: str = "tr",
        cancel_event: Optional[threading.Event] = None,
        options: Optional[TTSGenerationOptions] = None,
    ) -> Path:
        opts = options or TTSGenerationOptions(tts_engine=self.tts_mode)
        if opts.tts_engine is TTSEngineType.ELEVENLABS_STS:
            return self.speech_to_speech(
                source_audio_path=speaker_wav_path,
                speaker_wav_path=speaker_wav_path,
                output_wav_path=output_wav_path,
                options=opts,
                cancel_event=cancel_event,
            )
        return self.synthesize_tts(
            text=text,
            speaker_wav_path=speaker_wav_path,
            output_wav_path=output_wav_path,
            options=opts,
            cancel_event=cancel_event,
        )

    def unload(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Geçici olarak oluşturulan ElevenLabs IVC seslerini temizler ve delege motoru boşaltır."""
        with self._lock:
            to_delete = list(self._ephemeral_voice_ids)
            self._ephemeral_voice_ids.clear()
            self._ivc_cache.clear()

        for vid in to_delete:
            try:
                with httpx.Client(timeout=15.0) as client:
                    client.delete(f"{self.base_url}/voices/{vid}", headers=self._headers())
            except Exception:
                pass

        if to_delete and log_cb:
            log_cb(f"ElevenLabs oturumundaki {len(to_delete)} geçici klonlanmış ses temizlendi.")

        if self.transcribe_delegate is not None:
            try:
                self.transcribe_delegate.unload(log_cb=log_cb)
            except Exception:
                pass
