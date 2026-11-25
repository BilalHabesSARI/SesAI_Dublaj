"""Google Colab Pro (Bulut Modu) HTTP İstemci Motoru.

Desteklediği Bulut Yetenekleri:
- faster-whisper deşifre (`POST /transcribe`)
- Seçilebilir yeni nesil ses klonlama (`POST /synthesize`):
    * Chatterbox Multilingual (Resemble AI, language_id="tr", exaggeration, cfg_weight)
    * Coqui XTTS v2 (Klasik)
- GPU hızlandırmalı vokal / arka plan ayırma (`POST /separate`, Meta htdemucs)
"""

from __future__ import annotations

import io
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

import httpx

from core.models import EngineHealth, EngineMode, TTSEngineType, TTSGenerationOptions
from engines.base import SpeechEngine, TranscriptionResult


class ColabConnectionError(RuntimeError):
    """Colab bulut sunucusuyla iletişim hatası."""


class ColabCloudEngine(SpeechEngine):
    """Google Colab Pro + FastAPI + cloudflared istemcisi."""

    mode = EngineMode.COLAB_CLOUD

    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        timeout_sec: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        self.bearer_token = bearer_token.strip()
        self.timeout_sec = max(15.0, float(timeout_sec))
        self.max_retries = max(1, int(max_retries))

    def _validate_credentials(self) -> None:
        if not self.base_url:
            raise ColabConnectionError(
                "Colab Bağlantı URL'si boş! Lütfen Colab defterinin ürettiği "
                "'https://....trycloudflare.com' adresini girin."
            )
        if not (self.base_url.startswith("https://") or self.base_url.startswith("http://")):
            raise ColabConnectionError(
                "Geçersiz Colab URL formatı. Adres 'https://' ile başlamalıdır."
            )
        if not self.bearer_token:
            raise ColabConnectionError(
                "Colab Güvenlik Anahtarı (Bearer Token) boş! Lütfen defter çıktısındaki token'ı girin."
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json, audio/wav, application/zip",
        }

    def _check_cancel(self, cancel_event: Optional[threading.Event]) -> None:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Colab bulut isteği kullanıcı tarafından iptal edildi.")

    def _map_http_error(self, resp: httpx.Response, action: str) -> ColabConnectionError:
        if resp.status_code in (401, 403):
            return ColabConnectionError(
                f"{action} reddedildi (HTTP {resp.status_code}): Bearer Token hatalı veya süresi dolmuş."
            )
        if resp.status_code == 404:
            return ColabConnectionError(
                f"{action} bulunamadı (HTTP 404): Colab defterini en güncel sürüme yenileyin veya URL'yi kontrol edin."
            )
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text[:240]
        return ColabConnectionError(
            f"{action} başarısız (HTTP {resp.status_code}): {detail}"
        )

    def health_check(self) -> EngineHealth:
        try:
            self._validate_credentials()
        except ColabConnectionError as exc:
            return EngineHealth(ok=False, mode=self.mode, message=str(exc))

        start = time.perf_counter()
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                resp = client.get(f"{self.base_url}/health", headers=self._headers())
            latency = int(round((time.perf_counter() - start) * 1000.0))
            if resp.status_code != 200:
                err = self._map_http_error(resp, "Bağlantı testi")
                return EngineHealth(ok=False, mode=self.mode, message=str(err), latency_ms=latency)

            data = resp.json()
            gpu_name = str(data.get("gpu") or "Bulut GPU")
            vram_mb = int(data.get("vram_mb") or 0)
            whisper_model = str(data.get("whisper_model") or "faster-whisper")
            tts_model = str(data.get("tts_model") or "Chatterbox Multilingual / XTTS v2")
            avail = list(data.get("available_tts_engines") or [TTSEngineType.XTTS_V2.value])
            sep_ok = bool(data.get("supports_vocal_separation", False))
            return EngineHealth(
                ok=True,
                mode=self.mode,
                message=f"Colab bağlantısı başarılı ({gpu_name}, {latency} ms)",
                gpu_name=gpu_name,
                vram_total_mb=vram_mb,
                whisper_model=whisper_model,
                tts_model=tts_model,
                latency_ms=latency,
                available_tts_engines=avail,
                supports_vocal_separation=sep_ok,
            )
        except httpx.ConnectError:
            return EngineHealth(
                ok=False,
                mode=self.mode,
                message="Colab sunucusuna ulaşılamadı. Defterin çalıştığından ve URL'nin güncel olduğundan emin olun.",
            )
        except httpx.TimeoutException:
            return EngineHealth(
                ok=False,
                mode=self.mode,
                message="Colab bağlantı testi zaman aşımına uğradı (20 sn).",
            )
        except Exception as exc:
            return EngineHealth(ok=False, mode=self.mode, message=f"Bağlantı hatası: {exc}")

    def transcribe(
        self,
        audio_wav_path: Path,
        language: Optional[str] = "en",
        cancel_event: Optional[threading.Event] = None,
    ) -> TranscriptionResult:
        self._validate_credentials()
        self._check_cancel(cancel_event)

        audio_bytes = audio_wav_path.read_bytes()
        form_data = {"language": (language or "auto")}

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._check_cancel(cancel_event)
            try:
                with httpx.Client(timeout=self.timeout_sec, follow_redirects=True) as client:
                    resp = client.post(
                        f"{self.base_url}/transcribe",
                        headers=self._headers(),
                        data=form_data,
                        files={"audio": (audio_wav_path.name, audio_bytes, "audio/wav")},
                    )
                if resp.status_code == 200:
                    payload = resp.json()
                    return TranscriptionResult(
                        text=str(payload.get("text") or "").strip(),
                        detected_language=str(payload.get("language") or language or "en"),
                    )
                if resp.status_code in (502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise self._map_http_error(resp, "Colab deşifre")
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise ColabConnectionError(f"Colab deşifre ağ hatası: {exc}") from exc

        raise ColabConnectionError(f"Colab deşifre başarısız: {last_exc}")

    def synthesize_clone(
        self,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        language: str = "tr",
        cancel_event: Optional[threading.Event] = None,
        options: Optional[TTSGenerationOptions] = None,
    ) -> Path:
        self._validate_credentials()
        self._check_cancel(cancel_event)

        opts = options or TTSGenerationOptions()
        ref_bytes = speaker_wav_path.read_bytes()
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        form_data = {
            "text": text,
            "language": language,
            "tts_engine": opts.tts_engine.value,
            "exaggeration": f"{float(opts.exaggeration):.3f}",
            "cfg_weight": f"{float(opts.cfg_weight):.3f}",
            "voxcpm_cfg_value": f"{float(opts.voxcpm_cfg_value):.3f}",
            "voxcpm_timesteps": str(int(opts.voxcpm_timesteps)),
            "use_acting_prompt": "1" if opts.use_acting_prompt else "0",
            "acting_prompt": (opts.acting_prompt or "").strip(),
            "candidate_index": str(int(opts.candidate_index)),
            "random_seed": str(int(opts.random_seed or 0)),
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._check_cancel(cancel_event)
            try:
                with httpx.Client(timeout=self.timeout_sec, follow_redirects=True) as client:
                    resp = client.post(
                        f"{self.base_url}/synthesize",
                        headers=self._headers(),
                        data=form_data,
                        files={"speaker_wav": (speaker_wav_path.name, ref_bytes, "audio/wav")},
                    )
                if resp.status_code == 200 and len(resp.content) > 44:
                    output_wav_path.write_bytes(resp.content)
                    return output_wav_path
                if resp.status_code in (502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(2.5 * (attempt + 1))
                    continue
                raise self._map_http_error(resp, "Colab ses klonlama")
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2.5 * (attempt + 1))
                    continue
                raise ColabConnectionError(f"Colab ses klonlama ağ hatası: {exc}") from exc

        raise ColabConnectionError(f"Colab ses klonlama başarısız: {last_exc}")

    def separate_vocals(
        self,
        audio_path: Path,
        vocals_out: Path,
        background_out: Path,
        cancel_event: Optional[threading.Event] = None,
    ) -> bool:
        """Colab GPU Demucs (`/separate`) üzerinden konuşma ve arka planı ayırır."""
        self._validate_credentials()
        self._check_cancel(cancel_event)

        audio_bytes = audio_path.read_bytes()
        with httpx.Client(timeout=self.timeout_sec, follow_redirects=True) as client:
            resp = client.post(
                f"{self.base_url}/separate",
                headers=self._headers(),
                files={"audio": (audio_path.name, audio_bytes, "application/octet-stream")},
            )
        if resp.status_code == 404:
            # Eski Colab defteri /separate uç noktasına sahip değilse yerel yedeğe düş
            return False
        if resp.status_code != 200:
            raise self._map_http_error(resp, "Colab vokal ayırma")

        vocals_out.parent.mkdir(parents=True, exist_ok=True)
        background_out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content), "r") as zf:
            vocals_out.write_bytes(zf.read("vocals.wav"))
            background_out.write_bytes(zf.read("background.wav"))
        return vocals_out.is_file() and background_out.is_file()
