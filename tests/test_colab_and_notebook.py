"""colab_motoru.ipynb (v2), ColabCloudEngine (TTS + /separate) ve ElevenLabsEngine (TTS + STS + IVC) testleri."""

from __future__ import annotations

import ast
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

from core.models import TTSEngineType, TTSGenerationOptions
from engines.colab_engine import ColabCloudEngine
from engines.colab_notebook_generator import DEFAULT_NOTEBOOK_PATH, generate_colab_notebook
from engines.elevenlabs_engine import ElevenLabsEngine


def test_generate_valid_colab_notebook() -> None:
    nb_path = generate_colab_notebook(DEFAULT_NOTEBOOK_PATH)
    assert nb_path.is_file()
    data = json.loads(nb_path.read_text(encoding="utf-8"))

    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 3
    joined_src = "\n".join("".join(c["source"]) for c in data["cells"])

    # Tüm yeni nesil sinematik motorlar ve düzeltmeler mevcut olmalı
    assert "faster-whisper" in joined_src
    assert '"av>=11,<18"' in joined_src
    assert "_load_whisper_audio_16k" in joined_src
    assert "_safe_torchaudio_load" in joined_src
    assert "voxcpm" in joined_src
    assert "openbmb/VoxCPM2" in joined_src
    assert "Trendyol/Trendyol-TTS" in joined_src
    assert "load_denoiser=False" in joined_src
    assert "optimize=False" in joined_src
    assert "acting_prompt" in joined_src
    assert "chatterbox-tts" in joined_src
    assert "ChatterboxMultilingualTTS" in joined_src
    assert "exaggeration" in joined_src
    assert "cfg_weight" in joined_src
    assert "coqui-tts" in joined_src
    assert "xtts_v2" in joined_src
    assert "demucs" in joined_src
    assert "cloudflared" in joined_src
    assert "Bearer " in joined_src
    assert "/health" in joined_src
    assert "/transcribe" in joined_src
    assert "/synthesize" in joined_src
    assert "/separate" in joined_src

    # Tüm kod hücrelerinin Python sözdizimi geçerli olmalı
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            cleaned = []
            for ln in cell["source"]:
                if ln.lstrip().startswith("!"):
                    indent = ln[: len(ln) - len(ln.lstrip())]
                    cleaned.append(indent + "pass\n")
                else:
                    cleaned.append(ln)
            ast.parse("".join(cleaned))


def test_local_whisper_numpy_loader_bypasses_pyav(tmp_path: Path) -> None:
    from engines.local_gpu_engine import _load_whisper_audio_16k

    wav_48k = tmp_path / "stereo_48k.wav"
    t = np.linspace(0.0, 1.0, 48000, endpoint=False, dtype=np.float32)
    stereo = np.stack([0.2 * np.sin(2.0 * np.pi * 440.0 * t)] * 2, axis=1)
    sf.write(str(wav_48k), stereo, 48000, subtype="PCM_16")

    audio_16k = _load_whisper_audio_16k(wav_48k)
    assert audio_16k.ndim == 1
    assert audio_16k.dtype == np.float32
    assert 15950 <= len(audio_16k) <= 16050


def test_local_gpu_cuda_whisper_chatterbox_xtts_demucs_ready() -> None:
    import torch
    import faster_whisper  # noqa: F401
    import demucs  # noqa: F401
    import munch  # noqa: F401
    import sounddevice  # noqa: F401
    from voxcpm import VoxCPM  # noqa: F401
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: F401
    from TTS.api import TTS  # noqa: F401
    from core.portable_env import MODELS_DIR
    from engines.local_gpu_engine import LocalGPUEngine
    from engines.seed_vc.hf_utils import get_seed_vc_cache_dir
    from engines.seed_vc.seed_vc_wrapper import SeedVCWrapper

    assert torch.cuda.is_available() is True
    assert str(MODELS_DIR) in get_seed_vc_cache_dir()
    wrapper = SeedVCWrapper()
    assert wrapper.model_f0 is None and wrapper.model is None

    eng = LocalGPUEngine()
    assert eng.default_tts_engine is TTSEngineType.VOXCPM2
    backends = eng._detect_installed_backends()
    assert backends["seed_vc"] is True
    health = eng.health_check()
    assert health.ok is True
    assert "voxcpm2" in health.available_tts_engines
    assert "trendyol_tts" in health.available_tts_engines
    assert "chatterbox_multilingual" in health.available_tts_engines
    assert "xtts_v2" in health.available_tts_engines
    assert "Seed-VC STS" in health.message
    assert health.supports_vocal_separation is True

    default_opts = TTSGenerationOptions()
    assert abs(default_opts.voxcpm_cfg_value - 2.00) < 1e-4

    vox_opts = TTSGenerationOptions(
        tts_engine=TTSEngineType.VOXCPM2,
        voxcpm_cfg_value=2.0,
        voxcpm_timesteps=16,
        use_acting_prompt=True,
        acting_prompt="threatening, sinister dark tone, dramatic pause, intense",
    )
    assert (
        vox_opts.format_voxcpm_text("Sözleşmeniz feshedildi.")
        == "(threatening, sinister dark tone, dramatic pause, intense)Sözleşmeniz feshedildi."
    )
    vox_opts_off = TTSGenerationOptions(
        tts_engine=TTSEngineType.VOXCPM2,
        use_acting_prompt=False,
        acting_prompt="threatening, sinister dark tone",
    )
    assert vox_opts_off.format_voxcpm_text("Sözleşmeniz feshedildi.") == "Sözleşmeniz feshedildi."


def test_colab_cloud_engine_http_flow(tmp_path: Path) -> None:
    expected_token = "test_secret_token_987654"
    seen_synth_body: list[bytes] = []

    dummy_wav_path = tmp_path / "reply.wav"
    sf.write(str(dummy_wav_path), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    dummy_wav_bytes = dummy_wav_path.read_bytes()

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("vocals.wav", dummy_wav_bytes)
        zf.writestr("background.wav", dummy_wav_bytes)
    dummy_zip_bytes = zip_buf.getvalue()

    class MockColabHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _check_auth(self) -> bool:
            if self.headers.get("Authorization", "") != f"Bearer {expected_token}":
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b'{"detail":"Invalid token"}')
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                if not self._check_auth():
                    return
                payload = json.dumps(
                    {
                        "status": "ok",
                        "gpu": "NVIDIA L4 (Mock)",
                        "vram_mb": 23034,
                        "whisper_model": "large-v3-turbo",
                        "tts_model": "chatterbox_multilingual",
                        "available_tts_engines": ["chatterbox_multilingual", "xtts_v2"],
                        "supports_vocal_separation": True,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def do_POST(self):
            if not self._check_auth():
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if self.path == "/transcribe":
                payload = json.dumps({"text": "Hold the gate!", "language": "en"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/synthesize":
                seen_synth_body.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(dummy_wav_bytes)))
                self.end_headers()
                self.wfile.write(dummy_wav_bytes)
            elif self.path == "/separate":
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(dummy_zip_bytes)))
                self.end_headers()
                self.wfile.write(dummy_zip_bytes)

    server = HTTPServer(("127.0.0.1", 0), MockColabHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        bad_engine = ColabCloudEngine(f"http://127.0.0.1:{port}", "wrong_token")
        assert bad_engine.health_check().ok is False

        engine = ColabCloudEngine(f"http://127.0.0.1:{port}", expected_token)
        health = engine.health_check()
        assert health.ok is True
        assert "chatterbox_multilingual" in health.available_tts_engines
        assert health.supports_vocal_separation is True

        tr = engine.transcribe(dummy_wav_path, language="en")
        assert tr.text == "Hold the gate!"

        opts = TTSGenerationOptions(
            tts_engine=TTSEngineType.CHATTERBOX_MULTILINGUAL,
            exaggeration=0.72,
            cfg_weight=0.35,
        )
        out_dub = tmp_path / "dubbed_from_colab.wav"
        engine.synthesize_clone("Kapıyı tutun!", dummy_wav_path, out_dub, language="tr", options=opts)
        assert out_dub.is_file()
        assert b"chatterbox_multilingual" in seen_synth_body[0]
        assert b"0.720" in seen_synth_body[0]
        assert b"0.350" in seen_synth_body[0]

        voc_out = tmp_path / "voc.wav"
        bg_out = tmp_path / "bg.wav"
        assert engine.separate_vocals(dummy_wav_path, voc_out, bg_out) is True
        assert voc_out.is_file() and bg_out.is_file()
    finally:
        server.shutdown()
        server.server_close()


def test_elevenlabs_engine_tts_sts_and_auto_ivc_cleanup(tmp_path: Path) -> None:
    expected_key = "mock_elevenlabs_test_key_123"
    deleted_voices: list[str] = []
    called_paths: list[str] = []

    ref_wav = tmp_path / "char_ref.wav"
    sf.write(str(ref_wav), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    wav_bytes = ref_wav.read_bytes()

    class MockElevenLabsHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _check_key(self) -> bool:
            if self.headers.get("xi-api-key", "") != expected_key:
                self.send_response(401)
                self.end_headers()
                return False
            return True

        def do_GET(self):
            if not self._check_key():
                return
            called_paths.append(f"GET {self.path}")
            if self.path == "/v1/user":
                payload = json.dumps(
                    {"subscription": {"tier": "creator", "character_count": 1000, "character_limit": 100000}}
                ).encode("utf-8")
            elif self.path == "/v1/voices":
                payload = json.dumps(
                    {"voices": [{"voice_id": "voice_47_pro", "name": "Agent Deep", "category": "cloned"}]}
                ).encode("utf-8")
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if not self._check_key():
                return
            called_paths.append(f"POST {self.path}")
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if self.path == "/v1/voices/add":
                payload = json.dumps({"voice_id": "ivc_temp_999"}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path.startswith("/v1/text-to-speech/"):
                assert b"eleven_multilingual_v2" in body
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.end_headers()
                self.wfile.write(wav_bytes)
            elif self.path.startswith("/v1/speech-to-speech/"):
                assert b"eleven_multilingual_sts_v2" in body
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.end_headers()
                self.wfile.write(wav_bytes)
            else:
                self.send_response(404)
                self.end_headers()

        def do_DELETE(self):
            if not self._check_key():
                return
            called_paths.append(f"DELETE {self.path}")
            deleted_voices.append(self.path)
            self.send_response(200)
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), MockElevenLabsHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        el = ElevenLabsEngine(
            api_key=expected_key,
            base_url=f"http://127.0.0.1:{port}/v1",
        )
        health = el.health_check()
        assert health.ok is True
        assert "creator" in health.message

        voices = el.list_voices()
        assert voices == [("voice_47_pro", "Agent Deep (cloned)")]

        # 1. Boş Voice ID -> Otomatik IVC + Multilingual v2 TTS
        out_tts = tmp_path / "el_tts.wav"
        el.synthesize_clone(
            "Sözleşmeniz feshedildi.",
            ref_wav,
            out_tts,
            options=TTSGenerationOptions(tts_engine=TTSEngineType.ELEVENLABS_TTS),
        )
        assert out_tts.is_file()
        assert "POST /v1/voices/add" in called_paths
        assert "POST /v1/text-to-speech/ivc_temp_999" in called_paths

        # 2. Sabit Voice ID + Speech-to-Speech (Voice Changer)
        out_sts = tmp_path / "el_sts.wav"
        el.synthesize_clone(
            "",
            ref_wav,
            out_sts,
            options=TTSGenerationOptions(
                tts_engine=TTSEngineType.ELEVENLABS_STS,
                elevenlabs_voice_id="voice_47_pro",
            ),
        )
        assert out_sts.is_file()
        assert "POST /v1/speech-to-speech/voice_47_pro" in called_paths

        # 3. unload() geçici IVC sesini temizlemeli
        el.unload()
        assert "/v1/voices/ivc_temp_999" in deleted_voices
    finally:
        server.shutdown()
        server.server_close()
