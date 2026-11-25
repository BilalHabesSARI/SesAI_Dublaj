"""Google Colab Pro Bulut Motoru (colab_motoru.ipynb) Üreticisi.

Defter Google Colab üzerinde (L4 / A100 / T4 GPU) çalıştırıldığında:
- `faster-whisper` (+ uyumlu `av>=11,<18` + PyAV baypaslı `soundfile` okuma),
- Yeni nesil **Chatterbox Multilingual** (Resemble AI, `language_id="tr"`, `exaggeration`, `cfg_weight`),
- Klasik **Coqui XTTS v2**,
- **Meta HTDemucs (`demucs`)** vokal / arka plan ayrıştırma (`/separate`),
- Hesap/token gerektirmeyen ücretsiz `cloudflared` Quick Tunnel ve tek kullanımlık
  Bearer Token korumalı FastAPI sunucusunu ayağa kaldırır.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.portable_env import APP_ROOT

DEFAULT_NOTEBOOK_PATH: Path = APP_ROOT / "colab_motoru.ipynb"


def _lines(source: str) -> list[str]:
    return source.strip("\n").splitlines(keepends=True)


def build_colab_notebook_dict() -> dict:
    """Geçerli nbformat v4.5 JSON yapısında Sinematik Colab Pro defteri üretir."""
    cell_1_md = """# 🎬 SesAI Dublaj Stüdyosu — Google Colab Pro Sinematik Bulut Motoru (v3 — VoxCPM2 48 kHz)

Bu defter, **SesAI Dublaj Stüdyosu** masaüstü uygulamasının yerel ekran kartınızı yormadan **Google Colab Pro GPU** (L4 / A100 / T4) gücüyle sinema kalitesinde dublaj üretmesini sağlar.

### Desteklenen Bulut Yetenekleri
- **Deşifre:** `faster-whisper` (`large-v3-turbo`, PyAV `metadata_errors` korumalı)
- **Ses Motoru 1 (Yeni Varsayılan — 48 kHz):** **VoxCPM2** (`openbmb/VoxCPM2`, Controllable Voice Cloning + `acting_prompt` duygu yönlendirmesi)
- **Ses Motoru 2 (48 kHz Türkçe İnce Ayar):** **Trendyol-TTS** (`Trendyol/Trendyol-TTS`)
- **Ses Motoru 3 (24 kHz Sinematik):** **Chatterbox Multilingual** (Resemble AI, `language_id="tr"`, `exaggeration` & `cfg_weight` kontrollü)
- **Ses Motoru 4 (Klasik):** **Coqui XTTS v2**
- **Arka Plan Koruma:** **Meta HTDemucs** (`POST /separate` — konuşmayı müzik/efektten ayırır)
- **Güvenlik:** Hesap gerektirmeyen ücretsiz `cloudflared` tüneli + tek kullanımlık Bearer Token

### Nasıl Kullanılır?
1. **Çalışma Zamanı (Runtime) → Çalışma zamanı türünü değiştir → GPU (L4 / A100 / T4)** seçin.
2. **Çalışma Zamanı → Tümünü çalıştır** seçeneğine basın.
3. 2. hücrenin çıktısındaki **Colab Bağlantı URL** ve **Bearer Token** değerlerini masaüstü uygulamasına yapıştırıp **Bağlantıyı Test Et**'e tıklayın.
"""

    cell_2_setup = r'''# ==============================================================================
# HÜCRE 1: GPU Kontrolü, VoxCPM2 (48 kHz), Trendyol-TTS, Chatterbox, XTTS v2, Demucs, faster-whisper ve cloudflared
# ==============================================================================
import os
import subprocess

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

print("🔍 GPU kontrol ediliyor...")
gpu_check = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if gpu_check.returncode != 0:
    raise RuntimeError(
        "GPU algılanamadı! Lütfen 'Çalışma Zamanı -> Çalışma zamanı türünü değiştir -> GPU' seçin."
    )
print("✅ Algılanan GPU:", gpu_check.stdout.strip())

print("\n📦 Sinematik dublaj paketleri kuruluyor (VoxCPM2 48 kHz, Trendyol-TTS, Chatterbox, XTTS v2, Demucs, faster-whisper)...")
!apt-get -qq update && apt-get -qq install -y ffmpeg > /dev/null
# Python 3.12/3.13 + faster-whisper + VoxCPM2 + Chatterbox + Coqui TTS 0.28+ ortak uyumlu sürümler:
# - av>=11,<18 : PyAV 18+ metadata_errors hatasını önler
# - transformers==5.2.0 & librosa==0.11.0 : VoxCPM2, Chatterbox Multilingual (0.1.7) ve Coqui TTS (0.28+) ile tam uyumlu
!pip install -q --upgrade "librosa==0.11.0" "transformers==5.2.0"
!pip install -q -U faster-whisper "av>=11,<18" "coqui-tts>=0.28.0" demucs fastapi uvicorn python-multipart soundfile
# VoxCPM2 ve Chatterbox Multilingual runtime bağımlılıkları + --no-deps kurulumu
!pip install -q "resemble-perth>=1.0.0" s3tokenizer conformer==0.3.2 "diffusers==0.29.0" "safetensors>=0.5.3" omegaconf einops "pykakasi==2.3.0" spacy-pkuseg pyloudnorm six addict argbind simplejson sortedcontainers wetext
!pip install -q --no-deps chatterbox-tts voxcpm

print("\n🌐 Ücretsiz cloudflared tünel aracı indiriliyor (hesap/token gerektirmez)...")
if not os.path.isfile("/usr/local/bin/cloudflared"):
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
    !chmod +x /usr/local/bin/cloudflared

print("✅ Kurulum başarıyla tamamlandı! Şimdi 2. hücreyi çalıştırabilirsiniz.")
'''

    cell_3_server = r'''# ==============================================================================
# HÜCRE 2: Hibrit Bulut Sunucusu (Whisper + Chatterbox Multilingual + XTTS v2 + Demucs)
# ==============================================================================
import gc
import io
import math
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

# Yeni torchaudio sürümlerinde torchcodec eksikse soundfile yedeği
_orig_ta_load = torchaudio.load

def _safe_torchaudio_load(filepath, *args, **kwargs):
    try:
        return _orig_ta_load(filepath, *args, **kwargs)
    except Exception as _ta_err:
        err_l = str(_ta_err).lower()
        if "torchcodec" in err_l or "backend" in err_l:
            data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), int(sr)
        raise

torchaudio.load = _safe_torchaudio_load


def _load_whisper_audio_16k(wav_path: Path) -> np.ndarray:
    """WAV dosyasını PyAV (av.open) kullanmadan doğrudan 16kHz mono float32 NumPy dizisine yükler."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != 16000 and audio.size > 0:
        g = math.gcd(int(sr), 16000)
        audio = resample_poly(audio, 16000 // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


# ---- Ayarlar ----
WHISPER_MODEL_NAME = "large-v3-turbo"
SERVER_PORT = 8000
BEARER_TOKEN = secrets.token_urlsafe(32)
WORK_TEMP_DIR = Path("/tmp/sesai_colab_temp")
WORK_TEMP_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU hazır değil!")

GPU_NAME = torch.cuda.get_device_name(0)
VRAM_MB = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
print(f"🚀 GPU Hazır: {GPU_NAME} ({VRAM_MB} MB VRAM)")

# ---- Kurulu Motorları Kontrol Et ----
AVAILABLE_TTS = []
HAS_VOXCPM = False
HAS_CHATTERBOX = False
HAS_XTTS = False
HAS_DEMUCS = False

try:
    from voxcpm import VoxCPM
    HAS_VOXCPM = True
    AVAILABLE_TTS.extend(["voxcpm2", "trendyol_tts"])
except Exception as e:
    print(f"⚠️ VoxCPM2 yüklenemedi: {e}")

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    HAS_CHATTERBOX = True
    AVAILABLE_TTS.append("chatterbox_multilingual")
except Exception as e:
    print(f"⚠️ Chatterbox Multilingual yüklenemedi: {e}")

try:
    from TTS.api import TTS
    HAS_XTTS = True
    AVAILABLE_TTS.append("xtts_v2")
except Exception as e:
    print(f"⚠️ Coqui XTTS v2 yüklenemedi: {e}")

try:
    from demucs.apply import apply_model as demucs_apply_model
    from demucs.pretrained import get_model as demucs_get_model
    HAS_DEMUCS = True
except Exception as e:
    print(f"⚠️ Demucs vokal ayırıcı yüklenemedi: {e}")

if not AVAILABLE_TTS:
    raise RuntimeError("Hiçbir açık kaynak TTS motoru kurulu değil! HÜCRE 1'i kontrol edin.")

# ---- Whisper'ı Yükle + Varsayılan TTS Motorunu Önyükle ----
print(f"⏳ faster-whisper ({WHISPER_MODEL_NAME}) yükleniyor...")
whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="cuda", compute_type="float16")
print("✅ faster-whisper hazır.")

gpu_lock = threading.Lock()
_active_tts_name = None
_active_tts_model = None
_demucs_model = None


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def get_tts_model(engine_name: str):
    """İstenen TTS motorunu yükler; farklı motor seçilirse VRAM için eskisini boşaltır."""
    global _active_tts_name, _active_tts_model

    if engine_name not in AVAILABLE_TTS:
        engine_name = AVAILABLE_TTS[0]

    if _active_tts_model is not None and _active_tts_name == engine_name:
        return engine_name, _active_tts_model

    if _active_tts_model is not None:
        del _active_tts_model
        _active_tts_model = None
        _active_tts_name = None
        _free_vram()

    if engine_name in ("voxcpm2", "trendyol_tts"):
        repo_id = "Trendyol/Trendyol-TTS" if engine_name == "trendyol_tts" else "openbmb/VoxCPM2"
        print(f"⏳ {engine_name} ({repo_id}, 48 kHz) GPU'ya yükleniyor...")
        _active_tts_model = VoxCPM.from_pretrained(
            hf_model_id=repo_id,
            load_denoiser=False,
            optimize=False,
            device="cuda",
        )
        _active_tts_name = engine_name
        print(f"✅ {engine_name} (48 kHz) hazır.")
    elif engine_name == "chatterbox_multilingual":
        print("⏳ Chatterbox Multilingual (Resemble AI) GPU'ya yükleniyor...")
        _active_tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        _active_tts_name = "chatterbox_multilingual"
        print("✅ Chatterbox Multilingual hazır.")
    else:
        print("⏳ Coqui XTTS v2 GPU'ya yükleniyor...")
        _active_tts_model = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        ).to("cuda")
        _active_tts_name = "xtts_v2"
        print("✅ XTTS v2 hazır.")

    return _active_tts_name, _active_tts_model


def get_demucs_model():
    global _demucs_model
    if not HAS_DEMUCS:
        raise RuntimeError("Demucs bu oturumda kurulu değil.")
    if _demucs_model is None:
        print("⏳ Meta HTDemucs vokal ayırıcı yükleniyor...")
        _demucs_model = demucs_get_model("htdemucs").to("cuda")
        _demucs_model.eval()
        print("✅ HTDemucs hazır.")
    return _demucs_model


# Varsayılan motoru (VoxCPM2 48 kHz varsa onu) önyükle
default_tts = AVAILABLE_TTS[0]
get_tts_model(default_tts)

app = FastAPI(title="SesAI Dublaj Sinematik Colab Motoru v3 (VoxCPM2 48 kHz)", version="3.0.0")


def verify_bearer(authorization: str = Header(default="")) -> None:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Eksik Bearer Token.")
    incoming = authorization[len(prefix):].strip()
    if not secrets.compare_digest(incoming, BEARER_TOKEN):
        raise HTTPException(status_code=403, detail="Geçersiz Bearer Token.")


@app.get("/health", dependencies=[Depends(verify_bearer)])
def health():
    return {
        "status": "ok",
        "gpu": GPU_NAME,
        "vram_mb": VRAM_MB,
        "whisper_model": WHISPER_MODEL_NAME,
        "tts_model": _active_tts_name or default_tts,
        "available_tts_engines": AVAILABLE_TTS,
        "supports_vocal_separation": HAS_DEMUCS,
    }


@app.post("/transcribe", dependencies=[Depends(verify_bearer)])
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
):
    req_dir = Path(tempfile.mkdtemp(prefix="tr_", dir=str(WORK_TEMP_DIR)))
    try:
        src_path = req_dir / (audio.filename or "input.wav")
        src_path.write_bytes(await audio.read())
        lang_arg = None if (not language or language == "auto") else language

        with gpu_lock:
            audio_np = _load_whisper_audio_16k(src_path)
            segments, info = whisper_model.transcribe(
                audio_np,
                language=lang_arg,
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            text = " ".join((s.text or "").strip() for s in segments if (s.text or "").strip()).strip()
            detected_lang = getattr(info, "language", None) or (lang_arg or "en")

        return {"text": text, "language": detected_lang}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Deşifre hatası: {exc}")
    finally:
        shutil.rmtree(req_dir, ignore_errors=True)


@app.post("/synthesize", dependencies=[Depends(verify_bearer)])
async def synthesize(
    speaker_wav: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form(default="tr"),
    tts_engine: str = Form(default="voxcpm2"),
    exaggeration: float = Form(default=0.55),
    cfg_weight: float = Form(default=0.50),
    voxcpm_cfg_value: float = Form(default=1.70),
    voxcpm_timesteps: int = Form(default=16),
    use_acting_prompt: str = Form(default="1"),
    acting_prompt: str = Form(default=""),
    candidate_index: int = Form(default=1),
    random_seed: int = Form(default=0),
):
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Seslendirilecek metin boş olamaz.")

    req_dir = Path(tempfile.mkdtemp(prefix="syn_", dir=str(WORK_TEMP_DIR)))
    try:
        ref_path = req_dir / "speaker_ref.wav"
        out_path = req_dir / "dubbed_tr.wav"
        ref_path.write_bytes(await speaker_wav.read())

        with gpu_lock:
            seed_val = int(random_seed) if int(random_seed) > 0 else (secrets.randbits(31) if int(candidate_index) > 1 else 0)
            if seed_val > 0:
                torch.manual_seed(seed_val)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_val)
            active_name, model = get_tts_model(tts_engine)
            if active_name in ("voxcpm2", "trendyol_tts"):
                clean_act = (acting_prompt or "").strip().strip("()")
                if use_acting_prompt in ("1", "true", "True") and clean_act and not cleaned_text.startswith("("):
                    vox_text = f"({clean_act}){cleaned_text}"
                else:
                    vox_text = cleaned_text
                with torch.inference_mode():
                    wav_np = model.generate(
                        text=vox_text,
                        reference_wav_path=str(ref_path),
                        cfg_value=float(voxcpm_cfg_value),
                        inference_timesteps=int(voxcpm_timesteps),
                        normalize=False,
                        denoise=False,
                    )
                wav_arr = np.asarray(wav_np, dtype=np.float32)
                if wav_arr.ndim > 1:
                    wav_arr = wav_arr.squeeze(0)
                sr = int(
                    getattr(
                        getattr(model, "tts_model", None),
                        "sample_rate",
                        getattr(model, "sample_rate", 48000),
                    )
                )
                sf.write(str(out_path), wav_arr, sr, subtype="PCM_16")
            elif active_name == "chatterbox_multilingual":
                with torch.inference_mode():
                    wav_tensor = model.generate(
                        text=cleaned_text,
                        language_id=language or "tr",
                        audio_prompt_path=str(ref_path),
                        exaggeration=float(exaggeration),
                        cfg_weight=float(cfg_weight),
                    )
                wav_np = wav_tensor.detach().cpu().float().numpy()
                if wav_np.ndim > 1:
                    wav_np = wav_np.squeeze(0)
                sr = int(getattr(model, "sr", 24000))
                sf.write(str(out_path), wav_np, sr, subtype="PCM_16")
            else:
                model.tts_to_file(
                    text=cleaned_text,
                    speaker_wav=str(ref_path),
                    language=language or "tr",
                    file_path=str(out_path),
                    split_sentences=True,
                )
            _free_vram()

        if not out_path.is_file() or out_path.stat().st_size <= 44:
            raise RuntimeError(f"{active_name} geçerli WAV üretemedi.")
        return Response(content=out_path.read_bytes(), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ses klonlama hatası: {exc}")
    finally:
        shutil.rmtree(req_dir, ignore_errors=True)
        gc.collect()


@app.post("/separate", dependencies=[Depends(verify_bearer)])
async def separate_audio(audio: UploadFile = File(...)):
    if not HAS_DEMUCS:
        raise HTTPException(status_code=404, detail="Demucs bu oturumda etkin değil.")

    req_dir = Path(tempfile.mkdtemp(prefix="sep_", dir=str(WORK_TEMP_DIR)))
    try:
        raw_path = req_dir / (audio.filename or "input.wav")
        prep_wav = req_dir / "input_44k_stereo.wav"
        raw_path.write_bytes(await audio.read())

        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw_path),
             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", str(prep_wav)],
            check=True,
        )
        audio_np, sr = sf.read(str(prep_wav), dtype="float32", always_2d=True)
        wav_tensor = torch.from_numpy(audio_np.T.copy()).unsqueeze(0)

        with gpu_lock:
            d_model = get_demucs_model()
            with torch.no_grad():
                sources = demucs_apply_model(
                    d_model, wav_tensor.to("cuda"), split=True, overlap=0.25, device="cuda"
                )[0]
            names = list(getattr(d_model, "sources", ["drums", "bass", "other", "vocals"]))
            voc_idx = names.index("vocals") if "vocals" in names else -1
            voc_np = sources[voc_idx].detach().cpu().numpy().T
            bg_np = sum(sources[i] for i in range(len(names)) if i != voc_idx).detach().cpu().numpy().T
            torch.cuda.empty_cache()

        voc_buf = io.BytesIO()
        bg_buf = io.BytesIO()
        sf.write(voc_buf, voc_np, sr, format="WAV", subtype="PCM_16")
        sf.write(bg_buf, bg_np, sr, format="WAV", subtype="PCM_16")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("vocals.wav", voc_buf.getvalue())
            zf.writestr("background.wav", bg_buf.getvalue())

        return Response(content=zip_buf.getvalue(), media_type="application/zip")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vokal ayırma hatası: {exc}")
    finally:
        shutil.rmtree(req_dir, ignore_errors=True)
        gc.collect()


# ---- FastAPI'yi Arka Plan Thread'inde Başlat ----
uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=SERVER_PORT, log_level="warning")
uvicorn_server = uvicorn.Server(uvicorn_config)
server_thread = threading.Thread(target=uvicorn_server.run, daemon=True)
server_thread.start()
time.sleep(2.0)

# ---- Ücretsiz cloudflared Quick Tunnel Başlat ----
print("🌐 Cloudflare tüneli açılıyor...")
tunnel_proc = subprocess.Popen(
    [
        "/usr/local/bin/cloudflared",
        "tunnel",
        "--url",
        f"http://127.0.0.1:{SERVER_PORT}",
        "--no-autoupdate",
    ],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

public_url = None
url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
deadline = time.time() + 45.0

while time.time() < deadline:
    line = tunnel_proc.stdout.readline()
    if not line:
        if tunnel_proc.poll() is not None:
            break
        time.sleep(0.1)
        continue
    match = url_pattern.search(line)
    if match:
        public_url = match.group(0)
        break

if not public_url:
    tunnel_proc.terminate()
    raise RuntimeError("cloudflared tünel URL'si alınamadı. Hücreyi tekrar çalıştırın.")

print("\n" + "=" * 74)
print("🎬  SESAI DUBLAJ STÜDYOSU v2 — SİNEMATİK BULUT MOTORU HAZIR!")
print("=" * 74)
print(f"🔗 Colab Bağlantı URL : {public_url}")
print(f"🔑 Bearer Token       : {BEARER_TOKEN}")
print(f"💻 Aktif GPU          : {GPU_NAME} ({VRAM_MB} MB VRAM)")
print(f"🎙️ Ses Motorları      : {', '.join(AVAILABLE_TTS)}")
print(f"🎵 Vokal Ayırma       : {'HTDemucs Hazır' if HAS_DEMUCS else 'Kapalı'}")
print("=" * 74)
print("📌 URL ve Token bilgisini masaüstü uygulamasına yapıştırıp Bağlantıyı Test Et'e basın.\n")

try:
    while tunnel_proc.poll() is None:
        time.sleep(2.0)
except KeyboardInterrupt:
    print("\n🛑 Bulut motoru kullanıcı tarafından durduruluyor...")
finally:
    tunnel_proc.terminate()
    uvicorn_server.should_exit = True
    shutil.rmtree(WORK_TEMP_DIR, ignore_errors=True)
    print("🧹 Geçici dosyalar temizlendi ve sunucu kapatıldı.")
'''

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
            "colab": {
                "name": "colab_motoru.ipynb",
                "provenance": [],
                "gpuType": "L4",
                "machine_shape": "hm",
            },
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "id": "sesai-intro-md",
                "metadata": {},
                "source": _lines(cell_1_md),
            },
            {
                "cell_type": "code",
                "id": "sesai-setup-cell",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": _lines(cell_2_setup),
            },
            {
                "cell_type": "code",
                "id": "sesai-server-cell",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": _lines(cell_3_server),
            },
        ],
    }


def generate_colab_notebook(target_path: Path = DEFAULT_NOTEBOOK_PATH) -> Path:
    """`colab_motoru.ipynb` dosyasını belirtilen konuma yazar."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_colab_notebook_dict()
    tmp = target_path.with_suffix(".ipynb.tmp")
    tmp.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target_path)
    return target_path
