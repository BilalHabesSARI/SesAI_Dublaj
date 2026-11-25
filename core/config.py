"""Taşınabilir Yapılandırma Yöneticisi.

Tüm kullanıcı ayarları (Gemini API Key, ElevenLabs API Key, ses motoru,
VoxCPM2 CFG/adım/duygu yönlendirmesi, Chatterbox duygu/tutarlılık, fonetik
isimler, sabit referans ses, arka plan koruma, tempo vb.) yalnızca
./_veri/ayarlar.json dosyasına otomatik kaydedilir.
Windows Kayıt Defteri (Registry) veya %APPDATA% kesinlikle kullanılmaz.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from typing import Optional

from core.gemini_models import normalize_gemini_model_id
from core.models import EngineMode, TTSEngineType, TTSGenerationOptions
from core.portable_env import DATA_DIR

CONFIG_PATH: Path = DATA_DIR / "ayarlar.json"
CURRENT_CONFIG_VERSION: int = 7


@dataclass
class AppConfig:
    """Uygulama kalıcı ayarları."""

    config_version: int = CURRENT_CONFIG_VERSION
    engine_mode: str = EngineMode.COLAB_CLOUD.value
    tts_engine: str = TTSEngineType.VOXCPM2.value

    # Colab Bulut bağlantısı
    colab_url: str = ""
    colab_token: str = ""
    colab_timeout_sec: int = 300

    # VoxCPM2 & Trendyol-TTS (48 kHz Controllable Voice Cloning) kontrolleri
    voxcpm_cfg_value: float = 2.00
    voxcpm_timesteps: int = 16
    use_acting_prompt: bool = True
    candidate_count: int = 1

    # Yerel GPU Ses Dönüştürme (Seed-VC Speech-to-Speech) kontrolleri
    sts_diffusion_steps: int = 25
    sts_auto_f0_adjust: bool = True
    sts_pitch_shift: int = 0
    sts_cfg_rate: float = 0.70

    # Chatterbox Multilingual (Resemble AI) duygu & tutarlılık kontrolleri
    chatterbox_exaggeration: float = 0.55
    chatterbox_cfg_weight: float = 0.50

    # ElevenLabs API (Multilingual v2 TTS + Speech-to-Speech / Voice Changer)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_stability: float = 0.45
    elevenlabs_similarity: float = 0.80
    elevenlabs_style: float = 0.25
    elevenlabs_remove_bg_noise: bool = True

    # Gemini çeviri & otomatik dublaj yönetmeni ayarları
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_rpm: int = 4
    phonetic_foreign_names: bool = True

    # Dil, klasör ve karakter profili
    source_language: str = "en"
    target_language: str = "tr"
    last_game_dir: str = ""
    project_name: str = ""
    fixed_reference_wav: str = ""
    hitman_runtime_dir: str = r"C:\Program Files\Epic Games\HITMAN3\Runtime"

    # Arka plan koruma (vokal ayırma) & doğal tempo
    preserve_background: bool = False
    bg_mix_volume_db: float = 0.0
    trim_silence: bool = True
    max_tempo_speedup: float = 1.05
    cinematic_timing_lock: bool = False
    auto_apply_to_game: bool = False

    # VAD ve Yerel GPU (RTX 4070 12 GB VRAM) ayarları
    vad_threshold: float = 0.45
    vad_min_speech_ms: int = 220
    whisper_model_local: str = "large-v3-turbo"
    local_unload_between_stages: bool = True

    def get_engine_mode(self) -> EngineMode:
        try:
            return EngineMode(self.engine_mode)
        except ValueError:
            return EngineMode.COLAB_CLOUD

    def get_tts_engine(self) -> TTSEngineType:
        try:
            return TTSEngineType(self.tts_engine)
        except ValueError:
            return TTSEngineType.VOXCPM2

    def get_effective_candidate_count(self) -> int:
        """ElevenLabs API modunda kredi koruması için 1, yerel/açık kaynak motorlarda 1..4 döndürür."""
        if self.get_tts_engine().is_elevenlabs:
            return 1
        return max(1, min(4, int(self.candidate_count)))

    def build_tts_options(
        self,
        acting_prompt: str = "",
        candidate_index: int = 1,
        random_seed: Optional[int] = None,
    ) -> TTSGenerationOptions:
        return TTSGenerationOptions(
            tts_engine=self.get_tts_engine(),
            voxcpm_cfg_value=float(self.voxcpm_cfg_value),
            voxcpm_timesteps=int(self.voxcpm_timesteps),
            use_acting_prompt=bool(self.use_acting_prompt),
            acting_prompt=acting_prompt.strip(),
            sts_diffusion_steps=max(10, min(50, int(self.sts_diffusion_steps))),
            sts_auto_f0_adjust=bool(self.sts_auto_f0_adjust),
            sts_pitch_shift=max(-12, min(12, int(self.sts_pitch_shift))),
            sts_cfg_rate=float(self.sts_cfg_rate),
            exaggeration=float(self.chatterbox_exaggeration),
            cfg_weight=float(self.chatterbox_cfg_weight),
            elevenlabs_voice_id=self.elevenlabs_voice_id.strip(),
            elevenlabs_stability=float(self.elevenlabs_stability),
            elevenlabs_similarity=float(self.elevenlabs_similarity),
            elevenlabs_style=float(self.elevenlabs_style),
            elevenlabs_remove_bg_noise=bool(self.elevenlabs_remove_bg_noise),
            candidate_index=max(1, int(candidate_index)),
            random_seed=random_seed,
        )


class ConfigManager:
    """./_veri/ayarlar.json okuma/yazma yöneticisi."""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = path

    def load(self) -> AppConfig:
        if not self.path.exists():
            return AppConfig()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return AppConfig()
            valid_fields = AppConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in raw.items() if k in valid_fields}

            raw_model = str(filtered.get("gemini_model") or "").strip()
            normalized_model = normalize_gemini_model_id(raw_model)
            needs_resave = normalized_model != raw_model
            filtered["gemini_model"] = normalized_model

            prev_ver = int(filtered.get("config_version") or 1)
            if prev_ver < 2:
                if abs(float(filtered.get("max_tempo_speedup", 1.40)) - 1.40) < 1e-4:
                    filtered["max_tempo_speedup"] = 1.05
            if prev_ver < 3:
                if int(filtered.get("gemini_rpm", 15)) == 15:
                    filtered["gemini_rpm"] = 4
            if prev_ver < 5:
                # v5 geçişi: yeni varsayılan motor VoxCPM2 (48 kHz)
                if filtered.get("tts_engine") in (None, "", "chatterbox_multilingual"):
                    filtered["tts_engine"] = TTSEngineType.VOXCPM2.value
            if prev_ver < 7:
                # v7 geçişi: VoxCPM2 varsayılan CFG değeri 2.00
                if abs(float(filtered.get("voxcpm_cfg_value", 1.7)) - 1.7) < 1e-4:
                    filtered["voxcpm_cfg_value"] = 2.00
            if prev_ver < CURRENT_CONFIG_VERSION:
                filtered["config_version"] = CURRENT_CONFIG_VERSION
                needs_resave = True

            cfg = AppConfig(**filtered)
            cfg.candidate_count = max(1, min(4, int(cfg.candidate_count or 1)))
            if needs_resave:
                self.save(cfg)
            return cfg
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return AppConfig()

    def save(self, config: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        config.config_version = CURRENT_CONFIG_VERSION
        config.gemini_model = normalize_gemini_model_id(config.gemini_model)
        payload = json.dumps(asdict(config), ensure_ascii=False, indent=2)
        tmp_path = self.path.with_suffix(".json.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.path)
