"""SesAI Dublaj Stüdyosu - Veri Modelleri ve Durum Tanımları."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class EngineMode(str, Enum):
    """Hibrit işlem altyapısı modları."""

    COLAB_CLOUD = "colab_cloud"
    LOCAL_GPU = "local_gpu"

    @property
    def label_tr(self) -> str:
        if self is EngineMode.COLAB_CLOUD:
            return "Google Colab Pro (Bulut Modu)"
        return "Yerel GPU Modu (CUDA - Yedek)"


class TTSEngineType(str, Enum):
    """Seçilebilir yeni nesil seslendirme / ses klonlama motorları."""

    VOXCPM2 = "voxcpm2"
    TRENDYOL_TTS = "trendyol_tts"
    CHATTERBOX_MULTILINGUAL = "chatterbox_multilingual"
    XTTS_V2 = "xtts_v2"
    ELEVENLABS_TTS = "elevenlabs_tts"
    ELEVENLABS_STS = "elevenlabs_sts"

    @property
    def label_tr(self) -> str:
        mapping = {
            TTSEngineType.VOXCPM2: "VoxCPM2 (48 kHz — Duygu & Ses Klonlama Önerilen)",
            TTSEngineType.TRENDYOL_TTS: "Trendyol-TTS (VoxCPM2 Türkçe İnce Ayar — 48 kHz)",
            TTSEngineType.CHATTERBOX_MULTILINGUAL: "Chatterbox Multilingual (Resemble AI)",
            TTSEngineType.XTTS_V2: "Coqui XTTS v2 (Klasik Açık Kaynak)",
            TTSEngineType.ELEVENLABS_TTS: "ElevenLabs API — Multilingual v2 (TTS)",
            TTSEngineType.ELEVENLABS_STS: "ElevenLabs API — Speech-to-Speech (Voice Changer)",
        }
        return mapping[self]

    @property
    def is_voxcpm(self) -> bool:
        return self in (TTSEngineType.VOXCPM2, TTSEngineType.TRENDYOL_TTS)

    @property
    def voxcpm_repo_id(self) -> str:
        if self is TTSEngineType.TRENDYOL_TTS:
            return "Trendyol/Trendyol-TTS"
        return "openbmb/VoxCPM2"

    @property
    def is_elevenlabs(self) -> bool:
        return self in (TTSEngineType.ELEVENLABS_TTS, TTSEngineType.ELEVENLABS_STS)

    @property
    def is_open_source(self) -> bool:
        return self in (
            TTSEngineType.VOXCPM2,
            TTSEngineType.TRENDYOL_TTS,
            TTSEngineType.CHATTERBOX_MULTILINGUAL,
            TTSEngineType.XTTS_V2,
        )


class LineStatus(str, Enum):
    """Her ses dosyasının iş hattındaki durumu."""

    PENDING = "pending"
    SEPARATING_VOCALS = "separating_vocals"
    VAD_CHECKING = "vad_checking"
    SKIPPED_NON_SPEECH = "skipped_non_speech"
    TRANSCRIBING = "transcribing"
    SKIPPED_EMPTY = "skipped_empty"
    TRANSLATING = "translating"
    SYNTHESIZING = "synthesizing"
    SYNCING = "syncing"
    READY_PREVIEW = "ready_preview"
    APPLIED = "applied"
    ERROR = "error"
    CANCELLED = "cancelled"

    @property
    def label_tr(self) -> str:
        mapping = {
            LineStatus.PENDING: "Bekliyor",
            LineStatus.SEPARATING_VOCALS: "Vokal Ayrılıyor",
            LineStatus.VAD_CHECKING: "VAD Kontrolü",
            LineStatus.SKIPPED_NON_SPEECH: "Atlandı (Konuşma Yok)",
            LineStatus.TRANSCRIBING: "Deşifre Ediliyor",
            LineStatus.SKIPPED_EMPTY: "Atlandı (Metin Boş)",
            LineStatus.TRANSLATING: "Çevriliyor (Gemini)",
            LineStatus.SYNTHESIZING: "Seslendiriliyor",
            LineStatus.SYNCING: "Senkronize Ediliyor",
            LineStatus.READY_PREVIEW: "Önizlemeye Hazır",
            LineStatus.APPLIED: "Oyuna Uygulandı",
            LineStatus.ERROR: "Hata",
            LineStatus.CANCELLED: "İptal Edildi",
        }
        return mapping[self]


@dataclass
class TTSGenerationOptions:
    """Ses sentezleme, oyunculuk/duygu ve Seed-VC STS kontrol parametreleri."""

    tts_engine: TTSEngineType = TTSEngineType.VOXCPM2
    voxcpm_cfg_value: float = 2.00
    voxcpm_timesteps: int = 16
    use_acting_prompt: bool = True
    acting_prompt: str = ""
    sts_diffusion_steps: int = 25
    sts_auto_f0_adjust: bool = True
    sts_pitch_shift: int = 0
    sts_cfg_rate: float = 0.70
    exaggeration: float = 0.55
    cfg_weight: float = 0.50
    elevenlabs_voice_id: str = ""
    elevenlabs_stability: float = 0.45
    elevenlabs_similarity: float = 0.80
    elevenlabs_style: float = 0.25
    elevenlabs_remove_bg_noise: bool = True
    candidate_index: int = 1
    random_seed: Optional[int] = None

    def format_voxcpm_text(self, text: str) -> str:
        """VoxCPM2 Controllable Voice Cloning için `(acting_prompt)tr_text` metnini üretir."""
        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return ""
        if self.use_acting_prompt and self.acting_prompt and self.acting_prompt.strip():
            clean_act = self.acting_prompt.strip().strip("()")
            if clean_act:
                return f"({clean_act}){cleaned_text}"
        return cleaned_text


@dataclass
class DubCandidate:
    """Bir satır için üretilen aday dublaj ses kaydı ve Whisper hakem puanı."""

    candidate_index: int  # 1..N
    raw_wav_path: Optional[Path] = None
    synced_wav_path: Optional[Path] = None
    whisper_transcript: str = ""
    text_accuracy_score: float = 100.0  # 0..100 (%75 ağırlık)
    duration_fit_score: float = 100.0   # 0..100 (%25 ağırlık)
    total_score: float = 100.0          # 0..100 (0.75 * text + 0.25 * dur)
    raw_duration_ms: int = 0
    synced_duration_ms: int = 0
    tempo_factor: float = 1.0
    is_best: bool = False

    @property
    def score_pct_int(self) -> int:
        return max(0, min(100, int(round(self.total_score))))

    @property
    def button_label(self) -> str:
        return f"Aday {self.candidate_index} ({self.score_pct_int}%)"


@dataclass
class AudioMetadata:
    """Bir ses dosyasının teknik özellikleri."""

    duration_ms: int
    lufs: float
    sample_rate: int
    channels: int
    codec: str
    format_ext: str
    bit_rate: Optional[int] = None
    sample_fmt: Optional[str] = None

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def channel_label(self) -> str:
        if self.channels == 1:
            return "Mono"
        if self.channels == 2:
            return "Stereo"
        return f"{self.channels} Kanal"


@dataclass
class DiscoveredAudio:
    """Adaptör tarafından bulunan oyun ses dosyası."""

    rel_path: str
    original_path: Path
    adapter_name: str
    archive_container: Optional[Path] = None
    archive_entry_id: Optional[str] = None


@dataclass
class DialogueLine:
    """İş hattında takip edilen tek bir replik/ses dosyası."""

    line_id: str
    rel_path: str
    original_path: Path
    adapter_name: str = "loose_files"
    metadata: Optional[AudioMetadata] = None
    vad_speech_ratio: float = 0.0
    vad_max_prob: float = 0.0
    is_speech: bool = False
    source_text: str = ""
    source_lang: str = "en"
    translated_text: str = ""
    acting_prompt: str = ""
    from_cache: bool = False
    prepared_ref_wav: Optional[Path] = None
    raw_dub_wav: Optional[Path] = None
    synced_preview_path: Optional[Path] = None
    dub_duration_ms: int = 0
    tempo_factor: float = 1.0
    trimmed_silence_ms: int = 0
    background_preserved: bool = False
    status: LineStatus = LineStatus.PENDING
    error_message: str = ""
    applied: bool = False
    candidates: list[DubCandidate] = field(default_factory=list)
    selected_candidate_index: int = 1
    best_candidate_score: float = 0.0

    def get_candidate(self, candidate_index: int) -> Optional[DubCandidate]:
        for cand in self.candidates:
            if cand.candidate_index == candidate_index:
                return cand
        return None


@dataclass
class PipelineStats:
    """Çalışma özeti istatistikleri."""

    total_scanned: int = 0
    speech_detected: int = 0
    skipped_non_speech: int = 0
    transcribed: int = 0
    skipped_empty: int = 0
    translated: int = 0
    cache_hits: int = 0
    synthesized: int = 0
    synced: int = 0
    applied: int = 0
    errors: int = 0


@dataclass
class EngineHealth:
    """Bulut, Yerel GPU veya API motorunun sağlık/bağlantı bilgisi."""

    ok: bool
    mode: EngineMode
    message: str
    gpu_name: str = ""
    vram_total_mb: int = 0
    whisper_model: str = ""
    tts_model: str = ""
    latency_ms: int = 0
    available_tts_engines: list[str] = field(default_factory=list)
    supports_vocal_separation: bool = False


@dataclass
class BackupFileEntry:
    """Yedeklenen tek bir orijinal dosyanın SHA-256 kaydı."""

    rel_path: str
    original_abs_path: str
    backup_rel_path: str
    sha256: str
    size_bytes: int


@dataclass
class BackupManifest:
    """Bir oyun projesinin SHA-256 doğrulamalı yedek manifestosu."""

    project_name: str
    game_root: str
    created_at: str
    updated_at: str
    files: dict[str, BackupFileEntry] = field(default_factory=dict)


@dataclass
class RestoreResult:
    """Orijinale Dön (SHA-256) işleminin sonucu."""

    success: bool
    project_name: str
    total_files: int
    restored_files: int
    verified_files: int
    backup_deleted: bool
    errors: list[str] = field(default_factory=list)
