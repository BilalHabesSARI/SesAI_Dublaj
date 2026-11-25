"""SesAI Dublaj Stüdyosu - Sinematik Koyu Tema Ana Pencere (PySide6).

Özellikler:
- Ferah, okunaklı ve sıkışık olmayan 3 sekmeli üst kontrol paneli.
- 8 Ücretsiz Gemini Modeli, Otomatik Model Geçişi (Fallback) ve her gün 11:00 TRT
  sıfırlamalı Canlı Günlük Kota Takip Paneli (yeşil/kırmızı model sayaçlarıyla).
- Seçilebilir ses motorları: Chatterbox Multilingual (varsayılan), Coqui XTTS v2,
  ElevenLabs API (Multilingual v2 TTS + Speech-to-Speech / Voice Changer).
- Sabit 10–15 sn karakter referans WAV seçici ve "Arka Planı Koru (Vokal Ayırma)".
- Doğal tempo (varsayılan 1.05x, 1.00x = hızlandırma kapalı) + akıllı sessizlik kırpma.
- Tüm ayar, API anahtarı ve kota verilerini ./_veri/ içine otomatik kaydetme.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.backup_manager import (
    BackupManager,
    infer_smart_project_name,
    is_generic_audio_subfolder,
    sanitize_project_name,
)
from core.config import AppConfig, ConfigManager
from core.ffmpeg_manager import FFmpegManager
from core.gemini_models import (
    GEMINI_FREE_MODELS,
    GeminiQuotaManager,
    QuotaSnapshot,
    normalize_gemini_model_id,
    resolve_model_spec,
)
from core.glossary import GlossaryManager
from core.hitman_manager import (
    DEFAULT_HITMAN_RUNTIME_DIR,
    HitmanExtractResult,
    HitmanSceneProfile,
    SceneRuntimeStatus,
    discover_hitman_scenes,
    get_scenes_runtime_status,
)
from core.models import (
    DialogueLine,
    EngineHealth,
    EngineMode,
    LineStatus,
    PipelineStats,
    RestoreResult,
    TTSEngineType,
)
from core.pipeline import DubbingPipeline
from core.portable_env import (
    APP_ROOT,
    cleanup_temp_dir,
    get_temp_size_bytes,
    verify_portable_isolation,
)
from core.table_drafts import TableDraftManager
from core.translation_cache import TranslationCache
from core.translator import GeminiTranslator
from core.tts_pronunciation import TTSPronunciationManager
from engines.base import SpeechEngine
from engines.colab_engine import ColabCloudEngine
from engines.colab_notebook_generator import DEFAULT_NOTEBOOK_PATH, generate_colab_notebook
from engines.elevenlabs_engine import ElevenLabsEngine
from engines.local_gpu_engine import LocalGPUEngine
from ui.audio_player import AudioPreviewPlayer
from ui.glossary_dialog import GlossaryDialog
from ui.mic_recorder import MicrophoneNotAvailableError, PortableMicRecorder
from ui.workers import (
    ApplyDubbingWorker,
    BatchRedubAllWorker,
    ElevenLabsVoicesWorker,
    HealthCheckWorker,
    HitmanExtractWorker,
    HitmanRepackInstallWorker,
    HitmanUninstallWorker,
    PipelineWorker,
    RestoreOriginalWorker,
    SingleLineRedubWorker,
    SingleLineSTSWorker,
)

FOOTER_SIGNATURE_TEXT = (
    "Bilal Habeş SARI 🇹🇷 tarafından Google Antigravity kullanılarak geliştirilmiştir"
)


class MainWindow(QMainWindow):
    """SesAI Dublaj Stüdyosu ana penceresi."""

    quota_changed_signal = Signal(object)  # QuotaSnapshot (arka plan thread'lerinden güvenli)

    COL_FILE = 0
    COL_STATUS = 1
    COL_DUR = 2
    COL_LUFS = 3
    COL_ORIG = 4
    COL_TR = 5
    COL_ACTING = 6
    COL_TEMPO = 7

    def __init__(
        self,
        config_mgr: Optional[ConfigManager] = None,
        quota_mgr: Optional[GeminiQuotaManager] = None,
        draft_mgr: Optional[TableDraftManager] = None,
        pronunciation_mgr: Optional[TTSPronunciationManager] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("SesAI Dublaj Stüdyosu — Sinematik Yapay Zeka Oyun Dublajı")
        self.resize(1380, 910)
        self.setMinimumSize(1140, 760)

        # Çekirdek yöneticiler
        self.config_mgr = config_mgr or ConfigManager()
        self.config: AppConfig = self.config_mgr.load()
        self.quota_mgr = quota_mgr or GeminiQuotaManager()
        self.quota_mgr.set_active_model(self.config.gemini_model)
        self.draft_mgr = draft_mgr or TableDraftManager()

        self.glossary_mgr = GlossaryManager()
        self.pronunciation_mgr = pronunciation_mgr or TTSPronunciationManager()
        self.cache_mgr = TranslationCache()
        self.backup_mgr = BackupManager()
        self.ffmpeg_mgr = FFmpegManager()
        self.audio_player = AudioPreviewPlayer(self)
        self.mic_recorder = PortableMicRecorder(self)

        # Çalışma durumu
        self._lines: list[DialogueLine] = []
        self._line_row_map: dict[str, int] = {}
        self._quota_pill_buttons: dict[str, QPushButton] = {}
        self._candidate_buttons: list[QPushButton] = []
        self._active_worker: Optional[PipelineWorker] = None
        self._health_worker: Optional[HealthCheckWorker] = None
        self._el_voices_worker: Optional[ElevenLabsVoicesWorker] = None
        self._apply_worker: Optional[ApplyDubbingWorker] = None
        self._restore_worker: Optional[RestoreOriginalWorker] = None
        self._redub_worker: Optional[SingleLineRedubWorker] = None
        self._redub_all_worker: Optional[BatchRedubAllWorker] = None
        self._sts_worker: Optional[SingleLineSTSWorker] = None
        self._hitman_extract_worker: Optional[HitmanExtractWorker] = None
        self._hitman_install_worker: Optional[HitmanRepackInstallWorker] = None
        self._hitman_uninstall_worker: Optional[HitmanUninstallWorker] = None
        self._hitman_scenes: list[HitmanSceneProfile] = []
        self._hitman_statuses: dict[str, SceneRuntimeStatus] = {}
        self._active_pipeline: Optional[DubbingPipeline] = None
        self._updating_table = False
        self._loading_ui = True
        self._syncing_slider = False

        # Otomatik yerel kayıt zamanlayıcısı (debounce 350ms -> ./_veri/ayarlar.json)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(350)
        self._autosave_timer.timeout.connect(self._perform_autosave)

        # Saat 11:00 TRT geçişini canlı yakalamak için dakikalık kota yenileme zamanlayıcısı
        self._quota_tick_timer = QTimer(self)
        self._quota_tick_timer.setInterval(30_000)
        self._quota_tick_timer.timeout.connect(
            lambda: self._apply_quota_snapshot_to_ui(self.quota_mgr.get_snapshot())
        )
        self._quota_tick_timer.start()

        self._build_ui()
        self._load_config_into_ui()
        self._loading_ui = False
        self._connect_autosave_signals()

        self.quota_changed_signal.connect(self._apply_quota_snapshot_to_ui)
        self.quota_mgr.add_listener(self._on_quota_manager_notify)

        self._refresh_status_badges()
        self._refresh_summary_strip()
        self._apply_quota_snapshot_to_ui(self.quota_mgr.get_snapshot())

        self.audio_player.status_changed.connect(self.preview_status_label.setText)
        self.mic_recorder.status_changed.connect(self.preview_status_label.setText)

        iso = verify_portable_isolation()
        all_isolated = all(iso.values())
        self._log(
            "SesAI Dublaj Stüdyosu başlatıldı | "
            f"Taşınabilir izolasyon: {'%100 Doğrulandı' if all_isolated else 'Kontrol Gerekli'} | "
            f"FFmpeg: {self.ffmpeg_mgr.get_version_string()}"
        )
        snap = self.quota_mgr.get_snapshot()
        self._log(
            f"Aktif Ses Motoru: {self.config.get_tts_engine().label_tr} | "
            f"Gemini Modeli: {snap.active_short_name} ({snap.active_model_id}) | "
            f"Günlük Kota Kalan: {snap.total_remaining}/{snap.total_limit} (Yenilenme: {snap.next_reset_label})"
        )
        self._try_load_project_draft()

    # ==========================================================================
    # ARAYÜZ KURULUMU (FERAH VE OKUNAKLI YERLEŞİM + CANLI KOTA PANELİ)
    # ==========================================================================
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 10, 14, 0)
        root_layout.setSpacing(8)

        # 1. Üst Başlık ve Altyapı Modu Seçici
        root_layout.addLayout(self._build_header_bar())

        # 2. Ferah 4 Sekmeli Üst Kontrol Paneli
        self.control_tabs = QTabWidget()
        self.control_tabs.setObjectName("ControlTabs")
        self.control_tabs.addTab(self._build_tab_main_setup(), "🎬 Motor, Bağlantı ve Proje")
        self.control_tabs.addTab(
            self._build_tab_voice_cinema(), "🎭 Duygu, Sabit Referans ve Arka Plan"
        )
        self.control_tabs.addTab(
            self._build_tab_tempo_translation(), "🎚️ Doğal Tempo, Sessizlik ve Çeviri"
        )
        self.control_tabs.addTab(
            self._build_tab_hitman_rpkg(), "🎯 HITMAN 3 (RPKG)"
        )
        self.control_tabs.setMaximumHeight(248)
        root_layout.addWidget(self.control_tabs)

        # 3. Canlı Özet ve 8 Modelli Günlük Kota Takip Paneli
        root_layout.addWidget(self._build_live_quota_and_summary_panel())

        # 4. Ana Çalışma Alanı (Tablo + Önizleme / İlerleme + Canlı Log)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_table_and_preview_panel())
        splitter.addWidget(self._build_progress_and_log_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, 1)

        # 5. Ana Eylem Butonları
        root_layout.addLayout(self._build_action_bar())

        # 6. İmza Alt Bilgi (Footer)
        footer = QLabel(FOOTER_SIGNATURE_TEXT)
        footer.setObjectName("FooterSignature")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(footer)

    def _build_header_bar(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(10)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title = QLabel("🎙️ SesAI Dublaj Stüdyosu")
        title.setObjectName("AppTitle")
        subtitle = QLabel(
            "VoxCPM2 48 kHz • Trendyol-TTS • Chatterbox • XTTS v2 • ElevenLabs • Otomatik Dublaj Yönetmeni • SHA-256 Yedekli"
        )
        subtitle.setObjectName("AppSubtitle")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col)

        header.addStretch()

        self.autosave_label = QLabel("")
        self.autosave_label.setObjectName("MutedLabel")
        header.addWidget(self.autosave_label)

        self.portable_pill = QLabel("🔒 %100 Taşınabilir")
        self.portable_pill.setObjectName("StatusPillInfo")
        self.portable_pill.setToolTip(
            f"Tüm geçici dosyalar, modeller, yedekler ve ayarlar yalnızca şu dizinde tutulur:\n{APP_ROOT}"
        )
        header.addWidget(self.portable_pill)

        self.ffmpeg_pill = QLabel("FFmpeg kontrol ediliyor...")
        self.ffmpeg_pill.setObjectName("StatusPillOk")
        header.addWidget(self.ffmpeg_pill)

        mode_box = QHBoxLayout()
        mode_box.setSpacing(6)
        self.mode_colab_btn = QPushButton("☁️ Colab Bulut Modu")
        self.mode_colab_btn.setObjectName("ModeBtn")
        self.mode_colab_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_colab_btn.clicked.connect(lambda: self._set_engine_mode(EngineMode.COLAB_CLOUD))

        self.mode_local_btn = QPushButton("🎮 Yerel GPU Modu (CUDA)")
        self.mode_local_btn.setObjectName("ModeGpuBtn")
        self.mode_local_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_local_btn.clicked.connect(lambda: self._set_engine_mode(EngineMode.LOCAL_GPU))

        mode_box.addWidget(self.mode_colab_btn)
        mode_box.addWidget(self.mode_local_btn)
        header.addLayout(mode_box)

        return header

    # --- SEKME 1: MOTOR, BAĞLANTI VE PROJE ---
    def _build_tab_main_setup(self) -> QWidget:
        tab = QWidget()
        row = QHBoxLayout(tab)
        row.setContentsMargins(10, 10, 10, 10)
        row.setSpacing(12)

        # Sol Kart: Seslendirme Motoru & Bağlantı
        left_card = QFrame()
        left_card.setObjectName("Card")
        lg = QVBoxLayout(left_card)
        lg.setContentsMargins(12, 10, 12, 10)
        lg.setSpacing(8)

        eng_top = QHBoxLayout()
        eng_top.setSpacing(8)
        eng_lbl = QLabel("Ses Motoru:")
        eng_lbl.setObjectName("SectionTitle")
        self.tts_engine_combo = QComboBox()
        for eng_type in TTSEngineType:
            self.tts_engine_combo.addItem(eng_type.label_tr, eng_type.value)
        self.tts_engine_combo.setMinimumWidth(290)
        self.tts_engine_combo.currentIndexChanged.connect(self._on_tts_engine_changed)

        self.engine_health_label = QLabel("Test edilmedi")
        self.engine_health_label.setObjectName("MutedLabel")

        eng_top.addWidget(eng_lbl)
        eng_top.addWidget(self.tts_engine_combo, 1)
        eng_top.addWidget(self.engine_health_label)
        lg.addLayout(eng_top)

        self.engine_stack = QStackedWidget()

        # Sayfa 0: Colab Pro Bulut Bağlantısı
        colab_page = QWidget()
        cg = QGridLayout(colab_page)
        cg.setContentsMargins(0, 0, 0, 0)
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(7)

        cg.addWidget(QLabel("Colab URL:"), 0, 0)
        self.colab_url_input = QLineEdit()
        self.colab_url_input.setPlaceholderText("https://ornek-tunel.trycloudflare.com")
        cg.addWidget(self.colab_url_input, 0, 1, 1, 2)

        cg.addWidget(QLabel("Bearer Token:"), 1, 0)
        self.colab_token_input = QLineEdit()
        self.colab_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.colab_token_input.setPlaceholderText("Colab defterinin ürettiği tek kullanımlık anahtar")
        cg.addWidget(self.colab_token_input, 1, 1)

        self.toggle_token_btn = QPushButton("👁️")
        self.toggle_token_btn.setFixedWidth(38)
        self.toggle_token_btn.setToolTip("Token'ı Göster / Gizle")
        self.toggle_token_btn.clicked.connect(
            lambda: self._toggle_password_echo(self.colab_token_input)
        )
        cg.addWidget(self.toggle_token_btn, 1, 2)

        colab_btns = QHBoxLayout()
        colab_btns.setSpacing(8)
        self.test_conn_btn = QPushButton("🔌 Bağlantıyı Test Et")
        self.test_conn_btn.clicked.connect(self._on_test_connection)
        self.gen_notebook_btn = QPushButton("📓 Colab Defterini Oluştur")
        self.gen_notebook_btn.clicked.connect(self._on_generate_colab_notebook)
        self.open_notebook_dir_btn = QPushButton("📂 Defteri Göster")
        self.open_notebook_dir_btn.clicked.connect(self._on_reveal_colab_notebook)
        colab_btns.addWidget(self.test_conn_btn)
        colab_btns.addWidget(self.gen_notebook_btn)
        colab_btns.addWidget(self.open_notebook_dir_btn)
        cg.addLayout(colab_btns, 2, 0, 1, 3)
        self.engine_stack.addWidget(colab_page)

        # Sayfa 1: Yerel CUDA GPU (RTX 4070 12GB)
        local_page = QWidget()
        lgrid = QGridLayout(local_page)
        lgrid.setContentsMargins(0, 0, 0, 0)
        lgrid.setHorizontalSpacing(8)
        lgrid.setVerticalSpacing(7)

        lgrid.addWidget(QLabel("Whisper Modeli:"), 0, 0)
        self.local_whisper_combo = QComboBox()
        self.local_whisper_combo.addItems(["large-v3-turbo", "medium", "small", "large-v3"])
        self.local_whisper_combo.setMinimumWidth(160)
        lgrid.addWidget(self.local_whisper_combo, 0, 1)

        self.local_unload_chk = QCheckBox(
            "Aşamalar arası VRAM boşalt (RTX 4070 12 GB için önerilir)"
        )
        self.local_unload_chk.setChecked(True)
        lgrid.addWidget(self.local_unload_chk, 1, 0, 1, 2)

        local_btn_row = QHBoxLayout()
        local_btn_row.setSpacing(8)
        self.test_local_gpu_btn = QPushButton("🎮 Yerel CUDA Durumunu Test Et")
        self.test_local_gpu_btn.clicked.connect(self._on_test_connection)
        local_info = QLabel("Modeller yalnızca ./_modeller/ dizinine kaydedilir.")
        local_info.setObjectName("MutedLabel")
        local_btn_row.addWidget(self.test_local_gpu_btn)
        local_btn_row.addWidget(local_info, 1)
        lgrid.addLayout(local_btn_row, 2, 0, 1, 2)
        self.engine_stack.addWidget(local_page)

        # Sayfa 2: ElevenLabs API (Multilingual v2 TTS + Speech-to-Speech)
        el_page = QWidget()
        eg = QGridLayout(el_page)
        eg.setContentsMargins(0, 0, 0, 0)
        eg.setHorizontalSpacing(8)
        eg.setVerticalSpacing(7)

        eg.addWidget(QLabel("ElevenLabs Key:"), 0, 0)
        self.elevenlabs_key_input = QLineEdit()
        self.elevenlabs_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.elevenlabs_key_input.setPlaceholderText("ElevenLabs API anahtarınız (sk_...)")
        eg.addWidget(self.elevenlabs_key_input, 0, 1)

        self.toggle_el_key_btn = QPushButton("👁️")
        self.toggle_el_key_btn.setFixedWidth(38)
        self.toggle_el_key_btn.setToolTip("ElevenLabs Anahtarını Göster / Gizle")
        self.toggle_el_key_btn.clicked.connect(
            lambda: self._toggle_password_echo(self.elevenlabs_key_input)
        )
        eg.addWidget(self.toggle_el_key_btn, 0, 2)

        eg.addWidget(QLabel("Ses / Voice ID:"), 1, 0)
        self.elevenlabs_voice_combo = QComboBox()
        self.elevenlabs_voice_combo.setEditable(True)
        self.elevenlabs_voice_combo.lineEdit().setPlaceholderText(
            "Boş bırak = Referans WAV'dan Otomatik Klonla (IVC)"
        )
        eg.addWidget(self.elevenlabs_voice_combo, 1, 1, 1, 2)

        el_btns = QHBoxLayout()
        el_btns.setSpacing(8)
        self.test_el_btn = QPushButton("🔌 ElevenLabs Test Et")
        self.test_el_btn.clicked.connect(self._on_test_connection)
        self.load_el_voices_btn = QPushButton("🔄 Hesap Seslerini Yükle")
        self.load_el_voices_btn.clicked.connect(self._on_load_elevenlabs_voices)
        el_btns.addWidget(self.test_el_btn)
        el_btns.addWidget(self.load_el_voices_btn)
        el_btns.addStretch()
        eg.addLayout(el_btns, 2, 0, 1, 3)
        self.engine_stack.addWidget(el_page)

        lg.addWidget(self.engine_stack)

        # Sağ Kart: Gemini Çeviri (8 Ücretsiz Model) & Oyun Klasörü
        right_card = QFrame()
        right_card.setObjectName("Card")
        rg = QGridLayout(right_card)
        rg.setContentsMargins(12, 10, 12, 10)
        rg.setHorizontalSpacing(8)
        rg.setVerticalSpacing(8)

        r_title = QLabel("🧠 Gemini Çeviri (8 Ücretsiz Model • Otomatik Geçiş), Sözlük ve Klasör")
        r_title.setObjectName("SectionTitle")
        rg.addWidget(r_title, 0, 0, 1, 4)

        rg.addWidget(QLabel("Gemini API Key:"), 1, 0)
        gem_row = QHBoxLayout()
        gem_row.setSpacing(6)
        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key_input.setPlaceholderText("Google AI Studio Gemini API Anahtarı")

        self.toggle_gemini_btn = QPushButton("👁️")
        self.toggle_gemini_btn.setFixedWidth(38)
        self.toggle_gemini_btn.setToolTip("API Anahtarını Göster / Gizle")
        self.toggle_gemini_btn.clicked.connect(
            lambda: self._toggle_password_echo(self.gemini_key_input)
        )

        self.gemini_model_combo = QComboBox()
        for spec in GEMINI_FREE_MODELS:
            self.gemini_model_combo.addItem(
                f"{spec.short_name} ({spec.daily_limit}/gün)",
                spec.api_id,
            )
        self.gemini_model_combo.setMinimumWidth(195)
        self.gemini_model_combo.setToolTip(
            "429/503/404 veya günlük kota dolumunda otomatik olarak sıradaki müsait modele geçilir"
        )
        self.gemini_model_combo.currentIndexChanged.connect(self._on_gemini_model_combo_changed)

        self.glossary_btn = QPushButton("📖 Sözlük (0 | TTS:1)")
        self.glossary_btn.setMinimumWidth(144)
        self.glossary_btn.setToolTip(
            "Büyük/küçük harf duyarlı kalıcı çeviri sözlüğü ve TTS Telaffuz Sözlüğü (ör. ICA -> Ay-Si-Ey)"
        )
        self.glossary_btn.clicked.connect(self._on_open_glossary)

        gem_row.addWidget(self.gemini_key_input, 2)
        gem_row.addWidget(self.toggle_gemini_btn)
        gem_row.addWidget(self.gemini_model_combo, 1)
        gem_row.addWidget(self.glossary_btn)
        rg.addLayout(gem_row, 1, 1, 1, 3)

        rg.addWidget(QLabel("Oyun Klasörü:"), 2, 0)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        self.game_dir_input = QLineEdit()
        self.game_dir_input.setPlaceholderText(
            "Konuşma dosyalarını içeren oyun veya ses klasörünü seçin (.wav, .ogg, .mp3, .flac)"
        )
        self.game_dir_input.textChanged.connect(self._on_game_dir_changed)

        self.browse_dir_btn = QPushButton("📁 Klasör Seç")
        self.browse_dir_btn.setMinimumWidth(112)
        self.browse_dir_btn.clicked.connect(self._on_browse_folder)

        folder_row.addWidget(self.game_dir_input, 1)
        folder_row.addWidget(self.browse_dir_btn)
        rg.addLayout(folder_row, 2, 1, 1, 3)

        rg.addWidget(QLabel("Proje / Yedek:"), 3, 0)
        proj_row = QHBoxLayout()
        proj_row.setSpacing(10)
        self.project_name_input = QLineEdit()
        self.project_name_input.setPlaceholderText("./_yedekler/<proje_adi>")
        self.project_name_input.setMinimumWidth(180)
        self.project_name_input.setMaximumWidth(240)

        self.quick_bg_chk = QCheckBox("🎵 Arka Planı Koru (Vokal Ayırma)")
        self.quick_bg_chk.setToolTip(
            "Konuşmayı müzik/ses efektlerinden ayırıp dublaj sonrası orijinal efektlerle yeniden miksler"
        )
        self.quick_bg_chk.toggled.connect(self._sync_bg_from_quick)

        proj_row.addWidget(self.project_name_input)
        proj_row.addWidget(self.quick_bg_chk)
        proj_row.addStretch()
        rg.addLayout(proj_row, 3, 1, 1, 3)

        row.addWidget(left_card, 11)
        row.addWidget(right_card, 12)
        return tab

    # --- SEKME 2: DUYGU, SABİT REFERANS VE ARKA PLAN ---
    def _build_tab_voice_cinema(self) -> QWidget:
        tab = QWidget()
        row = QHBoxLayout(tab)
        row.setContentsMargins(10, 10, 10, 10)
        row.setSpacing(12)

        emo_card = QFrame()
        emo_card.setObjectName("Card")
        eg = QGridLayout(emo_card)
        eg.setContentsMargins(12, 8, 12, 8)
        eg.setHorizontalSpacing(10)
        eg.setVerticalSpacing(6)

        emo_title = QLabel(
            "🎭 VoxCPM2 (48 kHz) Oyunculuk Yönetmeni • Chatterbox • ElevenLabs İnce Ayar"
        )
        emo_title.setObjectName("SectionTitle")
        eg.addWidget(emo_title, 0, 0, 1, 4)

        vox_row = QHBoxLayout()
        vox_row.setSpacing(10)
        vox_row.addWidget(QLabel("VoxCPM2 CFG Değeri:"))
        self.voxcpm_cfg_spin = QDoubleSpinBox()
        self.voxcpm_cfg_spin.setRange(1.00, 2.50)
        self.voxcpm_cfg_spin.setSingleStep(0.05)
        self.voxcpm_cfg_spin.setDecimals(2)
        self.voxcpm_cfg_spin.setValue(2.00)
        self.voxcpm_cfg_spin.setMinimumWidth(88)
        self.voxcpm_cfg_spin.setToolTip(
            "VoxCPM2 / Trendyol-TTS 48 kHz CFG Değeri (Varsayılan: 2.00; 1.50 = daha esnek tonlama, 2.00 = daha temiz ve kararlı)"
        )
        vox_row.addWidget(self.voxcpm_cfg_spin)

        vox_row.addWidget(QLabel("Adım Sayısı:"))
        self.voxcpm_steps_spin = QSpinBox()
        self.voxcpm_steps_spin.setRange(8, 25)
        self.voxcpm_steps_spin.setValue(16)
        self.voxcpm_steps_spin.setMinimumWidth(78)
        self.voxcpm_steps_spin.setToolTip(
            "VoxCPM2 48 kHz inference_timesteps (Varsayılan: 16, Aralık: 8–25)"
        )
        vox_row.addWidget(self.voxcpm_steps_spin)

        self.use_acting_prompt_chk = QCheckBox("🎭 Duygu Yönlendirmesini Kullan")
        self.use_acting_prompt_chk.setChecked(True)
        self.use_acting_prompt_chk.setToolTip(
            "Açıkken Gemini'nin çıkardığı veya tabloda düzenlediğiniz oyunculuk/duygu talimatını "
            "'(acting_prompt)Türkçe Metin' formatında VoxCPM2 motoruna gönderir"
        )
        vox_row.addWidget(self.use_acting_prompt_chk)
        vox_row.addStretch()
        eg.addLayout(vox_row, 1, 0, 1, 4)

        cb_row = QHBoxLayout()
        cb_row.setSpacing(8)
        cb_row.addWidget(QLabel("Chatterbox Duygu:"))
        self.exag_slider = QSlider(Qt.Orientation.Horizontal)
        self.exag_slider.setRange(0, 150)
        self.exag_slider.setValue(55)
        self.exag_spin = QDoubleSpinBox()
        self.exag_spin.setRange(0.0, 1.50)
        self.exag_spin.setSingleStep(0.05)
        self.exag_spin.setDecimals(2)
        self.exag_spin.setValue(0.55)
        self.exag_spin.setMinimumWidth(82)
        self.exag_spin.setToolTip(
            "Chatterbox exaggeration: 0.35 = sakin, 0.55 = dengeli doğal, 0.85+ = dramatik"
        )
        self.exag_slider.valueChanged.connect(self._on_exag_slider_changed)
        self.exag_spin.valueChanged.connect(self._on_exag_spin_changed)
        cb_row.addWidget(self.exag_slider, 1)
        cb_row.addWidget(self.exag_spin)

        cb_row.addWidget(QLabel("CFG:"))
        self.cfg_slider = QSlider(Qt.Orientation.Horizontal)
        self.cfg_slider.setRange(0, 100)
        self.cfg_slider.setValue(50)
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(0.0, 1.00)
        self.cfg_spin.setSingleStep(0.05)
        self.cfg_spin.setDecimals(2)
        self.cfg_spin.setValue(0.50)
        self.cfg_spin.setMinimumWidth(82)
        self.cfg_spin.setToolTip(
            "Chatterbox cfg_weight: 0.25-0.40 aksan sızmasını azaltır; 0.50 varsayılan dengedir"
        )
        self.cfg_slider.valueChanged.connect(self._on_cfg_slider_changed)
        self.cfg_spin.valueChanged.connect(self._on_cfg_spin_changed)
        cb_row.addWidget(self.cfg_slider, 1)
        cb_row.addWidget(self.cfg_spin)
        eg.addLayout(cb_row, 2, 0, 1, 4)

        el_tune_row = QHBoxLayout()
        el_tune_row.setSpacing(10)
        el_tune_row.addWidget(QLabel("ElevenLabs Stability:"))
        self.el_stability_spin = QDoubleSpinBox()
        self.el_stability_spin.setRange(0.0, 1.0)
        self.el_stability_spin.setSingleStep(0.05)
        self.el_stability_spin.setValue(0.45)
        self.el_stability_spin.setMinimumWidth(88)
        el_tune_row.addWidget(self.el_stability_spin)

        el_tune_row.addWidget(QLabel("Benzerlik:"))
        self.el_similarity_spin = QDoubleSpinBox()
        self.el_similarity_spin.setRange(0.0, 1.0)
        self.el_similarity_spin.setSingleStep(0.05)
        self.el_similarity_spin.setValue(0.80)
        self.el_similarity_spin.setMinimumWidth(88)
        el_tune_row.addWidget(self.el_similarity_spin)

        el_tune_row.addWidget(QLabel("Stil:"))
        self.el_style_spin = QDoubleSpinBox()
        self.el_style_spin.setRange(0.0, 1.0)
        self.el_style_spin.setSingleStep(0.05)
        self.el_style_spin.setValue(0.25)
        self.el_style_spin.setMinimumWidth(88)
        el_tune_row.addWidget(self.el_style_spin)

        self.el_remove_noise_chk = QCheckBox("STS Gürültü Temizle")
        self.el_remove_noise_chk.setChecked(True)
        el_tune_row.addWidget(self.el_remove_noise_chk)
        el_tune_row.addStretch()
        eg.addLayout(el_tune_row, 3, 0, 1, 4)

        ref_card = QFrame()
        ref_card.setObjectName("Card")
        rfg = QGridLayout(ref_card)
        rfg.setContentsMargins(12, 8, 12, 8)
        rfg.setHorizontalSpacing(8)
        rfg.setVerticalSpacing(6)

        rf_title = QLabel("🧬 Karakter Referans Profili • Seed-VC STS Perde Uyumu • Arka Plan")
        rf_title.setObjectName("SectionTitle")
        rfg.addWidget(rf_title, 0, 0, 1, 4)

        rfg.addWidget(QLabel("Sabit Referans (.wav):"), 1, 0)
        self.fixed_ref_input = QLineEdit()
        self.fixed_ref_input.setPlaceholderText(
            "Opsiyonel: 10–15 sn temiz karakter sesi (boş = her satır kendi orijinal sesinden klonlanır)"
        )
        self.browse_ref_btn = QPushButton("📁 Seç")
        self.browse_ref_btn.setMinimumWidth(78)
        self.browse_ref_btn.clicked.connect(self._on_browse_fixed_ref)
        self.clear_ref_btn = QPushButton("✕")
        self.clear_ref_btn.setFixedWidth(34)
        self.clear_ref_btn.setToolTip("Sabit referansı temizle (satır bazlı klonlamaya dön)")
        self.clear_ref_btn.clicked.connect(lambda: self.fixed_ref_input.clear())

        rfg.addWidget(self.fixed_ref_input, 1, 1)
        rfg.addWidget(self.browse_ref_btn, 1, 2)
        rfg.addWidget(self.clear_ref_btn, 1, 3)

        bg_row = QHBoxLayout()
        bg_row.setSpacing(10)
        self.preserve_bg_chk = QCheckBox("🎵 Arka Planı Koru")
        self.preserve_bg_chk.toggled.connect(self._sync_bg_from_detail)
        bg_row.addWidget(self.preserve_bg_chk)

        bg_row.addWidget(QLabel("Seviye:"))
        self.bg_vol_spin = QDoubleSpinBox()
        self.bg_vol_spin.setRange(-18.0, 6.0)
        self.bg_vol_spin.setSingleStep(1.0)
        self.bg_vol_spin.setDecimals(1)
        self.bg_vol_spin.setSuffix(" dB")
        self.bg_vol_spin.setValue(0.0)
        self.bg_vol_spin.setMinimumWidth(90)
        bg_row.addWidget(self.bg_vol_spin)

        bg_row.addWidget(QLabel("Aday Sayısı (Whisper Puanlamalı):"))
        self.candidate_count_spin = QSpinBox()
        self.candidate_count_spin.setRange(1, 4)
        self.candidate_count_spin.setValue(1)
        self.candidate_count_spin.setMinimumWidth(78)
        self.candidate_count_spin.setToolTip(
            "Her satır için 1–4 arası aday ses üretir; >1 seçildiğinde faster-whisper Türkçe deşifre doğruluğu (%75) "
            "+ doğal süre uyumuna (%25) göre en iyi adayı otomatik seçer (ElevenLabs API'de 1 ile sınırlıdır)"
        )
        bg_row.addWidget(self.candidate_count_spin)
        bg_row.addStretch()
        rfg.addLayout(bg_row, 2, 0, 1, 4)

        sts_row = QHBoxLayout()
        sts_row.setSpacing(10)
        self.sts_auto_f0_chk = QCheckBox("🎙️ Ses Kalınlığı / Perde Uyumu (Otomatik)")
        self.sts_auto_f0_chk.setChecked(True)
        self.sts_auto_f0_chk.setToolTip(
            "Seed-VC STS modunda kullanıcının mikrofon veya dosya ses kalınlığını (F0 perdesini) "
            "Ajan 47 gibi kalın erkek veya kadın karakter sesine otomatik eşleştirir"
        )
        sts_row.addWidget(self.sts_auto_f0_chk)

        sts_row.addWidget(QLabel("Yarım Ton:"))
        self.sts_pitch_spin = QSpinBox()
        self.sts_pitch_spin.setRange(-12, 12)
        self.sts_pitch_spin.setValue(0)
        self.sts_pitch_spin.setSuffix(" st")
        self.sts_pitch_spin.setMinimumWidth(84)
        self.sts_pitch_spin.setToolTip(
            "Seed-VC STS yarım ton (semitone) ince ayarı (-12 = daha kalın, 0 = otomatik, +12 = daha ince)"
        )
        sts_row.addWidget(self.sts_pitch_spin)

        sts_row.addWidget(QLabel("Dönüştürme Adımı:"))
        self.sts_steps_spin = QSpinBox()
        self.sts_steps_spin.setRange(10, 50)
        self.sts_steps_spin.setValue(25)
        self.sts_steps_spin.setMinimumWidth(78)
        self.sts_steps_spin.setToolTip(
            "Seed-VC Ses Dönüştürme (STS) Diffusion Steps (Varsayılan: 25, Aralık: 10–50)"
        )
        sts_row.addWidget(self.sts_steps_spin)
        sts_row.addStretch()
        rfg.addLayout(sts_row, 3, 0, 1, 4)

        row.addWidget(emo_card, 11)
        row.addWidget(ref_card, 11)
        return tab

    # --- SEKME 3: DOĞAL TEMPO, SESSİZLİK VE ÇEVİRİ ---
    def _build_tab_tempo_translation(self) -> QWidget:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(12)

        tempo_card = QFrame()
        tempo_card.setObjectName("Card")
        tg = QGridLayout(tempo_card)
        tg.setContentsMargins(14, 12, 14, 12)
        tg.setHorizontalSpacing(14)
        tg.setVerticalSpacing(10)

        t_title = QLabel("🎚️ Doğal Tempo ve Akıllı Sessizlik Kırpma")
        t_title.setObjectName("SectionTitle")
        tg.addWidget(t_title, 0, 0, 1, 4)

        tg.addWidget(QLabel("Maksimum Hız (Tempo):"), 1, 0)
        self.tempo_spin = QDoubleSpinBox()
        self.tempo_spin.setRange(1.00, 1.50)
        self.tempo_spin.setSingleStep(0.01)
        self.tempo_spin.setDecimals(2)
        self.tempo_spin.setSuffix("x")
        self.tempo_spin.setSpecialValueText("1.00x (Kapalı)")
        self.tempo_spin.setValue(1.05)
        self.tempo_spin.setMinimumWidth(145)
        self.tempo_spin.setToolTip(
            "1.00x seçildiğinde hızlandırma tamamen kapatılır. Önerilen doğal üst sınır: 1.05x"
        )
        tg.addWidget(self.tempo_spin, 1, 1)

        self.trim_silence_chk = QCheckBox(
            "✂️ Hızlandırmadan önce baştaki, sondaki ve cümle arası gereksiz sessizlikleri kırp"
        )
        self.trim_silence_chk.setChecked(True)
        tg.addWidget(self.trim_silence_chk, 1, 2, 1, 2)

        tempo_hint = QLabel(
            "Önce gereksiz sessizlikler güvenli şekilde temizlenir. Süre hâlâ uzunsa yalnızca belirlenen üst sınıra kadar "
            "(varsayılan 1.05x) perde koruyucu yumuşak senkron uygulanır; 1.00x seçiliyse ses hiç hızlandırılmaz."
        )
        tempo_hint.setObjectName("MutedLabel")
        tempo_hint.setWordWrap(True)
        tg.addWidget(tempo_hint, 2, 0, 1, 4)

        self.cinematic_timing_lock_chk = QCheckBox(
            "🎬 Sinematik Zamanlama Kilidi (>= 15 sn uzun monologlarda orijinal esleri koru)"
        )
        self.cinematic_timing_lock_chk.setChecked(False)
        self.cinematic_timing_lock_chk.setToolTip(
            "Varsayılan olarak kapalıdır. Açıldığında yalnızca >= 15 saniyelik uzun monologlarda orijinal sesteki "
            ">= 0.65 sn sessizlik eslerini tespit eder, Türkçe cümleleri konuşma bloklarına paylaştırıp aynı referans "
            "sesle sentezler ve 15 ms yumuşak geçişle orijinal başlangıç saniyelerine yerleştirir. "
            "15 sn altındaki tüm repliklerde mevcut standart akış %100 aynı kalır."
        )
        tg.addWidget(self.cinematic_timing_lock_chk, 3, 0, 1, 4)

        tr_card = QFrame()
        tr_card.setObjectName("Card")
        cg = QGridLayout(tr_card)
        cg.setContentsMargins(14, 12, 14, 12)
        cg.setHorizontalSpacing(14)
        cg.setVerticalSpacing(10)

        c_title = QLabel("🌐 Çeviri Dili, Hız Limiti (RPM) ve Otomatik Uygulama")
        c_title.setObjectName("SectionTitle")
        cg.addWidget(c_title, 0, 0, 1, 4)

        cg.addWidget(QLabel("Kaynak Dil:"), 1, 0)
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["en", "auto", "de", "fr", "es", "ja", "ru", "it"])
        self.source_lang_combo.setMinimumWidth(110)
        cg.addWidget(self.source_lang_combo, 1, 1)

        cg.addWidget(QLabel("Gemini Hız Limiti (RPM):"), 1, 2)
        self.rpm_spin = QSpinBox()
        self.rpm_spin.setRange(1, 300)
        self.rpm_spin.setValue(4)
        self.rpm_spin.setSuffix(" /dk")
        self.rpm_spin.setMinimumWidth(120)
        self.rpm_spin.setToolTip(
            "Gemini API dakika başına maksimum istek sayısı (Ücretsiz 5 RPM kotası için önerilen: 4)"
        )
        cg.addWidget(self.rpm_spin, 1, 3)

        self.phonetic_names_chk = QCheckBox(
            "🗣️ Yabancı İsimleri Okunuşuyla Yaz (TTS)"
        )
        self.phonetic_names_chk.setChecked(True)
        self.phonetic_names_chk.setToolTip(
            "Açıkken özel yabancı isimleri Türkçe ses motorlarının pürüzsüz okuyacağı şekilde fonetik yazar "
            "(örn. Hamilton Lowe -> Hamilton Lov)"
        )
        cg.addWidget(self.phonetic_names_chk, 2, 0, 1, 2)

        self.auto_apply_chk = QCheckBox(
            "💾 İşlem bitince oyuna otomatik uygula (SHA-256 yedekli)"
        )
        cg.addWidget(self.auto_apply_chk, 2, 2, 1, 2)

        outer.addWidget(tempo_card, 12)
        outer.addWidget(tr_card, 10)
        return tab

    # --- SEKME 4: HITMAN 3 OYUN MODÜLÜ (RPKG ÇIKAR / PAKETLE / KUR) ---
    def _build_tab_hitman_rpkg(self) -> QWidget:
        tab = QWidget()
        outer = QHBoxLayout(tab)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(12)

        # Sol Kart: Runtime Klasörü, Sahne Seçimi, Filtreler ve Tek Tıkla Çıkarma
        ext_card = QFrame()
        ext_card.setObjectName("Card")
        eg = QGridLayout(ext_card)
        eg.setContentsMargins(12, 10, 12, 10)
        eg.setHorizontalSpacing(8)
        eg.setVerticalSpacing(7)

        eg.addWidget(QLabel("Runtime Klasörü:"), 0, 0)
        self.hitman_runtime_input = QLineEdit()
        self.hitman_runtime_input.setPlaceholderText(DEFAULT_HITMAN_RUNTIME_DIR)
        self.hitman_runtime_input.setToolTip(
            "HITMAN 3 / World of Assassination Runtime klasörü (varsayılan: C:\\Program Files\\Epic Games\\HITMAN3\\Runtime)"
        )
        eg.addWidget(self.hitman_runtime_input, 0, 1)

        self.hitman_browse_runtime_btn = QPushButton("📂 Gözat")
        self.hitman_browse_runtime_btn.setFixedWidth(84)
        self.hitman_browse_runtime_btn.clicked.connect(self._on_hitman_browse_runtime)
        eg.addWidget(self.hitman_browse_runtime_btn, 0, 2)

        self.hitman_refresh_btn = QPushButton("🔄 Tara")
        self.hitman_refresh_btn.setFixedWidth(74)
        self.hitman_refresh_btn.setToolTip(
            "Runtime içindeki tüm chunk*.rpkg dosyalarını ve kurulu sahne durumunu yenile"
        )
        self.hitman_refresh_btn.clicked.connect(self._on_hitman_refresh_clicked)
        eg.addWidget(self.hitman_refresh_btn, 0, 3)

        eg.addWidget(QLabel("Sahne / Chunk:"), 1, 0)
        self.hitman_scene_combo = QComboBox()
        self.hitman_scene_combo.setMinimumWidth(340)
        self.hitman_scene_combo.currentIndexChanged.connect(self._on_hitman_scene_changed)
        eg.addWidget(self.hitman_scene_combo, 1, 1, 1, 3)

        # Filtreler: Karakter/anahtar kelime + Min/Maks süre + Adet
        flt_row = QHBoxLayout()
        flt_row.setSpacing(6)
        flt_row.addWidget(QLabel("Filtre:"))
        self.hitman_keyword_input = QLineEdit()
        self.hitman_keyword_input.setPlaceholderText("ör. diana (boş = tüm konuşmalar)")
        self.hitman_keyword_input.setMinimumWidth(165)
        self.hitman_keyword_input.setToolTip(
            "IOI karakter/sahne anahtar kelimesi (ör. diana, burnwood, civmale). Boş bırakılırsa tüm konuşmalar taranır."
        )
        flt_row.addWidget(self.hitman_keyword_input, 1)

        flt_row.addWidget(QLabel("Süre:"))
        self.hitman_min_dur_spin = QDoubleSpinBox()
        self.hitman_min_dur_spin.setRange(0.1, 120.0)
        self.hitman_min_dur_spin.setSingleStep(0.5)
        self.hitman_min_dur_spin.setDecimals(1)
        self.hitman_min_dur_spin.setSuffix(" sn")
        self.hitman_min_dur_spin.setValue(0.5)
        self.hitman_min_dur_spin.setFixedWidth(78)
        flt_row.addWidget(self.hitman_min_dur_spin)

        flt_row.addWidget(QLabel("–"))
        self.hitman_max_dur_spin = QDoubleSpinBox()
        self.hitman_max_dur_spin.setRange(0.5, 300.0)
        self.hitman_max_dur_spin.setSingleStep(1.0)
        self.hitman_max_dur_spin.setDecimals(1)
        self.hitman_max_dur_spin.setSuffix(" sn")
        self.hitman_max_dur_spin.setValue(60.0)
        self.hitman_max_dur_spin.setFixedWidth(84)
        flt_row.addWidget(self.hitman_max_dur_spin)

        flt_row.addWidget(QLabel("Adet:"))
        self.hitman_count_combo = QComboBox()
        self.hitman_count_combo.setEditable(True)
        self.hitman_count_combo.addItems(["10", "20", "30", "50", "100"])
        self.hitman_count_combo.setCurrentText("20")
        self.hitman_count_combo.setFixedWidth(70)
        self.hitman_count_combo.setToolTip("Çıkarılacak yeni konuşma dosyası adedi (1–500)")
        flt_row.addWidget(self.hitman_count_combo)
        eg.addLayout(flt_row, 2, 0, 1, 4)

        self.hitman_extract_btn = QPushButton("📥 Yeni Sesleri Çıkar ve Projeye Yükle")
        self.hitman_extract_btn.setObjectName("PrimaryBtn")
        self.hitman_extract_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hitman_extract_btn.setToolTip(
            "Seçilen chunk'tan daha önce çıkarılmamış yeni WWES seslerini vgmstream-cli ile 48 kHz WAV'a çevirir, "
            "hitman_manifest.json oluşturur ve doğrudan ana stüdyonun Oyun Klasörü alanına yükler."
        )
        self.hitman_extract_btn.clicked.connect(self._on_hitman_extract)
        eg.addWidget(self.hitman_extract_btn, 3, 0, 1, 4)

        # Sağ Kart: Kurulu Sahneler Durum Kartı + Birleştir & 0xFFFF Kur + Kaldır
        dep_card = QFrame()
        dep_card.setObjectName("Card")
        dv = QVBoxLayout(dep_card)
        dv.setContentsMargins(12, 10, 12, 10)
        dv.setSpacing(8)

        dep_top = QHBoxLayout()
        dep_top.setSpacing(8)
        dep_title = QLabel("🛡️ Kurulu Sahneler & Aktif Dublaj Durumu")
        dep_title.setObjectName("SectionTitle")
        self.hitman_selected_badge = QLabel("")
        self.hitman_selected_badge.setObjectName("StatusPillInfo")
        dep_top.addWidget(dep_title)
        dep_top.addStretch()
        dep_top.addWidget(self.hitman_selected_badge)
        dv.addLayout(dep_top)

        self.hitman_status_box = QFrame()
        self.hitman_status_box.setObjectName("PreviewBar")
        sb_lay = QVBoxLayout(self.hitman_status_box)
        sb_lay.setContentsMargins(10, 6, 10, 6)
        sb_lay.setSpacing(2)
        self.hitman_status_label = QLabel("Runtime klasörü taranıyor...")
        self.hitman_status_label.setWordWrap(True)
        self.hitman_status_label.setStyleSheet("color: #CBD5E1; font-size: 11.5px;")
        sb_lay.addWidget(self.hitman_status_label)
        dv.addWidget(self.hitman_status_box, 1)

        dep_btns = QHBoxLayout()
        dep_btns.setSpacing(8)
        self.hitman_install_btn = QPushButton(
            "📦 Onaylı Dublajları Birleştir & HITMAN 3'e Kur (0xFFFF)"
        )
        self.hitman_install_btn.setObjectName("SuccessBtn")
        self.hitman_install_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hitman_install_btn.setToolTip(
            "Seçili sahneye ait tüm paketlerdeki onaylı Türkçe WAV dosyalarını birleştirir, "
            "0xFFFF Wwise Vorbis + ww2ogg + vgmstream + FFmpeg + SHA-256 doğrulamasıyla oyuna kurar."
        )
        self.hitman_install_btn.clicked.connect(self._on_hitman_repack_install)

        self.hitman_uninstall_btn = QPushButton("↩️ Seçili Sahne Yamasını Kaldır (Orijinale Dön)")
        self.hitman_uninstall_btn.setObjectName("DangerBtn")
        self.hitman_uninstall_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hitman_uninstall_btn.setToolTip(
            "Yalnızca seçili sahnenin bağımsız .rpkg yamasını kaldırır ve packagedefinition.txt'de "
            "diğer kurulu sahnelere dokunmadan orijinal patchlevel değerine döner."
        )
        self.hitman_uninstall_btn.clicked.connect(self._on_hitman_uninstall)

        dep_btns.addWidget(self.hitman_install_btn, 3)
        dep_btns.addWidget(self.hitman_uninstall_btn, 2)
        dv.addLayout(dep_btns)

        outer.addWidget(ext_card, 13)
        outer.addWidget(dep_card, 11)
        return tab

    # --- CANLI ÖZET & 8 MODELLİ GÜNLÜK KOTA TAKİP PANELİ ---
    def _build_live_quota_and_summary_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("PreviewBar")
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(10, 6, 10, 6)
        vbox.setSpacing(6)

        # Üst Satır: Aktif Profil Özeti + Toplam Kalan Günlük Kota (1120/1120 — Yenilenme: 11:00 TRT)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        self.chip_engine = QLabel("")
        self.chip_engine.setObjectName("SummaryChip")
        self.chip_candidates = QLabel("")
        self.chip_candidates.setObjectName("SummaryChip")
        self.chip_ref = QLabel("")
        self.chip_ref.setObjectName("SummaryChip")
        self.chip_bg = QLabel("")
        self.chip_bg.setObjectName("SummaryChip")
        self.chip_tempo = QLabel("")
        self.chip_tempo.setObjectName("SummaryChip")
        self.chip_rpm = QLabel("")
        self.chip_rpm.setObjectName("SummaryChip")

        self.quota_total_label = QLabel("Kalan: 1120/1120 — Yenilenme: 11:00 TRT")
        self.quota_total_label.setObjectName("QuotaHeaderChip")
        self.quota_total_label.setToolTip(
            "8 ücretsiz Gemini modelinin toplam günlük kalan istek kotası (Her gün 11:00 TRT'de otomatik sıfırlanır)"
        )

        top_row.addWidget(self.chip_engine)
        top_row.addWidget(self.chip_candidates)
        top_row.addWidget(self.chip_ref)
        top_row.addWidget(self.chip_bg)
        top_row.addWidget(self.chip_tempo)
        top_row.addWidget(self.chip_rpm)
        top_row.addStretch()
        top_row.addWidget(self.quota_total_label)
        vbox.addLayout(top_row)

        # Alt Satır: 8 Ücretsiz Gemini Modelinin Canlı Yeşil/Kırmızı Kota Rozetleri
        quota_row = QHBoxLayout()
        quota_row.setSpacing(6)

        self._quota_pill_buttons.clear()
        for spec in GEMINI_FREE_MODELS:
            btn = QPushButton(f"🟢 {spec.short_name}: 0/{spec.daily_limit}")
            btn.setObjectName("QuotaPillAvailable")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(
                f"Model: {spec.short_name} ({spec.api_id})\n"
                f"Günlük Ücretsiz Limit: {spec.daily_limit}/gün\n"
                "Bu modeli aktif çeviri modeli yapmak için tıklayın."
            )
            btn.clicked.connect(
                lambda _checked=False, aid=spec.api_id: self._on_quota_pill_clicked(aid)
            )
            self._quota_pill_buttons[spec.api_id] = btn
            quota_row.addWidget(btn, 1)

        vbox.addLayout(quota_row)
        return panel

    def _build_table_and_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Ses Dosyası",
                "Durum",
                "Süre",
                "LUFS",
                "Orijinal Metin (Deşifre)",
                "Türkçe Dublaj Metni (Çift Tıkla Düzenle)",
                "Oyunculuk / Duygu (Çift Tıkla Düzenle)",
                "Tempo",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.itemChanged.connect(self._on_table_item_edited)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(self.COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_DUR, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_LUFS, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(self.COL_ORIG, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self.COL_TR, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(self.COL_ACTING, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(self.COL_TEMPO, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(self.COL_FILE, 215)
        self.table.setColumnWidth(self.COL_ACTING, 235)
        layout.addWidget(self.table, 1)

        preview_bar = QFrame()
        preview_bar.setObjectName("PreviewBar")
        pb_vbox = QVBoxLayout(preview_bar)
        pb_vbox.setContentsMargins(10, 6, 10, 6)
        pb_vbox.setSpacing(6)

        # Üst Satır: Seçili dosya bilgisi + kırpılmayan tam okunaklı Aday 1..4 seçici butonları
        top_pb_row = QHBoxLayout()
        top_pb_row.setContentsMargins(0, 0, 0, 0)
        top_pb_row.setSpacing(10)

        self.selected_line_info = QLabel("Önizleme için tablodan bir konuşma satırı seçin.")
        self.selected_line_info.setObjectName("MutedLabel")
        top_pb_row.addWidget(self.selected_line_info, 1)

        self.candidate_bar_widget = QWidget()
        cand_layout = QHBoxLayout(self.candidate_bar_widget)
        cand_layout.setContentsMargins(0, 0, 0, 0)
        cand_layout.setSpacing(6)
        cand_lbl = QLabel("🎯 Aday Seçimi:")
        cand_lbl.setObjectName("MutedLabel")
        cand_layout.addWidget(cand_lbl)
        self._candidate_buttons.clear()
        for c_idx in range(1, 5):
            c_btn = QPushButton(f"Aday {c_idx} (100%)")
            c_btn.setObjectName("QuotaPillAvailable")
            c_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            c_btn.setMinimumWidth(124)
            c_btn.setMinimumHeight(28)
            c_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            c_btn.setVisible(False)
            c_btn.clicked.connect(
                lambda _checked=False, idx=c_idx: self._on_candidate_button_clicked(idx)
            )
            self._candidate_buttons.append(c_btn)
            cand_layout.addWidget(c_btn)
        self.candidate_bar_widget.setVisible(False)
        top_pb_row.addWidget(self.candidate_bar_widget)
        pb_vbox.addLayout(top_pb_row)

        # Alt Satır: Dinleme, Mikrofonla Oyunculuk Kaydı (STS), Ses Yükle (STS), Yeniden Dublaj ve Oyuna Uygula
        bot_pb_row = QHBoxLayout()
        bot_pb_row.setContentsMargins(0, 0, 0, 0)
        bot_pb_row.setSpacing(6)

        self.play_orig_btn = QPushButton("▶ Orijinali Dinle")
        self.play_orig_btn.setMinimumWidth(114)
        self.play_orig_btn.setEnabled(False)
        self.play_orig_btn.clicked.connect(self._on_play_original)

        self.play_dub_btn = QPushButton("🎙️ Türkçe Dublajı Dinle")
        self.play_dub_btn.setMinimumWidth(146)
        self.play_dub_btn.setEnabled(False)
        self.play_dub_btn.clicked.connect(self._on_play_dubbed)

        self.stop_audio_btn = QPushButton("⏹ Durdur")
        self.stop_audio_btn.setMinimumWidth(78)
        self.stop_audio_btn.clicked.connect(self.audio_player.stop)

        self.mic_record_btn = QPushButton("🎤 Mikrofonla Oyunculuk Kaydet")
        self.mic_record_btn.setMinimumWidth(196)
        self.mic_record_btn.setEnabled(False)
        self.mic_record_btn.setToolTip(
            "Kendi sesinizle (fısıltı, vurgu, alay, öfke, nefes) oynayarak 48 kHz kayıt alır "
            "ve Seed-VC ile doğrudan karakterin sesine dönüştürür"
        )
        self.mic_record_btn.clicked.connect(self._on_toggle_mic_recording)

        self.upload_sts_btn = QPushButton("📂 Ses Yükle (STS)")
        self.upload_sts_btn.setMinimumWidth(142)
        self.upload_sts_btn.setEnabled(False)
        self.upload_sts_btn.setToolTip(
            "Telefonunuzdan veya dışarıdan kaydettiğiniz oyunculuk sesini (.wav/.mp3/.ogg/.flac) "
            "Seed-VC ile doğrudan karakterin sesine dönüştürür"
        )
        self.upload_sts_btn.clicked.connect(self._on_upload_sts_audio)

        self.redub_line_btn = QPushButton("🔄 Satırı Yeniden Dublajla")
        self.redub_line_btn.setMinimumWidth(168)
        self.redub_line_btn.setEnabled(False)
        self.redub_line_btn.setToolTip(
            "Türkçe çeviriyi veya Oyunculuk/Duygu sütununu düzenledikten sonra seçili satırı güncel motorla yeniden seslendirir"
        )
        self.redub_line_btn.clicked.connect(self._on_redub_selected_line)

        self.redub_all_btn = QPushButton("🔁 Tümünü Yeniden Dublajla")
        self.redub_all_btn.setMinimumWidth(178)
        self.redub_all_btn.setEnabled(False)
        self.redub_all_btn.setToolTip(
            "Whisper deşifre ve Gemini çeviri aşamalarını TAMAMEN ATLAYARAK tablodaki tüm satırları "
            "mevcut Türkçe Dublaj Metni ve Oyunculuk/Duygu hücreleriyle seçili ses motorunda yeniden dublajlar"
        )
        self.redub_all_btn.clicked.connect(self._on_redub_all_lines)

        self.apply_game_btn = QPushButton("💾 Oyuna Uygula (SHA-256 Yedekli)")
        self.apply_game_btn.setObjectName("SuccessBtn")
        self.apply_game_btn.setMinimumWidth(210)
        self.apply_game_btn.setEnabled(False)
        self.apply_game_btn.clicked.connect(self._on_apply_to_game)

        bot_pb_row.addWidget(self.play_orig_btn)
        bot_pb_row.addWidget(self.play_dub_btn)
        bot_pb_row.addWidget(self.stop_audio_btn)
        bot_pb_row.addWidget(self.mic_record_btn)
        bot_pb_row.addWidget(self.upload_sts_btn)
        bot_pb_row.addWidget(self.redub_line_btn)
        bot_pb_row.addWidget(self.redub_all_btn)
        bot_pb_row.addStretch()
        bot_pb_row.addWidget(self.apply_game_btn)
        pb_vbox.addLayout(bot_pb_row)

        layout.addWidget(preview_bar)
        return panel

    def _build_progress_and_log_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        prog_header = QHBoxLayout()
        self.stage_label = QLabel("Hazır. Klasör seçip 'Dublajı Başlat' butonuna basın.")
        self.stage_label.setObjectName("SectionTitle")
        self.preview_status_label = QLabel("")
        self.preview_status_label.setObjectName("MutedLabel")
        self.clear_log_btn = QPushButton("Günlüğü Temizle")
        self.clear_log_btn.setFixedHeight(26)
        self.clear_log_btn.clicked.connect(lambda: self.log_console.clear())

        prog_header.addWidget(self.stage_label, 1)
        prog_header.addWidget(self.preview_status_label)
        prog_header.addWidget(self.clear_log_btn)
        layout.addLayout(prog_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_console = QPlainTextEdit()
        self.log_console.setObjectName("LogConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(1500)
        layout.addWidget(self.log_console, 1)

        return panel

    def _build_action_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 2, 0, 6)
        bar.setSpacing(8)

        self.start_btn = QPushButton("▶ Dublajı Başlat")
        self.start_btn.setObjectName("PrimaryBtn")
        self.start_btn.setMinimumHeight(38)
        self.start_btn.clicked.connect(self._on_start_pipeline)

        self.stop_btn = QPushButton("⏹ Durdur")
        self.stop_btn.setObjectName("DangerBtn")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_pipeline)

        self.restore_btn = QPushButton("↩ Orijinale Dön (SHA-256)")
        self.restore_btn.setObjectName("WarningBtn")
        self.restore_btn.setMinimumHeight(38)
        self.restore_btn.clicked.connect(self._on_restore_original)

        self.clean_temp_btn = QPushButton("🧹 Geçici Dosyaları Temizle")
        self.clean_temp_btn.setMinimumHeight(38)
        self.clean_temp_btn.clicked.connect(self._on_clean_temp)

        self.backup_badge = QLabel("Yedek: Yok")
        self.backup_badge.setObjectName("MutedLabel")

        bar.addWidget(self.start_btn)
        bar.addWidget(self.stop_btn)
        bar.addSpacing(10)
        bar.addWidget(self.restore_btn)
        bar.addWidget(self.clean_temp_btn)
        bar.addStretch()
        bar.addWidget(self.backup_badge)

        return bar

    # ==========================================================================
    # CANLI KOTA PANELİ VE OTOMATİK MODEL SEÇİMİ
    # ==========================================================================
    def _on_quota_manager_notify(self, snap: QuotaSnapshot) -> None:
        """Arka plan iş parçacığından da çağrılabileceği için Qt sinyaliyle ana thread'e aktarır."""
        self.quota_changed_signal.emit(snap)

    def _on_quota_pill_clicked(self, api_id: str) -> None:
        idx = self.gemini_model_combo.findData(api_id)
        if idx >= 0:
            self.gemini_model_combo.setCurrentIndex(idx)
        self.quota_mgr.set_active_model(api_id)
        self._schedule_autosave()

    def _on_gemini_model_combo_changed(self) -> None:
        if self._loading_ui:
            return
        api_id = self._current_gemini_model_api_id()
        self.quota_mgr.set_active_model(api_id)
        self._schedule_autosave()

    def _current_gemini_model_api_id(self) -> str:
        data = self.gemini_model_combo.currentData()
        if data:
            return normalize_gemini_model_id(str(data))
        return normalize_gemini_model_id(self.gemini_model_combo.currentText())

    def _apply_quota_snapshot_to_ui(self, snap: QuotaSnapshot) -> None:
        self.quota_total_label.setText(
            f"📊 Kalan: {snap.total_remaining}/{snap.total_limit} — Yenilenme: {snap.next_reset_label}"
        )

        # Aktif model otomatik fallback ile değiştiyse combo seçimini de senkronize et
        cur_combo_id = self._current_gemini_model_api_id()
        if snap.active_model_id and snap.active_model_id != cur_combo_id:
            idx = self.gemini_model_combo.findData(snap.active_model_id)
            if idx >= 0:
                prev_loading = self._loading_ui
                self._loading_ui = True
                self.gemini_model_combo.setCurrentIndex(idx)
                self._loading_ui = prev_loading

        for st in snap.models:
            btn = self._quota_pill_buttons.get(st.api_id)
            if btn is None:
                continue
            is_act = st.api_id == snap.active_model_id
            if st.is_available:
                dot = "🟢"
                prefix = "★ " if is_act else ""
                btn.setText(f"{dot} {prefix}{st.short_name}: {st.used}/{st.daily_limit}")
                obj_name = "QuotaPillActive" if is_act else "QuotaPillAvailable"
            else:
                dot = "🔴"
                btn.setText(f"{dot} {st.short_name}: {st.used}/{st.daily_limit}")
                obj_name = "QuotaPillExhausted"

            if btn.objectName() != obj_name:
                btn.setObjectName(obj_name)
                btn.style().unpolish(btn)
                btn.style().polish(btn)

        self._refresh_summary_strip()

    # ==========================================================================
    # SENKRONİZASYON, OTOMATİK KAYIT & AYAR YÖNETİMİ
    # ==========================================================================
    def _on_exag_slider_changed(self, val: int) -> None:
        if self._syncing_slider:
            return
        self._syncing_slider = True
        self.exag_spin.setValue(val / 100.0)
        self._syncing_slider = False

    def _on_exag_spin_changed(self, val: float) -> None:
        if self._syncing_slider:
            return
        self._syncing_slider = True
        self.exag_slider.setValue(int(round(val * 100)))
        self._syncing_slider = False

    def _on_cfg_slider_changed(self, val: int) -> None:
        if self._syncing_slider:
            return
        self._syncing_slider = True
        self.cfg_spin.setValue(val / 100.0)
        self._syncing_slider = False

    def _on_cfg_spin_changed(self, val: float) -> None:
        if self._syncing_slider:
            return
        self._syncing_slider = True
        self.cfg_slider.setValue(int(round(val * 100)))
        self._syncing_slider = False

    def _sync_bg_from_quick(self, checked: bool) -> None:
        if self.preserve_bg_chk.isChecked() != checked:
            self.preserve_bg_chk.setChecked(checked)
        self._schedule_autosave()

    def _sync_bg_from_detail(self, checked: bool) -> None:
        if self.quick_bg_chk.isChecked() != checked:
            self.quick_bg_chk.setChecked(checked)
        self._schedule_autosave()

    def _connect_autosave_signals(self) -> None:
        for w in (
            self.colab_url_input,
            self.colab_token_input,
            self.elevenlabs_key_input,
            self.gemini_key_input,
            self.game_dir_input,
            self.project_name_input,
            self.fixed_ref_input,
            self.hitman_runtime_input,
        ):
            w.textChanged.connect(self._schedule_autosave)

        for cb in (
            self.tts_engine_combo,
            self.local_whisper_combo,
            self.source_lang_combo,
            self.elevenlabs_voice_combo,
        ):
            cb.currentTextChanged.connect(self._schedule_autosave)

        for sp in (
            self.voxcpm_cfg_spin,
            self.voxcpm_steps_spin,
            self.candidate_count_spin,
            self.exag_spin,
            self.cfg_spin,
            self.el_stability_spin,
            self.el_similarity_spin,
            self.el_style_spin,
            self.bg_vol_spin,
            self.sts_pitch_spin,
            self.sts_steps_spin,
            self.tempo_spin,
            self.rpm_spin,
        ):
            sp.valueChanged.connect(self._schedule_autosave)

        for chk in (
            self.use_acting_prompt_chk,
            self.phonetic_names_chk,
            self.local_unload_chk,
            self.el_remove_noise_chk,
            self.sts_auto_f0_chk,
            self.trim_silence_chk,
            self.cinematic_timing_lock_chk,
            self.auto_apply_chk,
        ):
            chk.toggled.connect(self._schedule_autosave)

    def _schedule_autosave(self, *_args) -> None:
        if self._loading_ui:
            return
        self._refresh_summary_strip()
        self._autosave_timer.start()

    def _perform_autosave(self) -> None:
        if self._loading_ui:
            return
        self._collect_config_from_ui(save=True)
        self.autosave_label.setText("💾 Ayarlar kaydedildi")
        QTimer.singleShot(1800, lambda: self.autosave_label.setText(""))

    def _load_config_into_ui(self) -> None:
        self._set_engine_mode(self.config.get_engine_mode(), save=False)

        tts_val = self.config.get_tts_engine().value
        idx = self.tts_engine_combo.findData(tts_val)
        if idx >= 0:
            self.tts_engine_combo.setCurrentIndex(idx)

        self.colab_url_input.setText(self.config.colab_url)
        self.colab_token_input.setText(self.config.colab_token)
        self.elevenlabs_key_input.setText(self.config.elevenlabs_api_key)
        self.elevenlabs_voice_combo.setCurrentText(self.config.elevenlabs_voice_id)

        self.voxcpm_cfg_spin.setValue(float(self.config.voxcpm_cfg_value))
        self.voxcpm_steps_spin.setValue(int(self.config.voxcpm_timesteps))
        self.use_acting_prompt_chk.setChecked(bool(self.config.use_acting_prompt))
        self.candidate_count_spin.setValue(max(1, min(4, int(self.config.candidate_count))))
        self.phonetic_names_chk.setChecked(bool(self.config.phonetic_foreign_names))

        self.exag_spin.setValue(float(self.config.chatterbox_exaggeration))
        self.cfg_spin.setValue(float(self.config.chatterbox_cfg_weight))
        self.el_stability_spin.setValue(float(self.config.elevenlabs_stability))
        self.el_similarity_spin.setValue(float(self.config.elevenlabs_similarity))
        self.el_style_spin.setValue(float(self.config.elevenlabs_style))
        self.el_remove_noise_chk.setChecked(bool(self.config.elevenlabs_remove_bg_noise))

        self.gemini_key_input.setText(self.config.gemini_api_key)
        norm_gem = normalize_gemini_model_id(self.config.gemini_model)
        g_idx = self.gemini_model_combo.findData(norm_gem)
        if g_idx >= 0:
            self.gemini_model_combo.setCurrentIndex(g_idx)

        self.rpm_spin.setValue(int(self.config.gemini_rpm))
        self.tempo_spin.setValue(float(self.config.max_tempo_speedup))
        self.trim_silence_chk.setChecked(bool(self.config.trim_silence))
        self.cinematic_timing_lock_chk.setChecked(bool(self.config.cinematic_timing_lock))
        self.source_lang_combo.setCurrentText(self.config.source_language)
        self.game_dir_input.setText(self.config.last_game_dir)
        loaded_proj = (self.config.project_name or "").strip()
        if self.config.last_game_dir.strip() and (
            not loaded_proj or is_generic_audio_subfolder(loaded_proj)
        ):
            smart_proj = infer_smart_project_name(self.config.last_game_dir)
            loaded_proj = smart_proj
            self.config.project_name = smart_proj
            try:
                self.config_mgr.save(self.config)
            except Exception:
                pass
        self.project_name_input.setText(loaded_proj)
        self.fixed_ref_input.setText(self.config.fixed_reference_wav)
        self.preserve_bg_chk.setChecked(bool(self.config.preserve_background))
        self.quick_bg_chk.setChecked(bool(self.config.preserve_background))
        self.bg_vol_spin.setValue(float(self.config.bg_mix_volume_db))
        self.sts_auto_f0_chk.setChecked(bool(self.config.sts_auto_f0_adjust))
        self.sts_pitch_spin.setValue(int(self.config.sts_pitch_shift))
        self.sts_steps_spin.setValue(int(self.config.sts_diffusion_steps))
        self.auto_apply_chk.setChecked(bool(self.config.auto_apply_to_game))
        self.local_whisper_combo.setCurrentText(self.config.whisper_model_local)
        self.local_unload_chk.setChecked(bool(self.config.local_unload_between_stages))
        self.hitman_runtime_input.setText(
            (self.config.hitman_runtime_dir or DEFAULT_HITMAN_RUNTIME_DIR).strip()
        )

        self._update_engine_stack_page()
        self._update_glossary_button_label()
        self._refresh_hitman_scenes_and_status()

    def _current_tts_engine_type(self) -> TTSEngineType:
        raw = self.tts_engine_combo.currentData()
        try:
            return TTSEngineType(raw)
        except ValueError:
            return TTSEngineType.VOXCPM2

    def _current_elevenlabs_voice_id(self) -> str:
        cur_text = self.elevenlabs_voice_combo.currentText().strip()
        idx = self.elevenlabs_voice_combo.currentIndex()
        if idx >= 0 and self.elevenlabs_voice_combo.itemText(idx).strip() == cur_text:
            data = self.elevenlabs_voice_combo.itemData(idx)
            if data:
                return str(data).strip()
        return cur_text

    def _collect_config_from_ui(self, save: bool = True) -> AppConfig:
        mode = (
            EngineMode.COLAB_CLOUD
            if bool(self.mode_colab_btn.property("active"))
            else EngineMode.LOCAL_GPU
        )
        cfg = AppConfig(
            engine_mode=mode.value,
            tts_engine=self._current_tts_engine_type().value,
            colab_url=self.colab_url_input.text().strip(),
            colab_token=self.colab_token_input.text().strip(),
            colab_timeout_sec=self.config.colab_timeout_sec,
            voxcpm_cfg_value=round(self.voxcpm_cfg_spin.value(), 2),
            voxcpm_timesteps=int(self.voxcpm_steps_spin.value()),
            use_acting_prompt=self.use_acting_prompt_chk.isChecked(),
            candidate_count=int(self.candidate_count_spin.value()),
            phonetic_foreign_names=self.phonetic_names_chk.isChecked(),
            chatterbox_exaggeration=round(self.exag_spin.value(), 2),
            chatterbox_cfg_weight=round(self.cfg_spin.value(), 2),
            elevenlabs_api_key=self.elevenlabs_key_input.text().strip(),
            elevenlabs_voice_id=self._current_elevenlabs_voice_id(),
            elevenlabs_stability=round(self.el_stability_spin.value(), 2),
            elevenlabs_similarity=round(self.el_similarity_spin.value(), 2),
            elevenlabs_style=round(self.el_style_spin.value(), 2),
            elevenlabs_remove_bg_noise=self.el_remove_noise_chk.isChecked(),
            gemini_api_key=self.gemini_key_input.text().strip(),
            gemini_model=self._current_gemini_model_api_id(),
            gemini_rpm=self.rpm_spin.value(),
            source_language=self.source_lang_combo.currentText().strip() or "en",
            target_language="tr",
            last_game_dir=self.game_dir_input.text().strip(),
            project_name=self.project_name_input.text().strip(),
            fixed_reference_wav=self.fixed_ref_input.text().strip(),
            hitman_runtime_dir=(
                self.hitman_runtime_input.text().strip() or DEFAULT_HITMAN_RUNTIME_DIR
            ),
            preserve_background=self.preserve_bg_chk.isChecked(),
            bg_mix_volume_db=round(self.bg_vol_spin.value(), 1),
            sts_diffusion_steps=int(self.sts_steps_spin.value()),
            sts_auto_f0_adjust=self.sts_auto_f0_chk.isChecked(),
            sts_pitch_shift=int(self.sts_pitch_spin.value()),
            sts_cfg_rate=self.config.sts_cfg_rate,
            trim_silence=self.trim_silence_chk.isChecked(),
            max_tempo_speedup=round(self.tempo_spin.value(), 2),
            cinematic_timing_lock=self.cinematic_timing_lock_chk.isChecked(),
            auto_apply_to_game=self.auto_apply_chk.isChecked(),
            vad_threshold=self.config.vad_threshold,
            vad_min_speech_ms=self.config.vad_min_speech_ms,
            whisper_model_local=self.local_whisper_combo.currentText().strip() or "large-v3-turbo",
            local_unload_between_stages=self.local_unload_chk.isChecked(),
        )
        self.config = cfg
        if save:
            self.config_mgr.save(cfg)
        return cfg

    def _update_engine_stack_page(self) -> None:
        tts_type = self._current_tts_engine_type()
        if tts_type.is_elevenlabs:
            self.engine_stack.setCurrentIndex(2)
        elif bool(self.mode_colab_btn.property("active")):
            self.engine_stack.setCurrentIndex(0)
        else:
            self.engine_stack.setCurrentIndex(1)

    def _set_engine_mode(self, mode: EngineMode, save: bool = True) -> None:
        is_colab = mode is EngineMode.COLAB_CLOUD
        self.mode_colab_btn.setProperty("active", is_colab)
        self.mode_local_btn.setProperty("active", not is_colab)
        for btn in (self.mode_colab_btn, self.mode_local_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self._update_engine_stack_page()
        self.engine_health_label.setText("Test edilmedi")
        self.engine_health_label.setStyleSheet("color: #8B9BB4;")

        if save and not self._loading_ui:
            self._collect_config_from_ui(save=True)
            self._refresh_summary_strip()
            self._log(f"İşlem altyapısı değiştirildi: {mode.label_tr}")

    def _on_tts_engine_changed(self) -> None:
        self._update_engine_stack_page()
        self.engine_health_label.setText("Test edilmedi")
        self.engine_health_label.setStyleSheet("color: #8B9BB4;")
        if not self._loading_ui:
            tts = self._current_tts_engine_type()
            self._refresh_summary_strip()
            self._schedule_autosave()
            self._log(f"Seslendirme motoru seçildi: {tts.label_tr}")

    def _refresh_summary_strip(self) -> None:
        tts = self._current_tts_engine_type()
        if tts.is_voxcpm:
            v_lbl = "VoxCPM2 48kHz" if tts is TTSEngineType.VOXCPM2 else "Trendyol-TTS 48kHz"
            act_flag = " | 🎭Duygu" if self.use_acting_prompt_chk.isChecked() else ""
            self.chip_engine.setText(
                f"🎙️ {v_lbl} (CFG: {self.voxcpm_cfg_spin.value():.2f} | Adım: {self.voxcpm_steps_spin.value()}{act_flag})"
            )
        elif tts is TTSEngineType.CHATTERBOX_MULTILINGUAL:
            self.chip_engine.setText(
                f"🎙️ Chatterbox TR (Duygu: {self.exag_spin.value():.2f} | CFG: {self.cfg_spin.value():.2f})"
            )
        elif tts is TTSEngineType.XTTS_V2:
            self.chip_engine.setText("🎙️ Coqui XTTS v2")
        elif tts is TTSEngineType.ELEVENLABS_TTS:
            self.chip_engine.setText("🎙️ ElevenLabs Multilingual v2 (TTS)")
        else:
            self.chip_engine.setText("🎙️ ElevenLabs Speech-to-Speech (STS)")

        c_cnt = self.candidate_count_spin.value()
        if tts.is_elevenlabs or c_cnt <= 1:
            self.chip_candidates.setText("🎯 Aday: 1")
        else:
            self.chip_candidates.setText(f"🎯 Aday: {c_cnt} + Whisper")

        ref_txt = self.fixed_ref_input.text().strip()
        if ref_txt:
            self.chip_ref.setText(f"🧬 Sabit Ref: {Path(ref_txt).name}")
        else:
            self.chip_ref.setText("🧬 Ref: Satır Bazlı Orijinal")

        if self.preserve_bg_chk.isChecked():
            self.chip_bg.setText(f"🎵 Arka Plan: Korunuyor ({self.bg_vol_spin.value():+.1f} dB)")
        else:
            self.chip_bg.setText("🎵 Arka Plan: Kapalı")

        t_val = self.tempo_spin.value()
        trim_txt = " + Sessizlik Kırp" if self.trim_silence_chk.isChecked() else ""
        cin_txt = " + 🎬EsKilidi" if self.cinematic_timing_lock_chk.isChecked() else ""
        if t_val <= 1.002:
            self.chip_tempo.setText(f"🎚️ Tempo: Doğal (Kapalı){trim_txt}{cin_txt}")
        else:
            self.chip_tempo.setText(f"🎚️ Maks Hız: {t_val:.2f}x{trim_txt}{cin_txt}")

        spec = resolve_model_spec(self._current_gemini_model_api_id())
        self.chip_rpm.setText(f"⚡ {spec.short_name} • {self.rpm_spin.value()} RPM")

    def _refresh_status_badges(self) -> None:
        if self.ffmpeg_mgr.is_available():
            ver = self.ffmpeg_mgr.get_version_string()
            short_ver = ver.split()[2] if len(ver.split()) >= 3 else "Hazır"
            short_ver = short_ver.split("-")[0]
            self.ffmpeg_pill.setText(f"✅ FFmpeg {short_ver}")
            self.ffmpeg_pill.setObjectName("StatusPillOk")
            self.ffmpeg_pill.setToolTip(f"Konum: {self.ffmpeg_mgr.ffmpeg_exe}\n{ver}")
        else:
            self.ffmpeg_pill.setText("⚠️ FFmpeg Eksik")
            self.ffmpeg_pill.setObjectName("StatusPillWarn")
            self.ffmpeg_pill.setToolTip(
                "FFmpeg bulunamadı! ./_araclar/ffmpeg/ içine ffmpeg.exe yerleştirin."
            )
        self.ffmpeg_pill.style().unpolish(self.ffmpeg_pill)
        self.ffmpeg_pill.style().polish(self.ffmpeg_pill)

        projects = self.backup_mgr.list_projects()
        cur_proj = self._current_project_name()
        if cur_proj and self.backup_mgr.has_backup(cur_proj):
            m = self.backup_mgr.load_manifest(cur_proj)
            cnt = len(m.files) if m else 0
            self.backup_badge.setText(f"🛡️ Aktif Yedek: {cur_proj} ({cnt} dosya, SHA-256)")
            self.backup_badge.setStyleSheet("color: #34D399; font-weight: 600;")
        elif projects:
            self.backup_badge.setText(f"🛡️ Mevcut Yedekler: {', '.join(projects)}")
            self.backup_badge.setStyleSheet("color: #FBBF24; font-weight: 600;")
        else:
            self.backup_badge.setText("Yedek: Henüz yok")
            self.backup_badge.setStyleSheet("color: #8B9BB4;")

    def _update_glossary_button_label(self) -> None:
        cnt = len(self.glossary_mgr.list_entries())
        p_cnt = len(self.pronunciation_mgr.list_rules())
        self.glossary_btn.setText(f"📖 Sözlük ({cnt} | TTS:{p_cnt})")

    def _current_project_name(self) -> str:
        raw = self.project_name_input.text().strip()
        gdir = self.game_dir_input.text().strip()
        if raw and not is_generic_audio_subfolder(raw):
            return sanitize_project_name(raw)
        if gdir:
            return infer_smart_project_name(gdir)
        if raw:
            return sanitize_project_name(raw)
        return ""

    @staticmethod
    def _toggle_password_echo(line_edit: QLineEdit) -> None:
        if line_edit.echoMode() == QLineEdit.EchoMode.Password:
            line_edit.setEchoMode(QLineEdit.EchoMode.Normal)
        else:
            line_edit.setEchoMode(QLineEdit.EchoMode.Password)

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{ts}] {msg}")

    def _set_busy(self, busy: bool) -> None:
        self.start_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.restore_btn.setEnabled(not busy)
        self.clean_temp_btn.setEnabled(not busy)
        self.mode_colab_btn.setEnabled(not busy)
        self.mode_local_btn.setEnabled(not busy)
        self.tts_engine_combo.setEnabled(not busy)
        self.browse_dir_btn.setEnabled(not busy)
        self.browse_ref_btn.setEnabled(not busy)
        self.test_conn_btn.setEnabled(not busy)
        self.test_local_gpu_btn.setEnabled(not busy)
        self.test_el_btn.setEnabled(not busy)
        self.load_el_voices_btn.setEnabled(not busy)
        self.hitman_browse_runtime_btn.setEnabled(not busy)
        self.hitman_refresh_btn.setEnabled(not busy)
        self.hitman_scene_combo.setEnabled(not busy)
        self.hitman_extract_btn.setEnabled(not busy)
        self.hitman_install_btn.setEnabled(not busy)
        self.hitman_uninstall_btn.setEnabled(not busy)
        if busy:
            self.apply_game_btn.setEnabled(False)
            self.redub_line_btn.setEnabled(False)
            self.redub_all_btn.setEnabled(False)
        else:
            self._update_preview_buttons()

    def _build_speech_engine(
        self,
        cfg: AppConfig,
        translator: Optional[GeminiTranslator] = None,
    ) -> SpeechEngine:
        tts_type = cfg.get_tts_engine()
        compute_mode = cfg.get_engine_mode()

        if tts_type.is_elevenlabs:
            delegate: Optional[SpeechEngine] = None
            if compute_mode is EngineMode.COLAB_CLOUD and cfg.colab_url and cfg.colab_token:
                delegate = ColabCloudEngine(
                    base_url=cfg.colab_url,
                    bearer_token=cfg.colab_token,
                    timeout_sec=cfg.colab_timeout_sec,
                )
            elif compute_mode is EngineMode.LOCAL_GPU:
                local_eng = LocalGPUEngine(
                    whisper_model_name=cfg.whisper_model_local,
                    unload_between_stages=cfg.local_unload_between_stages,
                )
                if local_eng.health_check().ok:
                    delegate = local_eng

            gem_fn = translator.transcribe_audio_fallback if translator is not None else None
            return ElevenLabsEngine(
                api_key=cfg.elevenlabs_api_key,
                voice_id=cfg.elevenlabs_voice_id,
                tts_mode=tts_type,
                ffmpeg=self.ffmpeg_mgr,
                transcribe_delegate=delegate,
                gemini_transcribe_fn=gem_fn,
            )

        if compute_mode is EngineMode.COLAB_CLOUD:
            return ColabCloudEngine(
                base_url=cfg.colab_url,
                bearer_token=cfg.colab_token,
                timeout_sec=cfg.colab_timeout_sec,
            )
        return LocalGPUEngine(
            whisper_model_name=cfg.whisper_model_local,
            unload_between_stages=cfg.local_unload_between_stages,
            default_tts_engine=tts_type,
        )

    # ==========================================================================
    # KULLANICI OLAYLARI
    # ==========================================================================
    def _on_browse_folder(self) -> None:
        prev_dir = self.game_dir_input.text().strip()
        start_dir = prev_dir or str(APP_ROOT)
        chosen = QFileDialog.getExistingDirectory(
            self, "Oyun / Ses Klasörünü Seçin", start_dir
        )
        if chosen:
            if prev_dir and os.path.normcase(os.path.normpath(prev_dir)) != os.path.normcase(
                os.path.normpath(chosen)
            ):
                self._save_current_table_draft(show_status=False)
                self._lines.clear()
                self._line_row_map.clear()
                self.table.setRowCount(0)
            self.game_dir_input.setText(chosen)
            self.project_name_input.setText(infer_smart_project_name(chosen))
            self._collect_config_from_ui(save=True)
            self._refresh_status_badges()
            self._try_load_project_draft()
            self._update_preview_buttons()

    def _on_browse_fixed_ref(self) -> None:
        start_dir = self.game_dir_input.text().strip() or str(APP_ROOT)
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Sabit Karakter Referans Sesi Seçin (10–15 sn temiz WAV önerilir)",
            start_dir,
            "Ses Dosyaları (*.wav *.flac *.mp3 *.ogg);;Tüm Dosyalar (*.*)",
        )
        if chosen:
            self.fixed_ref_input.setText(chosen)
            self._collect_config_from_ui(save=True)
            self._refresh_summary_strip()
            try:
                clean_ref = self.ffmpeg_mgr.prepare_clean_fixed_reference(
                    Path(chosen), sample_rate=48000, max_duration_sec=20.0
                )
                ref_meta = self.ffmpeg_mgr.probe_audio(clean_ref, measure_lufs=False)
                self._log(
                    f"Sabit karakter referans sesi seçildi ve ./_temp/ altında 48 kHz Mono "
                    f"(baş/son sessizlik kırpılmış, {ref_meta.duration_sec:.2f} sn) temiz kopyası hazırlandı: {clean_ref.name}"
                )
            except Exception as exc:
                self._log(f"Sabit karakter referans sesi seçildi: {chosen} (Ön hazırlık uyarısı: {exc})")

    def _on_game_dir_changed(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            cur_proj = self.project_name_input.text().strip()
            if not cur_proj or is_generic_audio_subfolder(cur_proj):
                self.project_name_input.setText(infer_smart_project_name(cleaned))
        self._refresh_status_badges()

    def _on_open_glossary(self) -> None:
        dlg = GlossaryDialog(
            self.glossary_mgr,
            self,
            pronunciation_manager=self.pronunciation_mgr,
        )
        if dlg.exec():
            self._update_glossary_button_label()
            self._log(
                f"Sözlük güncellendi: {len(self.glossary_mgr.list_entries())} çeviri terimi, "
                f"{len(self.pronunciation_mgr.list_rules())} TTS telaffuz kuralı."
            )

    def _on_generate_colab_notebook(self) -> None:
        try:
            nb_path = generate_colab_notebook(DEFAULT_NOTEBOOK_PATH)
            self._log(f"Sinematik Google Colab defteri (v2) oluşturuldu: {nb_path}")
            QMessageBox.information(
                self,
                "Colab Defteri Hazır (v2)",
                f"Chatterbox Multilingual + XTTS v2 + HTDemucs destekli defter oluşturuldu:\n\n{nb_path}\n\n"
                "Bu dosyayı Google Colab'e (colab.research.google.com) yükleyip GPU (L4/A100/T4) "
                "çalışma zamanında çalıştırın; ardından ürettiği URL ve Bearer Token bilgisini buraya yapıştırın.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Hata", f"Colab defteri oluşturulamadı:\n{exc}")

    def _on_reveal_colab_notebook(self) -> None:
        if not DEFAULT_NOTEBOOK_PATH.is_file():
            generate_colab_notebook(DEFAULT_NOTEBOOK_PATH)
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", "/select,", str(DEFAULT_NOTEBOOK_PATH)])
        except Exception as exc:
            self._log(f"Klasör açılamadı: {exc}")

    def _on_test_connection(self) -> None:
        cfg = self._collect_config_from_ui(save=True)
        engine = self._build_speech_engine(cfg)
        self.engine_health_label.setText("⏳ Test ediliyor...")
        self.engine_health_label.setStyleSheet("color: #60A5FA;")
        self.test_conn_btn.setEnabled(False)
        self.test_local_gpu_btn.setEnabled(False)
        self.test_el_btn.setEnabled(False)

        self._health_worker = HealthCheckWorker(engine)
        self._health_worker.finished_health.connect(self._on_health_check_done)
        self._health_worker.start()

    def _on_health_check_done(self, res: EngineHealth) -> None:
        self.test_conn_btn.setEnabled(True)
        self.test_local_gpu_btn.setEnabled(True)
        self.test_el_btn.setEnabled(True)
        if res.ok:
            self.engine_health_label.setText(f"✅ {res.message}")
            self.engine_health_label.setStyleSheet("color: #34D399; font-weight: 600;")
            avail = ", ".join(res.available_tts_engines) if res.available_tts_engines else res.tts_model
            sep_str = "HTDemucs Hazır" if res.supports_vocal_separation else "FFmpeg Yedek"
            self._log(
                f"Motor testi BAŞARILI: {res.message} | Motorlar: {avail} | Vokal Ayırma: {sep_str}"
            )
        else:
            self.engine_health_label.setText("❌ Bağlantı / Hazırlık Hatası")
            self.engine_health_label.setStyleSheet("color: #F87171; font-weight: 600;")
            self._log(f"Motor testi BAŞARISIZ: {res.message}")
            QMessageBox.warning(self, "Motor Testi Başarısız", res.message)

    def _on_load_elevenlabs_voices(self) -> None:
        cfg = self._collect_config_from_ui(save=True)
        if not cfg.elevenlabs_api_key:
            QMessageBox.warning(
                self, "API Anahtarı Eksik", "Önce geçerli bir ElevenLabs API anahtarı girin."
            )
            return
        self.load_el_voices_btn.setEnabled(False)
        self.engine_health_label.setText("⏳ Sesler yükleniyor...")
        self._el_voices_worker = ElevenLabsVoicesWorker(cfg.elevenlabs_api_key)
        self._el_voices_worker.voices_loaded.connect(self._on_el_voices_loaded)
        self._el_voices_worker.failed_signal.connect(self._on_el_voices_failed)
        self._el_voices_worker.start()

    def _on_el_voices_loaded(self, voices: list) -> None:
        self.load_el_voices_btn.setEnabled(True)
        prev_id = self._current_elevenlabs_voice_id()
        self._loading_ui = True
        try:
            self.elevenlabs_voice_combo.clear()
            self.elevenlabs_voice_combo.addItem("Otomatik Referans Klonla (IVC)", "")
            for vid, label in voices:
                self.elevenlabs_voice_combo.addItem(f"{label} [{vid[:8]}]", vid)
            if prev_id:
                idx = self.elevenlabs_voice_combo.findData(prev_id)
                if idx >= 0:
                    self.elevenlabs_voice_combo.setCurrentIndex(idx)
                else:
                    self.elevenlabs_voice_combo.setCurrentText(prev_id)
        finally:
            self._loading_ui = False

        self.engine_health_label.setText(f"✅ {len(voices)} ElevenLabs sesi yüklendi")
        self.engine_health_label.setStyleSheet("color: #34D399; font-weight: 600;")
        self._log(f"ElevenLabs hesabından {len(voices)} ses yüklendi.")

    def _on_el_voices_failed(self, err: str) -> None:
        self.load_el_voices_btn.setEnabled(True)
        self.engine_health_label.setText("❌ Sesler yüklenemedi")
        self.engine_health_label.setStyleSheet("color: #F87171;")
        self._log(f"ElevenLabs ses listesi hatası: {err}")
        QMessageBox.warning(self, "ElevenLabs Hatası", err)

    # ==========================================================================
    # ANA DUBLAJ İŞ HATTI
    # ==========================================================================
    def _on_start_pipeline(self) -> None:
        cfg = self._collect_config_from_ui(save=True)
        game_dir = Path(cfg.last_game_dir)
        if not cfg.last_game_dir or not game_dir.is_dir():
            QMessageBox.warning(
                self, "Klasör Seçilmedi", "Lütfen geçerli bir oyun/ses klasörü seçin."
            )
            return

        if not self.ffmpeg_mgr.is_available():
            QMessageBox.critical(
                self,
                "FFmpeg Eksik",
                "FFmpeg bulunamadı! Lütfen taşınabilir ffmpeg.exe dosyasını ./_araclar/ffmpeg/ klasörüne koyun.",
            )
            return

        if cfg.fixed_reference_wav and not Path(cfg.fixed_reference_wav).is_file():
            QMessageBox.warning(
                self,
                "Referans Ses Bulunamadı",
                f"Seçilen sabit referans ses dosyası mevcut değil:\n{cfg.fixed_reference_wav}",
            )
            return

        tts_type = cfg.get_tts_engine()
        if tts_type is not TTSEngineType.ELEVENLABS_STS and not cfg.gemini_api_key:
            QMessageBox.warning(
                self,
                "Gemini API Anahtarı Eksik",
                "Çeviri adımı için lütfen Google Gemini API anahtarınızı girin.",
            )
            return

        if tts_type.is_elevenlabs:
            if not cfg.elevenlabs_api_key:
                QMessageBox.warning(
                    self,
                    "ElevenLabs API Anahtarı Eksik",
                    "ElevenLabs motoru seçili. Lütfen ElevenLabs API anahtarınızı girin.",
                )
                return
        elif cfg.get_engine_mode() is EngineMode.COLAB_CLOUD:
            if not cfg.colab_url or not cfg.colab_token:
                QMessageBox.warning(
                    self,
                    "Colab Bilgileri Eksik",
                    "Colab Bulut Modu seçili. Lütfen Colab URL ve Bearer Token bilgilerini girin "
                    "veya üstten 'Yerel GPU Modu (CUDA)' seçeneğine geçin.",
                )
                return

        project_name = self._current_project_name() or infer_smart_project_name(game_dir)
        self.project_name_input.setText(project_name)

        self.audio_player.release()
        self._lines.clear()
        self._line_row_map.clear()
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.stage_label.setText("Sinematik iş hattı başlatılıyor...")

        translator = GeminiTranslator(
            api_key=cfg.gemini_api_key,
            model_name=cfg.gemini_model,
            rpm=cfg.gemini_rpm,
            cache=self.cache_mgr,
            glossary=self.glossary_mgr,
            quota_manager=self.quota_mgr,
            phonetic_foreign_names=cfg.phonetic_foreign_names,
        )
        engine = self._build_speech_engine(cfg, translator=translator)
        self._active_pipeline = DubbingPipeline(
            config=cfg,
            speech_engine=engine,
            translator=translator,
            ffmpeg=self.ffmpeg_mgr,
            backup_manager=self.backup_mgr,
            pronunciation_manager=self.pronunciation_mgr,
        )

        self._active_worker = PipelineWorker(self._active_pipeline, game_dir, project_name)
        self._active_worker.log_signal.connect(self._log)
        self._active_worker.progress_signal.connect(self._on_pipeline_progress)
        self._active_worker.discovered_signal.connect(self._on_pipeline_discovered)
        self._active_worker.line_updated_signal.connect(self._on_line_updated)
        self._active_worker.completed_signal.connect(self._on_pipeline_completed)
        self._active_worker.cancelled_signal.connect(self._on_pipeline_cancelled)
        self._active_worker.failed_signal.connect(self._on_pipeline_failed)

        self._set_busy(True)
        self._active_worker.start()

    def _on_stop_pipeline(self) -> None:
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.request_cancel()
            self.stop_btn.setEnabled(False)
            self.stage_label.setText("Durduruluyor, geçici dosyalar temizlenecek...")
            self._log("Durdurma isteği gönderildi...")
        elif self._redub_all_worker and self._redub_all_worker.isRunning():
            self._redub_all_worker.request_cancel()
            self.stop_btn.setEnabled(False)
            self.stage_label.setText(
                "Durduruluyor — o ana kadar tamamlanan satırlar korunacak..."
            )
            self._log(
                "Toplu yeniden dublaj için durdurma isteği gönderildi (biten satırlar korunacak)..."
            )
        elif self._hitman_extract_worker and self._hitman_extract_worker.isRunning():
            self._hitman_extract_worker.request_cancel()
            self.stop_btn.setEnabled(False)
            self.stage_label.setText("HITMAN 3 ses çıkarma işlemi durduruluyor...")
            self._log("[HITMAN 3] Ses çıkarma işlemi için durdurma isteği gönderildi...")
        elif self._hitman_install_worker and self._hitman_install_worker.isRunning():
            self._hitman_install_worker.request_cancel()
            self.stop_btn.setEnabled(False)
            self.stage_label.setText("HITMAN 3 RPKG paketleme işlemi durduruluyor...")
            self._log("[HITMAN 3] RPKG paketleme işlemi için durdurma isteği gönderildi...")

    def _on_pipeline_progress(self, cur: int, total: int, stage: str) -> None:
        pct = int((cur / max(1, total)) * 100)
        self.progress_bar.setValue(max(0, min(100, pct)))
        self.stage_label.setText(stage)

    def _on_pipeline_discovered(self, lines: list[DialogueLine]) -> None:
        self._lines = lines
        self._line_row_map.clear()
        self._updating_table = True
        try:
            self.table.setRowCount(len(lines))
            for row, ln in enumerate(lines):
                self._line_row_map[ln.line_id] = row
                self._populate_table_row(row, ln)
        finally:
            self._updating_table = False

    def _on_line_updated(self, line: DialogueLine) -> None:
        row = self._line_row_map.get(line.line_id)
        if row is None:
            return
        self._updating_table = True
        try:
            self._populate_table_row(row, line)
        finally:
            self._updating_table = False
        if self.table.currentRow() == row:
            self._update_preview_buttons()

    def _populate_table_row(self, row: int, ln: DialogueLine) -> None:
        def make_item(
            text: str, editable: bool = False, color: Optional[str] = None
        ) -> QTableWidgetItem:
            it = QTableWidgetItem(text)
            if not editable:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if color:
                it.setForeground(QColor(color))
            return it

        status_color = {
            LineStatus.READY_PREVIEW: "#34D399",
            LineStatus.APPLIED: "#60A5FA",
            LineStatus.SKIPPED_NON_SPEECH: "#64748B",
            LineStatus.SKIPPED_EMPTY: "#64748B",
            LineStatus.ERROR: "#F87171",
            LineStatus.CANCELLED: "#FBBF24",
        }.get(ln.status, "#CBD5E1")

        dur_str = f"{ln.metadata.duration_sec:.2f}s" if ln.metadata else "—"
        if ln.dub_duration_ms > 0 and ln.metadata:
            dur_str = f"{ln.metadata.duration_sec:.2f}s → {ln.dub_duration_ms / 1000.0:.2f}s"

        lufs_str = f"{ln.metadata.lufs:.1f}" if ln.metadata else "—"
        tempo_str = f"{ln.tempo_factor:.2f}x" if ln.dub_duration_ms > 0 else "—"
        if ln.background_preserved and ln.dub_duration_ms > 0:
            tempo_str += " 🎵"

        status_text = (
            "Hata" if (ln.status is LineStatus.ERROR and ln.error_message) else ln.status.label_tr
        )

        self.table.setItem(row, self.COL_FILE, make_item(ln.rel_path))
        st_item = make_item(status_text, color=status_color)
        if ln.error_message:
            st_item.setToolTip(ln.error_message)
        self.table.setItem(row, self.COL_STATUS, st_item)
        self.table.setItem(row, self.COL_DUR, make_item(dur_str))
        self.table.setItem(row, self.COL_LUFS, make_item(lufs_str))
        self.table.setItem(row, self.COL_ORIG, make_item(ln.source_text))
        self.table.setItem(row, self.COL_TR, make_item(ln.translated_text, editable=True))
        act_item = make_item(ln.acting_prompt, editable=True, color="#A78BFA")
        act_item.setToolTip(
            "VoxCPM2 48 kHz Oyunculuk / Duygu / Tempo Talimatı (İngilizce 4–8 kelime).\n"
            "Çift tıklayarak değiştirebilir ve 'Satırı Yeniden Dublajla' ile anında test edebilirsiniz."
        )
        self.table.setItem(row, self.COL_ACTING, act_item)
        self.table.setItem(row, self.COL_TEMPO, make_item(tempo_str))

    def _on_pipeline_completed(self, lines: list[DialogueLine], stats: PipelineStats) -> None:
        self._lines = lines
        self._set_busy(False)
        self._refresh_status_badges()
        self._update_preview_buttons()
        self._apply_quota_snapshot_to_ui(self.quota_mgr.get_snapshot())
        self._save_current_table_draft(show_status=False)
        self.progress_bar.setValue(100)

        ready_cnt = sum(
            1 for ln in self._lines if ln.synced_preview_path and ln.synced_preview_path.is_file()
        )
        if ready_cnt > 0 and not self.config.auto_apply_to_game:
            self.stage_label.setText(
                f"Önizlemeye hazır ({ready_cnt} satır). Dinleyip beğendiğinizde 'Oyuna Uygula'ya basın."
            )
        else:
            self.stage_label.setText(
                f"Tamamlandı: {stats.synced} dublaj üretildi, {stats.applied} dosyaya uygulandı."
            )

    def _on_pipeline_cancelled(self, msg: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("İşlem durduruldu ve geçici dosyalar temizlendi.")
        self._log(msg)
        self._refresh_status_badges()
        self._apply_quota_snapshot_to_ui(self.quota_mgr.get_snapshot())

    def _on_pipeline_failed(self, err: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("Hata oluştu; geçici dosyalar temizlendi.")
        self._log(f"KRİTİK HATA: {err}")
        self._apply_quota_snapshot_to_ui(self.quota_mgr.get_snapshot())
        QMessageBox.critical(self, "Dublaj Hatası", err)

    # ==========================================================================
    # ÖNİZLEME, TEK SATIR YENİDEN DUBLAJ & OYUNA UYGULAMA
    # ==========================================================================
    def _selected_line(self) -> Optional[DialogueLine]:
        row = self.table.currentRow()
        if 0 <= row < len(self._lines):
            return self._lines[row]
        return None

    def _on_table_selection_changed(self) -> None:
        self._update_preview_buttons()

    def _on_table_item_edited(self, item: QTableWidgetItem) -> None:
        if self._updating_table:
            return
        row = item.row()
        if not (0 <= row < len(self._lines)):
            return
        col = item.column()
        if col == self.COL_TR:
            self._lines[row].translated_text = item.text().strip()
            self._update_preview_buttons()
            self._save_current_table_draft(show_status=True)
        elif col == self.COL_ACTING:
            self._lines[row].acting_prompt = item.text().strip().strip("()")
            self._update_preview_buttons()
            self._save_current_table_draft(show_status=True)

    def _save_current_table_draft(self, show_status: bool = False) -> None:
        """Tablodaki mevcut satırları ve elle yapılan düzenlemeleri `./_veri/taslaklar/` içine kaydeder."""
        if not self._lines:
            return
        proj = self._current_project_name()
        gdir = self.game_dir_input.text().strip()
        if not proj or not gdir:
            return
        try:
            self.draft_mgr.save_lines(proj, gdir, self._lines)
            if show_status:
                self.autosave_label.setText("💾 Tablo taslağı kaydedildi")
                QTimer.singleShot(
                    1800,
                    lambda: (
                        self.autosave_label.setText("")
                        if self.autosave_label.text() == "💾 Tablo taslağı kaydedildi"
                        else None
                    ),
                )
        except Exception as exc:
            self._log(f"Uyarı: Tablo taslağı kaydedilemedi: {exc}")

    def _try_load_project_draft(self) -> int:
        """Aktif proje için `./_veri/taslaklar/` içinde kayıtlı taslak varsa tabloya otomatik yükler."""
        proj = self._current_project_name()
        gdir = self.game_dir_input.text().strip()
        if not proj or not gdir:
            return 0
        restored = self.draft_mgr.load_lines(proj, expected_game_dir=gdir)
        if not restored:
            return 0
        self._on_pipeline_discovered(restored)
        self._update_preview_buttons()
        draft_file = self.draft_mgr.draft_path(proj).name
        self._log(
            f"📝 Proje tablo taslağı otomatik yüklendi ({len(restored)} satır): "
            f"./_veri/taslaklar/{draft_file} (elle yapılan düzenlemeler korundu)"
        )
        return len(restored)

    def _resolve_original_playback_path(self, ln: DialogueLine) -> Optional[Path]:
        if ln.applied:
            proj = self._current_project_name()
            manifest = self.backup_mgr.load_manifest(proj) if proj else None
            if manifest and ln.rel_path in manifest.files:
                b_entry = manifest.files[ln.rel_path]
                backup_file = self.backup_mgr.project_dir(proj) / b_entry.backup_rel_path
                if backup_file.is_file():
                    return backup_file
        return ln.original_path if ln.original_path.is_file() else None

    def _resolve_dubbed_playback_path(self, ln: DialogueLine) -> Optional[Path]:
        if ln.synced_preview_path and ln.synced_preview_path.is_file():
            return ln.synced_preview_path
        if ln.applied and ln.original_path.is_file():
            return ln.original_path
        return None

    def _update_preview_buttons(self) -> None:
        ln = self._selected_line()
        has_any_ready = any(
            x.synced_preview_path and x.synced_preview_path.is_file() for x in self._lines
        )
        running = bool(
            (self._active_worker and self._active_worker.isRunning())
            or (self._apply_worker and self._apply_worker.isRunning())
            or (self._redub_worker and self._redub_worker.isRunning())
            or (self._redub_all_worker and self._redub_all_worker.isRunning())
            or (self._sts_worker and self._sts_worker.isRunning())
        )
        is_sts_mode = self._current_tts_engine_type() is TTSEngineType.ELEVENLABS_STS
        has_redub_all_targets = any(
            x.is_speech
            and x.original_path.is_file()
            and (is_sts_mode or bool((x.translated_text or "").strip()))
            for x in self._lines
        )
        self.apply_game_btn.setEnabled(has_any_ready and not running)
        self.redub_all_btn.setEnabled(has_redub_all_targets and not running)

        if ln is None:
            self.selected_line_info.setText("Önizleme için tablodan bir konuşma satırı seçin.")
            self.candidate_bar_widget.setVisible(False)
            self.play_orig_btn.setEnabled(False)
            self.play_dub_btn.setEnabled(False)
            self.mic_record_btn.setEnabled(False)
            self.upload_sts_btn.setEnabled(False)
            self.redub_line_btn.setEnabled(False)
            return

        meta_txt = ""
        if ln.metadata:
            meta_txt = (
                f" | {ln.metadata.duration_sec:.2f} sn | {ln.metadata.lufs:.1f} LUFS | "
                f"{ln.metadata.sample_rate} Hz {ln.metadata.channel_label}"
            )
        self.selected_line_info.setText(f"Seçili: {ln.rel_path}{meta_txt}")

        # Çoklu aday (>1) varsa aday seçici butonları doğruluk puanlarıyla göster
        if len(ln.candidates) > 1:
            self.candidate_bar_widget.setVisible(True)
            for idx, btn in enumerate(self._candidate_buttons):
                if idx < len(ln.candidates):
                    cand = ln.candidates[idx]
                    is_sel = cand.candidate_index == ln.selected_candidate_index
                    prefix = "★ " if is_sel else ""
                    btn.setText(f"{prefix}Aday {cand.candidate_index} ({cand.score_pct_int}%)")
                    btn.setToolTip(
                        f"Aday {cand.candidate_index} — Toplam Puan: %{cand.score_pct_int}\n"
                        f"• Metin Doğruluğu (%75): %{int(round(cand.text_accuracy_score))}\n"
                        f"• Doğal Süre Uyumu (%25): %{int(round(cand.duration_fit_score))}\n"
                        f"• Whisper Deşifresi: \"{cand.whisper_transcript}\"\n"
                        "Bu adayı aktif önizleme ve oyuna uygulama seçimi yapmak için tıklayın."
                    )
                    obj_name = "QuotaPillActive" if is_sel else "QuotaPillAvailable"
                    if btn.objectName() != obj_name:
                        btn.setObjectName(obj_name)
                        btn.style().unpolish(btn)
                        btn.style().polish(btn)
                    btn.setEnabled(not running)
                    btn.setVisible(True)
                else:
                    btn.setVisible(False)
        else:
            self.candidate_bar_widget.setVisible(False)

        self.play_orig_btn.setEnabled(self._resolve_original_playback_path(ln) is not None)
        self.play_dub_btn.setEnabled(self._resolve_dubbed_playback_path(ln) is not None)
        self.mic_record_btn.setEnabled((not running) or self.mic_recorder.is_recording)
        self.upload_sts_btn.setEnabled((not running) and (not self.mic_recorder.is_recording))
        self.redub_line_btn.setEnabled(
            bool(ln.translated_text.strip()) and ln.is_speech and not running
        )

    def _on_candidate_button_clicked(self, candidate_index: int) -> None:
        """Kullanıcı önizleme çubuğundan başka bir adaya (Aday 1..4) tıkladığında aktif seçimi değiştirir."""
        ln = self._selected_line()
        if ln is None or len(ln.candidates) <= 1:
            return
        try:
            # Windows ses oynatıcı dosya kilidini serbest bırak
            self.audio_player.release()
            QApplication.processEvents()

            cfg = self._collect_config_from_ui(save=False)
            pipeline = self._active_pipeline
            if pipeline is None:
                translator = GeminiTranslator(
                    api_key=cfg.gemini_api_key or "DUMMY",
                    model_name=cfg.gemini_model,
                    rpm=cfg.gemini_rpm,
                    cache=self.cache_mgr,
                    glossary=self.glossary_mgr,
                    quota_manager=self.quota_mgr,
                    phonetic_foreign_names=cfg.phonetic_foreign_names,
                )
                pipeline = DubbingPipeline(
                    config=cfg,
                    speech_engine=self._build_speech_engine(cfg, translator=translator),
                    translator=translator,
                    ffmpeg=self.ffmpeg_mgr,
                    backup_manager=self.backup_mgr,
                    pronunciation_manager=self.pronunciation_mgr,
                )
            else:
                pipeline.config = cfg

            pipeline.activate_candidate(ln, candidate_index, log_cb=self._log)
            self._on_line_updated(ln)
            self._update_preview_buttons()
        except Exception as exc:
            self._log(f"Aday değiştirme hatası: {exc}")
            QMessageBox.warning(self, "Aday Seçilemedi", str(exc))

    def _on_play_original(self) -> None:
        ln = self._selected_line()
        if not ln:
            return
        path = self._resolve_original_playback_path(ln)
        if path:
            self.audio_player.play_file(path, f"Orijinal — {ln.rel_path}")
        else:
            QMessageBox.warning(self, "Dosya Yok", "Orijinal ses dosyasına erişilemedi.")

    def _on_play_dubbed(self) -> None:
        ln = self._selected_line()
        if not ln:
            return
        path = self._resolve_dubbed_playback_path(ln)
        if path:
            self.audio_player.play_file(path, f"Türkçe Dublaj — {ln.rel_path}")
        else:
            QMessageBox.warning(
                self, "Dublaj Yok", "Bu satır için henüz Türkçe dublaj üretilmedi."
            )

    def _on_redub_selected_line(self) -> None:
        ln = self._selected_line()
        if not ln or not ln.translated_text.strip():
            return

        cfg = self._collect_config_from_ui(save=True)
        # Windows dosya kilidi (Permission denied) önlemi: oynatıcıyı durdur ve kaynağını boşalt
        self.audio_player.release()
        QApplication.processEvents()
        translator = GeminiTranslator(
            api_key=cfg.gemini_api_key,
            model_name=cfg.gemini_model,
            rpm=cfg.gemini_rpm,
            cache=self.cache_mgr,
            glossary=self.glossary_mgr,
            quota_manager=self.quota_mgr,
            phonetic_foreign_names=cfg.phonetic_foreign_names,
        )
        engine = self._build_speech_engine(cfg, translator=translator)
        pipeline = DubbingPipeline(
            config=cfg,
            speech_engine=engine,
            translator=translator,
            ffmpeg=self.ffmpeg_mgr,
            backup_manager=self.backup_mgr,
            pronunciation_manager=self.pronunciation_mgr,
        )

        self._set_busy(True)
        self.stage_label.setText(f"Yeniden dublajlanıyor: {ln.rel_path}...")
        self._redub_worker = SingleLineRedubWorker(
            pipeline, ln, ln.translated_text, ln.acting_prompt
        )
        self._redub_worker.log_signal.connect(self._log)
        self._redub_worker.line_done_signal.connect(self._on_single_redub_done)
        self._redub_worker.failed_signal.connect(self._on_single_redub_failed)
        self._redub_worker.start()

    def _on_single_redub_done(self, line: DialogueLine) -> None:
        self._on_line_updated(line)
        self._save_current_table_draft(show_status=False)
        self._set_busy(False)
        self.stage_label.setText(f"Satır yeniden dublajlandı: {line.rel_path}")

    def _on_single_redub_failed(self, err: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("Satır yeniden dublajlanamadı.")
        self._log(f"HATA (Yeniden Dublaj): {err}")
        QMessageBox.critical(self, "Yeniden Dublaj Hatası", err)

    def _on_redub_all_lines(self) -> None:
        """Tablodaki tüm konuşma satırlarını Whisper/Gemini aşamalarını TAMAMEN ATLAYARAK
        mevcut Türkçe Dublaj Metni ve Oyunculuk/Duygu hücreleriyle yeniden dublajlar.
        """
        cfg = self._collect_config_from_ui(save=True)
        tts_type = cfg.get_tts_engine()
        is_sts_mode = tts_type is TTSEngineType.ELEVENLABS_STS

        targets = [
            ln
            for ln in self._lines
            if ln.is_speech
            and ln.original_path.is_file()
            and (is_sts_mode or bool((ln.translated_text or "").strip()))
        ]
        if not targets:
            QMessageBox.information(
                self,
                "Satır Bulunamadı",
                "Yeniden dublajlanacak Türkçe metinli konuşma satırı bulunamadı.",
            )
            return

        if not self.ffmpeg_mgr.is_available():
            QMessageBox.critical(
                self,
                "FFmpeg Eksik",
                "FFmpeg bulunamadı! Lütfen taşınabilir ffmpeg.exe dosyasını ./_araclar/ffmpeg/ klasörüne koyun.",
            )
            return

        if cfg.fixed_reference_wav and not Path(cfg.fixed_reference_wav).is_file():
            QMessageBox.warning(
                self,
                "Referans Ses Bulunamadı",
                f"Seçilen sabit referans ses dosyası mevcut değil:\n{cfg.fixed_reference_wav}",
            )
            return

        if tts_type.is_elevenlabs:
            if not cfg.elevenlabs_api_key:
                QMessageBox.warning(
                    self,
                    "ElevenLabs API Anahtarı Eksik",
                    "ElevenLabs motoru seçili. Lütfen ElevenLabs API anahtarınızı girin.",
                )
                return
        elif cfg.get_engine_mode() is EngineMode.COLAB_CLOUD:
            if not cfg.colab_url or not cfg.colab_token:
                QMessageBox.warning(
                    self,
                    "Colab Bilgileri Eksik",
                    "Colab Bulut Modu seçili. Lütfen Colab URL ve Bearer Token bilgilerini girin "
                    "veya üstten 'Yerel GPU Modu (CUDA)' seçeneğine geçin.",
                )
                return

        self._save_current_table_draft(show_status=False)
        self.audio_player.release()
        QApplication.processEvents()

        translator = GeminiTranslator(
            api_key=cfg.gemini_api_key or "DUMMY",
            model_name=cfg.gemini_model,
            rpm=cfg.gemini_rpm,
            cache=self.cache_mgr,
            glossary=self.glossary_mgr,
            quota_manager=self.quota_mgr,
            phonetic_foreign_names=cfg.phonetic_foreign_names,
        )
        engine = self._build_speech_engine(cfg, translator=translator)
        self._active_pipeline = DubbingPipeline(
            config=cfg,
            speech_engine=engine,
            translator=translator,
            ffmpeg=self.ffmpeg_mgr,
            backup_manager=self.backup_mgr,
            pronunciation_manager=self.pronunciation_mgr,
        )

        self.progress_bar.setValue(0)
        self.stage_label.setText(
            f"Tümünü Yeniden Dublajla ({len(targets)} satır — Whisper & Gemini atlandı)..."
        )
        self._redub_all_worker = BatchRedubAllWorker(self._active_pipeline, self._lines)
        self._redub_all_worker.log_signal.connect(self._log)
        self._redub_all_worker.progress_signal.connect(self._on_pipeline_progress)
        self._redub_all_worker.line_updated_signal.connect(self._on_line_updated)
        self._redub_all_worker.completed_signal.connect(self._on_redub_all_completed)
        self._redub_all_worker.cancelled_signal.connect(self._on_redub_all_cancelled)
        self._redub_all_worker.failed_signal.connect(self._on_redub_all_failed)

        self._set_busy(True)
        self._redub_all_worker.start()

    def _on_redub_all_completed(self, lines: list[DialogueLine], stats: PipelineStats) -> None:
        self._lines = lines
        self._set_busy(False)
        self._refresh_status_badges()
        self._update_preview_buttons()
        self._save_current_table_draft(show_status=False)
        self.progress_bar.setValue(100)
        self.stage_label.setText(
            f"✅ Tümünü Yeniden Dublajla tamamlandı ({stats.synced} satır önizlemeye hazır)."
        )

    def _on_redub_all_cancelled(
        self, lines: list[DialogueLine], stats: PipelineStats, msg: str
    ) -> None:
        self._lines = lines
        self._set_busy(False)
        self._refresh_status_badges()
        self._update_preview_buttons()
        self._save_current_table_draft(show_status=False)
        self.stage_label.setText(
            f"⏹ Durduruldu — o ana kadar tamamlanan {stats.synced} satır korundu."
        )
        self._log(msg)

    def _on_redub_all_failed(self, err: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("Toplu yeniden dublaj başarısız oldu.")
        self._log(f"HATA (Tümünü Yeniden Dublajla): {err}")
        QMessageBox.critical(self, "Toplu Yeniden Dublaj Hatası", err)

    def _on_toggle_mic_recording(self) -> None:
        """
        Mikrofonla Oyunculuk Kaydet / Durdur butonu.
        Kayıt aktifse kaydı sonlandırıp Seed-VC (Speech-to-Speech) dönüştürmesini başlatır;
        değilse mikrofon varlığını kontrol ederek 48 kHz mono kaydı ./_temp/ içine başlatır.
        """
        ln = self._selected_line()
        if not ln:
            return

        if self.mic_recorder.is_recording:
            try:
                recorded_wav = self.mic_recorder.stop_recording()
                self._reset_mic_record_button_ui()
                self._log(
                    f"🎤 Mikrofon oyunculuk kaydı tamamlandı ({recorded_wav.name}). "
                    f"Seed-VC ile karakter sesine dönüştürülüyor..."
                )
                self._start_sts_conversion_for_line(ln, recorded_wav)
            except Exception as exc:
                self._reset_mic_record_button_ui()
                self._log(f"HATA (Mikrofon Kaydı): {exc}")
                QMessageBox.warning(self, "Mikrofon Kaydı Hatası", str(exc))
            return

        # Kayıt başlatılıyor
        self.audio_player.release()
        QApplication.processEvents()
        try:
            out_wav = self.mic_recorder.start_recording(prefix=f"mic_{Path(ln.rel_path).stem}")
            self.mic_record_btn.setText("⏹ Kaydı Bitir && Dönüştür")
            self.mic_record_btn.setObjectName("DangerBtn")
            self.mic_record_btn.style().unpolish(self.mic_record_btn)
            self.mic_record_btn.style().polish(self.mic_record_btn)
            self.upload_sts_btn.setEnabled(False)
            self.stage_label.setText(
                f"🎤 Mikrofon kaydı sürüyor ({ln.rel_path})... Bitirmek için tekrar tıklayın."
            )
            self._log(
                f"🎤 Mikrofonla oyunculuk kaydı başlatıldı (48 kHz Mono -> ./_temp/mikrofon_kayit/{out_wav.name}). "
                f"Repliği kendi duygunuzla okuyup '⏹ Kaydı Bitir & Dönüştür' butonuna basın."
            )
        except MicrophoneNotAvailableError as mic_err:
            self._reset_mic_record_button_ui()
            self._log(f"⚠️ Mikrofon Algılanamadı: {mic_err}")
            QMessageBox.warning(self, "Mikrofon Algılanamadı", str(mic_err))
        except Exception as exc:
            self._reset_mic_record_button_ui()
            self._log(f"HATA (Mikrofon Başlatılamadı): {exc}")
            QMessageBox.warning(self, "Mikrofon Başlatılamadı", str(exc))

    def _reset_mic_record_button_ui(self) -> None:
        self.mic_record_btn.setText("🎤 Mikrofonla Oyunculuk Kaydet")
        if self.mic_record_btn.objectName() != "":
            self.mic_record_btn.setObjectName("")
            self.mic_record_btn.style().unpolish(self.mic_record_btn)
            self.mic_record_btn.style().polish(self.mic_record_btn)
        self._update_preview_buttons()

    def _on_upload_sts_audio(self) -> None:
        """
        Dışarıdan (telefon veya kayıt cihazından) kaydedilmiş kullanıcı oyunculuk sesini seçip
        Seed-VC ile seçili satırın karakter sesine dönüştürür.
        """
        ln = self._selected_line()
        if not ln:
            return

        start_dir = self.game_dir_input.text().strip() or str(APP_ROOT)
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Kendi Oyunculuk Sesinizi Seçin (Speech-to-Speech / Seed-VC)",
            start_dir,
            "Ses Dosyaları (*.wav *.mp3 *.ogg *.flac)",
        )
        if not chosen:
            return

        user_audio = Path(chosen)
        if not user_audio.is_file():
            QMessageBox.warning(self, "Dosya Bulunamadı", f"Seçilen ses dosyası bulunamadı:\n{chosen}")
            return

        self._start_sts_conversion_for_line(ln, user_audio)

    def _start_sts_conversion_for_line(self, ln: DialogueLine, user_audio_path: Path) -> None:
        """Seçili satır için arka planda SingleLineSTSWorker (Seed-VC) başlatır."""
        cfg = self._collect_config_from_ui(save=True)
        # Windows dosya kilidi (Permission denied) önlemi: oynatıcıyı durdur ve kaynağını boşalt
        self.audio_player.release()
        QApplication.processEvents()

        translator = GeminiTranslator(
            api_key=cfg.gemini_api_key or "DUMMY",
            model_name=cfg.gemini_model,
            rpm=cfg.gemini_rpm,
            cache=self.cache_mgr,
            glossary=self.glossary_mgr,
            quota_manager=self.quota_mgr,
            phonetic_foreign_names=cfg.phonetic_foreign_names,
        )
        engine = self._build_speech_engine(cfg, translator=translator)
        pipeline = DubbingPipeline(
            config=cfg,
            speech_engine=engine,
            translator=translator,
            ffmpeg=self.ffmpeg_mgr,
            backup_manager=self.backup_mgr,
            pronunciation_manager=self.pronunciation_mgr,
        )

        self._set_busy(True)
        self.stage_label.setText(
            f"🎙️ Seed-VC ile oyunculuğunuz karakter sesine dönüştürülüyor: {ln.rel_path}..."
        )
        self._sts_worker = SingleLineSTSWorker(pipeline, ln, user_audio_path)
        self._sts_worker.log_signal.connect(self._log)
        self._sts_worker.line_done_signal.connect(self._on_single_sts_done)
        self._sts_worker.failed_signal.connect(self._on_single_sts_failed)
        self._sts_worker.start()

    def _on_single_sts_done(self, line: DialogueLine) -> None:
        self._on_line_updated(line)
        self._set_busy(False)
        self.stage_label.setText(
            f"✅ Oyunculuğunuz karakter sesine aktarıldı (Seed-VC STS): {line.rel_path}"
        )

    def _on_single_sts_failed(self, err: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("Ses dönüştürme (STS) başarısız oldu.")
        self._log(f"HATA (Ses Dönüştürme / STS): {err}")
        QMessageBox.critical(self, "Ses Dönüştürme (Seed-VC STS) Hatası", err)

    def _on_apply_to_game(self) -> None:
        ready = [
            ln for ln in self._lines if ln.synced_preview_path and ln.synced_preview_path.is_file()
        ]
        if not ready:
            QMessageBox.information(
                self, "Hazır Dosya Yok", "Oyuna uygulanacak hazır Türkçe dublaj bulunamadı."
            )
            return

        cfg = self._collect_config_from_ui(save=True)
        game_dir = Path(cfg.last_game_dir)
        project_name = self._current_project_name() or infer_smart_project_name(game_dir)

        reply = QMessageBox.question(
            self,
            "Oyuna Uygula (SHA-256 Yedekli)",
            f"{len(ready)} Türkçe dublaj dosyası oyuna uygulanacak.\n\n"
            f"• Önce orijinal dosyaların SHA-256 özeti alınıp './_yedekler/{project_name}/' "
            "klasörüne yedeklenecek.\n"
            "• İstediğiniz zaman 'Orijinale Dön (SHA-256)' ile tek tıkla geri alabilirsiniz.\n"
            "• Uygulama bitince './_temp' geçici dosyaları otomatik temizlenecek.\n\n"
            "Devam etmek istiyor musunuz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.audio_player.release()
        translator = GeminiTranslator(
            api_key=cfg.gemini_api_key,
            model_name=cfg.gemini_model,
            cache=self.cache_mgr,
            glossary=self.glossary_mgr,
            quota_manager=self.quota_mgr,
        )
        pipeline = self._active_pipeline or DubbingPipeline(
            config=cfg,
            speech_engine=self._build_speech_engine(cfg, translator=translator),
            translator=translator,
            ffmpeg=self.ffmpeg_mgr,
            backup_manager=self.backup_mgr,
            pronunciation_manager=self.pronunciation_mgr,
        )

        self._set_busy(True)
        self.stage_label.setText("SHA-256 yedekleniyor ve oyuna uygulanıyor...")
        self._apply_worker = ApplyDubbingWorker(pipeline, game_dir, project_name, ready)
        self._apply_worker.log_signal.connect(self._log)
        self._apply_worker.line_updated_signal.connect(self._on_line_updated)
        self._apply_worker.done_signal.connect(self._on_apply_done)
        self._apply_worker.failed_signal.connect(self._on_apply_failed)
        self._apply_worker.start()

    def _on_apply_done(self, count: int) -> None:
        self._set_busy(False)
        self._refresh_status_badges()
        self._refresh_hitman_scenes_and_status()
        self.stage_label.setText(
            f"✅ {count} Türkçe dublaj SHA-256 yedekli olarak oyuna uygulandı!"
        )
        QMessageBox.information(
            self,
            "Uygulama Tamamlandı",
            f"{count} ses dosyası başarıyla oyuna uygulandı.\n"
            "Orijinal dosyalar SHA-256 doğrulamalı olarak yedeklendi ve geçici dosyalar temizlendi.",
        )

    def _on_apply_failed(self, err: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("Oyuna uygulama başarısız!")
        self._log(f"HATA (Uygulama): {err}")
        QMessageBox.critical(self, "Uygulama Hatası", err)

    # ==========================================================================
    # ORİJİNALE DÖN (SHA-256) VE GEÇİCİ TEMİZLİK
    # ==========================================================================
    def _on_restore_original(self) -> None:
        projects = self.backup_mgr.list_projects()
        if not projects:
            QMessageBox.information(
                self,
                "Yedek Bulunamadı",
                "'./_yedekler/' klasöründe geri yüklenecek SHA-256 yedeği bulunamadı.",
            )
            return

        cur_proj = self._current_project_name()
        if cur_proj in projects:
            chosen_project = cur_proj
        elif len(projects) == 1:
            chosen_project = projects[0]
        else:
            item, ok = QInputDialog.getItem(
                self,
                "Geri Yüklenecek Projeyi Seçin",
                "Orijinale döndürülecek yedek projesini seçin:",
                projects,
                0,
                False,
            )
            if not ok or not item:
                return
            chosen_project = item

        manifest = self.backup_mgr.load_manifest(chosen_project)
        file_cnt = len(manifest.files) if manifest else 0

        reply = QMessageBox.question(
            self,
            "Orijinale Dön (SHA-256 Doğrulamalı)",
            f"Proje: {chosen_project}\n"
            f"Geri yüklenecek orijinal dosya sayısı: {file_cnt}\n\n"
            "Tüm orijinal ses dosyaları oyundaki yerlerine geri yüklenecek, "
            "her dosya için SHA-256 bütünlük kontrolü yapılacak ve %100 doğrulama "
            "sonrası yedek klasörü kendiliğinden silinecektir.\n\n"
            "Onaylıyor musunuz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.audio_player.release()
        self._set_busy(True)
        self.progress_bar.setValue(0)
        self.stage_label.setText(f"Orijinale dönülüyor ({chosen_project})...")

        self._restore_worker = RestoreOriginalWorker(self.backup_mgr, chosen_project)
        self._restore_worker.log_signal.connect(self._log)
        self._restore_worker.progress_signal.connect(
            lambda cur, tot: self.progress_bar.setValue(int((cur / max(1, tot)) * 100))
        )
        self._restore_worker.done_signal.connect(self._on_restore_done)
        self._restore_worker.start()

    def _on_restore_done(self, res: RestoreResult) -> None:
        self._set_busy(False)
        self._refresh_status_badges()
        self._refresh_hitman_scenes_and_status()
        if res.success:
            self.progress_bar.setValue(100)
            self.stage_label.setText(
                f"✅ Orijinale dönüldü ({res.verified_files}/{res.total_files} SHA-256 doğrulandı)."
            )
            for ln in self._lines:
                ln.applied = False
            QMessageBox.information(
                self,
                "Orijinale Dön Başarılı",
                f"Proje: {res.project_name}\n"
                f"Geri yüklenen ve SHA-256 ile doğrulanan dosya: {res.verified_files}/{res.total_files}\n"
                "Yedek klasörü %100 doğrulama sonrası otomatik olarak silindi.",
            )
        else:
            err_txt = "\n".join(res.errors[:10])
            self.stage_label.setText("⚠️ Orijinale dönme sırasında hata oluştu (yedek korundu).")
            QMessageBox.critical(
                self,
                "Geri Yükleme Hatası",
                f"Doğrulanan dosya: {res.verified_files}/{res.total_files}\n"
                f"Yedek klasörü güvenlik amacıyla korundu.\n\nHatalar:\n{err_txt}",
            )

    def _on_clean_temp(self) -> None:
        size_before = get_temp_size_bytes()
        self.audio_player.release()
        deleted, failed = cleanup_temp_dir(keep_root=True)
        for ln in self._lines:
            if ln.synced_preview_path and not ln.synced_preview_path.exists():
                ln.synced_preview_path = None
            ln.candidates.clear()
        self._update_preview_buttons()
        mb = size_before / (1024 * 1024)
        if failed:
            self._log(
                f"Geçici temizlik: {deleted} öğe ({mb:.2f} MB) silindi, {len(failed)} öğe kilitli."
            )
        else:
            self._log(
                f"./_temp klasörü tamamen temizlendi ({deleted} öğe, {mb:.2f} MB serbest bırakıldı)."
            )
            self.stage_label.setText(f"🧹 Geçici dosyalar temizlendi ({mb:.2f} MB).")

    # ==========================================================================
    # HITMAN 3 (RPKG) OYUN MODÜLÜ — ÇIKARMA, BİRLEŞTİRME, 0xFFFF KURULUM VE KALDIRMA
    # ==========================================================================
    def _current_hitman_runtime_dir(self) -> Path:
        raw = self.hitman_runtime_input.text().strip() or DEFAULT_HITMAN_RUNTIME_DIR
        return Path(raw)

    def _current_hitman_scene(self) -> Optional[HitmanSceneProfile]:
        key = self.hitman_scene_combo.currentData()
        if key:
            for sc in self._hitman_scenes:
                if sc.scene_key == key:
                    return sc
        return self._hitman_scenes[0] if self._hitman_scenes else None

    def _on_hitman_browse_runtime(self) -> None:
        start = self.hitman_runtime_input.text().strip() or DEFAULT_HITMAN_RUNTIME_DIR
        if not Path(start).is_dir():
            start = str(APP_ROOT)
        chosen = QFileDialog.getExistingDirectory(
            self, "HITMAN 3 Runtime Klasörünü Seçin", start
        )
        if chosen:
            self.hitman_runtime_input.setText(chosen)
            self._collect_config_from_ui(save=True)
            self._refresh_hitman_scenes_and_status()
            self._log(f"[HITMAN 3] Runtime klasörü güncellendi: {chosen}")

    def _on_hitman_refresh_clicked(self) -> None:
        self._refresh_hitman_scenes_and_status()
        rt = self._current_hitman_runtime_dir()
        installed_cnt = sum(1 for st in self._hitman_statuses.values() if st.is_installed)
        self._log(
            f"[HITMAN 3] Runtime tarandı ({rt}): {len(self._hitman_scenes)} sahne/chunk profili, "
            f"{installed_cnt} kurulu Türkçe dublaj yaması tespit edildi."
        )

    def _refresh_hitman_scenes_and_status(self) -> None:
        prev_key = self.hitman_scene_combo.currentData()
        rt_dir = self._current_hitman_runtime_dir()
        self._hitman_scenes = discover_hitman_scenes(rt_dir)
        status_list = get_scenes_runtime_status(rt_dir, self._hitman_scenes)
        self._hitman_statuses = {st.profile.scene_key: st for st in status_list}

        self.hitman_scene_combo.blockSignals(True)
        try:
            self.hitman_scene_combo.clear()
            for sc in self._hitman_scenes:
                st = self._hitman_statuses.get(sc.scene_key)
                prefix = "✅ " if (st and st.is_installed) else ""
                self.hitman_scene_combo.addItem(f"{prefix}{sc.label_tr}", sc.scene_key)
            if prev_key:
                idx = self.hitman_scene_combo.findData(prev_key)
                if idx >= 0:
                    self.hitman_scene_combo.setCurrentIndex(idx)
        finally:
            self.hitman_scene_combo.blockSignals(False)

        self._update_hitman_status_card()

    def _on_hitman_scene_changed(self) -> None:
        self._update_hitman_status_card()

    def _update_hitman_status_card(self) -> None:
        rt_dir = self._current_hitman_runtime_dir()
        installed = [st for st in self._hitman_statuses.values() if st.is_installed]
        sel = self._current_hitman_scene()
        sel_st = self._hitman_statuses.get(sel.scene_key) if sel else None

        if sel and sel_st:
            pkg_cnt = len(sel_st.local_package_dirs)
            self.hitman_selected_badge.setText(
                f"{sel.short_title}: {sel_st.local_approved_dubbed}/{sel_st.local_total_extracted} onaylı ({pkg_cnt} paket)"
            )
            self.hitman_selected_badge.setObjectName(
                "StatusPillOk" if sel_st.is_installed else "StatusPillInfo"
            )
            self.hitman_selected_badge.style().unpolish(self.hitman_selected_badge)
            self.hitman_selected_badge.style().polish(self.hitman_selected_badge)
        else:
            self.hitman_selected_badge.setText("Sahne seçilmedi")

        lines: list[str] = []
        if not rt_dir.is_dir():
            lines.append(f"⚠️ Runtime klasörü bulunamadı: {rt_dir}")
        else:
            if installed:
                total_dubs = sum(st.installed_wwes_count for st in installed)
                parts_txt = "  •  ".join(
                    f"✅ {st.profile.short_title} ({st.profile.patch_rpkg_filename}: {st.installed_wwes_count} ses)"
                    for st in installed
                )
                lines.append(f"<b>Kurulu Yamalar ({total_dubs} aktif dublaj):</b> {parts_txt}")
            else:
                lines.append("Kurulu SesAI Türkçe dublaj yaması henüz tespit edilmedi.")

            if sel and sel_st:
                pkg_names = ", ".join(p.name for p in sel_st.local_package_dirs) or "Henüz paket yok"
                inst_txt = (
                    f"Oyunda kurulu ({sel_st.installed_wwes_count} aktif ses)"
                    if sel_st.is_installed
                    else "Oyunda kurulu değil"
                )
                lines.append(
                    f"<b>Seçili Sahne [{sel.short_title}]:</b> {inst_txt} | "
                    f"Birleştirmeye hazır onaylı WAV: <b>{sel_st.local_approved_dubbed}</b> "
                    f"(Yerel paketler: {pkg_names})"
                )

        self.hitman_status_label.setText("<br>".join(lines))

    def _on_hitman_extract(self) -> None:
        rt_dir = self._current_hitman_runtime_dir()
        if not rt_dir.is_dir():
            QMessageBox.warning(
                self,
                "HITMAN 3 Runtime Bulunamadı",
                f"Seçilen HITMAN 3 Runtime klasörü mevcut değil:\n{rt_dir}",
            )
            return

        scene = self._current_hitman_scene()
        if scene is None:
            QMessageBox.warning(self, "Sahne Seçilmedi", "Lütfen bir HITMAN 3 sahnesi / chunk seçin.")
            return

        min_dur = float(self.hitman_min_dur_spin.value())
        max_dur = float(self.hitman_max_dur_spin.value())
        if max_dur <= min_dur:
            QMessageBox.warning(
                self,
                "Geçersiz Süre Aralığı",
                "Maksimum süre, minimum süreden büyük olmalıdır.",
            )
            return

        try:
            limit_cnt = max(1, min(500, int(self.hitman_count_combo.currentText().strip() or "20")))
        except ValueError:
            QMessageBox.warning(
                self,
                "Geçersiz Ses Adedi",
                "Lütfen 1 ile 500 arasında geçerli bir ses adedi girin (ör. 10, 20, 30, 50).",
            )
            return

        keyword = self.hitman_keyword_input.text().strip()
        self._collect_config_from_ui(save=True)
        self.audio_player.release()

        self.progress_bar.setValue(0)
        self.stage_label.setText(
            f"HITMAN 3 ses çıkarılıyor ({scene.short_title} / chunk{scene.chunk_number})..."
        )
        self._hitman_extract_worker = HitmanExtractWorker(
            runtime_dir=rt_dir,
            scene=scene,
            keyword=keyword,
            min_duration_sec=min_dur,
            max_duration_sec=max_dur,
            limit_count=limit_cnt,
        )
        self._hitman_extract_worker.log_signal.connect(self._log)
        self._hitman_extract_worker.progress_signal.connect(self._on_pipeline_progress)
        self._hitman_extract_worker.completed_signal.connect(self._on_hitman_extract_done)
        self._hitman_extract_worker.cancelled_signal.connect(self._on_hitman_extract_cancelled)
        self._hitman_extract_worker.failed_signal.connect(self._on_hitman_extract_failed)

        self._set_busy(True)
        self._hitman_extract_worker.start()

    def _on_hitman_extract_done(self, res: HitmanExtractResult) -> None:
        self._set_busy(False)
        # Eski proje taslağını koruyup yeni çıkarılan paket klasörünü ana stüdyoya otomatik yükle
        self._save_current_table_draft(show_status=False)
        self._lines.clear()
        self._line_row_map.clear()
        self.table.setRowCount(0)

        self.game_dir_input.setText(str(res.wav_dir))
        self.project_name_input.setText(res.project_name)
        self._collect_config_from_ui(save=True)
        self._refresh_status_badges()
        self._refresh_hitman_scenes_and_status()
        self._update_preview_buttons()

        self.progress_bar.setValue(100)
        self.stage_label.setText(
            f"✅ {res.extracted_count} yeni HITMAN 3 sesi çıkarıldı ve projeye yüklendi ({res.project_name})."
        )
        self._log(
            f"[HITMAN 3] Yeni paket ana stüdyoya yüklendi: Oyun Klasörü='{res.wav_dir}' | "
            f"Proje='{res.project_name}'. Şimdi doğrudan '🚀 Dublajı Başlat' butonuna basabilirsiniz!"
        )
        QMessageBox.information(
            self,
            "HITMAN 3 Sesleri Çıkarıldı ve Yüklendi",
            f"Sahne: {res.scene_title} (chunk{res.chunk_number})\n"
            f"Çıkarılan yeni 48 kHz WAV: {res.extracted_count} dosya\n"
            f"Önceki paketlerden hariç tutulan: {res.skipped_existing_count} hash\n"
            f"Paket Klasörü: {res.package_dir}\n\n"
            "Klasör ana stüdyonun 'Oyun Klasörü' ve 'Proje / Yedek' alanlarına otomatik yüklendi.\n"
            "Doğrudan '🚀 Dublajı Başlat' butonuna basarak Türkçe dublaj üretebilirsiniz!",
        )

    def _on_hitman_extract_cancelled(self, msg: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("⏹ HITMAN 3 ses çıkarma durduruldu.")
        self._log(msg)
        self._refresh_hitman_scenes_and_status()

    def _on_hitman_extract_failed(self, err: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("❌ HITMAN 3 ses çıkarma başarısız oldu.")
        self._log(f"HATA (HITMAN 3 Çıkarma): {err}")
        QMessageBox.warning(self, "HITMAN 3 Ses Çıkarma Hatası", err)

    def _on_hitman_repack_install(self) -> None:
        rt_dir = self._current_hitman_runtime_dir()
        if not rt_dir.is_dir():
            QMessageBox.warning(
                self,
                "HITMAN 3 Runtime Bulunamadı",
                f"Seçilen HITMAN 3 Runtime klasörü mevcut değil:\n{rt_dir}",
            )
            return

        scene = self._current_hitman_scene()
        if scene is None:
            QMessageBox.warning(self, "Sahne Seçilmedi", "Lütfen paketlenecek HITMAN 3 sahnesini seçin.")
            return

        self._collect_config_from_ui(save=True)
        self.audio_player.release()
        self.progress_bar.setValue(0)
        self.stage_label.setText(
            f"HITMAN 3 onaylı dublajlar birleştirilip 0xFFFF RPKG kuruluyor ({scene.short_title})..."
        )

        self._hitman_install_worker = HitmanRepackInstallWorker(
            runtime_dir=rt_dir,
            scene=scene,
        )
        self._hitman_install_worker.log_signal.connect(self._log)
        self._hitman_install_worker.progress_signal.connect(self._on_pipeline_progress)
        self._hitman_install_worker.completed_signal.connect(self._on_hitman_install_done)
        self._hitman_install_worker.cancelled_signal.connect(self._on_hitman_install_cancelled)
        self._hitman_install_worker.failed_signal.connect(self._on_hitman_install_failed)

        self._set_busy(True)
        self._hitman_install_worker.start()

    def _on_hitman_install_done(self, report: dict) -> None:
        self._set_busy(False)
        self._refresh_hitman_scenes_and_status()
        pkg = report.get("rpkg_package") or {}
        dep = report.get("deployment") or {}
        cnt = int(report.get("merged_approved_count") or pkg.get("wwes_count") or 0)
        rpkg_name = str(pkg.get("rpkg_filename") or "")
        rpkg_sha = str(pkg.get("rpkg_sha256") or "")
        src_pkgs = report.get("merged_source_packages") or []

        self.progress_bar.setValue(100)
        if dep.get("installed_directly"):
            self.stage_label.setText(
                f"✅ {rpkg_name} ({cnt} onaylı 0xFFFF dublaj) HITMAN 3'e kuruldu ve SHA-256 doğrulandı!"
            )
            QMessageBox.information(
                self,
                "HITMAN 3 Kurulumu Başarılı (0xFFFF)",
                f"Sahne: {report.get('scene_label_tr', '')}\n"
                f"Birleştirilen Paket Sayısı: {len(src_pkgs)}\n"
                f"Kurulan Aktif Dublaj: {cnt} ses (0xFFFF Wwise Vorbis + ww2ogg/vgmstream/FFmpeg doğrulandı)\n"
                f"RPKG Paketi: {rpkg_name} ({pkg.get('rpkg_size_mb', 0)} MB)\n"
                f"SHA-256: {rpkg_sha}\n\n"
                "Oyunu açıp doğrudan Türkçe dublajla test edebilirsiniz!",
            )
        else:
            bat_path = dep.get("install_bat_path", "")
            self.stage_label.setText(
                f"⚠️ {rpkg_name} hazır — Program Files yazma izni için yönetici .bat çalıştırın."
            )
            QMessageBox.warning(
                self,
                "Yönetici (UAC) İzni Gerekli",
                f"{rpkg_name} ({cnt} dublaj) başarıyla üretildi ve doğrulandı ancak Runtime klasörüne "
                f"doğrudan yazılamadı:\n{dep.get('error', '')}\n\n"
                f"Kurulumu tamamlamak için şu betiğe çift tıklayın:\n{bat_path}",
            )

    def _on_hitman_install_cancelled(self, msg: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("⏹ HITMAN 3 paketleme işlemi durduruldu.")
        self._log(msg)
        self._refresh_hitman_scenes_and_status()

    def _on_hitman_install_failed(self, err: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("❌ HITMAN 3 paketleme/kurulum başarısız oldu.")
        self._log(f"HATA (HITMAN 3 Kurulum): {err}")
        QMessageBox.critical(self, "HITMAN 3 Paketleme / Kurulum Hatası", err)

    def _on_hitman_uninstall(self) -> None:
        rt_dir = self._current_hitman_runtime_dir()
        if not rt_dir.is_dir():
            QMessageBox.warning(
                self,
                "HITMAN 3 Runtime Bulunamadı",
                f"Seçilen HITMAN 3 Runtime klasörü mevcut değil:\n{rt_dir}",
            )
            return

        scene = self._current_hitman_scene()
        if scene is None:
            return

        st = self._hitman_statuses.get(scene.scene_key)
        if not st or not st.is_installed:
            QMessageBox.information(
                self,
                "Kurulu Yama Yok",
                f"{scene.short_title} ({scene.patch_rpkg_filename}) şu anda oyunda kurulu görünmüyor.",
            )
            return

        reply = QMessageBox.question(
            self,
            "HITMAN 3 Sahne Yamasını Kaldır",
            f"Sahne: {scene.short_title}\n"
            f"Kaldırılacak Yama: {scene.patch_rpkg_filename} ({st.installed_wwes_count} aktif dublaj)\n\n"
            "Yalnızca bu sahnenin yama paketi silinecek ve packagedefinition.txt içinde yalnızca "
            f"'{scene.partition_name}' patchlevel değeri orijinal seviyesine döndürülecektir.\n"
            "Diğer kurulu sahne yamalarına (ör. ICA Tesisi / Serbest Çalışan) dokunulmayacaktır.\n\n"
            "Devam etmek istiyor musunuz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.audio_player.release()
        self.progress_bar.setValue(0)
        self.stage_label.setText(f"HITMAN 3 sahne yaması kaldırılıyor ({scene.short_title})...")

        self._hitman_uninstall_worker = HitmanUninstallWorker(runtime_dir=rt_dir, scene=scene)
        self._hitman_uninstall_worker.log_signal.connect(self._log)
        self._hitman_uninstall_worker.progress_signal.connect(self._on_pipeline_progress)
        self._hitman_uninstall_worker.completed_signal.connect(self._on_hitman_uninstall_done)
        self._hitman_uninstall_worker.failed_signal.connect(self._on_hitman_uninstall_failed)

        self._set_busy(True)
        self._hitman_uninstall_worker.start()

    def _on_hitman_uninstall_done(self, res: dict) -> None:
        self._set_busy(False)
        self._refresh_hitman_scenes_and_status()
        self.progress_bar.setValue(100)
        self.stage_label.setText(
            f"✅ {res.get('scene_title', '')} yaması ({res.get('rpkg_filename', '')}) kaldırıldı."
        )
        QMessageBox.information(
            self,
            "Sahne Yaması Kaldırıldı",
            f"Sahne: {res.get('scene_title', '')}\n"
            f"Silinen Paket: {res.get('rpkg_filename', '')}\n"
            f"Geri Yüklenen Partition: {res.get('partition_name', '')} patchlevel={res.get('restored_patchlevel', 0)}\n"
            f"packagedefinition.txt SHA-256: {res.get('live_pkgdef_sha256', '')}\n\n"
            "Diğer kurulu sahne yamaları korundu.",
        )

    def _on_hitman_uninstall_failed(self, err: str) -> None:
        self._set_busy(False)
        self.stage_label.setText("❌ HITMAN 3 yama kaldırma başarısız oldu.")
        self._log(f"HATA (HITMAN 3 Kaldırma): {err}")
        QMessageBox.critical(self, "HITMAN 3 Kaldırma Hatası", err)

    # ==========================================================================
    # PENCERE KAPANIRKEN GARANTİLİ KAYIT VE TEMİZLİK
    # ==========================================================================
    def closeEvent(self, event: QCloseEvent) -> None:
        self._autosave_timer.stop()
        self._quota_tick_timer.stop()
        self.quota_mgr.remove_listener(self._on_quota_manager_notify)
        self._save_current_table_draft(show_status=False)
        self._collect_config_from_ui(save=True)
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.request_cancel()
            self._active_worker.wait(3000)
        if self._redub_all_worker and self._redub_all_worker.isRunning():
            self._redub_all_worker.request_cancel()
            self._redub_all_worker.wait(3000)
        if self._hitman_extract_worker and self._hitman_extract_worker.isRunning():
            self._hitman_extract_worker.request_cancel()
            self._hitman_extract_worker.wait(3000)
        if self._hitman_install_worker and self._hitman_install_worker.isRunning():
            self._hitman_install_worker.request_cancel()
            self._hitman_install_worker.wait(3000)
        if self._hitman_uninstall_worker and self._hitman_uninstall_worker.isRunning():
            self._hitman_uninstall_worker.wait(3000)

        if self.mic_recorder.is_recording:
            try:
                self.mic_recorder.stop_recording()
            except Exception:
                pass

        self.audio_player.shutdown()
        cleanup_temp_dir(keep_root=True)
        event.accept()
