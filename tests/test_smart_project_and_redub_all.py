"""Akıllı Proje Adı, Manifest acting_prompt, Tümünü Yeniden Dublajla ve Otomatik Tablo Taslağı Testleri."""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from core.backup_manager import (
    BackupManager,
    infer_smart_project_name,
    is_generic_audio_subfolder,
)
from core.config import AppConfig, ConfigManager
from core.ffmpeg_manager import FFmpegManager
from core.gemini_models import GeminiQuotaManager
from core.glossary import GlossaryManager
from core.models import (
    DialogueLine,
    EngineHealth,
    EngineMode,
    LineStatus,
    TTSEngineType,
    TTSGenerationOptions,
)
from core.pipeline import DubbingPipeline
from core.portable_env import APP_ROOT, TEMP_DIR, bootstrap_portable_environment, cleanup_temp_dir
from core.table_drafts import TableDraftManager
from core.translation_cache import TranslationCache
from core.translator import GeminiTranslator
from core.vad_filter import SileroVADFilter, VADResult
from engines.base import SpeechEngine, TranscriptionResult


def _write_test_wav(path: Path, duration_sec: float = 1.4, freq_hz: float = 280.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = 48000
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    sig = (0.22 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(str(path), sig, sr, subtype="PCM_16")
    return path


class _RecordingMockEngine(SpeechEngine):
    mode = EngineMode.LOCAL_GPU

    def __init__(
        self,
        fail_on_transcribe: bool = False,
        cancel_after_n_synth: int = 0,
        cancel_event_to_trip: threading.Event | None = None,
    ) -> None:
        self.fail_on_transcribe = fail_on_transcribe
        self.cancel_after_n_synth = cancel_after_n_synth
        self.cancel_event_to_trip = cancel_event_to_trip
        self.transcribe_calls: list[tuple[Path, str]] = []
        self.synth_calls: list[dict] = []

    def health_check(self) -> EngineHealth:
        return EngineHealth(ok=True, mode=self.mode, message="Mock GPU OK")

    def transcribe(
        self, audio_wav_path: Path, language: str = "en", cancel_event=None
    ) -> TranscriptionResult:
        if self.fail_on_transcribe:
            raise AssertionError("Whisper deşifre çağrılmamalıydı!")
        self.transcribe_calls.append((audio_wav_path, language))
        return TranscriptionResult(text="Ajan programına hoş geldin, aday.", detected_language=language)

    def synthesize_clone(
        self,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        language: str = "tr",
        cancel_event=None,
        options: TTSGenerationOptions | None = None,
    ) -> Path:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("cancelled")
        self.synth_calls.append(
            {
                "text": text,
                "acting_prompt": options.acting_prompt if options else "",
                "candidate_index": options.candidate_index if options else 1,
                "speaker": speaker_wav_path,
            }
        )
        sr = 24000
        t = np.linspace(0.0, 1.3, int(sr * 1.3), endpoint=False, dtype=np.float32)
        audio = (0.25 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), audio, sr, subtype="PCM_16")
        if (
            self.cancel_after_n_synth > 0
            and self.cancel_event_to_trip is not None
            and len(self.synth_calls) >= self.cancel_after_n_synth
        ):
            self.cancel_event_to_trip.set()
        return output_wav_path


class _StrictNoCallTranslator(GeminiTranslator):
    def translate_line(self, *args, **kwargs):
        raise AssertionError("Gemini translate_line çağrılmamalıydı!")

    def translate_batch(self, *args, **kwargs):
        raise AssertionError("Gemini translate_batch çağrılmamalıydı!")


class _StrictNoCallVAD(SileroVADFilter):
    def analyze_file(self, audio_path: Path, work_dir=None) -> VADResult:
        raise AssertionError("VAD çağrılmamalıydı!")


class _AlwaysSpeechVAD(SileroVADFilter):
    def analyze_file(self, audio_path: Path, work_dir=None) -> VADResult:
        return VADResult(
            is_speech=True,
            speech_ratio=0.90,
            max_speech_prob=0.98,
            speech_duration_ms=1300,
            total_duration_ms=1400,
            engine_used="mock_vad",
        )


def test_smart_project_name_inference_and_ui_upgrade(tmp_path: Path) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    bootstrap_portable_environment()

    assert is_generic_audio_subfolder("sesler") is True
    assert is_generic_audio_subfolder("AUDIO") is True
    assert is_generic_audio_subfolder("sounds") is True
    assert is_generic_audio_subfolder("wav") is True
    assert is_generic_audio_subfolder("voices") is True
    assert is_generic_audio_subfolder("Hitman_IlkSahne") is False

    assert infer_smart_project_name(r"C:\Games\Hitman_IlkSahne\sesler") == "Hitman_IlkSahne"
    assert infer_smart_project_name(r"C:\Games\Hitman_SerbestEv\sesler") == "Hitman_SerbestEv"
    assert infer_smart_project_name(r"C:\Games\CyberDemo\audio") == "CyberDemo"
    assert infer_smart_project_name(r"C:\Games\SkyDemo\sounds") == "SkyDemo"
    assert infer_smart_project_name(r"C:\Games\WavPack\wav") == "WavPack"
    assert infer_smart_project_name(r"C:\Games\VoicePack\voices") == "VoicePack"
    assert infer_smart_project_name(r"C:\Games\OzelProje") == "OzelProje"

    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    cfg_mgr = ConfigManager(path=tmp_path / "ayarlar_smart.json")
    cfg_mgr.save(
        AppConfig(
            last_game_dir=str(tmp_path / "Hitman_IlkSahne" / "sesler"),
            project_name="sesler",
        )
    )
    quota_mgr = GeminiQuotaManager(path=tmp_path / "kota_smart.json")
    draft_mgr = TableDraftManager(drafts_dir=tmp_path / "taslaklar")

    win = MainWindow(config_mgr=cfg_mgr, quota_mgr=quota_mgr, draft_mgr=draft_mgr)
    try:
        assert win.project_name_input.text() == "Hitman_IlkSahne"
        assert win._current_project_name() == "Hitman_IlkSahne"
        reloaded = cfg_mgr.load()
        assert reloaded.project_name == "Hitman_IlkSahne"

        # Klasör adı genel alt klasörle değiştirildiğinde (proje adı hala genel ise) otomatik üst klasörü almalı
        win.project_name_input.setText("voices")
        win.game_dir_input.setText(str(tmp_path / "Hitman_SerbestEv" / "voices"))
        app.processEvents()
        assert win.project_name_input.text() == "Hitman_SerbestEv"
    finally:
        win.close()


def test_hitman_ilksahne_manifest_15_entries_and_verbatim_acting_prompt(tmp_path: Path) -> None:
    bootstrap_portable_environment()

    # 1. Yerelde Hitman_IlkSahne/hitman_manifest.json mevcutsa 15 kaydın birebir doğruluğunu denetle
    #    (Temiz açık kaynak klonlarda Hitman_* .gitignore dışında olduğundan bu blok güvenle atlanır)
    manifest_path = APP_ROOT / "Hitman_IlkSahne" / "hitman_manifest.json"
    if manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_hash = {str(item["hash_hex"]).upper(): item for item in raw["files"]}
        assert len(by_hash) == 15

        expected_samples = {
            "00EF877A8B0484FC": (
                "Ajan programına hoş geldin, aday. Seni içeride bekliyor olacağım. Lütfen ışıkları takip et.",
                "formal yet inviting, confident, calm polite guidance",
            ),
            "00FD5831537234D0": (
                "İleri düzey görev eğitimine hoş geldin. Bu operasyon aslen Sidney'de gerçekleşti. Hedef, Serçe olarak da bilinen ünlü usta hırsız Kalvin Ritter'dı. Yata sızman, hedefini izole edip ortadan kaldırman ve bölgeden ayrılman gerekecek; üstelik hiç şüphe uyandırmadan. Ve unutma; bir ICA ajanı olarak girdiğin her odadaki en tehlikeli kişi sensin. Ama kaba kuvvet seni bu işte hiçbir yere götürmez ve gerçek bir suikastçı asla dikkatleri üzerine çekmez. İyi şanslar, aday.",
                "serious authoritative briefing, calm poise, sharp disciplined warning, supportive closing",
            ),
            "0017B0FE07C356AF": (
                "İçeridesin. Aferin, aday.",
                "calm confirmation, approving professional satisfaction",
            ),
            "00525B83F0F4E91F": (
                "Tekrar hoş geldin, aday. Bir ICA ajanı olarak karşılaştığın her zorluğun üstesinden birden fazla yolla gelinebilir. Bu alıştırmayı tekrar tamamla; bu kez olaya farklı bir açıdan yaklaş. Stratejini çeşitlendir. Doğaçlama yap. Seni izliyor olacağız.",
                "calm professional welcome, encouraging firm guidance, lowered mysterious watchful closing",
            ),
            "0051196C5F3819F4": (
                "Oo, böyle bir araç mı? Yazık olurdu doğrusu. Aslında kız kardeşimin kocasında da tıpkı bunun gibi bir model var.",
                "admiring exclamation, slight regret, relaxed friendly chatty tone",
            ),
        }
        for h_hex, (exp_tr, exp_act) in expected_samples.items():
            rec = by_hash[h_hex]
            assert rec["dialogue_subtitle_tr"] == exp_tr
            assert rec["dialogue_subtitle_tr_mod"] == exp_tr
            assert rec["acting_prompt"] == exp_act
        for rec in raw["files"]:
            assert rec.get("dialogue_subtitle_tr", "").strip()
            assert rec.get("acting_prompt", "").strip()

    # 2. Pipeline'ın manifestteki özel acting_prompt'u (calm/polite/professional silmeden) birebir kullandığını doğrula
    sandbox = tmp_path / "manifest_act_sandbox"
    sesler_dir = sandbox / "Hitman_Test" / "sesler"
    wav_file = _write_test_wav(sesler_dir / "01_welcome_00EF877A8B0484FC.wav")

    verbatim_prompt = "formal yet inviting, confident, calm polite guidance"
    verbatim_tr = (
        "Ajan programına hoş geldin, aday. Seni içeride bekliyor olacağım. Lütfen ışıkları takip et."
    )
    (sandbox / "Hitman_Test" / "hitman_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "index": 1,
                        "hash_hex": "00EF877A8B0484FC",
                        "wav_filename": wav_file.name,
                        "dialogue_subtitle_en": "Welcome to the agency program, initiate.",
                        "dialogue_subtitle_tr": verbatim_tr,
                        "acting_prompt": verbatim_prompt,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        ff = FFmpegManager()
        engine = _RecordingMockEngine(fail_on_transcribe=True)
        translator = _StrictNoCallTranslator(
            api_key="DUMMY",
            cache=TranslationCache(sandbox / "cache.db"),
            glossary=GlossaryManager(sandbox / "sozluk.json"),
        )
        pipeline = DubbingPipeline(
            config=AppConfig(
                tts_engine=TTSEngineType.VOXCPM2.value,
                candidate_count=1,
                auto_apply_to_game=False,
            ),
            speech_engine=engine,
            translator=translator,
            ffmpeg=ff,
            vad=_AlwaysSpeechVAD(ffmpeg=ff),
            backup_manager=BackupManager(backups_root=sandbox / "_yedekler"),
        )
        lines, stats = pipeline.run(game_dir=sesler_dir, project_name="Hitman_Test")
        assert stats.synced == 1
        assert len(lines) == 1
        assert lines[0].translated_text == verbatim_tr
        assert lines[0].acting_prompt == verbatim_prompt
        assert len(engine.synth_calls) == 1
        assert engine.synth_calls[0]["text"] == verbatim_tr
        assert engine.synth_calls[0]["acting_prompt"] == verbatim_prompt
    finally:
        cleanup_temp_dir(keep_root=True)
    assert list(TEMP_DIR.iterdir()) == []


def test_redub_all_skips_whisper_gemini_preserves_edits_and_keeps_completed_on_cancel(
    tmp_path: Path,
) -> None:
    bootstrap_portable_environment()

    game_dir = tmp_path / "Hitman_RedubAll" / "sesler"
    wav1 = _write_test_wav(game_dir / "01_line.wav", duration_sec=1.3, freq_hz=260.0)
    wav2 = _write_test_wav(game_dir / "02_line.wav", duration_sec=1.3, freq_hz=310.0)
    wav3 = _write_test_wav(game_dir / "03_line.wav", duration_sec=1.3, freq_hz=360.0)

    def _make_lines() -> list[DialogueLine]:
        return [
            DialogueLine(
                line_id="L001",
                rel_path="01_line.wav",
                original_path=wav1,
                is_speech=True,
                source_text="Welcome, initiate.",
                translated_text="Elle düzenlenmiş birinci satır.",
                acting_prompt="calm polite guidance, warm poise",
            ),
            DialogueLine(
                line_id="L002",
                rel_path="02_line.wav",
                original_path=wav2,
                is_speech=True,
                source_text="Move along.",
                translated_text="Elle düzenlenmiş ikinci satır.",
                acting_prompt="low whispered urgency, wary",
            ),
            DialogueLine(
                line_id="L003",
                rel_path="03_line.wav",
                original_path=wav3,
                is_speech=True,
                source_text="Well done.",
                translated_text="Elle düzenlenmiş üçüncü satır.",
                acting_prompt="pleased dry confidence",
            ),
        ]

    try:
        ff = FFmpegManager()

        # Senaryo A: Tam çalışma (candidate_count=1) — 0 Whisper, 0 Gemini, 0 VAD
        lines_full = _make_lines()
        engine_full = _RecordingMockEngine(fail_on_transcribe=True)
        translator_strict = _StrictNoCallTranslator(
            api_key="DUMMY",
            cache=TranslationCache(tmp_path / "cache_a.db"),
            glossary=GlossaryManager(tmp_path / "sozluk_a.json"),
        )
        pipeline_full = DubbingPipeline(
            config=AppConfig(
                tts_engine=TTSEngineType.VOXCPM2.value,
                candidate_count=1,
                auto_apply_to_game=False,
            ),
            speech_engine=engine_full,
            translator=translator_strict,
            ffmpeg=ff,
            vad=_StrictNoCallVAD(ffmpeg=ff),
            backup_manager=BackupManager(backups_root=tmp_path / "_yedekler_a"),
        )
        out_lines, stats, was_cancelled = pipeline_full.redub_all_lines(lines_full)
        assert was_cancelled is False
        assert stats.synced == 3
        assert len(engine_full.synth_calls) == 3
        assert [c["text"] for c in engine_full.synth_calls] == [
            "Elle düzenlenmiş birinci satır.",
            "Elle düzenlenmiş ikinci satır.",
            "Elle düzenlenmiş üçüncü satır.",
        ]
        assert [c["acting_prompt"] for c in engine_full.synth_calls] == [
            "calm polite guidance, warm poise",
            "low whispered urgency, wary",
            "pleased dry confidence",
        ]
        for ln in out_lines:
            assert ln.status is LineStatus.READY_PREVIEW
            assert ln.synced_preview_path is not None and ln.synced_preview_path.is_file()

        # Senaryo B: İşlem ortasında "Durdur" (candidate_count=2 ile 1. satır bittikten sonra iptal)
        # İlk satırın 2 adayı tamamlandığında iptal tetiklenir -> 1. satır puanlanıp senkronize edilerek KORUNUR!
        lines_cancel = _make_lines()
        cancel_ev = threading.Event()
        engine_cancel = _RecordingMockEngine(
            fail_on_transcribe=False,
            cancel_after_n_synth=2,  # 1. satırın 2 adayı bitince iptal et
            cancel_event_to_trip=cancel_ev,
        )
        pipeline_cancel = DubbingPipeline(
            config=AppConfig(
                tts_engine=TTSEngineType.VOXCPM2.value,
                candidate_count=2,
                auto_apply_to_game=False,
            ),
            speech_engine=engine_cancel,
            translator=translator_strict,
            ffmpeg=ff,
            vad=_StrictNoCallVAD(ffmpeg=ff),
            backup_manager=BackupManager(backups_root=tmp_path / "_yedekler_b"),
        )
        out_c, stats_c, was_cancelled_c = pipeline_cancel.redub_all_lines(
            lines_cancel, cancel_event=cancel_ev
        )
        assert was_cancelled_c is True
        assert stats_c.synced == 1
        assert out_c[0].status is LineStatus.READY_PREVIEW
        assert out_c[0].synced_preview_path is not None and out_c[0].synced_preview_path.is_file()
        assert len(out_c[0].candidates) == 2
        # Whisper yalnızca Türkçe aday hakemi olarak çalıştı (kaynak deşifre için değil!)
        assert all(lang == "tr" for _, lang in engine_cancel.transcribe_calls)
        # İşlenmeyen 2. ve 3. satırların metin ve duygu düzenlemeleri bozulmadan kaldı
        assert out_c[1].translated_text == "Elle düzenlenmiş ikinci satır."
        assert out_c[1].acting_prompt == "low whispered urgency, wary"
        assert out_c[2].translated_text == "Elle düzenlenmiş üçüncü satır."
        assert out_c[2].acting_prompt == "pleased dry confidence"
    finally:
        cleanup_temp_dir(keep_root=True)
    assert list(TEMP_DIR.iterdir()) == []


def test_table_draft_autosave_restore_and_redub_all_ui_button(tmp_path: Path) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    bootstrap_portable_environment()

    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])

    game_dir = tmp_path / "Hitman_IlkSahne" / "sesler"
    wav1 = _write_test_wav(game_dir / "01_intro.wav")
    wav2 = _write_test_wav(game_dir / "02_briefing.wav")

    cfg_mgr = ConfigManager(path=tmp_path / "draft_ayarlar.json")
    cfg_mgr.save(AppConfig(last_game_dir=str(game_dir), project_name="sesler"))
    quota_mgr = GeminiQuotaManager(path=tmp_path / "draft_kota.json")
    draft_mgr = TableDraftManager(drafts_dir=tmp_path / "taslaklar")

    win1 = MainWindow(config_mgr=cfg_mgr, quota_mgr=quota_mgr, draft_mgr=draft_mgr)
    try:
        # Proje adı otomatik Hitman_IlkSahne olmalı ve "Tümünü Yeniden Dublajla" butonu hazır olmalı
        assert win1.project_name_input.text() == "Hitman_IlkSahne"
        assert "Tümünü Yeniden Dublajla" in win1.redub_all_btn.text()
        assert win1.redub_all_btn.isEnabled() is False

        initial_lines = [
            DialogueLine(
                line_id="L01",
                rel_path="01_intro.wav",
                original_path=wav1,
                is_speech=True,
                source_text="Welcome, initiate.",
                translated_text="İlk metin.",
                acting_prompt="calm welcome",
            ),
            DialogueLine(
                line_id="L02",
                rel_path="02_briefing.wav",
                original_path=wav2,
                is_speech=True,
                source_text="Good luck.",
                translated_text="İyi şanslar.",
                acting_prompt="firm guidance",
            ),
        ]
        win1._on_pipeline_discovered(initial_lines)
        win1._update_preview_buttons()
        app.processEvents()
        assert win1.redub_all_btn.isEnabled() is True

        # Kullanıcı tabloda 1. satırın Türkçe metnini ve 2. satırın Oyunculuk/Duygu hücresini elle düzenlesin
        tr_item = win1.table.item(0, win1.COL_TR)
        act_item = win1.table.item(1, win1.COL_ACTING)
        assert tr_item is not None and act_item is not None
        tr_item.setText("Ajan programına hoş geldin, aday — özel düzenleme.")
        act_item.setText("serious authoritative briefing, supportive closing")
        app.processEvents()

        draft_file = draft_mgr.draft_path("Hitman_IlkSahne")
        assert draft_file.is_file()
    finally:
        win1.close()

    # Uygulama kapatılıp yeniden açıldığında elle yapılan düzenlemeler otomatik geri yüklenmeli!
    win2 = MainWindow(config_mgr=cfg_mgr, quota_mgr=quota_mgr, draft_mgr=draft_mgr)
    try:
        app.processEvents()
        assert win2.table.rowCount() == 2
        assert (
            win2.table.item(0, win2.COL_TR).text()
            == "Ajan programına hoş geldin, aday — özel düzenleme."
        )
        assert (
            win2.table.item(1, win2.COL_ACTING).text()
            == "serious authoritative briefing, supportive closing"
        )
        assert win2.redub_all_btn.isEnabled() is True
    finally:
        win2.close()
    assert list(TEMP_DIR.iterdir()) == []
