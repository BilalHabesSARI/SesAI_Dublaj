"""TTS Telaffuz Sözlüğü ve Sinematik Zamanlama Kilidi (>= 15 sn) Birim Testleri."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf

from core.backup_manager import BackupManager
from core.cinematic_timing import (
    SpeechBlock,
    assemble_timed_blocks,
    detect_speech_blocks,
    split_turkish_text_to_blocks,
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
from core.portable_env import TEMP_DIR, bootstrap_portable_environment, cleanup_temp_dir
from core.table_drafts import TableDraftManager
from core.translation_cache import TranslationCache
from core.translator import GeminiTranslator
from core.tts_pronunciation import TTSPronunciationManager
from engines.base import SpeechEngine, TranscriptionResult


def _make_segmented_wav(
    path: Path,
    total_sec: float,
    speech_intervals_sec: list[tuple[float, float]],
    sr: int = 48000,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = np.zeros(int(sr * total_sec), dtype=np.float32)
    for s0, s1 in speech_intervals_sec:
        i0 = int(s0 * sr)
        i1 = min(len(buf), int(s1 * sr))
        t = np.linspace(0.0, (i1 - i0) / sr, i1 - i0, endpoint=False, dtype=np.float32)
        buf[i0:i1] = 0.25 * np.sin(2.0 * np.pi * 280.0 * t)
    sf.write(str(path), buf, sr, subtype="PCM_16")
    return path


class _SpySpeechEngine(SpeechEngine):
    mode = EngineMode.LOCAL_GPU

    def __init__(self, chunk_sec: float = 1.6) -> None:
        self.chunk_sec = chunk_sec
        self.synth_calls: list[dict] = []

    def health_check(self) -> EngineHealth:
        return EngineHealth(ok=True, mode=self.mode, message="Spy OK")

    def transcribe(self, audio_wav_path: Path, language: str = "en", cancel_event=None) -> TranscriptionResult:
        return TranscriptionResult(text="test", detected_language=language)

    def synthesize_clone(
        self,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        language: str = "tr",
        cancel_event=None,
        options: TTSGenerationOptions | None = None,
    ) -> Path:
        self.synth_calls.append(
            {
                "text": text,
                "speaker": str(speaker_wav_path),
                "acting_prompt": options.acting_prompt if options else "",
            }
        )
        sr = 48000
        t = np.linspace(0.0, self.chunk_sec, int(sr * self.chunk_sec), endpoint=False, dtype=np.float32)
        audio = (0.28 * np.sin(2.0 * np.pi * 240.0 * t)).astype(np.float32)
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), audio, sr, subtype="PCM_16")
        return output_wav_path


def test_tts_pronunciation_default_ica_whole_word_and_persistence(tmp_path: Path) -> None:
    pron_path = tmp_path / "tts_telaffuz.json"
    mgr = TTSPronunciationManager(path=pron_path)

    # Varsayılan başlangıç kuralı ICA -> Ay-Si-Ey olmalı ve dosyaya yazılmalı
    assert pron_path.is_file()
    rules = mgr.list_rules()
    assert any(r.source == "ICA" and r.spoken == "Ay-Si-Ey" for r in rules)

    # Tam kelime eşleşmesi: ICA ve ICA'nın dönüşmeli, fakat 'kalıca' / 'ica' bozulmamalı
    orig_text = "Bir ICA ajanı olarak ICA'nın kurallarını uygula; kalıca ve ica değişmesin."
    spoken = mgr.apply(orig_text)
    assert spoken == "Bir Ay-Si-Ey ajanı olarak Ay-Si-Ey'nın kurallarını uygula; kalıca ve ica değişmesin."

    # Yeni kural ekleme ve yeniden yükleme kalıcılığı
    mgr.upsert("FBI", "Ef-Bi-Ay", "Ajans kısaltması")
    mgr2 = TTSPronunciationManager(path=pron_path)
    assert mgr2.apply("ICA ve FBI birlikte.") == "Ay-Si-Ey ve Ef-Bi-Ay birlikte."


def test_cinematic_timing_lock_only_applies_to_15s_plus_and_preserves_pauses(tmp_path: Path) -> None:
    bootstrap_portable_environment()

    # 1. 18.0 sn uzun monolog: 3 konuşma bloğu, aralarında 1.5 sn ve 2.0 sn uzun esler
    long_wav = _make_segmented_wav(
        tmp_path / "long_briefing_18s.wav",
        total_sec=18.0,
        speech_intervals_sec=[(0.5, 4.0), (5.5, 10.0), (12.0, 17.0)],
    )
    # 2. 5.0 sn kısa replik: içinde 1.0 sn es olsa bile < 15 sn olduğu için standart akış korunmalı
    short_wav = _make_segmented_wav(
        tmp_path / "short_line_5s.wav",
        total_sec=5.0,
        speech_intervals_sec=[(0.3, 1.8), (2.8, 4.5)],
    )

    blocks, total_ms = detect_speech_blocks(long_wav, min_pause_sec=0.65)
    assert len(blocks) == 3
    assert total_ms == 18000
    assert abs(blocks[0].start_ms - 500) <= 50
    assert abs(blocks[1].start_ms - 5500) <= 50
    assert abs(blocks[2].start_ms - 12000) <= 50

    tr_long = (
        "İleri düzey görev eğitimine hoş geldin. Bu operasyon Sidney'de gerçekleşti. "
        "Hedefini sessizce ortadan kaldırman gerekecek. "
        "Unutma; bir ICA ajanı olarak her odadaki en tehlikeli kişi sensin. İyi şanslar."
    )
    matched_blocks, chunks = split_turkish_text_to_blocks(tr_long, blocks)
    assert len(matched_blocks) == 3
    assert len(chunks) == 3
    assert all(c.strip() for c in chunks)
    assert "ICA" in " ".join(chunks)

    ff = FFmpegManager()
    pron_mgr = TTSPronunciationManager(path=tmp_path / "tts_telaffuz.json")
    translator = GeminiTranslator(
        api_key="DUMMY",
        cache=TranslationCache(tmp_path / "cache.db"),
        glossary=GlossaryManager(tmp_path / "sozluk.json"),
    )

    try:
        # A) Kilit AÇIK iken < 15 sn kısa replik: standart 1 çağrı + TTS telaffuz dönüşümü (tablo metni korunur)
        spy_short = _SpySpeechEngine(chunk_sec=1.5)
        pipe_on = DubbingPipeline(
            config=AppConfig(
                tts_engine=TTSEngineType.VOXCPM2.value,
                candidate_count=1,
                cinematic_timing_lock=True,
                auto_apply_to_game=False,
            ),
            speech_engine=spy_short,
            translator=translator,
            ffmpeg=ff,
            backup_manager=BackupManager(backups_root=tmp_path / "_yedek"),
            pronunciation_manager=pron_mgr,
        )
        short_line = DialogueLine(
            line_id="S1",
            rel_path="short_line_5s.wav",
            original_path=short_wav,
            is_speech=True,
            source_text="Welcome to the ICA.",
            translated_text="ICA eğitimine hoş geldin. Hazır mısın?",
            acting_prompt="calm polite guidance",
        )
        pipe_on.redub_single_line(short_line, short_line.translated_text)
        assert len(spy_short.synth_calls) == 1
        assert spy_short.synth_calls[0]["text"] == "Ay-Si-Ey eğitimine hoş geldin. Hazır mısın?"
        assert short_line.translated_text == "ICA eğitimine hoş geldin. Hazır mısın?"

        # B) Kilit KAPALI iken >= 15 sn uzun monolog: yine standart 1 çağrı
        spy_off = _SpySpeechEngine(chunk_sec=2.0)
        pipe_off = DubbingPipeline(
            config=AppConfig(
                tts_engine=TTSEngineType.VOXCPM2.value,
                candidate_count=1,
                cinematic_timing_lock=False,
                auto_apply_to_game=False,
            ),
            speech_engine=spy_off,
            translator=translator,
            ffmpeg=ff,
            backup_manager=BackupManager(backups_root=tmp_path / "_yedek"),
            pronunciation_manager=pron_mgr,
        )
        long_line_off = DialogueLine(
            line_id="L0",
            rel_path="long_briefing_18s.wav",
            original_path=long_wav,
            is_speech=True,
            source_text="Long briefing.",
            translated_text=tr_long,
            acting_prompt="serious authoritative briefing",
        )
        pipe_off.redub_single_line(long_line_off, tr_long)
        assert len(spy_off.synth_calls) == 1

        # C) Kilit AÇIK iken >= 15 sn uzun monolog: 3 ayrı blok, aynı referans ses,
        #    ICA -> Ay-Si-Ey yalnızca TTS'e gider, tablo metni 'ICA' kalır ve orijinal esler korunur!
        spy_long = _SpySpeechEngine(chunk_sec=1.6)
        pipe_long = DubbingPipeline(
            config=AppConfig(
                tts_engine=TTSEngineType.VOXCPM2.value,
                candidate_count=1,
                cinematic_timing_lock=True,
                trim_silence=True,
                auto_apply_to_game=False,
            ),
            speech_engine=spy_long,
            translator=translator,
            ffmpeg=ff,
            backup_manager=BackupManager(backups_root=tmp_path / "_yedek"),
            pronunciation_manager=pron_mgr,
        )
        long_line_on = DialogueLine(
            line_id="L1",
            rel_path="long_briefing_18s.wav",
            original_path=long_wav,
            is_speech=True,
            source_text="Long briefing.",
            translated_text=tr_long,
            acting_prompt="serious authoritative briefing",
        )
        pipe_long.redub_single_line(long_line_on, tr_long)
        assert len(spy_long.synth_calls) == 3
        assert len({c["speaker"] for c in spy_long.synth_calls}) == 1
        joined_tts = " ".join(c["text"] for c in spy_long.synth_calls)
        assert "Ay-Si-Ey" in joined_tts
        assert "ICA" not in joined_tts
        assert "ICA" in long_line_on.translated_text
        assert long_line_on.status is LineStatus.READY_PREVIEW
        assert long_line_on.synced_preview_path is not None and long_line_on.synced_preview_path.is_file()

        # Üretilen final seste orijinal >= 0.65 sn eslerin korunduğunu doğrula (3 ayrı konuşma bloğu!)
        out_blocks, out_total_ms = detect_speech_blocks(
            long_line_on.synced_preview_path, min_pause_sec=0.65
        )
        assert len(out_blocks) == 3
        assert out_total_ms >= 15000
        assert abs(out_blocks[1].start_ms - 5500) <= 150
        assert abs(out_blocks[2].start_ms - 12000) <= 150
    finally:
        cleanup_temp_dir(keep_root=True)
    assert list(TEMP_DIR.iterdir()) == []


def test_ui_cinematic_lock_checkbox_and_pronunciation_dialog(tmp_path: Path) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    bootstrap_portable_environment()

    from PySide6.QtWidgets import QApplication
    from ui.glossary_dialog import GlossaryDialog
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    cfg_mgr = ConfigManager(path=tmp_path / "ui_ayar.json")
    quota_mgr = GeminiQuotaManager(path=tmp_path / "ui_kota.json")
    draft_mgr = TableDraftManager(drafts_dir=tmp_path / "ui_taslak")
    pron_mgr = TTSPronunciationManager(path=tmp_path / "ui_telaffuz.json")

    win = MainWindow(
        config_mgr=cfg_mgr,
        quota_mgr=quota_mgr,
        draft_mgr=draft_mgr,
        pronunciation_mgr=pron_mgr,
    )
    try:
        # Varsayılan olarak kapalı olmalı
        assert win.cinematic_timing_lock_chk.isChecked() is False
        assert "TTS:1" in win.glossary_btn.text()

        # Açıldığında ayarlara otomatik kaydedilmeli
        win.cinematic_timing_lock_chk.setChecked(True)
        app.processEvents()
        win._collect_config_from_ui(save=True)
        assert cfg_mgr.load().cinematic_timing_lock is True
        assert "EsKilidi" in win.chip_tempo.text()

        # Sözlük penceresindeki 2. sekmede varsayılan ICA -> Ay-Si-Ey olmalı ve yeni kural kaydedilebilmeli
        dlg = GlossaryDialog(
            GlossaryManager(path=tmp_path / "ui_sozluk.json"),
            win,
            pronunciation_manager=pron_mgr,
        )
        assert dlg.tabs.count() == 2
        assert dlg.pron_table.rowCount() == 1
        assert dlg.pron_table.item(0, 0).text() == "ICA"
        assert dlg.pron_table.item(0, 1).text() == "Ay-Si-Ey"

        dlg.pron_src_input.setText("CIA")
        dlg.pron_spoken_input.setText("Si-Ay-Ey")
        dlg._on_pron_add()
        dlg._on_save_and_close()

        reloaded = TTSPronunciationManager(path=tmp_path / "ui_telaffuz.json")
        assert reloaded.apply("ICA ve CIA ajanları.") == "Ay-Si-Ey ve Si-Ay-Ey ajanları."
    finally:
        win.close()
    assert list(TEMP_DIR.iterdir()) == []
