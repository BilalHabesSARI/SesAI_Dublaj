"""Uçtan uca (End-to-End) Sinematik İş Hattı + Kalıcı Ayarlar + Ferah PySide6 Arayüz Testleri."""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from core.backup_manager import BackupManager, compute_sha256
from core.config import AppConfig, ConfigManager
from core.ffmpeg_manager import FFmpegManager
from core.glossary import GlossaryManager
from core.models import EngineHealth, EngineMode, LineStatus, TTSEngineType, TTSGenerationOptions
from core.pipeline import DubbingPipeline, PipelineCancelledError
from core.portable_env import APP_ROOT, TEMP_DIR, bootstrap_portable_environment
from core.translation_cache import TranslationCache
from core.translator import GeminiTranslator, TranslationResult
from core.vad_filter import SileroVADFilter, VADResult
from engines.base import SpeechEngine, TranscriptionResult


class _MockSpeechEngine(SpeechEngine):
    mode = EngineMode.COLAB_CLOUD

    def __init__(self) -> None:
        self.speaker_refs_seen: list[Path] = []
        self.options_seen: list[TTSGenerationOptions] = []
        self.sts_calls: list[dict] = []

    def health_check(self) -> EngineHealth:
        return EngineHealth(ok=True, mode=self.mode, message="Mock OK")

    def transcribe(self, audio_wav_path: Path, language="en", cancel_event=None) -> TranscriptionResult:
        return TranscriptionResult(text="Stay sharp, soldier!", detected_language="en")

    def convert_speech_to_speech(
        self,
        source_acting_wav: Path,
        target_speaker_wav: Path,
        output_wav_path: Path,
        diffusion_steps: int = 25,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        cancel_event=None,
        options=None,
    ) -> Path:
        assert source_acting_wav.is_file()
        assert target_speaker_wav.is_file()
        self.sts_calls.append(
            {
                "source": source_acting_wav,
                "target": target_speaker_wav,
                "diffusion_steps": diffusion_steps,
                "auto_f0_adjust": auto_f0_adjust,
                "pitch_shift": pitch_shift,
            }
        )
        sr = 22050
        t = np.linspace(0.0, 1.75, int(sr * 1.75), endpoint=False, dtype=np.float32)
        converted = (0.30 * np.sin(2.0 * np.pi * 185.0 * t)).astype(np.float32)
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav_path), converted, sr, subtype="PCM_16")
        return output_wav_path

    def synthesize_clone(
        self,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        language: str = "tr",
        cancel_event=None,
        options: TTSGenerationOptions | None = None,
    ) -> Path:
        assert speaker_wav_path.is_file()
        self.speaker_refs_seen.append(speaker_wav_path)
        if options is not None:
            self.options_seen.append(options)
        # Başı/sonu hafif sessiz + 1.9s konuşma üret (sessizlik kırpma + doğal 1.05x testi!)
        sr = 24000
        sil = np.zeros(int(sr * 0.35), dtype=np.float32)
        t = np.linspace(0.0, 1.9, int(sr * 1.9), endpoint=False, dtype=np.float32)
        speech = (0.28 * np.sin(2.0 * np.pi * 260.0 * t)).astype(np.float32)
        audio = np.concatenate([sil, speech, sil])
        sf.write(str(output_wav_path), audio, sr, subtype="PCM_16")
        return output_wav_path


class _MockVAD(SileroVADFilter):
    def analyze_file(self, audio_path: Path, work_dir=None) -> VADResult:
        low_name = audio_path.name.lower()
        is_sp = any(k in low_name for k in ("dialog", "vocals", "diana", "onboarding"))
        return VADResult(
            is_speech=is_sp,
            speech_ratio=0.82 if is_sp else 0.0,
            max_speech_prob=0.95 if is_sp else 0.03,
            speech_duration_ms=1400 if is_sp else 0,
            total_duration_ms=1800,
            engine_used="mock_vad",
        )


class _MockTranslator(GeminiTranslator):
    def translate_line(
        self,
        source_text,
        duration_sec,
        source_lang="en",
        target_lang="tr",
        use_cache=True,
        cancel_event=None,
        log_cb=None,
    ):
        return TranslationResult(
            translated_text="Gözünü dört aç asker!",
            from_cache=False,
            matched_glossary_count=0,
        )


def test_full_e2e_cinema_pipeline_fixed_ref_bg_preserve_backup_and_temp_clean() -> None:
    bootstrap_portable_environment()

    sandbox = APP_ROOT / "tests" / "_e2e_sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        game_dir = sandbox / "OrnekOyun"
        (game_dir / "voices").mkdir(parents=True)
        dialog_wav = game_dir / "voices" / "hero_dialog_01.wav"
        sfx_wav = game_dir / "voices" / "footstep_sfx.wav"
        fixed_ref = sandbox / "agent47_12sec_ref.wav"

        # 1.8s 48kHz Stereo konuşma + 12s temiz sabit karakter referansı
        t = np.linspace(0.0, 1.8, int(48000 * 1.8), endpoint=False, dtype=np.float32)
        sig = (0.20 * np.sin(2.0 * np.pi * 320.0 * t)).astype(np.float32)
        stereo = np.stack([sig, sig], axis=1)
        sf.write(str(dialog_wav), stereo, 48000, subtype="PCM_16")
        sf.write(str(sfx_wav), np.zeros((48000, 2), dtype=np.float32), 48000, subtype="PCM_16")

        t_ref = np.linspace(0.0, 12.0, int(24000 * 12.0), endpoint=False, dtype=np.float32)
        sf.write(
            str(fixed_ref),
            (0.22 * np.sin(2.0 * np.pi * 160.0 * t_ref)).astype(np.float32),
            24000,
            subtype="PCM_16",
        )

        orig_dialog_hash = compute_sha256(dialog_wav)
        orig_sfx_hash = compute_sha256(sfx_wav)

        cfg = AppConfig(
            tts_engine=TTSEngineType.CHATTERBOX_MULTILINGUAL.value,
            chatterbox_exaggeration=0.68,
            chatterbox_cfg_weight=0.40,
            fixed_reference_wav=str(fixed_ref),
            preserve_background=True,
            trim_silence=True,
            max_tempo_speedup=1.05,
            auto_apply_to_game=False,
        )
        engine = _MockSpeechEngine()
        translator = _MockTranslator(
            api_key="DUMMY",
            cache=TranslationCache(sandbox / "cache.db"),
            glossary=GlossaryManager(sandbox / "sozluk.json"),
        )
        backup_mgr = BackupManager(backups_root=sandbox / "_yedekler")
        ff = FFmpegManager()

        pipeline = DubbingPipeline(
            config=cfg,
            speech_engine=engine,
            translator=translator,
            ffmpeg=ff,
            vad=_MockVAD(ffmpeg=ff),
            backup_manager=backup_mgr,
        )

        # 1. Önizleme modunda çalıştır
        lines, stats = pipeline.run(game_dir=game_dir, project_name="OrnekOyun")
        assert stats.total_scanned == 2
        assert stats.synced >= 1
        assert stats.applied == 0
        assert compute_sha256(dialog_wav) == orig_dialog_hash

        # Sabit referans WAV + Chatterbox parametreleri motora eksiksiz aktarılmış olmalı!
        assert engine.speaker_refs_seen[0].name == "fixed_character_ref.wav"
        assert engine.options_seen[0].tts_engine is TTSEngineType.CHATTERBOX_MULTILINGUAL
        assert abs(engine.options_seen[0].exaggeration - 0.68) < 1e-4
        assert abs(engine.options_seen[0].cfg_weight - 0.40) < 1e-4

        ready_line = next(ln for ln in lines if ln.rel_path.endswith("hero_dialog_01.wav"))
        assert ready_line.status is LineStatus.READY_PREVIEW
        assert ready_line.background_preserved is True
        assert ready_line.tempo_factor <= 1.055  # Yeni doğal 1.05x üst sınırı!
        assert ready_line.acting_prompt  # Otomatik oyunculuk talimatı üretilmiş olmalı

        # 1b. Windows dosya kilidi (Permission denied) testi + acting_prompt düzenleyip Satırı Yeniden Dublajla:
        # Eski önizleme dosyasını açık tutarken bile redub_single_line benzersiz yeni dosya adına yazarak başarılı olmalı!
        old_preview_path = ready_line.synced_preview_path
        assert old_preview_path is not None and old_preview_path.is_file()
        with open(old_preview_path, "rb") as _locked_handle:
            redubbed = pipeline.redub_single_line(
                line=ready_line,
                new_turkish_text="Sözleşmeniz derhal feshedildi!",
                new_acting_prompt="threatening, sinister dark tone, dramatic pause",
            )
            assert redubbed.synced_preview_path is not None
            assert redubbed.synced_preview_path != old_preview_path
            assert redubbed.synced_preview_path.is_file()
            assert redubbed.acting_prompt == "threatening, sinister dark tone, dramatic pause"
            assert engine.options_seen[-1].acting_prompt == "threatening, sinister dark tone, dramatic pause"

        # 1c. Yerel GPU Ses Dönüştürme (Speech-to-Speech / Seed-VC) testi:
        # Kullanıcının kendi mikrofon/dosya oyunculuk kaydını karakter sesine dönüştürüp LUFS & arka plan miksiyle bağlamalı!
        user_acting_wav = sandbox / "user_mic_acting.wav"
        t_u = np.linspace(0.0, 1.75, int(48000 * 1.75), endpoint=False, dtype=np.float32)
        sf.write(
            str(user_acting_wav),
            (0.25 * np.sin(2.0 * np.pi * 240.0 * t_u)).astype(np.float32),
            48000,
            subtype="PCM_16",
        )
        prev_before_sts = ready_line.synced_preview_path
        assert prev_before_sts is not None
        with open(prev_before_sts, "rb") as _locked_sts_handle:
            sts_line = pipeline.convert_line_speech_to_speech(
                line=ready_line,
                user_acting_audio_path=user_acting_wav,
            )
            assert len(engine.sts_calls) == 1
            assert engine.sts_calls[0]["diffusion_steps"] == 25
            assert engine.sts_calls[0]["auto_f0_adjust"] is True
            assert engine.sts_calls[0]["pitch_shift"] == 0
            assert sts_line.synced_preview_path is not None
            assert sts_line.synced_preview_path != prev_before_sts
            assert "_sts_" in sts_line.synced_preview_path.name
            assert sts_line.synced_preview_path.is_file()
            assert len(sts_line.candidates) == 1
            assert sts_line.candidates[0].is_best is True

        # 2. SHA-256 yedekli uygula -> ./_temp temizlenmeli
        applied_cnt = pipeline.apply_lines_to_game(
            game_dir=game_dir,
            project_name="OrnekOyun",
            lines=[ready_line],
            cleanup_temp_after=True,
        )
        assert applied_cnt == 1
        assert compute_sha256(dialog_wav) != orig_dialog_hash
        assert compute_sha256(sfx_wav) == orig_sfx_hash
        assert list(TEMP_DIR.iterdir()) == []

        dubbed_meta = ff.probe_audio(dialog_wav, measure_lufs=False)
        assert dubbed_meta.sample_rate == 48000
        assert dubbed_meta.channels == 2

        # 3. Orijinale Dön (SHA-256)
        restore_res = backup_mgr.restore_project("OrnekOyun")
        assert restore_res.success is True
        assert restore_res.backup_deleted is True
        assert compute_sha256(dialog_wav) == orig_dialog_hash

        # 4. İptal durumunda ./_temp temizlenmeli
        cancel_ev = threading.Event()
        cancel_ev.set()
        try:
            pipeline.run(game_dir=game_dir, project_name="OrnekOyun", cancel_event=cancel_ev)
            assert False, "İptal hatası bekleniyordu"
        except PipelineCancelledError:
            pass
        assert list(TEMP_DIR.iterdir()) == []
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_config_persistence_and_v1_migration(tmp_path: Path) -> None:
    cfg_path = tmp_path / "ayarlar.json"
    cm = ConfigManager(path=cfg_path)

    saved = AppConfig(
        tts_engine=TTSEngineType.ELEVENLABS_STS.value,
        chatterbox_exaggeration=0.75,
        chatterbox_cfg_weight=0.35,
        gemini_api_key="GEM_KEY_TEST",
        elevenlabs_api_key="EL_KEY_TEST",
        elevenlabs_voice_id="voice_abc",
        fixed_reference_wav="C:/ornek/ref.wav",
        preserve_background=True,
        max_tempo_speedup=1.00,
        gemini_rpm=25,
    )
    cm.save(saved)
    loaded = cm.load()
    assert loaded.gemini_api_key == "GEM_KEY_TEST"
    assert loaded.elevenlabs_api_key == "EL_KEY_TEST"
    assert loaded.elevenlabs_voice_id == "voice_abc"
    assert loaded.get_tts_engine() is TTSEngineType.ELEVENLABS_STS
    assert abs(loaded.chatterbox_exaggeration - 0.75) < 1e-4
    assert abs(loaded.chatterbox_cfg_weight - 0.35) < 1e-4
    assert loaded.preserve_background is True
    assert abs(loaded.max_tempo_speedup - 1.00) < 1e-4
    assert loaded.gemini_rpm == 25


def test_pyside6_spacious_main_window_and_autosave(tmp_path: Path) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    bootstrap_portable_environment()

    from core.gemini_models import GeminiQuotaManager
    from PySide6.QtWidgets import QApplication
    from ui.main_window import FOOTER_SIGNATURE_TEXT, MainWindow
    from ui.theme import DARK_THEME_QSS

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(DARK_THEME_QSS)

    isolated_cfg = ConfigManager(path=tmp_path / "test_ayarlar.json")
    isolated_quota = GeminiQuotaManager(path=tmp_path / "test_kota.json")
    win = MainWindow(config_mgr=isolated_cfg, quota_mgr=isolated_quota)

    assert "SesAI Dublaj Stüdyosu" in win.windowTitle()
    assert FOOTER_SIGNATURE_TEXT == (
        "Bilal Habeş SARI 🇹🇷 tarafından Google Antigravity kullanılarak geliştirilmiştir"
    )

    # 8 ücretsiz Gemini modelinin tamamı combo ve canlı kota panelinde bulunmalı
    assert win.gemini_model_combo.count() == 8
    assert len(win._quota_pill_buttons) == 8
    assert "1120/1120" in win.quota_total_label.text()
    assert "11:00 TRT" in win.quota_total_label.text()

    # Bir modelin kotası dolduğunda rozeti kırmızı (QuotaPillExhausted) olmalı ve kalan 1100/1120'ye düşmeli
    isolated_quota.mark_model_exhausted("gemini-4-hydrogen")
    app.processEvents()
    assert "1100/1120" in win.quota_total_label.text()
    assert win._quota_pill_buttons["gemini-4-hydrogen"].objectName() == "QuotaPillExhausted"
    assert "🔴" in win._quota_pill_buttons["gemini-4-hydrogen"].text()

    # Varsayılanlar: VoxCPM2 (48 kHz, CFG 2.00) + 8 sütunlu tablo (Oyunculuk/Duygu dahil) + Aday Sayısı 1..4 + Maks Hız 1.05x
    assert win._current_tts_engine_type() is TTSEngineType.VOXCPM2
    assert win.table.columnCount() == 8
    assert abs(win.voxcpm_cfg_spin.value() - 2.00) < 1e-4
    assert win.voxcpm_steps_spin.value() == 16
    assert win.use_acting_prompt_chk.isChecked() is True
    assert win.phonetic_names_chk.isChecked() is True
    assert win.sts_auto_f0_chk.isChecked() is True
    assert win.sts_pitch_spin.value() == 0
    assert win.sts_steps_spin.value() == 25
    assert win.mic_record_btn.minimumWidth() >= 190
    assert win.upload_sts_btn.minimumWidth() >= 140
    assert win.candidate_count_spin.value() == 1
    assert win.candidate_count_spin.minimum() == 1
    assert win.candidate_count_spin.maximum() == 4
    assert "Aday: 1" in win.chip_candidates.text()
    win.candidate_count_spin.setValue(4)
    assert "Aday: 4 + Whisper" in win.chip_candidates.text()
    assert abs(win.tempo_spin.value() - 1.05) < 1e-4
    assert win.tempo_spin.minimum() == 1.00
    assert "Kapalı" in win.tempo_spin.specialValueText()
    assert win.rpm_spin.minimumWidth() >= 115
    assert win.tempo_spin.minimumWidth() >= 140

    # Çoklu aday üretilen bir satır seçildiğinde önizleme çubuğunda Aday 1 (XX%), Aday 2 (XX%) butonları kesilmeden görünmeli
    from core.models import DialogueLine, DubCandidate
    dummy_line = DialogueLine(
        line_id="L00001",
        rel_path="test.wav",
        original_path=tmp_path / "test.wav",
        is_speech=True,
        translated_text="Sözleşmeniz feshedildi.",
        selected_candidate_index=2,
        candidates=[
            DubCandidate(candidate_index=1, total_score=84.0, text_accuracy_score=80.0, duration_fit_score=96.0),
            DubCandidate(candidate_index=2, total_score=97.0, text_accuracy_score=98.0, duration_fit_score=94.0, is_best=True),
            DubCandidate(candidate_index=3, total_score=91.0, text_accuracy_score=90.0, duration_fit_score=94.0),
        ],
    )
    win._on_pipeline_discovered([dummy_line])
    win.table.selectRow(0)
    app.processEvents()
    assert win.candidate_bar_widget.isHidden() is False
    for c_btn in win._candidate_buttons:
        assert c_btn.minimumWidth() >= 120
    assert win._candidate_buttons[0].isHidden() is False
    assert "Aday 1 (84%)" in win._candidate_buttons[0].text()
    assert "★ Aday 2 (97%)" in win._candidate_buttons[1].text()
    assert "Aday 3 (91%)" in win._candidate_buttons[2].text()
    assert win._candidate_buttons[3].isHidden() is True
    assert win.mic_record_btn.isEnabled() is True
    assert win.upload_sts_btn.isEnabled() is True

    # Slider <-> SpinBox senkronizasyonu
    win.exag_slider.setValue(85)
    assert abs(win.exag_spin.value() - 0.85) < 1e-4
    win.cfg_spin.setValue(0.30)
    assert win.cfg_slider.value() == 30

    # ElevenLabs moduna geçiş stack'i 2. sayfaya almalı ve özet çipinde Aday: 1 göstermeli
    idx_el = win.tts_engine_combo.findData(TTSEngineType.ELEVENLABS_TTS.value)
    win.tts_engine_combo.setCurrentIndex(idx_el)
    assert win.engine_stack.currentIndex() == 2
    assert "Aday: 1" in win.chip_candidates.text()

    # API anahtarları ve STS ayarları girilip kapatılınca yerel config dosyasına otomatik kaydedilmeli
    win.gemini_key_input.setText("TEST_GEM_AUTO")
    win.elevenlabs_key_input.setText("TEST_EL_AUTO")
    win.tempo_spin.setValue(1.00)
    win.preserve_bg_chk.setChecked(True)
    win.sts_auto_f0_chk.setChecked(False)
    win.sts_pitch_spin.setValue(-3)
    win.sts_steps_spin.setValue(35)
    win.close()

    reloaded = isolated_cfg.load()
    assert reloaded.gemini_api_key == "TEST_GEM_AUTO"
    assert reloaded.elevenlabs_api_key == "TEST_EL_AUTO"
    assert reloaded.candidate_count == 4
    assert abs(reloaded.max_tempo_speedup - 1.00) < 1e-4
    assert reloaded.preserve_background is True
    assert reloaded.sts_auto_f0_adjust is False
    assert reloaded.sts_pitch_shift == -3
    assert reloaded.sts_diffusion_steps == 35
    assert list(TEMP_DIR.iterdir()) == []


def test_multi_candidate_whisper_judge_elevenlabs_cap_manual_switch_and_clean_ref() -> None:
    bootstrap_portable_environment()
    from core.candidate_ranker import score_candidate, select_best_candidate
    from core.models import DubCandidate
    from core.portable_env import cleanup_temp_dir

    sandbox = APP_ROOT / "tests" / "_cand_sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        ff = FFmpegManager()

        # 1. Otomatik 48 kHz Mono + Baş/Son Sessizlik Kırpmalı Referans Ses Ön Hazırlığı
        raw_stereo_ref = sandbox / "raw_stereo_with_silence.wav"
        sr_in = 24000
        lead_sil = np.zeros((int(sr_in * 1.1), 2), dtype=np.float32)
        t_v = np.linspace(0.0, 2.0, int(sr_in * 2.0), endpoint=False, dtype=np.float32)
        voice_part = np.stack([0.25 * np.sin(2.0 * np.pi * 220.0 * t_v)] * 2, axis=1)
        trail_sil = np.zeros((int(sr_in * 1.1), 2), dtype=np.float32)
        sf.write(
            str(raw_stereo_ref),
            np.concatenate([lead_sil, voice_part, trail_sil], axis=0),
            sr_in,
            subtype="PCM_16",
        )
        orig_ref_hash = compute_sha256(raw_stereo_ref)

        clean_48k_ref = ff.prepare_clean_fixed_reference(raw_stereo_ref, sample_rate=48000)
        assert clean_48k_ref.is_file()
        assert str(TEMP_DIR) in str(clean_48k_ref)
        # Orijinal dosyaya kesinlikle dokunulmamış olmalı
        assert compute_sha256(raw_stereo_ref) == orig_ref_hash
        clean_meta = ff.probe_audio(clean_48k_ref, measure_lufs=False)
        assert clean_meta.sample_rate == 48000
        assert clean_meta.channels == 1
        assert 1700 <= clean_meta.duration_ms <= 2400  # 4.2 sn'den ~2.0 sn'ye sessizlikler kırpılmış olmalı!

        # 2. Whisper Hakemi Puanlama Birim Doğrulaması (%75 Metin Doğruluğu + %25 Doğal Süre)
        target_tr = "Gözünü dört aç asker!"
        c1_corrupted = DubCandidate(
            candidate_index=1,
            whisper_transcript="Gözünü aç",  # Kelime yutmuş
            raw_duration_ms=3400,  # Aşırı uzun
        )
        c2_perfect = DubCandidate(
            candidate_index=2,
            whisper_transcript="Gözünü dört aç asker!",  # Tam isabet
            raw_duration_ms=1780,  # 1800ms orijinal süreye doğal uyum
        )
        c3_hallucinated = DubCandidate(
            candidate_index=3,
            whisper_transcript="Gözünü dört dört dört aç asker ekstra uydurma",  # Kekeleme/halüsinasyon
            raw_duration_ms=2600,
        )
        for c in (c1_corrupted, c2_perfect, c3_hallucinated):
            score_candidate(c, target_tr_text=target_tr, original_duration_ms=1800)

        best_c = select_best_candidate([c1_corrupted, c2_perfect, c3_hallucinated])
        assert best_c.candidate_index == 2
        assert c2_perfect.is_best is True
        assert c2_perfect.total_score >= 95.0
        assert c2_perfect.total_score > c1_corrupted.total_score
        assert c2_perfect.total_score > c3_hallucinated.total_score
        assert c1_corrupted.total_score < 50.0
        assert c3_hallucinated.total_score < 50.0

        # 3. VRAM Dostu Aşamalı Üretim (3 Aday) + Whisper Hakemi + Manuel Aday Değiştirme
        game_dir = sandbox / "OyunAdayTest"
        game_dir.mkdir(parents=True)
        dialog_wav = game_dir / "dialog_line.wav"
        t_d = np.linspace(0.0, 1.8, int(48000 * 1.8), endpoint=False, dtype=np.float32)
        sf.write(
            str(dialog_wav),
            (0.22 * np.sin(2.0 * np.pi * 300.0 * t_d)).astype(np.float32),
            48000,
            subtype="PCM_16",
        )

        class _MultiCandidateMockEngine(_MockSpeechEngine):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[str] = []
                self._tr_judge_idx = 0

            def unload_tts(self, log_cb=None) -> None:
                self.events.append("unload_tts")

            def unload_whisper(self, log_cb=None) -> None:
                self.events.append("unload_whisper")

            def synthesize_clone(
                self,
                text: str,
                speaker_wav_path: Path,
                output_wav_path: Path,
                language: str = "tr",
                cancel_event=None,
                options: TTSGenerationOptions | None = None,
            ) -> Path:
                c_idx = options.candidate_index if options else 1
                self.events.append(f"synth_cand_{c_idx}")
                # Aday 1: çok uzun (2.9s), Aday 2: tam doğal süre (1.76s), Aday 3: hafif kısa (1.55s)
                dur_map = {1: 2.90, 2: 1.76, 3: 1.55}
                dur_s = dur_map.get(c_idx, 1.80)
                sr = 48000
                t_c = np.linspace(0.0, dur_s, int(sr * dur_s), endpoint=False, dtype=np.float32)
                freq = 240.0 + c_idx * 45.0
                sig = (0.26 * np.sin(2.0 * np.pi * freq * t_c)).astype(np.float32)
                sf.write(str(output_wav_path), sig, sr, subtype="PCM_16")
                return output_wav_path

            def transcribe(self, audio_wav_path: Path, language="en", cancel_event=None) -> TranscriptionResult:
                if language == "tr":
                    self._tr_judge_idx = (self._tr_judge_idx % 3) + 1
                    self.events.append(f"whisper_judge_cand_{self._tr_judge_idx}")
                    if self._tr_judge_idx == 1:
                        return TranscriptionResult(text="Gözünü aç", detected_language="tr")
                    if self._tr_judge_idx == 2:
                        return TranscriptionResult(text="Gözünü dört aç asker!", detected_language="tr")
                    return TranscriptionResult(text="Gözünü dört aç!", detected_language="tr")
                return TranscriptionResult(text="Stay sharp, soldier!", detected_language="en")

        mc_engine = _MultiCandidateMockEngine()
        cfg = AppConfig(
            tts_engine=TTSEngineType.VOXCPM2.value,
            candidate_count=3,
            fixed_reference_wav=str(raw_stereo_ref),
            auto_apply_to_game=False,
        )
        translator = _MockTranslator(
            api_key="DUMMY",
            cache=TranslationCache(sandbox / "cache.db"),
            glossary=GlossaryManager(sandbox / "sozluk.json"),
        )
        backup_mgr = BackupManager(backups_root=sandbox / "_yedekler")
        pipeline = DubbingPipeline(
            config=cfg,
            speech_engine=mc_engine,
            translator=translator,
            ffmpeg=ff,
            vad=_MockVAD(ffmpeg=ff),
            backup_manager=backup_mgr,
        )

        lines, stats = pipeline.run(game_dir=game_dir, project_name="OyunAdayTest")
        assert stats.synthesized == 3
        assert stats.synced == 1

        # Önce 3 adayın tamamı TTS ile üretilmeli, ardından tek sefer unload_tts çağrılıp Whisper hakemi çalışmalı!
        assert mc_engine.events[:7] == [
            "synth_cand_1",
            "synth_cand_2",
            "synth_cand_3",
            "unload_tts",
            "whisper_judge_cand_1",
            "whisper_judge_cand_2",
            "whisper_judge_cand_3",
        ]

        ln = lines[0]
        assert len(ln.candidates) == 3
        assert ln.selected_candidate_index == 2  # En yüksek puanlı Aday 2 otomatik seçilmeli
        cand2_preview = ln.synced_preview_path
        assert cand2_preview is not None and cand2_preview.is_file()

        # Kullanıcı arayüzden manuel olarak Aday 3'e geçiş yaptığında:
        pipeline.activate_candidate(ln, 3)
        assert ln.selected_candidate_index == 3
        assert ln.synced_preview_path is not None and ln.synced_preview_path.is_file()
        assert ln.synced_preview_path != cand2_preview

        # Satırı Yeniden Dublajla da seçili aday sayısı (3) kadar üretim + Whisper hakemi çalıştırmalı
        mc_engine._tr_judge_idx = 0
        pipeline.redub_single_line(ln, "Gözünü dört aç asker!")
        assert len(ln.candidates) == 3
        assert ln.selected_candidate_index == 2

        # 4. ElevenLabs API Kredi Koruması: candidate_count=4 olsa bile otomatik 1 ile sınırlanmalı
        el_logs: list[str] = []
        pipeline.config.tts_engine = TTSEngineType.ELEVENLABS_TTS.value
        pipeline.config.candidate_count = 4
        assert pipeline._resolve_effective_candidate_count(el_logs.append) == 1
        assert any("1 ile sınırlandı" in m for m in el_logs)
    finally:
        cleanup_temp_dir(keep_root=True)
        shutil.rmtree(sandbox, ignore_errors=True)
    assert list(TEMP_DIR.iterdir()) == []


def test_whisper_transcription_cache_and_hitman_manifest_tr_subtitle_priority() -> None:
    import json
    from core.portable_env import cleanup_temp_dir

    bootstrap_portable_environment()
    sandbox = APP_ROOT / "tests" / "_hitman_cache_sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        hitman_root = sandbox / "Hitman_SerbestEv_Test"
        sesler_dir = hitman_root / "sesler"
        sesler_dir.mkdir(parents=True, exist_ok=True)

        wav1 = sesler_dir / "01_ed002_onboarding_diana_001_0063350CB7FDD8A4.wav"
        wav2 = sesler_dir / "02_dialog_guard_005B09B5A328F67F.wav"

        t = np.linspace(0.0, 1.8, int(48000 * 1.8), endpoint=False, dtype=np.float32)
        sig = (0.22 * np.sin(2.0 * np.pi * 280.0 * t)).astype(np.float32)
        sf.write(str(wav1), sig, 48000, subtype="PCM_16")
        sf.write(str(wav2), sig, 48000, subtype="PCM_16")

        # Üst dizine hitman_manifest.json yaz (1. dosya için hazır Türkçe altyazı mevcut)
        manifest_payload = {
            "schema_version": 1,
            "files": [
                {
                    "index": 1,
                    "hash_hex": "0063350CB7FDD8A4",
                    "wav_filename": wav1.name,
                    "friendly_name": "ed002_onboarding_diana_001",
                    "dialogue_subtitle_en": "Welcome to your safehouse, 47.",
                    "dialogue_subtitle_tr_mod": "Güvenli evine hoş geldin, 47!",
                },
                {
                    "index": 2,
                    "hash_hex": "005B09B5A328F67F",
                    "wav_filename": wav2.name,
                    "friendly_name": "dialog_guard",
                    "dialogue_subtitle_en": "",
                    "dialogue_subtitle_tr_mod": "",
                },
            ],
        }
        (hitman_root / "hitman_manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        class _CountingEngine(_MockSpeechEngine):
            def __init__(self) -> None:
                super().__init__()
                self.transcribe_calls = 0

            def transcribe(self, audio_wav_path: Path, language="en", cancel_event=None) -> TranscriptionResult:
                self.transcribe_calls += 1
                return TranscriptionResult(text="Stay sharp, 47!", detected_language="en")

        class _CountingTranslator(_MockTranslator):
            def __init__(self, **kwargs) -> None:
                super().__init__(**kwargs)
                self.translated_item_ids: list[str] = []

            def translate_batch(self, items, use_cache=True, cancel_event=None, log_cb=None, progress_cb=None):
                self.translated_item_ids.extend(it.item_id for it in items)
                return {
                    it.item_id: TranslationResult(
                        translated_text="Tetikte kal 47!",
                        from_cache=False,
                        matched_glossary_count=0,
                        acting_prompt="threatening, sinister dark tone, sharp dramatic emphasis",
                    )
                    for it in items
                }

        engine = _CountingEngine()
        cache = TranslationCache(sandbox / "ceviri_ve_desifre.db")
        translator = _CountingTranslator(
            api_key="DUMMY",
            cache=cache,
            glossary=GlossaryManager(sandbox / "sozluk.json"),
        )
        ff = FFmpegManager()
        pipeline = DubbingPipeline(
            config=AppConfig(auto_apply_to_game=False),
            speech_engine=engine,
            translator=translator,
            ffmpeg=ff,
            vad=_MockVAD(ffmpeg=ff),
            backup_manager=BackupManager(backups_root=sandbox / "_yedekler"),
        )

        # 1. Çalıştırma:
        # - 1. dosyada hitman_manifest.json içinde dialogue_subtitle_en dolu olduğu için Whisper ATLANMALI!
        # - Yalnızca 2. dosya (dialogue_subtitle_en boş) Whisper ile deşifre edilmeli (transcribe_calls == 1)!
        # - 1. dosya ayrıca hitman_manifest.json içindeki hazır Türkçe altyazıyı kullanmalı (Gemini'ye gitmemeli!)
        logs1: list[str] = []
        lines1, stats1 = pipeline.run(
            game_dir=sesler_dir,
            project_name="HitmanSerbestEvTest",
            log_cb=logs1.append,
        )
        assert engine.transcribe_calls == 1
        assert cache.transcription_count() == 2
        assert len(translator.translated_item_ids) == 1  # Yalnızca 2. dosya çevirmene gönderildi!
        l1 = next(x for x in lines1 if x.rel_path == wav1.name)
        l2 = next(x for x in lines1 if x.rel_path == wav2.name)
        assert l1.source_text == "Welcome to your safehouse, 47."
        assert l1.translated_text == "Güvenli evine hoş geldin, 47!"
        assert l1.acting_prompt
        assert l2.source_text == "Stay sharp, 47!"
        assert l2.translated_text == "Tetikte kal 47!"
        assert any("Hitman Orijinal İngilizce Altyazı" in m for m in logs1)
        assert stats1.synced == 2

        # 2. Çalıştırma ("Dublajı Başlat" tekrar tıklandığında):
        # Daha önce deşifre edilmiş sesler Whisper'dan tekrar geçmeden (transcribe_calls değişmeden) önbellekten yüklenmeli!
        logs2: list[str] = []
        lines2, stats2 = pipeline.run(
            game_dir=sesler_dir,
            project_name="HitmanSerbestEvTest",
            log_cb=logs2.append,
        )
        assert engine.transcribe_calls == 1  # 0 yeni Whisper çağrısı!
        assert any("Whisper Deşifre Önbelleği" in m for m in logs2)
        assert stats2.synced == 2
    finally:
        cleanup_temp_dir(keep_root=True)
        shutil.rmtree(sandbox, ignore_errors=True)
    assert list(TEMP_DIR.iterdir()) == []


def test_hitman_manifest_en_only_skips_whisper_and_populates_ui_table(tmp_path: Path) -> None:
    import json
    from core.gemini_models import GeminiQuotaManager
    from core.portable_env import cleanup_temp_dir
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    bootstrap_portable_environment()

    sandbox = APP_ROOT / "tests" / "_hitman_en_only_sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)

    try:
        hitman_root = sandbox / "Hitman_EN_Test"
        sesler_dir = hitman_root / "sesler"
        sesler_dir.mkdir(parents=True, exist_ok=True)

        wav_diana = sesler_dir / "01_ed002_onboarding_diana_001_0063350CB7FDD8A4.wav"
        t = np.linspace(0.0, 1.8, int(48000 * 1.8), endpoint=False, dtype=np.float32)
        sig = (0.22 * np.sin(2.0 * np.pi * 260.0 * t)).astype(np.float32)
        sf.write(str(wav_diana), sig, 48000, subtype="PCM_16")

        exact_en_text = (
            "Welcome to your safehouse, 47. Everything is at stake. Good luck, 47."
        )
        manifest_payload = {
            "schema_version": 1,
            "files": [
                {
                    "index": 1,
                    "hash_hex": "0063350CB7FDD8A4",
                    "wav_filename": wav_diana.name,
                    "friendly_name": "ed002_onboarding_diana_001",
                    "dialogue_subtitle_en": exact_en_text,
                    "dialogue_subtitle_tr_mod": "",
                }
            ],
        }
        (hitman_root / "hitman_manifest.json").write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        class _NeverWhisperEngine(_MockSpeechEngine):
            def __init__(self) -> None:
                super().__init__()
                self.transcribe_calls = 0

            def transcribe(self, audio_wav_path: Path, language="en", cancel_event=None) -> TranscriptionResult:
                self.transcribe_calls += 1
                raise AssertionError(
                    "İngilizce altyazısı dolu olan Hitman dosyası için Whisper çağrılmamalıydı!"
                )

        engine = _NeverWhisperEngine()
        translator = _MockTranslator(
            api_key="DUMMY",
            cache=TranslationCache(sandbox / "cache.db"),
            glossary=GlossaryManager(sandbox / "sozluk.json"),
        )
        ff = FFmpegManager()
        pipeline = DubbingPipeline(
            config=AppConfig(auto_apply_to_game=False),
            speech_engine=engine,
            translator=translator,
            ffmpeg=ff,
            vad=_MockVAD(ffmpeg=ff),
            backup_manager=BackupManager(backups_root=sandbox / "_yedekler"),
        )

        lines, stats = pipeline.run(game_dir=sesler_dir, project_name="HitmanEnOnlyTest")
        assert engine.transcribe_calls == 0
        assert len(lines) == 1
        assert lines[0].source_text == exact_en_text
        assert stats.transcribed == 1
        assert stats.synced == 1

        # Arayüz tablosunun Orijinal Metin sütununa birebir imlası kusursuz oyun metni yazılmalı
        app = QApplication.instance() or QApplication([])
        win = MainWindow(
            config_mgr=ConfigManager(path=tmp_path / "en_ui_ayarlar.json"),
            quota_mgr=GeminiQuotaManager(path=tmp_path / "en_ui_kota.json"),
        )
        win._on_pipeline_discovered(lines)
        app.processEvents()
        orig_cell = win.table.item(0, win.COL_ORIG)
        assert orig_cell is not None
        assert orig_cell.text() == exact_en_text
        win.close()
    finally:
        cleanup_temp_dir(keep_root=True)
        shutil.rmtree(sandbox, ignore_errors=True)
    assert list(TEMP_DIR.iterdir()) == []
