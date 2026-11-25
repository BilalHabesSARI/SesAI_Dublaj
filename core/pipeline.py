"""Sinematik Oyun Dublajı İş Hattı (Pipeline) Orkestratörü.

Desteklenen Yetenekler:
- Seçilebilir ses motorları: Chatterbox Multilingual (Resemble AI), Coqui XTTS v2,
  ElevenLabs API (Multilingual v2 TTS + Speech-to-Speech / Voice Changer).
- Karakter Referans Profili: İsteğe bağlı 10–15 sn sabit referans WAV (boşsa her
  satırın kendi orijinal sesi referans olarak kullanılır).
- Arka Plan Koruma (Vokal Ayırma): İsteğe bağlı konuşma / müzik-efekt ayrıştırma
  ve dublaj sonrası orijinal arka plan katmanıyla kırpmasız (clipping'siz) yeniden miksleme.
- Akıllı sessizlik kırpma + doğal tempo (varsayılan 1.05x, 1.00x = hızlandırma kapalı).
- SHA-256 doğrulamalı yedekleme ve `try/finally` ile garantili `./_temp` temizliği.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from adapters import AdapterRegistry, create_default_registry
from core.backup_manager import BackupManager, sanitize_project_name
from core.candidate_ranker import score_candidate, select_best_candidate
from core.cinematic_timing import (
    EDGE_FADE_MS,
    MIN_CINEMATIC_MONOLOGUE_SEC,
    MIN_PAUSE_SEC,
    assemble_timed_blocks,
    detect_speech_blocks,
    split_turkish_text_to_blocks,
)
from core.config import AppConfig
from core.ffmpeg_manager import FFmpegManager
from core.models import (
    DialogueLine,
    DiscoveredAudio,
    DubCandidate,
    LineStatus,
    PipelineStats,
    TTSEngineType,
)
from core.portable_env import TEMP_DIR, cleanup_temp_dir, make_temp_subdir
from core.translator import (
    BatchLineInput,
    GeminiTranslator,
    clean_acting_prompt,
    infer_fallback_acting_prompt,
)
from core.tts_pronunciation import TTSPronunciationManager
from core.vad_filter import SileroVADFilter
from core.vocal_separator import VocalSeparator
from engines.base import SpeechEngine


class PipelineCancelledError(InterruptedError):
    """Kullanıcı Durdur butonuna bastığında fırlatılır."""


class DubbingPipeline:
    """Tüm dublaj aşamalarını yöneten, iptal edilebilir ve güvenli sinematik iş hattı."""

    def __init__(
        self,
        config: AppConfig,
        speech_engine: SpeechEngine,
        translator: GeminiTranslator,
        ffmpeg: Optional[FFmpegManager] = None,
        vad: Optional[SileroVADFilter] = None,
        backup_manager: Optional[BackupManager] = None,
        registry: Optional[AdapterRegistry] = None,
        vocal_separator: Optional[VocalSeparator] = None,
        pronunciation_manager: Optional[TTSPronunciationManager] = None,
    ) -> None:
        self.config = config
        self.engine = speech_engine
        self.translator = translator
        self.ffmpeg = ffmpeg or FFmpegManager()
        self.vad = vad or SileroVADFilter(
            threshold=config.vad_threshold,
            min_speech_ms=config.vad_min_speech_ms,
            ffmpeg=self.ffmpeg,
        )
        self.backup_manager = backup_manager or BackupManager()
        self.registry = registry or create_default_registry()
        self.vocal_separator = vocal_separator or VocalSeparator(ffmpeg=self.ffmpeg)
        self.pronunciation = pronunciation_manager or TTSPronunciationManager()

    @staticmethod
    def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
        if cancel_event and cancel_event.is_set():
            raise PipelineCancelledError("İşlem kullanıcı tarafından durduruldu.")

    @staticmethod
    def _load_hitman_manifest_subtitles(
        game_root: Path,
        log: Optional[Callable[[str], None]] = None,
    ) -> dict[str, dict[str, str]]:
        """Seçili oyun klasöründe veya bir üst dizininde `hitman_manifest.json` varsa
        hazır İngilizce/Türkçe altyazı eşleşmelerini yükler.
        """
        candidates = (
            game_root / "hitman_manifest.json",
            game_root.parent / "hitman_manifest.json",
        )
        manifest_path: Optional[Path] = None
        for cand in candidates:
            if cand.is_file():
                manifest_path = cand
                break
        if manifest_path is None:
            return {}

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}

        files_list = raw.get("files")
        if not isinstance(files_list, list):
            return {}

        lookup: dict[str, dict[str, str]] = {}
        en_ready_count = 0
        tr_ready_count = 0
        acting_ready_count = 0
        for item in files_list:
            if not isinstance(item, dict):
                continue
            en_sub = str(
                item.get("dialogue_subtitle_en")
                or item.get("english_subtitle")
                or item.get("subtitle_en")
                or item.get("source_text")
                or ""
            ).strip()
            tr_sub = str(
                item.get("dialogue_subtitle_tr_mod")
                or item.get("dialogue_subtitle_tr")
                or item.get("turkish_subtitle")
                or ""
            ).strip()
            act_sub = str(
                item.get("acting_prompt")
                or item.get("dialogue_acting_prompt")
                or ""
            ).strip().strip("()")
            if not en_sub and not tr_sub and not act_sub:
                continue
            if en_sub:
                en_ready_count += 1
            if tr_sub:
                tr_ready_count += 1
            if act_sub:
                acting_ready_count += 1
            entry_data = {"en": en_sub, "tr": tr_sub, "acting_prompt": act_sub}

            for key_raw in (
                item.get("wav_filename"),
                Path(str(item.get("wav_rel_path") or "")).name,
                item.get("hash_hex"),
                item.get("friendly_name"),
            ):
                k = str(key_raw or "").strip().lower()
                if k:
                    lookup[k] = entry_data
                    stem_k = Path(k).stem.lower()
                    if stem_k:
                        lookup[stem_k] = entry_data

        if lookup and log:
            act_info = (
                f", {acting_ready_count} adet özel oyunculuk/duygu etiketi"
                if acting_ready_count > 0
                else ""
            )
            log(
                f"🎯 Hitman manifest algılandı ({manifest_path.name}): "
                f"{en_ready_count} adet orijinal İngilizce altyazı (Whisper atlanacak), "
                f"{tr_ready_count} adet hazır Türkçe altyazı metni{act_info} öncelikli kullanılacak."
            )
        return lookup

    @staticmethod
    def _find_hitman_manifest_entry(
        line: DialogueLine,
        hitman_map: dict[str, dict[str, str]],
    ) -> Optional[dict[str, str]]:
        if not hitman_map:
            return None
        rel_p = Path(line.rel_path)
        candidates = [
            rel_p.name.lower(),
            rel_p.stem.lower(),
        ]
        m_hash = re.search(r"([0-9a-f]{16})$", rel_p.stem, flags=re.IGNORECASE)
        if m_hash:
            candidates.append(m_hash.group(1).lower())
        for c in candidates:
            if c in hitman_map:
                return hitman_map[c]
        return None

    def _prepare_fixed_reference_if_set(
        self,
        work_dir: Path,
        log: Callable[[str], None],
        sample_rate: int = 48000,
    ) -> Optional[Path]:
        raw_ref = (self.config.fixed_reference_wav or "").strip()
        if not raw_ref:
            return None
        src_ref = Path(raw_ref)
        if not src_ref.is_file():
            raise FileNotFoundError(
                f"Seçilen Sabit Referans Ses (.wav) bulunamadı: {src_ref}"
            )
        prepared = work_dir / "fixed_character_ref.wav"
        self.ffmpeg.prepare_clean_fixed_reference(
            src_path=src_ref,
            dst_wav=prepared,
            sample_rate=sample_rate,
            max_duration_sec=20.0,
        )
        ref_meta = self.ffmpeg.probe_audio(prepared, measure_lufs=False)
        log(
            f"Sabit Karakter Referans Profili aktif: {src_ref.name} "
            f"({ref_meta.duration_sec:.2f} sn, {sample_rate} Hz Mono, baş/son sessizlik kırpıldı)"
        )
        return prepared

    def _resolve_effective_candidate_count(self, log: Callable[[str], None]) -> int:
        """Aktif motora göre üretilecek aday sayısını (1..4) belirler; ElevenLabs'ta 1'e sabitler."""
        requested = max(1, min(4, int(getattr(self.config, "candidate_count", 1) or 1)))
        tts_eng = self.config.get_tts_engine()
        if tts_eng.is_elevenlabs:
            if requested > 1:
                log(
                    "Bilgi: ElevenLabs API motorunda boşuna kredi yakmamak için "
                    f"aday sayısı otomatik olarak 1 ile sınırlandı (istenen: {requested})."
                )
            return 1
        return requested

    def _measure_trimmed_candidate_duration_ms(
        self, raw_wav: Path, work_dir: Path, skip_trim: bool = False
    ) -> int:
        """Aday sesin sessizlik kırpma sonrası doğal konuşma süresini (ms) ölçer."""
        if skip_trim or not self.config.trim_silence:
            return self.ffmpeg.probe_audio(raw_wav, measure_lufs=False).duration_ms
        tmp_trim = work_dir / f"_dur_probe_{uuid.uuid4().hex[:8]}.wav"
        try:
            trimmed_wav, _ = self.ffmpeg.trim_silence_safely(raw_wav, tmp_trim)
            return self.ffmpeg.probe_audio(trimmed_wav, measure_lufs=False).duration_ms
        except Exception:
            return self.ffmpeg.probe_audio(raw_wav, measure_lufs=False).duration_ms
        finally:
            tmp_trim.unlink(missing_ok=True)

    def _synthesize_candidate_wav(
        self,
        line: DialogueLine,
        text: str,
        speaker_wav_path: Path,
        output_wav_path: Path,
        tts_options,
        work_dir: Path,
        cancel_event: Optional[threading.Event],
        log: Callable[[str], None],
    ) -> bool:
        """TTS telaffuz sözlüğünü yalnızca ses motoruna giden metne uygular ve
        isteğe bağlı 'Sinematik Zamanlama Kilidi' (>= 15 sn uzun monolog) açıksa
        orijinal esleri (>= 0.65 sn) 15 ms yumuşak geçişle koruyarak sentezler.

        Returns:
            Sinematik zamanlama kilidinin bu satırda gerçekten uygulanıp uygulanmadığı.
        """
        is_sts_mode = tts_options.tts_engine is TTSEngineType.ELEVENLABS_STS

        # Yalnızca >= 15 sn uzun monologlarda ve onay kutusu açıkken blok bazlı sentez yap
        if (
            self.config.cinematic_timing_lock
            and not is_sts_mode
            and line.metadata is not None
            and line.metadata.duration_sec >= MIN_CINEMATIC_MONOLOGUE_SEC
            and line.original_path.is_file()
            and (text or "").strip()
        ):
            blocks, orig_total_ms = detect_speech_blocks(
                line.original_path,
                min_pause_sec=MIN_PAUSE_SEC,
                ffmpeg=self.ffmpeg,
                work_dir=work_dir,
            )
            if len(blocks) >= 2:
                matched_blocks, text_chunks = split_turkish_text_to_blocks(text, blocks)
                if len(matched_blocks) >= 2:
                    block_wavs: list[tuple] = []
                    tmp_chunk_paths: list[Path] = []
                    try:
                        for b_idx, (blk, chunk_txt) in enumerate(
                            zip(matched_blocks, text_chunks), start=1
                        ):
                            self._check_cancel(cancel_event)
                            spoken_chunk = self.pronunciation.apply(chunk_txt)
                            chunk_wav = (
                                work_dir
                                / f"{line.line_id}_cin_a{tts_options.candidate_index}_b{b_idx}_{uuid.uuid4().hex[:5]}.wav"
                            )
                            tmp_chunk_paths.append(chunk_wav)
                            self.engine.synthesize_clone(
                                text=spoken_chunk,
                                speaker_wav_path=speaker_wav_path,
                                output_wav_path=chunk_wav,
                                language=self.config.target_language,
                                cancel_event=cancel_event,
                                options=tts_options,
                            )
                            block_wavs.append((blk, chunk_wav))

                        assemble_timed_blocks(
                            block_wavs=block_wavs,
                            total_orig_duration_ms=orig_total_ms,
                            output_wav_path=output_wav_path,
                            fade_ms=EDGE_FADE_MS,
                        )
                        setattr(line, "_cinematic_locked", True)
                        if tts_options.candidate_index == 1:
                            log(
                                f"🎬 Sinematik Zamanlama Kilidi [{line.rel_path}]: "
                                f"{len(matched_blocks)} konuşma bloğu orijinal esleriyle (>= {MIN_PAUSE_SEC:.2f} sn) "
                                f"ve {int(EDGE_FADE_MS)} ms yumuşak geçişle birleştirildi."
                            )
                        return True
                    finally:
                        for tp in tmp_chunk_paths:
                            tp.unlink(missing_ok=True)

        # 15 saniyeden kısa tüm dosyalarda (veya kilit kapalıyken) mevcut standart akış %100 aynı kalır
        spoken_text = text if is_sts_mode else self.pronunciation.apply(text)
        if not is_sts_mode and spoken_text != text and tts_options.candidate_index == 1:
            log(f"🗣️ TTS Telaffuz uygulandı [{line.rel_path}] (tablo metni korundu).")
        self.engine.synthesize_clone(
            text=spoken_text,
            speaker_wav_path=speaker_wav_path,
            output_wav_path=output_wav_path,
            language=self.config.target_language,
            cancel_event=cancel_event,
            options=tts_options,
        )
        setattr(line, "_cinematic_locked", False)
        return False

    def _score_candidates_for_line(
        self,
        line: DialogueLine,
        work_dir: Path,
        cancel_event: Optional[threading.Event],
        log: Callable[[str], None],
    ) -> DubCandidate:
        """Satırın adaylarını faster-whisper Türkçe deşifresi (%75) + doğal süre (%25) ile puanlayıp en iyisini seçer."""
        orig_dur_ms = line.metadata.duration_ms if line.metadata else 2000
        if len(line.candidates) == 1:
            only = line.candidates[0]
            only.whisper_transcript = line.translated_text
            score_candidate(
                only,
                target_tr_text=line.translated_text,
                original_duration_ms=orig_dur_ms,
                max_tempo_speedup=self.config.max_tempo_speedup,
            )
            only.is_best = True
            line.selected_candidate_index = only.candidate_index
            line.best_candidate_score = only.total_score
            return only

        for cand in line.candidates:
            self._check_cancel(cancel_event)
            assert cand.raw_wav_path is not None and cand.raw_wav_path.is_file()
            cand_16k = work_dir / f"{line.line_id}_cand{cand.candidate_index}_whisper16k.wav"
            try:
                self.ffmpeg.convert_to_wav(
                    cand.raw_wav_path, cand_16k, sample_rate=16000, channels=1
                )
                tr_res = self.engine.transcribe(
                    cand_16k,
                    language=self.config.target_language or "tr",
                    cancel_event=cancel_event,
                )
                cand.whisper_transcript = (tr_res.text or "").strip()
            except PipelineCancelledError:
                raise
            except InterruptedError as exc:
                raise PipelineCancelledError(str(exc)) from exc
            except Exception as exc:
                log(
                    f"Uyarı: [{line.rel_path}] Aday {cand.candidate_index} Whisper hakem deşifresi başarısız ({exc})."
                )
                cand.whisper_transcript = ""
            finally:
                cand_16k.unlink(missing_ok=True)

            score_candidate(
                cand,
                target_tr_text=line.translated_text,
                original_duration_ms=orig_dur_ms,
                max_tempo_speedup=self.config.max_tempo_speedup,
            )

        best = select_best_candidate(line.candidates)
        line.selected_candidate_index = best.candidate_index
        line.best_candidate_score = best.total_score
        summary = ", ".join(
            f"Aday {c.candidate_index}: %{c.score_pct_int} "
            f"(Metin:%{int(round(c.text_accuracy_score))}, Süre:%{int(round(c.duration_fit_score))})"
            for c in line.candidates
        )
        log(
            f"🎯 Whisper Hakemi [{line.rel_path}] -> En İyi: Aday {best.candidate_index} "
            f"(%{best.score_pct_int}) | [{summary}]"
        )
        return best

    def activate_candidate(
        self,
        line: DialogueLine,
        candidate_index: int,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> DialogueLine:
        """Kullanıcı arayüzden başka bir adaya tıkladığında o adayı aktif önizleme seçimi yapar."""
        cand = line.get_candidate(candidate_index)
        if cand is None:
            raise ValueError(f"Aday {candidate_index} bulunamadı.")

        if line.metadata is None:
            line.metadata = self.ffmpeg.probe_audio(line.original_path, measure_lufs=True)

        is_sts_mode = self.config.get_tts_engine() is TTSEngineType.ELEVENLABS_STS

        # Aday daha önce senkronize edilmemişse veya dosyası yoksa benzersiz yeni dosya adına senkronize et
        if cand.synced_wav_path is None or not cand.synced_wav_path.is_file():
            if cand.raw_wav_path is None or not cand.raw_wav_path.is_file():
                raise FileNotFoundError(
                    f"Aday {candidate_index} ham ses dosyası ./_temp/ içinde bulunamadı."
                )
            rel_p = Path(line.rel_path)
            if line.synced_preview_path is not None and line.synced_preview_path.parent.exists():
                base_dir = line.synced_preview_path.parent
            else:
                base_dir = make_temp_subdir("onizleme") / rel_p.parent
            base_dir.mkdir(parents=True, exist_ok=True)
            unique_tag = uuid.uuid4().hex[:8]
            out_path = base_dir / f"{rel_p.stem}_aday{cand.candidate_index}_{unique_tag}{rel_p.suffix}"

            synced_path, dub_ms, tempo = self.ffmpeg.sync_and_package_dub(
                dub_wav_path=cand.raw_wav_path,
                original_meta=line.metadata,
                output_path=out_path,
                max_tempo_speedup=self.config.max_tempo_speedup,
                trim_silence=(
                    self.config.trim_silence
                    and not is_sts_mode
                    and not bool(getattr(line, "_cinematic_locked", False))
                ),
                background_wav_path=getattr(line, "_bg_wav_path", None),
                bg_volume_db=self.config.bg_mix_volume_db,
                target_vocal_lufs=getattr(line, "_vocal_lufs", None),
            )
            cand.synced_wav_path = synced_path
            cand.synced_duration_ms = dub_ms
            cand.tempo_factor = tempo

        line.selected_candidate_index = cand.candidate_index
        line.synced_preview_path = cand.synced_wav_path
        line.dub_duration_ms = cand.synced_duration_ms
        line.tempo_factor = cand.tempo_factor
        line.status = LineStatus.READY_PREVIEW
        if log_cb:
            log_cb(
                f"Manuel aday seçildi [{line.rel_path}]: Aday {cand.candidate_index} "
                f"(%{cand.score_pct_int} | {cand.synced_duration_ms}ms, tempo {cand.tempo_factor:.2f}x)"
            )
        return line

    def _separate_if_enabled(
        self,
        line: DialogueLine,
        work_dir: Path,
        cancel_event: Optional[threading.Event],
        log: Callable[[str], None],
    ) -> tuple[Path, Optional[Path], Optional[float]]:
        """Arka plan koruma açıksa vokali ve arka planı ayırır.

        Returns:
            (vokal_kaynak_yolu, arka_plan_wav_veya_None, vokal_lufs_veya_None)
        """
        if not self.config.preserve_background:
            return line.original_path, None, None

        stems_dir = work_dir / "stems"

        def _cloud_sep(src: Path, v_out: Path, b_out: Path) -> bool:
            return self.engine.separate_vocals(src, v_out, b_out, cancel_event=cancel_event)

        sep = self.vocal_separator.separate(
            src_audio=line.original_path,
            out_dir=stems_dir,
            stem_prefix=line.line_id,
            cloud_separator_fn=_cloud_sep,
            prefer_local_demucs=True,
            cancel_event=cancel_event,
            log_cb=log,
        )
        line.background_preserved = True
        vocal_lufs, vocal_rms = self.ffmpeg.measure_active_speech_lufs(sep.vocals_wav)
        log(
            f"Vokal ayrıldı [{line.rel_path}] ({sep.backend_used}, "
            f"aktif konuşma şiddeti: {vocal_lufs:.1f} LUFS / {vocal_rms:.1f} dBFS RMS)"
        )
        return sep.vocals_wav, sep.background_wav, vocal_lufs

    def run(
        self,
        game_dir: Path,
        project_name: str,
        cancel_event: Optional[threading.Event] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        line_cb: Optional[Callable[[DialogueLine], None]] = None,
        discovered_cb: Optional[Callable[[list[DialogueLine]], None]] = None,
    ) -> tuple[list[DialogueLine], PipelineStats]:
        """Seçilen oyun klasöründe sinematik dublaj iş hattını çalıştırır."""

        def log(msg: str) -> None:
            if log_cb:
                log_cb(msg)

        def report(step_cur: int, step_total: int, stage_text: str) -> None:
            if progress_cb:
                progress_cb(step_cur, step_total, stage_text)

        def emit_line(line: DialogueLine) -> None:
            if line_cb:
                line_cb(line)

        if not self.ffmpeg.is_available():
            raise RuntimeError(
                "FFmpeg bulunamadı! Lütfen ./_araclar/ffmpeg/ klasörüne ffmpeg.exe koyun."
            )

        game_root = game_dir.resolve()
        if not game_root.is_dir():
            raise FileNotFoundError(f"Seçilen klasör mevcut değil: {game_root}")

        safe_project = sanitize_project_name(project_name or game_root.name)
        tts_options = self.config.build_tts_options()
        is_sts_mode = tts_options.tts_engine is TTSEngineType.ELEVENLABS_STS

        stats = PipelineStats()
        lines: list[DialogueLine] = []
        discovered_map: dict[str, DiscoveredAudio] = {}
        # Her satır için varsa ayrılmış (vocals, background, vocal_lufs) önbelleği
        line_stems: dict[str, tuple[Path, Optional[Path], Optional[float]]] = {}

        work_dir = make_temp_subdir("calisma")
        preview_dir = make_temp_subdir("onizleme")
        had_fatal_or_cancel = False

        try:
            # ==================================================================
            # ADIM 1: Özyinelemeli Tarama (Adaptör Katmanı)
            # ==================================================================
            self._check_cancel(cancel_event)
            report(0, 100, "Adım 1/6: Ses dosyaları taranıyor...")
            tempo_label = (
                "Kapalı (1.00x Doğal)"
                if self.config.max_tempo_speedup <= 1.002
                else f"Maks {self.config.max_tempo_speedup:.2f}x"
            )
            log(
                f"Tarama başlatıldı: {game_root} | Altyapı: {self.engine.mode.label_tr} | "
                f"Ses Motoru: {tts_options.tts_engine.label_tr} | Tempo: {tempo_label} | "
                f"Arka Plan Koruma: {'Açık' if self.config.preserve_background else 'Kapalı'}"
            )

            ref_sample_rate = 48000 if tts_options.tts_engine.is_voxcpm else 24000
            fixed_ref_wav = self._prepare_fixed_reference_if_set(
                work_dir, log, sample_rate=ref_sample_rate
            )

            discovered = self.registry.discover_all(game_root)
            stats.total_scanned = len(discovered)
            log(f"Toplam {len(discovered)} ses dosyası bulundu (.wav, .ogg, .mp3, .flac).")

            if not discovered:
                report(100, 100, "Desteklenen ses dosyası bulunamadı.")
                return [], stats

            for idx, item in enumerate(discovered, start=1):
                line_id = f"L{idx:05d}"
                discovered_map[line_id] = item
                lines.append(
                    DialogueLine(
                        line_id=line_id,
                        rel_path=item.rel_path,
                        original_path=item.original_path,
                        adapter_name=item.adapter_name,
                        status=LineStatus.PENDING,
                    )
                )

            if discovered_cb:
                discovered_cb(lines)

            # ==================================================================
            # ADIM 2: (Opsiyonel Vokal Ayırma) + Yerel CPU Silero VAD Filtresi
            # ==================================================================
            speech_lines: list[DialogueLine] = []
            total_files = len(lines)
            for idx, line in enumerate(lines, start=1):
                self._check_cancel(cancel_event)
                try:
                    vocal_src = line.original_path
                    if self.config.preserve_background:
                        line.status = LineStatus.SEPARATING_VOCALS
                        emit_line(line)
                        report(
                            int((idx / total_files) * 18),
                            100,
                            f"Adım 2/6 (Vokal Ayırma): {idx}/{total_files} — {line.rel_path}",
                        )
                        vocal_src, bg_wav, voc_lufs = self._separate_if_enabled(
                            line, work_dir, cancel_event, log
                        )
                        line_stems[line.line_id] = (vocal_src, bg_wav, voc_lufs)
                    else:
                        line_stems[line.line_id] = (line.original_path, None, None)

                    line.status = LineStatus.VAD_CHECKING
                    emit_line(line)
                    report(
                        int((idx / total_files) * 22),
                        100,
                        f"Adım 2/6 (VAD): {idx}/{total_files} — {line.rel_path}",
                    )
                    vad_res = self.vad.analyze_file(vocal_src, work_dir=work_dir)
                    line.vad_speech_ratio = vad_res.speech_ratio
                    line.vad_max_prob = vad_res.max_speech_prob
                    line.is_speech = vad_res.is_speech

                    if vad_res.is_speech:
                        stats.speech_detected += 1
                        speech_lines.append(line)
                    else:
                        stats.skipped_non_speech += 1
                        line.status = LineStatus.SKIPPED_NON_SPEECH
                        emit_line(line)
                except PipelineCancelledError:
                    raise
                except InterruptedError as exc:
                    raise PipelineCancelledError(str(exc)) from exc
                except Exception as exc:
                    stats.errors += 1
                    line.status = LineStatus.ERROR
                    line.error_message = f"VAD/Vokal hatası: {exc}"
                    emit_line(line)
                    log(f"HATA (VAD/Vokal) [{line.rel_path}]: {exc}")

            # Vokal ayırma için yerel Demucs yüklendiyse VRAM'i hemen boşalt
            self.vocal_separator.unload()

            log(
                f"VAD tamamlandı ({self.vad._backend_name}): "
                f"{stats.speech_detected} konuşma dosyası seçildi, "
                f"{stats.skipped_non_speech} konuşmasız dosya (müzik/SFX) pas geçildi."
            )

            if not speech_lines:
                report(100, 100, "İnsan konuşması içeren dosya bulunamadı.")
                return lines, stats

            # ==================================================================
            # ADIM 3: Teknik Ölçüm & Deşifre (Whisper Önbellekli + Hitman Manifest Destekli)
            # ==================================================================
            hitman_subtitles_map = self._load_hitman_manifest_subtitles(game_root, log=log)
            transcribed_lines: list[DialogueLine] = []
            total_sp = len(speech_lines)
            whisper_cache_hits = 0
            for idx, line in enumerate(speech_lines, start=1):
                self._check_cancel(cancel_event)
                line.status = LineStatus.TRANSCRIBING
                emit_line(line)
                report(
                    22 + int((idx / total_sp) * 23),
                    100,
                    f"Adım 3/6 (Ölçüm & Deşifre): {idx}/{total_sp} — {line.rel_path}",
                )
                vocal_src, _, _ = line_stems[line.line_id]
                whisper_wav = work_dir / f"{line.line_id}_whisper.wav"
                try:
                    line.metadata = self.ffmpeg.probe_audio(line.original_path, measure_lufs=True)
                    if is_sts_mode:
                        # ElevenLabs Speech-to-Speech orijinal ses performansını doğrudan hedef sese dönüştürür
                        line.source_text = "[Speech-to-Speech: Orijinal Duygu & Performans Aktarımı]"
                        line.translated_text = "[Speech-to-Speech / Voice Changer]"
                        stats.transcribed += 1
                        transcribed_lines.append(line)
                        emit_line(line)
                        log(
                            f"STS Hazır [{line.rel_path}] ({line.metadata.duration_sec:.2f}s, "
                            f"{line.metadata.lufs:.1f} LUFS)"
                        )
                        continue

                    # 3a) Hitman manifest içinde orijinal İngilizce oyun altyazısı (DLGE) varsa
                    #     Whisper tahminini tamamen atla ve imlası kusursuz oyun metnini doğrudan kullan!
                    hm_entry = self._find_hitman_manifest_entry(line, hitman_subtitles_map)
                    if hm_entry and hm_entry.get("en"):
                        line.source_text = hm_entry["en"].strip()
                        line.source_lang = self.config.source_language or "en"
                        self.translator.cache.put_transcription(
                            line.original_path,
                            source_lang=self.config.source_language,
                            transcribed_text=line.source_text,
                            detected_lang=line.source_lang,
                        )
                        stats.transcribed += 1
                        transcribed_lines.append(line)
                        emit_line(line)
                        log(
                            f"Deşifre [Hitman Orijinal İngilizce Altyazı] [{line.rel_path}] "
                            f"({line.metadata.duration_sec:.2f}s, {line.metadata.lufs:.1f} LUFS): "
                            f"\"{line.source_text}\""
                        )
                        continue

                    # 3b) ./_veri/ceviri_onbellek.db içindeki Whisper deşifre önbelleğini kontrol et
                    cached_tr = self.translator.cache.get_transcription(
                        line.original_path,
                        source_lang=self.config.source_language,
                    )
                    if cached_tr is not None and cached_tr[0].strip():
                        line.source_text = cached_tr[0].strip()
                        line.source_lang = cached_tr[1] or self.config.source_language
                        whisper_cache_hits += 1
                        stats.transcribed += 1
                        transcribed_lines.append(line)
                        emit_line(line)
                        log(
                            f"Deşifre [Önbellek] [{line.rel_path}] ({line.metadata.duration_sec:.2f}s, "
                            f"{line.metadata.lufs:.1f} LUFS): \"{line.source_text}\""
                        )
                        continue

                    self.ffmpeg.convert_to_wav(
                        vocal_src, whisper_wav, sample_rate=16000, channels=1
                    )
                    tr_res = self.engine.transcribe(
                        whisper_wav,
                        language=self.config.source_language,
                        cancel_event=cancel_event,
                    )
                    line.source_text = tr_res.text.strip()
                    line.source_lang = tr_res.detected_language

                    if not line.source_text:
                        stats.skipped_empty += 1
                        line.status = LineStatus.SKIPPED_EMPTY
                        emit_line(line)
                        log(f"Pas geçildi (metin boş): {line.rel_path}")
                    else:
                        # Başarılı Whisper deşifresini ./_veri/ altına önbelleğe kaydet
                        self.translator.cache.put_transcription(
                            line.original_path,
                            source_lang=self.config.source_language,
                            transcribed_text=line.source_text,
                            detected_lang=line.source_lang or self.config.source_language,
                        )
                        stats.transcribed += 1
                        transcribed_lines.append(line)
                        emit_line(line)
                        log(
                            f"Deşifre [{line.rel_path}] ({line.metadata.duration_sec:.2f}s, "
                            f"{line.metadata.lufs:.1f} LUFS): \"{line.source_text}\""
                        )
                except PipelineCancelledError:
                    raise
                except InterruptedError as exc:
                    raise PipelineCancelledError(str(exc)) from exc
                except Exception as exc:
                    stats.errors += 1
                    line.status = LineStatus.ERROR
                    line.error_message = f"Deşifre hatası: {exc}"
                    emit_line(line)
                    log(f"HATA (Deşifre) [{line.rel_path}]: {exc}")
                finally:
                    whisper_wav.unlink(missing_ok=True)

            if whisper_cache_hits > 0:
                log(
                    f"⚡ Whisper Deşifre Önbelleği (./_veri/): {whisper_cache_hits}/{total_sp} ses dosyası "
                    "tekrar Whisper'dan geçirilmeden anında yüklendi."
                )

            if not transcribed_lines:
                report(100, 100, "Deşifre edilebilir konuşma metni çıkmadı.")
                return lines, stats

            # ==================================================================
            # ADIM 4: Sinematik Oyun Dublajı Senaristi Toplu (Batch JSON) Gemini Çevirisi
            #         (+ Hitman Hazır Türkçe Altyazı Önceliği & 503 Ücretsiz Yedek Çevirmen)
            # ==================================================================
            translated_lines: list[DialogueLine] = []
            if is_sts_mode:
                translated_lines = list(transcribed_lines)
            else:
                total_tr = len(transcribed_lines)
                batch_inputs: list[BatchLineInput] = []
                pre_translated_ids: set[str] = set()

                for line in transcribed_lines:
                    line.status = LineStatus.TRANSLATING
                    emit_line(line)
                    dur_sec = line.metadata.duration_sec if line.metadata else 2.0
                    lufs_val = line.metadata.lufs if line.metadata else -23.0

                    # Hitman manifest'te hazır Türkçe altyazı metni varsa Gemini'ye istek atmadan doğrudan kullan!
                    hm_entry = self._find_hitman_manifest_entry(line, hitman_subtitles_map)
                    if hm_entry and hm_entry.get("tr"):
                        ready_tr = hm_entry["tr"].strip()
                        custom_manifest_act = (hm_entry.get("acting_prompt") or "").strip().strip("()")
                        if custom_manifest_act:
                            # Manifestte elle hazırlanmış özel oyunculuk/duygu etiketi birebir korunur!
                            act_prompt = custom_manifest_act
                        else:
                            act_prompt = clean_acting_prompt(
                                infer_fallback_acting_prompt(
                                    line.source_text, duration_sec=dur_sec, lufs=lufs_val
                                ),
                                source_text=line.source_text,
                                duration_sec=dur_sec,
                                lufs=lufs_val,
                            )
                        line.translated_text = ready_tr
                        line.acting_prompt = act_prompt
                        line.from_cache = True
                        pre_translated_ids.add(line.line_id)
                        stats.translated += 1
                        stats.cache_hits += 1
                        try:
                            matched = self.translator.glossary.find_matches(line.source_text)
                            g_key = self.translator._cache_glossary_key(
                                self.translator.glossary.signature_hash(matched)
                            )
                            self.translator.cache.put(
                                source_text=line.source_text,
                                source_lang=line.source_lang or self.config.source_language,
                                target_lang=self.config.target_language,
                                duration_sec=dur_sec,
                                glossary_hash=g_key,
                                model_name="hitman-manifest-tr",
                                translated_text=ready_tr,
                                acting_prompt=act_prompt,
                            )
                        except Exception:
                            pass
                        emit_line(line)
                        act_src_tag = "Manifest Duygu" if custom_manifest_act else "Duygu"
                        log(
                            f"Çeviri [Hitman Hazır Türkçe Altyazı] [{line.rel_path}]: "
                            f"\"{line.translated_text}\" | {act_src_tag}: ({line.acting_prompt})"
                        )
                        continue

                    batch_inputs.append(
                        BatchLineInput(
                            item_id=line.line_id,
                            source_text=line.source_text,
                            duration_sec=dur_sec,
                            source_lang=line.source_lang or self.config.source_language,
                            target_lang=self.config.target_language,
                            lufs=lufs_val,
                        )
                    )

                report(
                    45,
                    100,
                    f"Adım 4/6 (Gemini Toplu Çeviri & Oyunculuk Yönetmeni): {total_tr} replik çevriliyor...",
                )

                try:
                    batch_results = (
                        self.translator.translate_batch(
                            items=batch_inputs,
                            use_cache=True,
                            cancel_event=cancel_event,
                            log_cb=log,
                            progress_cb=lambda cur, tot: report(
                                45 + int((cur / max(1, tot)) * 18),
                                100,
                                f"Adım 4/6 (Gemini Toplu Çeviri): {cur}/{tot} replik tamamlandı",
                            ),
                        )
                        if batch_inputs
                        else {}
                    )
                    for line in transcribed_lines:
                        if line.line_id in pre_translated_ids:
                            translated_lines.append(line)
                            continue
                        t_res = batch_results.get(line.line_id)
                        if t_res and t_res.translated_text:
                            line.translated_text = t_res.translated_text
                            hm_entry_post = self._find_hitman_manifest_entry(line, hitman_subtitles_map)
                            custom_post_act = (
                                (hm_entry_post.get("acting_prompt") or "").strip().strip("()")
                                if hm_entry_post
                                else ""
                            )
                            line.acting_prompt = custom_post_act or t_res.acting_prompt
                            line.from_cache = t_res.from_cache
                            stats.translated += 1
                            if t_res.from_cache:
                                stats.cache_hits += 1
                            translated_lines.append(line)
                            emit_line(line)
                            cache_tag = " [Önbellek]" if t_res.from_cache else ""
                            gloss_tag = (
                                f" [Sözlük:{t_res.matched_glossary_count}]"
                                if t_res.matched_glossary_count
                                else ""
                            )
                            act_tag = f" | Duygu: ({line.acting_prompt})" if line.acting_prompt else ""
                            log(
                                f"Çeviri{cache_tag}{gloss_tag} [{line.rel_path}]: \"{line.translated_text}\"{act_tag}"
                            )
                        else:
                            stats.errors += 1
                            line.status = LineStatus.ERROR
                            line.error_message = "Çeviri sonucu boş döndü."
                            emit_line(line)
                except PipelineCancelledError:
                    raise
                except InterruptedError as exc:
                    raise PipelineCancelledError(str(exc)) from exc
                except Exception as exc:
                    for line in transcribed_lines:
                        if not line.translated_text:
                            stats.errors += 1
                            line.status = LineStatus.ERROR
                            line.error_message = f"Çeviri hatası: {exc}"
                            emit_line(line)
                    log(f"HATA (Toplu Çeviri): {exc}")

            if not translated_lines:
                report(100, 100, "Çevrilen satır bulunamadı.")
                return lines, stats

            # ==================================================================
            # ADIM 5 & 6: VRAM Dostu Aşamalı Çoklu Aday Üretimi + Whisper Hakemi + Senkron & Miks
            # ==================================================================
            effective_candidates = self._resolve_effective_candidate_count(log)
            candidates_dir = preview_dir / "_adaylar"
            candidates_dir.mkdir(parents=True, exist_ok=True)
            stems_persist_dir = preview_dir / "_stems"

            # AŞAMA 5A: Tüm repliklerin N adet aday sesini aktif TTS motoruyla üret (VRAM'de tek model!)
            synthesized_lines: list[DialogueLine] = []
            total_syn = len(translated_lines)
            for idx, line in enumerate(translated_lines, start=1):
                self._check_cancel(cancel_event)
                line.status = LineStatus.SYNTHESIZING
                line.candidates.clear()
                emit_line(line)

                vocal_src, bg_wav, voc_lufs = line_stems[line.line_id]
                if bg_wav is not None and bg_wav.is_file():
                    stems_persist_dir.mkdir(parents=True, exist_ok=True)
                    persisted_bg = stems_persist_dir / f"{line.line_id}_bg.wav"
                    shutil.copy2(bg_wav, persisted_bg)
                    bg_wav = persisted_bg
                    line_stems[line.line_id] = (vocal_src, bg_wav, voc_lufs)
                setattr(line, "_bg_wav_path", bg_wav)
                setattr(line, "_vocal_lufs", voc_lufs)

                per_line_ref_wav: Optional[Path] = None
                try:
                    if is_sts_mode:
                        per_line_ref_wav = work_dir / f"{line.line_id}_sts_src.wav"
                        self.ffmpeg.convert_to_wav(vocal_src, per_line_ref_wav, sample_rate=24000, channels=1)
                        speaker_for_engine = per_line_ref_wav
                    elif fixed_ref_wav is not None:
                        speaker_for_engine = fixed_ref_wav
                    else:
                        per_line_ref_wav = work_dir / f"{line.line_id}_ref.wav"
                        self.ffmpeg.prepare_reference_wav(
                            vocal_src, per_line_ref_wav, sample_rate=ref_sample_rate
                        )
                        speaker_for_engine = per_line_ref_wav

                    for c_idx in range(1, effective_candidates + 1):
                        self._check_cancel(cancel_event)
                        cand_label = (
                            f" (Aday {c_idx}/{effective_candidates})"
                            if effective_candidates > 1
                            else ""
                        )
                        report(
                            63 + int((((idx - 1) + (c_idx / effective_candidates) * 0.65) / total_syn) * 20),
                            100,
                            f"Adım 5/6 ({tts_options.tts_engine.label_tr}{cand_label}): {idx}/{total_syn} — {line.rel_path}",
                        )
                        cand_raw_wav = (
                            candidates_dir
                            / f"{line.line_id}_aday{c_idx}_{uuid.uuid4().hex[:6]}_raw.wav"
                        )
                        line_tts_options = self.config.build_tts_options(
                            acting_prompt=line.acting_prompt,
                            candidate_index=c_idx,
                        )
                        used_cinematic = self._synthesize_candidate_wav(
                            line=line,
                            text=line.translated_text,
                            speaker_wav_path=speaker_for_engine,
                            output_wav_path=cand_raw_wav,
                            tts_options=line_tts_options,
                            work_dir=work_dir,
                            cancel_event=cancel_event,
                            log=log,
                        )
                        raw_dur_ms = self._measure_trimmed_candidate_duration_ms(
                            cand_raw_wav, work_dir, skip_trim=used_cinematic
                        )
                        line.candidates.append(
                            DubCandidate(
                                candidate_index=c_idx,
                                raw_wav_path=cand_raw_wav,
                                raw_duration_ms=raw_dur_ms,
                            )
                        )
                        stats.synthesized += 1

                    synthesized_lines.append(line)
                except PipelineCancelledError:
                    raise
                except InterruptedError as exc:
                    raise PipelineCancelledError(str(exc)) from exc
                except Exception as exc:
                    stats.errors += 1
                    line.status = LineStatus.ERROR
                    line.error_message = f"Seslendirme hatası: {exc}"
                    emit_line(line)
                    log(f"HATA (Seslendirme) [{line.rel_path}]: {exc}")
                finally:
                    if per_line_ref_wav is not None:
                        per_line_ref_wav.unlink(missing_ok=True)

            # AŞAMA 5B: Çoklu aday varsa önce TTS modelini VRAM'den boşalt, sonra faster-whisper'ı TEK SEFERDE yükleyip puanla!
            if effective_candidates > 1 and synthesized_lines:
                self.engine.unload_tts(log_cb=log)
                log(
                    f"🎯 Whisper Hakemi başlatıldı: {len(synthesized_lines)} replik × "
                    f"{effective_candidates} aday Türkçe deşifre ve doğal süreyle puanlanıyor..."
                )

            ready_lines: list[DialogueLine] = []
            total_ready = len(synthesized_lines)
            for idx, line in enumerate(synthesized_lines, start=1):
                self._check_cancel(cancel_event)
                vocal_src, bg_wav, voc_lufs = line_stems[line.line_id]
                try:
                    if effective_candidates > 1:
                        report(
                            83 + int(((idx - 0.5) / max(1, total_ready)) * 6),
                            100,
                            f"Adım 5B/6 (Whisper Aday Hakemi): {idx}/{total_ready} — {line.rel_path}",
                        )
                    best_cand = self._score_candidates_for_line(
                        line=line,
                        work_dir=work_dir,
                        cancel_event=cancel_event,
                        log=log,
                    )

                    # ADIM 6: En Yüksek Puanlı Adayı Akıllı Sessizlik Kırpma, Doğal Tempo, LUFS ve Arka Plan Miksinden Geçir
                    self._check_cancel(cancel_event)
                    line.status = LineStatus.SYNCING
                    emit_line(line)
                    report(
                        89 + int((idx / max(1, total_ready)) * 6),
                        100,
                        f"Adım 6/6 (Sessizlik Kırpma & Senkron): {idx}/{total_ready} — {line.rel_path}",
                    )

                    assert line.metadata is not None
                    assert best_cand.raw_wav_path is not None
                    preview_out = preview_dir / Path(line.rel_path)
                    synced_path, dub_ms, tempo = self.ffmpeg.sync_and_package_dub(
                        dub_wav_path=best_cand.raw_wav_path,
                        original_meta=line.metadata,
                        output_path=preview_out,
                        max_tempo_speedup=self.config.max_tempo_speedup,
                        trim_silence=(
                            self.config.trim_silence
                            and not is_sts_mode
                            and not bool(getattr(line, "_cinematic_locked", False))
                        ),
                        background_wav_path=bg_wav,
                        bg_volume_db=self.config.bg_mix_volume_db,
                        target_vocal_lufs=voc_lufs,
                    )
                    best_cand.synced_wav_path = synced_path
                    best_cand.synced_duration_ms = dub_ms
                    best_cand.tempo_factor = tempo

                    line.synced_preview_path = synced_path
                    line.dub_duration_ms = dub_ms
                    line.tempo_factor = tempo
                    line.status = LineStatus.READY_PREVIEW
                    stats.synced += 1
                    ready_lines.append(line)
                    emit_line(line)
                    bg_tag = " + ArkaPlanMiks" if bg_wav else ""
                    cand_tag = (
                        f" | Seçilen: Aday {best_cand.candidate_index} (%{best_cand.score_pct_int})"
                        if len(line.candidates) > 1
                        else ""
                    )
                    log(
                        f"Hazır [{line.rel_path}]: {line.metadata.duration_ms}ms -> {dub_ms}ms "
                        f"(tempo: {tempo:.2f}x, hedef: {(voc_lufs if voc_lufs is not None else line.metadata.lufs):.1f} LUFS, "
                        f"{line.metadata.sample_rate}Hz {line.metadata.channel_label}{bg_tag}{cand_tag})"
                    )
                except PipelineCancelledError:
                    raise
                except InterruptedError as exc:
                    raise PipelineCancelledError(str(exc)) from exc
                except Exception as exc:
                    stats.errors += 1
                    line.status = LineStatus.ERROR
                    line.error_message = f"Hakem/Senkron hatası: {exc}"
                    emit_line(line)
                    log(f"HATA (Hakem/Senkron) [{line.rel_path}]: {exc}")

            if effective_candidates > 1 and synthesized_lines:
                self.engine.unload_whisper(log_cb=log)

            # ==================================================================
            # Otomatik Uygulama Seçiliyse SHA-256 Yedekli Uygula
            # ==================================================================
            if self.config.auto_apply_to_game and ready_lines:
                report(95, 100, "SHA-256 yedekleniyor ve oyuna uygulanıyor...")
                applied_count = self.apply_lines_to_game(
                    game_dir=game_root,
                    project_name=safe_project,
                    lines=ready_lines,
                    discovered_map=discovered_map,
                    log_cb=log,
                    line_cb=emit_line,
                )
                stats.applied = applied_count

            report(100, 100, "Sinematik dublaj iş hattı tamamlandı!")
            log(
                f"Özet: Tarandı={stats.total_scanned}, Konuşma={stats.speech_detected}, "
                f"Atlandı={stats.skipped_non_speech}, Çevrildi={stats.translated} "
                f"(Önbellek={stats.cache_hits}), Senkron={stats.synced}, "
                f"Uygulandı={stats.applied}, Hata={stats.errors}"
            )
            return lines, stats

        except PipelineCancelledError:
            had_fatal_or_cancel = True
            for ln in lines:
                if ln.status not in (
                    LineStatus.SKIPPED_NON_SPEECH,
                    LineStatus.SKIPPED_EMPTY,
                    LineStatus.APPLIED,
                    LineStatus.ERROR,
                ):
                    ln.status = LineStatus.CANCELLED
                    emit_line(ln)
            log("Dublaj işlemi durduruldu; tüm geçici dosyalar temizleniyor.")
            raise
        except Exception:
            had_fatal_or_cancel = True
            raise
        finally:
            try:
                self.vocal_separator.unload()
            except Exception:
                pass
            try:
                self.engine.unload(log_cb=log)
            except Exception:
                pass

            shutil.rmtree(work_dir, ignore_errors=True)

            if had_fatal_or_cancel or self.config.auto_apply_to_game or stats.synced == 0:
                cleanup_temp_dir(keep_root=True)
                for ln in lines:
                    if ln.synced_preview_path and not ln.synced_preview_path.exists():
                        ln.synced_preview_path = None

    def redub_single_line(
        self,
        line: DialogueLine,
        new_turkish_text: str,
        cancel_event: Optional[threading.Event] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        new_acting_prompt: Optional[str] = None,
    ) -> DialogueLine:
        """Kullanıcının elle düzenlediği tek bir satırı (metin + acting_prompt) güncel motor ayarlarıyla yeniden seslendirir."""
        work_dir = make_temp_subdir("tek_satir")
        if new_acting_prompt is not None:
            line.acting_prompt = new_acting_prompt.strip().strip("()")
        tts_options = self.config.build_tts_options(acting_prompt=line.acting_prompt)
        is_sts_mode = tts_options.tts_engine is TTSEngineType.ELEVENLABS_STS
        ref_sample_rate = 48000 if tts_options.tts_engine.is_voxcpm else 24000

        def log(msg: str) -> None:
            if log_cb:
                log_cb(msg)

        try:
            cleaned = new_turkish_text.strip()
            if not is_sts_mode and not cleaned:
                raise ValueError("Türkçe dublaj metni boş olamaz.")
            line.translated_text = cleaned
            if line.metadata is None:
                line.metadata = self.ffmpeg.probe_audio(line.original_path, measure_lufs=True)

            # Kullanıcının tabloda yaptığı çeviri ve oyunculuk/duygu düzenlemesini SQLite önbelleğine de işle
            if line.source_text and line.translated_text:
                try:
                    matched = self.translator.glossary.find_matches(line.source_text)
                    g_key = self.translator._cache_glossary_key(
                        self.translator.glossary.signature_hash(matched)
                    )
                    self.translator.cache.put(
                        source_text=line.source_text,
                        source_lang=line.source_lang or self.config.source_language,
                        target_lang=self.config.target_language,
                        duration_sec=line.metadata.duration_sec,
                        glossary_hash=g_key,
                        model_name=self.translator.model_name,
                        translated_text=line.translated_text,
                        acting_prompt=line.acting_prompt,
                    )
                except Exception:
                    pass

            vocal_src, bg_wav, voc_lufs = self._separate_if_enabled(
                line, work_dir, cancel_event, log
            )
            fixed_ref = self._prepare_fixed_reference_if_set(
                work_dir, log, sample_rate=ref_sample_rate
            )

            if is_sts_mode:
                ref_wav = work_dir / f"{line.line_id}_sts_src.wav"
                self.ffmpeg.convert_to_wav(vocal_src, ref_wav, sample_rate=24000, channels=1)
            elif fixed_ref is not None:
                ref_wav = fixed_ref
            else:
                ref_wav = work_dir / f"{line.line_id}_ref.wav"
                self.ffmpeg.prepare_reference_wav(
                    vocal_src, ref_wav, sample_rate=ref_sample_rate
                )

            old_preview = line.synced_preview_path
            rel_p = Path(line.rel_path)
            if old_preview is not None and old_preview.parent.exists():
                base_preview_dir = old_preview.parent
            else:
                preview_root = make_temp_subdir("onizleme")
                base_preview_dir = preview_root / rel_p.parent
            base_preview_dir.mkdir(parents=True, exist_ok=True)
            candidates_dir = base_preview_dir / "_adaylar"
            candidates_dir.mkdir(parents=True, exist_ok=True)

            if bg_wav is not None and bg_wav.is_file():
                persisted_bg = candidates_dir / f"{line.line_id}_redub_bg_{uuid.uuid4().hex[:6]}.wav"
                shutil.copy2(bg_wav, persisted_bg)
                bg_wav = persisted_bg
            setattr(line, "_bg_wav_path", bg_wav)
            setattr(line, "_vocal_lufs", voc_lufs)

            effective_candidates = self._resolve_effective_candidate_count(log)
            line.candidates.clear()
            for c_idx in range(1, effective_candidates + 1):
                self._check_cancel(cancel_event)
                cand_raw_wav = (
                    candidates_dir
                    / f"{line.line_id}_redub_aday{c_idx}_{uuid.uuid4().hex[:6]}_raw.wav"
                )
                cand_opts = self.config.build_tts_options(
                    acting_prompt=line.acting_prompt,
                    candidate_index=c_idx,
                )
                used_cinematic = self._synthesize_candidate_wav(
                    line=line,
                    text=cleaned,
                    speaker_wav_path=ref_wav,
                    output_wav_path=cand_raw_wav,
                    tts_options=cand_opts,
                    work_dir=work_dir,
                    cancel_event=cancel_event,
                    log=log,
                )
                raw_dur_ms = self._measure_trimmed_candidate_duration_ms(
                    cand_raw_wav, work_dir, skip_trim=used_cinematic
                )
                line.candidates.append(
                    DubCandidate(
                        candidate_index=c_idx,
                        raw_wav_path=cand_raw_wav,
                        raw_duration_ms=raw_dur_ms,
                    )
                )

            if effective_candidates > 1:
                self.engine.unload_tts(log_cb=log)

            best_cand = self._score_candidates_for_line(
                line=line,
                work_dir=work_dir,
                cancel_event=cancel_event,
                log=log,
            )

            # Windows Permission denied dosya kilidi hatasını kalıcı olarak önle:
            # Eski önizleme dosyasının üzerine doğrudan yazmak yerine ./_temp/ altında
            # benzersiz yeni bir dosya adına yazıp line.synced_preview_path'e bağla.
            assert best_cand.raw_wav_path is not None
            unique_tag = uuid.uuid4().hex[:8]
            preview_out = base_preview_dir / f"{rel_p.stem}_redub_{unique_tag}{rel_p.suffix}"

            synced_path, dub_ms, tempo = self.ffmpeg.sync_and_package_dub(
                dub_wav_path=best_cand.raw_wav_path,
                original_meta=line.metadata,
                output_path=preview_out,
                max_tempo_speedup=self.config.max_tempo_speedup,
                trim_silence=(
                    self.config.trim_silence
                    and not is_sts_mode
                    and not bool(getattr(line, "_cinematic_locked", False))
                ),
                background_wav_path=bg_wav,
                bg_volume_db=self.config.bg_mix_volume_db,
                target_vocal_lufs=voc_lufs,
            )
            best_cand.synced_wav_path = synced_path
            best_cand.synced_duration_ms = dub_ms
            best_cand.tempo_factor = tempo

            line.synced_preview_path = synced_path
            line.dub_duration_ms = dub_ms
            line.tempo_factor = tempo
            line.status = LineStatus.READY_PREVIEW
            line.error_message = ""
            if old_preview is not None and old_preview != synced_path:
                try:
                    old_preview.unlink(missing_ok=True)
                except OSError:
                    pass
            act_info = f", duygu: ({line.acting_prompt})" if line.acting_prompt else ""
            cand_info = (
                f", seçilen: Aday {best_cand.candidate_index} (%{best_cand.score_pct_int})"
                if len(line.candidates) > 1
                else ""
            )
            log(
                f"Satır yeniden seslendirildi [{line.rel_path}] ({dub_ms}ms, tempo {tempo:.2f}x{act_info}{cand_info})."
            )
            return line
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            try:
                self.vocal_separator.unload()
            except Exception:
                pass
            try:
                self.engine.unload(log_cb=log_cb)
            except Exception:
                pass

    def redub_all_lines(
        self,
        lines: list[DialogueLine],
        cancel_event: Optional[threading.Event] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        line_cb: Optional[Callable[[DialogueLine], None]] = None,
    ) -> tuple[list[DialogueLine], PipelineStats, bool]:
        """Tablodaki tüm konuşma satırlarını Whisper deşifre ve Gemini çeviri adımlarını
        TAMAMEN ATLAYARAK mevcut Türkçe metin ve Oyunculuk/Duygu hücreleriyle yeniden dublajlar.

        Durdurma (cancel_event) tetiklenirse o ana kadar tamamlanan satırlar ve önizleme dosyaları
        KESİNLİKLE korunur (./_temp temizlenmez, yarım kalan satır eski durumuna geri alınır).
        """
        work_dir = make_temp_subdir("toplu_yeniden")
        preview_dir = make_temp_subdir("onizleme")
        candidates_dir = preview_dir / "_adaylar"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        stems_persist_dir = preview_dir / "_stems"

        tts_options = self.config.build_tts_options()
        is_sts_mode = tts_options.tts_engine is TTSEngineType.ELEVENLABS_STS
        ref_sample_rate = 48000 if tts_options.tts_engine.is_voxcpm else 24000

        def log(msg: str) -> None:
            if log_cb:
                log_cb(msg)

        def report(cur: int, total: int, stage: str) -> None:
            if progress_cb:
                progress_cb(cur, total, stage)

        def emit_line(ln: DialogueLine) -> None:
            if line_cb:
                line_cb(ln)

        target_lines = [
            ln
            for ln in lines
            if ln.is_speech
            and ln.original_path.is_file()
            and (is_sts_mode or bool((ln.translated_text or "").strip()))
        ]
        stats = PipelineStats(
            total_scanned=len(lines),
            speech_detected=len(target_lines),
        )
        was_cancelled = False

        if not target_lines:
            log("Yeniden dublajlanacak uygun konuşma satırı bulunamadı.")
            shutil.rmtree(work_dir, ignore_errors=True)
            return lines, stats, False

        effective_candidates = self._resolve_effective_candidate_count(log)
        log(
            f"🔁 Tümünü Yeniden Dublajla başlatıldı: {len(target_lines)} satır | "
            "Whisper deşifre ve Gemini çeviri ATLANDI (mevcut Türkçe metin ve oyunculuk/duygu korunuyor) | "
            f"Motor: {tts_options.tts_engine.label_tr} | Aday Sayısı: {effective_candidates}"
        )

        def _sync_single_line(
            line: DialogueLine,
            best_cand: DubCandidate,
            bg_wav: Optional[Path],
            voc_lufs: Optional[float],
        ) -> None:
            assert line.metadata is not None
            assert best_cand.raw_wav_path is not None
            old_preview = line.synced_preview_path
            rel_p = Path(line.rel_path)
            base_preview_dir = preview_dir / rel_p.parent
            base_preview_dir.mkdir(parents=True, exist_ok=True)
            unique_tag = uuid.uuid4().hex[:8]
            preview_out = base_preview_dir / f"{rel_p.stem}_reduball_{unique_tag}{rel_p.suffix}"

            synced_path, dub_ms, tempo = self.ffmpeg.sync_and_package_dub(
                dub_wav_path=best_cand.raw_wav_path,
                original_meta=line.metadata,
                output_path=preview_out,
                max_tempo_speedup=self.config.max_tempo_speedup,
                trim_silence=(
                    self.config.trim_silence
                    and not is_sts_mode
                    and not bool(getattr(line, "_cinematic_locked", False))
                ),
                background_wav_path=bg_wav,
                bg_volume_db=self.config.bg_mix_volume_db,
                target_vocal_lufs=voc_lufs,
            )
            best_cand.synced_wav_path = synced_path
            best_cand.synced_duration_ms = dub_ms
            best_cand.tempo_factor = tempo

            line.synced_preview_path = synced_path
            line.dub_duration_ms = dub_ms
            line.tempo_factor = tempo
            line.background_preserved = bg_wav is not None
            line.status = LineStatus.READY_PREVIEW
            line.error_message = ""
            if old_preview is not None and old_preview != synced_path:
                try:
                    old_preview.unlink(missing_ok=True)
                except OSError:
                    pass

        try:
            fixed_ref_wav = self._prepare_fixed_reference_if_set(
                work_dir, log, sample_rate=ref_sample_rate
            )
            total_target = len(target_lines)

            # Kullanıcının tablodaki güncel metin ve oyunculuk/duygu düzenlemelerini önbelleğe işle
            for line in target_lines:
                cleaned_tr = (line.translated_text or "").strip()
                cleaned_act = (line.acting_prompt or "").strip().strip("()")
                line.translated_text = cleaned_tr
                line.acting_prompt = cleaned_act
                if line.source_text and cleaned_tr:
                    try:
                        dur_sec = line.metadata.duration_sec if line.metadata else 2.0
                        matched = self.translator.glossary.find_matches(line.source_text)
                        g_key = self.translator._cache_glossary_key(
                            self.translator.glossary.signature_hash(matched)
                        )
                        self.translator.cache.put(
                            source_text=line.source_text,
                            source_lang=line.source_lang or self.config.source_language,
                            target_lang=self.config.target_language,
                            duration_sec=dur_sec,
                            glossary_hash=g_key,
                            model_name=self.translator.model_name,
                            translated_text=cleaned_tr,
                            acting_prompt=cleaned_act,
                        )
                    except Exception:
                        pass

            if effective_candidates == 1:
                # Tek adayda her satır anında sentezlenip senkronize edilir -> Durdur'da biten her satır hazır!
                for idx, line in enumerate(target_lines, start=1):
                    if cancel_event and cancel_event.is_set():
                        was_cancelled = True
                        break

                    prev_status = line.status
                    prev_candidates = list(line.candidates)
                    prev_sel = line.selected_candidate_index
                    prev_score = line.best_candidate_score

                    per_line_ref_wav: Optional[Path] = None
                    try:
                        if line.metadata is None:
                            line.metadata = self.ffmpeg.probe_audio(
                                line.original_path, measure_lufs=True
                            )
                        vocal_src, bg_wav, voc_lufs = self._separate_if_enabled(
                            line, work_dir, cancel_event, log
                        )
                        if bg_wav is not None and bg_wav.is_file():
                            stems_persist_dir.mkdir(parents=True, exist_ok=True)
                            persisted_bg = stems_persist_dir / f"{line.line_id}_reduball_bg.wav"
                            shutil.copy2(bg_wav, persisted_bg)
                            bg_wav = persisted_bg
                        setattr(line, "_bg_wav_path", bg_wav)
                        setattr(line, "_vocal_lufs", voc_lufs)

                        line.status = LineStatus.SYNTHESIZING
                        emit_line(line)
                        report(
                            int(((idx - 0.5) / total_target) * 95),
                            100,
                            f"Yeniden Dublaj ({tts_options.tts_engine.label_tr}): {idx}/{total_target} — {line.rel_path}",
                        )

                        if is_sts_mode:
                            per_line_ref_wav = work_dir / f"{line.line_id}_sts_src.wav"
                            self.ffmpeg.convert_to_wav(
                                vocal_src, per_line_ref_wav, sample_rate=24000, channels=1
                            )
                            speaker_for_engine = per_line_ref_wav
                        elif fixed_ref_wav is not None:
                            speaker_for_engine = fixed_ref_wav
                        else:
                            per_line_ref_wav = work_dir / f"{line.line_id}_ref.wav"
                            self.ffmpeg.prepare_reference_wav(
                                vocal_src, per_line_ref_wav, sample_rate=ref_sample_rate
                            )
                            speaker_for_engine = per_line_ref_wav

                        cand_raw_wav = (
                            candidates_dir
                            / f"{line.line_id}_reduball_aday1_{uuid.uuid4().hex[:6]}_raw.wav"
                        )
                        line_tts_options = self.config.build_tts_options(
                            acting_prompt=line.acting_prompt,
                            candidate_index=1,
                        )
                        used_cinematic = self._synthesize_candidate_wav(
                            line=line,
                            text=line.translated_text,
                            speaker_wav_path=speaker_for_engine,
                            output_wav_path=cand_raw_wav,
                            tts_options=line_tts_options,
                            work_dir=work_dir,
                            cancel_event=cancel_event,
                            log=log,
                        )
                        raw_dur_ms = self._measure_trimmed_candidate_duration_ms(
                            cand_raw_wav, work_dir, skip_trim=used_cinematic
                        )
                        line.candidates = [
                            DubCandidate(
                                candidate_index=1,
                                raw_wav_path=cand_raw_wav,
                                raw_duration_ms=raw_dur_ms,
                            )
                        ]
                        stats.synthesized += 1

                        best_cand = self._score_candidates_for_line(
                            line=line,
                            work_dir=work_dir,
                            cancel_event=None,
                            log=log,
                        )
                        line.status = LineStatus.SYNCING
                        emit_line(line)
                        _sync_single_line(line, best_cand, bg_wav, voc_lufs)
                        stats.synced += 1
                        emit_line(line)
                        report(
                            int((idx / total_target) * 95),
                            100,
                            f"Yeniden Dublaj Tamamlandı: {idx}/{total_target} — {line.rel_path}",
                        )
                        act_tag = f", duygu: ({line.acting_prompt})" if line.acting_prompt else ""
                        log(
                            f"Yeniden dublajlandı [{line.rel_path}] "
                            f"({line.dub_duration_ms}ms, tempo {line.tempo_factor:.2f}x{act_tag})."
                        )
                    except (PipelineCancelledError, InterruptedError):
                        line.status = prev_status
                        line.candidates = prev_candidates
                        line.selected_candidate_index = prev_sel
                        line.best_candidate_score = prev_score
                        emit_line(line)
                        was_cancelled = True
                        break
                    except Exception as exc:
                        stats.errors += 1
                        line.status = LineStatus.ERROR
                        line.error_message = f"Yeniden dublaj hatası: {exc}"
                        emit_line(line)
                        log(f"HATA (Yeniden Dublaj) [{line.rel_path}]: {exc}")
                    finally:
                        if per_line_ref_wav is not None:
                            per_line_ref_wav.unlink(missing_ok=True)
            else:
                # Çoklu aday (>1): VRAM dostu Aşama A (TTS aday üretimi) + Aşama B (Whisper hakemi + senkron)
                synthesized_items: list[tuple[DialogueLine, Optional[Path], Optional[float]]] = []
                for idx, line in enumerate(target_lines, start=1):
                    if cancel_event and cancel_event.is_set():
                        was_cancelled = True
                        break

                    prev_status = line.status
                    prev_candidates = list(line.candidates)
                    prev_sel = line.selected_candidate_index
                    prev_score = line.best_candidate_score

                    per_line_ref_wav = None
                    new_candidates: list[DubCandidate] = []
                    try:
                        if line.metadata is None:
                            line.metadata = self.ffmpeg.probe_audio(
                                line.original_path, measure_lufs=True
                            )
                        vocal_src, bg_wav, voc_lufs = self._separate_if_enabled(
                            line, work_dir, cancel_event, log
                        )
                        if bg_wav is not None and bg_wav.is_file():
                            stems_persist_dir.mkdir(parents=True, exist_ok=True)
                            persisted_bg = stems_persist_dir / f"{line.line_id}_reduball_bg.wav"
                            shutil.copy2(bg_wav, persisted_bg)
                            bg_wav = persisted_bg
                        setattr(line, "_bg_wav_path", bg_wav)
                        setattr(line, "_vocal_lufs", voc_lufs)

                        line.status = LineStatus.SYNTHESIZING
                        emit_line(line)

                        if is_sts_mode:
                            per_line_ref_wav = work_dir / f"{line.line_id}_sts_src.wav"
                            self.ffmpeg.convert_to_wav(
                                vocal_src, per_line_ref_wav, sample_rate=24000, channels=1
                            )
                            speaker_for_engine = per_line_ref_wav
                        elif fixed_ref_wav is not None:
                            speaker_for_engine = fixed_ref_wav
                        else:
                            per_line_ref_wav = work_dir / f"{line.line_id}_ref.wav"
                            self.ffmpeg.prepare_reference_wav(
                                vocal_src, per_line_ref_wav, sample_rate=ref_sample_rate
                            )
                            speaker_for_engine = per_line_ref_wav

                        for c_idx in range(1, effective_candidates + 1):
                            self._check_cancel(cancel_event)
                            report(
                                int(
                                    (((idx - 1) + (c_idx / effective_candidates) * 0.85) / total_target)
                                    * 68
                                ),
                                100,
                                f"Yeniden Dublaj ({tts_options.tts_engine.label_tr} — Aday {c_idx}/{effective_candidates}): "
                                f"{idx}/{total_target} — {line.rel_path}",
                            )
                            cand_raw_wav = (
                                candidates_dir
                                / f"{line.line_id}_reduball_aday{c_idx}_{uuid.uuid4().hex[:6]}_raw.wav"
                            )
                            line_tts_options = self.config.build_tts_options(
                                acting_prompt=line.acting_prompt,
                                candidate_index=c_idx,
                            )
                            used_cinematic = self._synthesize_candidate_wav(
                                line=line,
                                text=line.translated_text,
                                speaker_wav_path=speaker_for_engine,
                                output_wav_path=cand_raw_wav,
                                tts_options=line_tts_options,
                                work_dir=work_dir,
                                cancel_event=cancel_event,
                                log=log,
                            )
                            raw_dur_ms = self._measure_trimmed_candidate_duration_ms(
                                cand_raw_wav, work_dir, skip_trim=used_cinematic
                            )
                            new_candidates.append(
                                DubCandidate(
                                    candidate_index=c_idx,
                                    raw_wav_path=cand_raw_wav,
                                    raw_duration_ms=raw_dur_ms,
                                )
                            )
                            stats.synthesized += 1

                        line.candidates = new_candidates
                        synthesized_items.append((line, bg_wav, voc_lufs))
                    except (PipelineCancelledError, InterruptedError):
                        line.status = prev_status
                        line.candidates = prev_candidates
                        line.selected_candidate_index = prev_sel
                        line.best_candidate_score = prev_score
                        emit_line(line)
                        was_cancelled = True
                        break
                    except Exception as exc:
                        stats.errors += 1
                        line.status = LineStatus.ERROR
                        line.error_message = f"Yeniden dublaj hatası: {exc}"
                        emit_line(line)
                        log(f"HATA (Yeniden Dublaj) [{line.rel_path}]: {exc}")
                    finally:
                        if per_line_ref_wav is not None:
                            per_line_ref_wav.unlink(missing_ok=True)

                if synthesized_items:
                    self.engine.unload_tts(log_cb=log)
                    if was_cancelled:
                        log(
                            f"⏹ Durduruldu — o ana kadar seslendirilen {len(synthesized_items)} satır "
                            "Whisper hakemiyle puanlanıp senkronize edilerek korunuyor..."
                        )
                    else:
                        log(
                            f"🎯 Whisper Hakemi başlatıldı: {len(synthesized_items)} replik × "
                            f"{effective_candidates} aday puanlanıp senkronize ediliyor..."
                        )

                    # Aşama A'da iptal olduysa tamamlanmış satırları kaybetmemek için puanlama+senkronu bitir
                    stage_b_cancel = None if was_cancelled else cancel_event
                    total_syn = len(synthesized_items)
                    for idx, (line, bg_wav, voc_lufs) in enumerate(synthesized_items, start=1):
                        if stage_b_cancel and stage_b_cancel.is_set():
                            was_cancelled = True
                            break
                        prev_status = line.status
                        try:
                            report(
                                68 + int(((idx - 0.5) / max(1, total_syn)) * 20),
                                100,
                                f"Whisper Aday Hakemi: {idx}/{total_syn} — {line.rel_path}",
                            )
                            best_cand = self._score_candidates_for_line(
                                line=line,
                                work_dir=work_dir,
                                cancel_event=stage_b_cancel,
                                log=log,
                            )
                            line.status = LineStatus.SYNCING
                            emit_line(line)
                            report(
                                88 + int((idx / max(1, total_syn)) * 10),
                                100,
                                f"Senkron & LUFS: {idx}/{total_syn} — {line.rel_path}",
                            )
                            _sync_single_line(line, best_cand, bg_wav, voc_lufs)
                            stats.synced += 1
                            emit_line(line)
                            act_tag = f", duygu: ({line.acting_prompt})" if line.acting_prompt else ""
                            log(
                                f"Yeniden dublajlandı [{line.rel_path}] "
                                f"(Aday {best_cand.candidate_index} %{best_cand.score_pct_int}, "
                                f"{line.dub_duration_ms}ms, tempo {line.tempo_factor:.2f}x{act_tag})."
                            )
                        except (PipelineCancelledError, InterruptedError):
                            line.status = prev_status
                            emit_line(line)
                            was_cancelled = True
                            break
                        except Exception as exc:
                            stats.errors += 1
                            line.status = LineStatus.ERROR
                            line.error_message = f"Puanlama / senkron hatası: {exc}"
                            emit_line(line)
                            log(f"HATA (Yeniden Dublaj Senkron) [{line.rel_path}]: {exc}")

            if was_cancelled:
                log(
                    f"⏹ Toplu yeniden dublaj kullanıcı tarafından durduruldu — "
                    f"o ana kadar tamamlanan {stats.synced}/{total_target} satır korundu."
                )
            else:
                report(100, 100, f"Tümünü Yeniden Dublajla tamamlandı ({stats.synced}/{total_target}).")
                log(
                    f"✅ Tümünü Yeniden Dublajla tamamlandı: {stats.synced}/{total_target} satır "
                    f"hazır (Hata: {stats.errors})."
                )
            return lines, stats, was_cancelled
        finally:
            # Yalnızca geçici çalışma klasörünü sil; tamamlanmış satırların önizlemelerini KESİNLİKLE koru!
            shutil.rmtree(work_dir, ignore_errors=True)
            try:
                self.vocal_separator.unload()
            except Exception:
                pass
            try:
                self.engine.unload(log_cb=log)
            except Exception:
                pass

    def convert_line_speech_to_speech(
        self,
        line: DialogueLine,
        user_acting_audio_path: Path,
        cancel_event: Optional[threading.Event] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> DialogueLine:
        """Kullanıcının mikrofondan kaydettiği veya dışarıdan yüklediği gerçek oyunculuk sesini
        Seed-VC (Sıfır Eğitimli STS) ile karakterin sesine dönüştürüp FFmpeg LUFS/senkron/arka-plan-miks
        hattından geçirir ve benzersiz önizleme dosyasına bağlar.
        """
        if not user_acting_audio_path.is_file():
            raise FileNotFoundError(
                f"Dönüştürülecek oyunculuk ses dosyası bulunamadı: {user_acting_audio_path}"
            )

        work_dir = make_temp_subdir("sts_donusum")
        tts_options = self.config.build_tts_options(acting_prompt=line.acting_prompt)
        local_sts_fallback = None

        def log(msg: str) -> None:
            if log_cb:
                log_cb(msg)

        try:
            self._check_cancel(cancel_event)
            if line.metadata is None:
                line.metadata = self.ffmpeg.probe_audio(line.original_path, measure_lufs=True)

            # 1. Kullanıcının oyunculuk kaydını nefes/duraksama ritmini bozmadan 48 kHz Mono WAV'a hazırla
            user_acting_48k = work_dir / f"{line.line_id}_user_acting_48k.wav"
            self.ffmpeg.convert_to_wav(
                user_acting_audio_path,
                user_acting_48k,
                sample_rate=48000,
                channels=1,
            )

            # 2. Hedef karakter referans sesini (48 kHz temiz Sabit Referans veya satırın kendi vokali) hazırla
            vocal_src, bg_wav, voc_lufs = self._separate_if_enabled(
                line, work_dir, cancel_event, log
            )
            fixed_ref = self._prepare_fixed_reference_if_set(
                work_dir, log, sample_rate=48000
            )
            if fixed_ref is not None:
                target_ref_wav = fixed_ref
            else:
                target_ref_wav = work_dir / f"{line.line_id}_target_char_48k.wav"
                self.ffmpeg.prepare_reference_wav(
                    vocal_src,
                    target_ref_wav,
                    sample_rate=48000,
                    max_duration_sec=20.0,
                    trim_edge_silence=True,
                )

            # 3. Seed-VC (veya aktif STS motoru) ile kullanıcının oyunculuğunu karakter sesine dönüştür
            raw_converted_wav = work_dir / f"{line.line_id}_sts_raw_converted.wav"
            steps = max(10, min(50, int(self.config.sts_diffusion_steps)))
            auto_f0 = bool(self.config.sts_auto_f0_adjust)
            semi_shift = max(-12, min(12, int(self.config.sts_pitch_shift)))

            log(
                f"🎙️ Seed-VC Ses Dönüştürme (STS) başlatıldı [{line.rel_path}] "
                f"(Adım: {steps}, Otomatik Perde: {'Açık' if auto_f0 else 'Kapalı'}, "
                f"Yarım Ton: {semi_shift:+d})..."
            )

            try:
                self.engine.convert_speech_to_speech(
                    source_acting_wav=user_acting_48k,
                    target_speaker_wav=target_ref_wav,
                    output_wav_path=raw_converted_wav,
                    diffusion_steps=steps,
                    auto_f0_adjust=auto_f0,
                    pitch_shift=semi_shift,
                    cancel_event=cancel_event,
                    options=tts_options,
                )
            except NotImplementedError:
                from engines.local_gpu_engine import LocalGPUEngine

                local_sts_fallback = LocalGPUEngine(
                    whisper_model_name=self.config.whisper_model_local,
                    unload_between_stages=True,
                    default_tts_engine=tts_options.tts_engine,
                )
                local_sts_fallback.convert_speech_to_speech(
                    source_acting_wav=user_acting_48k,
                    target_speaker_wav=target_ref_wav,
                    output_wav_path=raw_converted_wav,
                    diffusion_steps=steps,
                    auto_f0_adjust=auto_f0,
                    pitch_shift=semi_shift,
                    cancel_event=cancel_event,
                    options=tts_options,
                )

            # 4. Windows dosya kilidi (Permission denied) korumalı benzersiz önizleme dosyasına senkronize et
            old_preview = line.synced_preview_path
            rel_p = Path(line.rel_path)
            if old_preview is not None and old_preview.parent.exists():
                base_preview_dir = old_preview.parent
            else:
                preview_root = make_temp_subdir("onizleme")
                base_preview_dir = preview_root / rel_p.parent
            base_preview_dir.mkdir(parents=True, exist_ok=True)
            unique_tag = uuid.uuid4().hex[:8]
            preview_out = base_preview_dir / f"{rel_p.stem}_sts_{unique_tag}{rel_p.suffix}"

            # İnsan oyunculuğunun kendi nefes ve duraksama ritmini korumak için trim_silence=False
            synced_path, dub_ms, tempo = self.ffmpeg.sync_and_package_dub(
                dub_wav_path=raw_converted_wav,
                original_meta=line.metadata,
                output_path=preview_out,
                max_tempo_speedup=self.config.max_tempo_speedup,
                trim_silence=False,
                background_wav_path=bg_wav,
                bg_volume_db=self.config.bg_mix_volume_db,
                target_vocal_lufs=voc_lufs,
            )

            sts_cand = DubCandidate(
                candidate_index=1,
                raw_wav_path=None,
                synced_wav_path=synced_path,
                whisper_transcript=line.translated_text or "[Gerçek İnsan Oyunculuğu — STS]",
                text_accuracy_score=100.0,
                duration_fit_score=100.0,
                total_score=100.0,
                raw_duration_ms=dub_ms,
                synced_duration_ms=dub_ms,
                tempo_factor=tempo,
                is_best=True,
            )
            line.candidates = [sts_cand]
            line.selected_candidate_index = 1
            line.best_candidate_score = 100.0
            line.synced_preview_path = synced_path
            line.dub_duration_ms = dub_ms
            line.tempo_factor = tempo
            line.status = LineStatus.READY_PREVIEW
            line.error_message = ""
            if not line.translated_text.strip():
                line.translated_text = "[Gerçek İnsan Oyunculuğu — Seed-VC STS]"

            if old_preview is not None and old_preview != synced_path:
                try:
                    old_preview.unlink(missing_ok=True)
                except OSError:
                    pass

            log(
                f"✅ Gerçek oyunculuk sesi karakter sesine dönüştürüldü [{line.rel_path}] "
                f"({dub_ms}ms, tempo {tempo:.2f}x)."
            )
            return line
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            try:
                self.vocal_separator.unload()
            except Exception:
                pass
            if local_sts_fallback is not None:
                try:
                    local_sts_fallback.unload(log_cb=log_cb)
                except Exception:
                    pass
            try:
                self.engine.unload(log_cb=log_cb)
            except Exception:
                pass

    def apply_lines_to_game(
        self,
        game_dir: Path,
        project_name: str,
        lines: list[DialogueLine],
        discovered_map: Optional[dict[str, DiscoveredAudio]] = None,
        log_cb: Optional[Callable[[str], None]] = None,
        line_cb: Optional[Callable[[DialogueLine], None]] = None,
        cleanup_temp_after: bool = True,
    ) -> int:
        """Önizlemeye hazır tüm satırları SHA-256 doğrulamalı yedek alarak oyuna uygular."""
        ready = [
            ln
            for ln in lines
            if ln.synced_preview_path and ln.synced_preview_path.is_file()
        ]
        if not ready:
            return 0

        game_root = game_dir.resolve()
        safe_project = sanitize_project_name(project_name or game_root.name)

        items_by_line: list[tuple[DialogueLine, DiscoveredAudio]] = []
        backup_targets: list[Path] = []
        for ln in ready:
            item = (
                discovered_map[ln.line_id]
                if (discovered_map and ln.line_id in discovered_map)
                else DiscoveredAudio(
                    rel_path=ln.rel_path,
                    original_path=ln.original_path,
                    adapter_name=ln.adapter_name,
                )
            )
            items_by_line.append((ln, item))
            adapter = self.registry.get(item.adapter_name)
            for t in adapter.get_backup_targets([item]):
                if t not in backup_targets:
                    backup_targets.append(t)

        self.backup_manager.backup_files(
            project_name=safe_project,
            game_root=game_root,
            file_paths=backup_targets,
            log_cb=log_cb,
        )

        applied = 0
        for ln, item in items_by_line:
            assert ln.synced_preview_path is not None
            adapter = self.registry.get(item.adapter_name)
            adapter.apply_dubbed(item, ln.synced_preview_path)
            ln.applied = True
            ln.status = LineStatus.APPLIED
            applied += 1
            if line_cb:
                line_cb(ln)

        if log_cb:
            log_cb(f"{applied} Türkçe dublaj dosyası oyuna güvenle uygulandı (SHA-256 yedekli).")

        if cleanup_temp_after:
            deleted, failed = cleanup_temp_dir(keep_root=True)
            for ln in ready:
                if ln.synced_preview_path and not ln.synced_preview_path.exists():
                    ln.synced_preview_path = None
            if log_cb:
                if failed:
                    log_cb(f"Geçici temizlik: {deleted} öğe silindi, {len(failed)} dosya kilitli.")
                else:
                    log_cb(f"İşlem bitti: ./_temp klasöründeki {deleted} geçici öğe otomatik silindi.")

        return applied
