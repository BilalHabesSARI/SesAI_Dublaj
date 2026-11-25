"""Arayüzü Kilitlemeden Çalışan Arka Plan İş Parçacıkları (QThread Workers)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from core.backup_manager import BackupManager
from core.hitman_manager import (
    HitmanCancelledError,
    HitmanExtractResult,
    HitmanSceneProfile,
    extract_scene_voices,
    merge_repack_and_install_scene,
    uninstall_scene_patch,
)
from core.models import DialogueLine, EngineHealth, PipelineStats, RestoreResult
from core.pipeline import DubbingPipeline, PipelineCancelledError
from engines.base import SpeechEngine
from engines.elevenlabs_engine import ElevenLabsEngine


class HealthCheckWorker(QThread):
    """Colab Bulut, Yerel CUDA GPU veya ElevenLabs API sağlık test işçisi."""

    finished_health = Signal(object)  # EngineHealth

    def __init__(self, engine: SpeechEngine) -> None:
        super().__init__()
        self.engine = engine

    def run(self) -> None:
        try:
            res = self.engine.health_check()
        except Exception as exc:
            res = EngineHealth(ok=False, mode=self.engine.mode, message=f"Test hatası: {exc}")
        self.finished_health.emit(res)


class ElevenLabsVoicesWorker(QThread):
    """ElevenLabs hesabındaki kullanılabilir sesleri arka planda yükler."""

    voices_loaded = Signal(list)  # list[tuple[str, str]]
    failed_signal = Signal(str)

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self.api_key = api_key

    def run(self) -> None:
        try:
            el = ElevenLabsEngine(api_key=self.api_key)
            voices = el.list_voices()
            self.voices_loaded.emit(voices)
        except Exception as exc:
            self.failed_signal.emit(str(exc))


class PipelineWorker(QThread):
    """6 aşamalı sinematik dublaj iş hattını arka planda yürüten işçi."""

    log_signal = Signal(str)
    progress_signal = Signal(int, int, str)
    discovered_signal = Signal(list)
    line_updated_signal = Signal(object)
    completed_signal = Signal(list, object)  # lines, PipelineStats
    cancelled_signal = Signal(str)
    failed_signal = Signal(str)

    def __init__(
        self,
        pipeline: DubbingPipeline,
        game_dir: Path,
        project_name: str,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.game_dir = game_dir
        self.project_name = project_name
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            lines, stats = self.pipeline.run(
                game_dir=self.game_dir,
                project_name=self.project_name,
                cancel_event=self.cancel_event,
                log_cb=self.log_signal.emit,
                progress_cb=self.progress_signal.emit,
                line_cb=self.line_updated_signal.emit,
                discovered_cb=self.discovered_signal.emit,
            )
            self.completed_signal.emit(lines, stats)
        except PipelineCancelledError as exc:
            self.cancelled_signal.emit(str(exc))
        except Exception as exc:
            self.failed_signal.emit(str(exc))


class SingleLineRedubWorker(QThread):
    """Düzenlenen tek bir satırı (metin + acting_prompt) yeniden seslendirip senkronize eden işçi."""

    log_signal = Signal(str)
    line_done_signal = Signal(object)
    failed_signal = Signal(str)

    def __init__(
        self,
        pipeline: DubbingPipeline,
        line: DialogueLine,
        new_text: str,
        new_acting_prompt: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.line = line
        self.new_text = new_text
        self.new_acting_prompt = new_acting_prompt
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            updated = self.pipeline.redub_single_line(
                line=self.line,
                new_turkish_text=self.new_text,
                new_acting_prompt=self.new_acting_prompt,
                cancel_event=self.cancel_event,
                log_cb=self.log_signal.emit,
            )
            self.line_done_signal.emit(updated)
        except Exception as exc:
            self.failed_signal.emit(str(exc))


class BatchRedubAllWorker(QThread):
    """Tablodaki tüm konuşma satırlarını Whisper/Gemini atlayarak yeniden dublajlayan işçi."""

    log_signal = Signal(str)
    progress_signal = Signal(int, int, str)
    line_updated_signal = Signal(object)
    completed_signal = Signal(list, object)  # lines, PipelineStats
    cancelled_signal = Signal(list, object, str)  # lines, PipelineStats, message
    failed_signal = Signal(str)

    def __init__(
        self,
        pipeline: DubbingPipeline,
        lines: list[DialogueLine],
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.lines = lines
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            updated_lines, stats, was_cancelled = self.pipeline.redub_all_lines(
                lines=self.lines,
                cancel_event=self.cancel_event,
                log_cb=self.log_signal.emit,
                progress_cb=self.progress_signal.emit,
                line_cb=self.line_updated_signal.emit,
            )
            if was_cancelled:
                self.cancelled_signal.emit(
                    updated_lines,
                    stats,
                    f"Toplu yeniden dublaj durduruldu ({stats.synced} tamamlanan satır korundu).",
                )
            else:
                self.completed_signal.emit(updated_lines, stats)
        except Exception as exc:
            self.failed_signal.emit(str(exc))


class SingleLineSTSWorker(QThread):
    """Mikrofon veya dosyadan gelen gerçek oyunculuk sesini Seed-VC ile karakter sesine dönüştüren işçi."""

    log_signal = Signal(str)
    line_done_signal = Signal(object)
    failed_signal = Signal(str)

    def __init__(
        self,
        pipeline: DubbingPipeline,
        line: DialogueLine,
        user_acting_audio_path: Path,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.line = line
        self.user_acting_audio_path = user_acting_audio_path
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            updated = self.pipeline.convert_line_speech_to_speech(
                line=self.line,
                user_acting_audio_path=self.user_acting_audio_path,
                cancel_event=self.cancel_event,
                log_cb=self.log_signal.emit,
            )
            self.line_done_signal.emit(updated)
        except Exception as exc:
            self.failed_signal.emit(str(exc))


class ApplyDubbingWorker(QThread):
    """Önizlenmiş Türkçe dublajları SHA-256 yedek alarak oyuna uygulayan işçi."""

    log_signal = Signal(str)
    line_updated_signal = Signal(object)
    done_signal = Signal(int)
    failed_signal = Signal(str)

    def __init__(
        self,
        pipeline: DubbingPipeline,
        game_dir: Path,
        project_name: str,
        lines: list[DialogueLine],
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.game_dir = game_dir
        self.project_name = project_name
        self.lines = lines

    def run(self) -> None:
        try:
            count = self.pipeline.apply_lines_to_game(
                game_dir=self.game_dir,
                project_name=self.project_name,
                lines=self.lines,
                log_cb=self.log_signal.emit,
                line_cb=self.line_updated_signal.emit,
                cleanup_temp_after=True,
            )
            self.done_signal.emit(count)
        except Exception as exc:
            self.failed_signal.emit(str(exc))


class RestoreOriginalWorker(QThread):
    """SHA-256 doğrulamalı Orijinale Dön işçisi."""

    log_signal = Signal(str)
    progress_signal = Signal(int, int)
    done_signal = Signal(object)  # RestoreResult

    def __init__(self, backup_manager: BackupManager, project_name: str) -> None:
        super().__init__()
        self.backup_manager = backup_manager
        self.project_name = project_name

    def run(self) -> None:
        res = self.backup_manager.restore_project(
            project_name=self.project_name,
            log_cb=self.log_signal.emit,
            progress_cb=self.progress_signal.emit,
        )
        self.done_signal.emit(res)


class HitmanExtractWorker(QThread):
    """HITMAN 3 RPKG arşivinden yeni `.WWES` konuşmalarını çıkarıp 48 kHz `.wav` üreten işçi."""

    log_signal = Signal(str)
    progress_signal = Signal(int, int, str)
    completed_signal = Signal(object)  # HitmanExtractResult
    cancelled_signal = Signal(str)
    failed_signal = Signal(str)

    def __init__(
        self,
        runtime_dir: Path,
        scene: HitmanSceneProfile,
        keyword: str,
        min_duration_sec: float,
        max_duration_sec: float,
        limit_count: int,
    ) -> None:
        super().__init__()
        self.runtime_dir = runtime_dir
        self.scene = scene
        self.keyword = keyword
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.limit_count = limit_count
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            res: HitmanExtractResult = extract_scene_voices(
                runtime_dir=self.runtime_dir,
                scene=self.scene,
                keyword=self.keyword,
                min_duration_sec=self.min_duration_sec,
                max_duration_sec=self.max_duration_sec,
                limit_count=self.limit_count,
                cancel_event=self.cancel_event,
                log_cb=self.log_signal.emit,
                progress_cb=self.progress_signal.emit,
            )
            self.completed_signal.emit(res)
        except HitmanCancelledError as exc:
            self.cancelled_signal.emit(str(exc))
        except Exception as exc:
            self.failed_signal.emit(str(exc))


class HitmanRepackInstallWorker(QThread):
    """Aynı sahnenin onaylı dublajlarını birleştirip `0xFFFF` Wwise Vorbis `.rpkg` kuran işçi."""

    log_signal = Signal(str)
    progress_signal = Signal(int, int, str)
    completed_signal = Signal(dict)  # report_data
    cancelled_signal = Signal(str)
    failed_signal = Signal(str)

    def __init__(
        self,
        runtime_dir: Path,
        scene: HitmanSceneProfile,
    ) -> None:
        super().__init__()
        self.runtime_dir = runtime_dir
        self.scene = scene
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            report = merge_repack_and_install_scene(
                runtime_dir=self.runtime_dir,
                scene=self.scene,
                cancel_event=self.cancel_event,
                log_cb=self.log_signal.emit,
                progress_cb=self.progress_signal.emit,
            )
            self.completed_signal.emit(report)
        except HitmanCancelledError as exc:
            self.cancelled_signal.emit(str(exc))
        except Exception as exc:
            self.failed_signal.emit(str(exc))


class HitmanUninstallWorker(QThread):
    """Seçili HITMAN 3 sahnesinin yamasını diğer sahnelere dokunmadan kaldıran işçi."""

    log_signal = Signal(str)
    progress_signal = Signal(int, int, str)
    completed_signal = Signal(dict)
    failed_signal = Signal(str)

    def __init__(
        self,
        runtime_dir: Path,
        scene: HitmanSceneProfile,
    ) -> None:
        super().__init__()
        self.runtime_dir = runtime_dir
        self.scene = scene

    def run(self) -> None:
        try:
            res = uninstall_scene_patch(
                runtime_dir=self.runtime_dir,
                scene=self.scene,
                log_cb=self.log_signal.emit,
                progress_cb=self.progress_signal.emit,
            )
            self.completed_signal.emit(res)
        except Exception as exc:
            self.failed_signal.emit(str(exc))
