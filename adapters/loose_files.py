"""Gevşek Ses Dosyaları Adaptörü (.wav, .ogg, .mp3, .flac)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from adapters.base import GameAudioAdapter
from core.models import DiscoveredAudio


class LooseFilesAdapter(GameAudioAdapter):
    """Standart ses dosyalarını (.wav, .ogg, .mp3, .flac) özyinelemeli tarar ve uygular."""

    name = "loose_files"
    display_name = "Standart Ses Dosyaları (.wav, .ogg, .mp3, .flac)"
    supported_extensions = (".wav", ".ogg", ".mp3", ".flac")
    is_implemented = True

    IGNORED_DIR_NAMES = {
        "_temp",
        "_yedekler",
        "_modeller",
        "_veri",
        ".git",
        "__pycache__",
    }

    def can_handle(self, root_dir: Path) -> bool:
        return root_dir.is_dir()

    def discover(self, root_dir: Path) -> list[DiscoveredAudio]:
        root_resolved = root_dir.resolve()
        discovered: list[DiscoveredAudio] = []

        for dirpath, dirnames, filenames in os.walk(root_resolved):
            # Sistem/yedek klasörlerini pas geç
            dirnames[:] = sorted(
                d for d in dirnames if d.lower() not in self.IGNORED_DIR_NAMES
            )
            for fname in sorted(filenames):
                ext = Path(fname).suffix.lower()
                if ext not in self.supported_extensions:
                    continue
                full_path = Path(dirpath) / fname
                rel_path = full_path.relative_to(root_resolved).as_posix()
                discovered.append(
                    DiscoveredAudio(
                        rel_path=rel_path,
                        original_path=full_path,
                        adapter_name=self.name,
                    )
                )

        return discovered

    def apply_dubbed(self, item: DiscoveredAudio, dubbed_file: Path) -> Path:
        """Dublaj dosyasını atomik değişimle orijinal oyun dosyasının yerine yazar."""
        target = item.original_path
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.sesai_tmp")
        try:
            shutil.copy2(dubbed_file, staging)
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
        return target
