"""Oyun Ses Kaynağı Adaptörü Soyut Temel Sınıfı."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from core.models import DiscoveredAudio


class GameAudioAdapter(ABC):
    """Tüm oyun ses kaynakları (gevşek dosyalar veya arşiv paketleri) için ortak arayüz."""

    name: str = "base"
    display_name: str = "Temel Adaptör"
    supported_extensions: tuple[str, ...] = ()
    is_implemented: bool = False

    @abstractmethod
    def can_handle(self, root_dir: Path) -> bool:
        """Verilen oyun dizininde bu adaptörün işleyebileceği içerik olup olmadığını kontrol eder."""

    @abstractmethod
    def discover(self, root_dir: Path) -> list[DiscoveredAudio]:
        """Oyun dizinindeki ses dosyalarını özyinelemeli olarak tarar ve listeler."""

    def extract_to_temp(self, item: DiscoveredAudio, temp_dir: Path) -> Path:
        """Arşiv içindeki bir sesi ./_temp altına çıkarır. Gevşek dosyalarda doğrudan orijinal yolu döner."""
        return item.original_path

    @abstractmethod
    def apply_dubbed(self, item: DiscoveredAudio, dubbed_file: Path) -> Path:
        """Hazırlanan Türkçe dublajlı ses dosyasını oyundaki hedef konumuna uygular."""

    def get_backup_targets(self, items: Sequence[DiscoveredAudio]) -> list[Path]:
        """SHA-256 yedeklemesi alınacak fiziksel dosyaların listesini döndürür.

        Arşiv adaptörlerinde bu metod, değiştirilecek .bnk/.bank/.pak kapsayıcılarını
        döndürerek arşiv düzeyinde SHA-256 güvenliği sağlar.
        """
        seen: set[Path] = set()
        targets: list[Path] = []
        for item in items:
            target = item.archive_container if item.archive_container else item.original_path
            resolved = target.resolve()
            if resolved not in seen:
                seen.add(resolved)
                targets.append(resolved)
        return targets
