"""Gelecekteki Oyun Ses Arşiv Açıcıları (Wwise / FMOD / Unreal) İçin Genişletme Katmanı.

Bu modül, ileride Wwise (.bnk/.wem), FMOD (.bank/.fsb) veya Unreal Engine
(.pak/.uasset/.ubulk) ses arşivlerini açıp paketleyen eklentilerin çekirdek
iş hattını değiştirmeden takılabilmesi için temiz bir adaptör kayıt defteri sunar.
"""

from __future__ import annotations

from pathlib import Path

from adapters.base import GameAudioAdapter
from adapters.loose_files import LooseFilesAdapter
from core.models import DiscoveredAudio


class WwiseBankAdapter(GameAudioAdapter):
    """Wwise (.bnk / .pck / .wem) ses bankaları için hazır genişletme yuvası."""

    name = "wwise_bank"
    display_name = "Wwise Ses Arşivi (.bnk / .pck / .wem) [Planlanan]"
    supported_extensions = (".bnk", ".pck", ".wem")
    is_implemented = False

    def can_handle(self, root_dir: Path) -> bool:
        return False

    def discover(self, root_dir: Path) -> list[DiscoveredAudio]:
        return []

    def apply_dubbed(self, item: DiscoveredAudio, dubbed_file: Path) -> Path:
        raise NotImplementedError(
            "Wwise (.bnk/.wem) paketleme adaptörü henüz etkin değil; vgmstream/wwiser "
            "entegre edildiğinde doğrudan bu sınıfa bağlanabilir."
        )


class FMODBankAdapter(GameAudioAdapter):
    """FMOD Studio (.bank / .fsb) ses bankaları için hazır genişletme yuvası."""

    name = "fmod_bank"
    display_name = "FMOD Ses Arşivi (.bank / .fsb) [Planlanan]"
    supported_extensions = (".bank", ".fsb")
    is_implemented = False

    def can_handle(self, root_dir: Path) -> bool:
        return False

    def discover(self, root_dir: Path) -> list[DiscoveredAudio]:
        return []

    def apply_dubbed(self, item: DiscoveredAudio, dubbed_file: Path) -> Path:
        raise NotImplementedError(
            "FMOD (.bank/.fsb) adaptörü henüz etkin değil; FSBExtractor/FMOD araçları "
            "eklendiğinde bu sınıf üzerinden otomatik devreye alınabilir."
        )


class UnrealAudioAdapter(GameAudioAdapter):
    """Unreal Engine (.pak / .uasset / .ubulk) ses paketleri için hazır genişletme yuvası."""

    name = "unreal_audio"
    display_name = "Unreal Engine Ses Paketi (.pak / .uasset) [Planlanan]"
    supported_extensions = (".pak", ".uasset", ".ubulk")
    is_implemented = False

    def can_handle(self, root_dir: Path) -> bool:
        return False

    def discover(self, root_dir: Path) -> list[DiscoveredAudio]:
        return []

    def apply_dubbed(self, item: DiscoveredAudio, dubbed_file: Path) -> Path:
        raise NotImplementedError(
            "Unreal Engine ses adaptörü henüz etkin değil; retoc/repak araçları "
            "eklendiğinde bu sınıf üzerinden iş hattına bağlanabilir."
        )


class AdapterRegistry:
    """Aktif ve planlanan oyun ses adaptörlerini yöneten kayıt defteri."""

    def __init__(self) -> None:
        self._adapters: dict[str, GameAudioAdapter] = {}

    def register(self, adapter: GameAudioAdapter) -> None:
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> GameAudioAdapter:
        if name not in self._adapters:
            raise KeyError(f"Bilinmeyen ses adaptörü: {name}")
        return self._adapters[name]

    def active_adapters(self) -> list[GameAudioAdapter]:
        return [a for a in self._adapters.values() if a.is_implemented]

    def all_adapters(self) -> list[GameAudioAdapter]:
        return list(self._adapters.values())

    def discover_all(self, root_dir: Path) -> list[DiscoveredAudio]:
        """Etkin tüm adaptörlerle oyun klasörünü tarar."""
        results: list[DiscoveredAudio] = []
        for adapter in self.active_adapters():
            if adapter.can_handle(root_dir):
                results.extend(adapter.discover(root_dir))
        return results


def create_default_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(LooseFilesAdapter())
    registry.register(WwiseBankAdapter())
    registry.register(FMODBankAdapter())
    registry.register(UnrealAudioAdapter())
    return registry
