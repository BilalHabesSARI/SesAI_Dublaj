"""Oyun Ses Kaynağı Adaptör Katmanı.

Mevcut gevşek ses dosyaları (.wav, .ogg, .mp3, .flac) yanında ileride
Wwise (.bnk/.wem), FMOD (.bank/.fsb) ve Unreal (.pak/.uasset) arşiv
adaptörlerinin iş hattına dokunmadan eklenebilmesi için tasarlanmıştır.
"""

from adapters.archive_stubs import (
    AdapterRegistry,
    FMODBankAdapter,
    UnrealAudioAdapter,
    WwiseBankAdapter,
    create_default_registry,
)
from adapters.base import GameAudioAdapter
from adapters.loose_files import LooseFilesAdapter

__all__ = [
    "GameAudioAdapter",
    "LooseFilesAdapter",
    "WwiseBankAdapter",
    "FMODBankAdapter",
    "UnrealAudioAdapter",
    "AdapterRegistry",
    "create_default_registry",
]
