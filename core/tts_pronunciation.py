"""TTS Telaffuz Sözlüğü (Kısaltma ve Özel Okunuş Eşleştirme Katmanı).

Tablodaki Türkçe dublaj metni okunaklı haliyle (`ICA`) kalırken, yalnızca TTS
seslendirme motoruna (VoxCPM2, Trendyol-TTS, Chatterbox, XTTS, ElevenLabs)
gönderilen metin üzerinde tam kelime eşleşmesiyle telaffuz dönüşümü uygular.

Veriler yalnızca proje içindeki `./_veri/tts_telaffuz.json` dosyasında saklanır.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from core.portable_env import DATA_DIR

DEFAULT_TTS_PRONUNCIATION_PATH: Path = DATA_DIR / "tts_telaffuz.json"


@dataclass
class PronunciationRule:
    """Tek bir TTS telaffuz dönüşüm kuralı (örn. 'ICA' -> 'Ay-Si-Ey')."""

    source: str
    spoken: str
    note: str = ""


DEFAULT_PRONUNCIATION_RULES: tuple[PronunciationRule, ...] = (
    PronunciationRule(
        source="ICA",
        spoken="Ay-Si-Ey",
        note="Varsayılan kısaltma okunuşu (tablo metni 'ICA' kalır)",
    ),
)


class TTSPronunciationManager:
    """./_veri/tts_telaffuz.json üzerinde kalıcı TTS telaffuz eşleştirme yöneticisi."""

    def __init__(self, path: Path = DEFAULT_TTS_PRONUNCIATION_PATH) -> None:
        self.path = Path(path)
        self._rules: dict[str, PronunciationRule] = {}
        self.load()

    def load(self) -> list[PronunciationRule]:
        self._rules.clear()
        if not self.path.is_file():
            # İlk açılışta varsayılan başlangıç kurallarını (ICA -> Ay-Si-Ey) oluştur
            for rule in DEFAULT_PRONUNCIATION_RULES:
                self._rules[rule.source] = PronunciationRule(
                    source=rule.source,
                    spoken=rule.spoken,
                    note=rule.note,
                )
            try:
                self.save()
            except OSError:
                pass
            return self.list_rules()

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            items = raw.get("entries", []) if isinstance(raw, dict) else raw
            for item in items:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("source", "")).strip()
                spoken = str(item.get("spoken", item.get("target", ""))).strip()
                note = str(item.get("note", "")).strip()
                if src and spoken:
                    self._rules[src] = PronunciationRule(source=src, spoken=spoken, note=note)
        except (OSError, json.JSONDecodeError):
            self._rules.clear()
        return self.list_rules()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "case_sensitive": True,
            "whole_word": True,
            "entries": [asdict(r) for r in self.list_rules()],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def list_rules(self) -> list[PronunciationRule]:
        return sorted(self._rules.values(), key=lambda r: (-len(r.source), r.source))

    def upsert(self, source: str, spoken: str, note: str = "") -> PronunciationRule:
        src = source.strip()
        spk = spoken.strip()
        if not src or not spk:
            raise ValueError("Yazılı terim ve TTS okunuşu boş olamaz.")
        rule = PronunciationRule(source=src, spoken=spk, note=note.strip())
        self._rules[src] = rule
        self.save()
        return rule

    def remove(self, source: str) -> bool:
        if source in self._rules:
            del self._rules[source]
            self.save()
            return True
        return False

    def replace_all(self, rules: Iterable[PronunciationRule]) -> None:
        self._rules.clear()
        for r in rules:
            src = r.source.strip()
            spk = r.spoken.strip()
            if src and spk:
                self._rules[src] = PronunciationRule(source=src, spoken=spk, note=r.note.strip())
        self.save()

    def apply(self, text: str) -> str:
        """Yalnızca TTS motoruna gönderilen metin üzerinde tam kelime eşleşmeli telaffuz dönüşümü yapar.

        Orijinal tablo metnine dokunmaz; noktalama ve kesme işaretli ekleri (örn. `ICA'nın`) korur.
        """
        if not text or not self._rules:
            return text
        result = text
        for rule in self.list_rules():
            src = rule.source
            spk = rule.spoken
            if re.match(r"^\w.*\w$|^\w$", src):
                pattern = r"(?<!\w)" + re.escape(src) + r"(?!\w)"
                result = re.sub(pattern, lambda _m, s=spk: s, result)
            else:
                result = result.replace(src, spk)
        return result
