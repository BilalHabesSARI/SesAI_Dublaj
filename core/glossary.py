"""Büyük/Küçük Harf Duyarlı Kalıcı Sözlük (Glossary) Yöneticisi.

Sözlük verileri yalnızca ./_veri/sozluk.json dosyasında tutulur.
Eşleştirmeler kesin olarak büyük/küçük harf duyarlıdır (case-sensitive):
örneğin 'Shade' ile 'shade' ayrı terimler olarak değerlendirilir.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from core.portable_env import DATA_DIR

DEFAULT_GLOSSARY_PATH: Path = DATA_DIR / "sozluk.json"


@dataclass
class GlossaryEntry:
    """Tek bir büyük/küçük harf duyarlı sözlük girdisi."""

    source: str
    target: str
    note: str = ""


class GlossaryManager:
    """Kalıcı, büyük/küçük harf duyarlı oyun terminoloji sözlüğü."""

    def __init__(self, path: Path = DEFAULT_GLOSSARY_PATH) -> None:
        self.path = path
        self._entries: dict[str, GlossaryEntry] = {}
        self.load()

    def load(self) -> list[GlossaryEntry]:
        self._entries.clear()
        if not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            items = raw.get("entries", []) if isinstance(raw, dict) else raw
            for item in items:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("source", "")).strip()
                tgt = str(item.get("target", "")).strip()
                note = str(item.get("note", "")).strip()
                if src and tgt:
                    # Anahtar birebir case-sensitive saklanır
                    self._entries[src] = GlossaryEntry(source=src, target=tgt, note=note)
        except (OSError, json.JSONDecodeError):
            self._entries.clear()
        return self.list_entries()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_sensitive": True,
            "entries": [asdict(e) for e in self.list_entries()],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def list_entries(self) -> list[GlossaryEntry]:
        # Uzun terimleri önce döndürmek eşleştirme ve görünüm tutarlılığı sağlar
        return sorted(self._entries.values(), key=lambda e: (-len(e.source), e.source))

    def upsert(self, source: str, target: str, note: str = "") -> GlossaryEntry:
        src = source.strip()
        tgt = target.strip()
        if not src or not tgt:
            raise ValueError("Kaynak terim ve Türkçe karşılık boş olamaz.")
        entry = GlossaryEntry(source=src, target=tgt, note=note.strip())
        self._entries[src] = entry
        self.save()
        return entry

    def remove(self, source: str) -> bool:
        if source in self._entries:
            del self._entries[source]
            self.save()
            return True
        return False

    def clear(self) -> None:
        self._entries.clear()
        self.save()

    def replace_all(self, entries: Iterable[GlossaryEntry]) -> None:
        self._entries.clear()
        for e in entries:
            src = e.source.strip()
            tgt = e.target.strip()
            if src and tgt:
                self._entries[src] = GlossaryEntry(source=src, target=tgt, note=e.note.strip())
        self.save()

    def find_matches(self, text: str) -> list[GlossaryEntry]:
        """Metin içinde geçen sözlük terimlerini BÜYÜK/KÜÇÜK HARF DUYARLI olarak bulur."""
        if not text or not self._entries:
            return []
        matched: list[GlossaryEntry] = []
        for entry in self.list_entries():
            src = entry.source
            # Kelime karakteri ile başlayıp biten terimlerde tam kelime sınırı uygula;
            # noktalama/özel karakter içeren terimlerde kesin alt dize eşleşmesi yap.
            if re.match(r"^\w.*\w$|^\w$", src):
                pattern = r"(?<!\w)" + re.escape(src) + r"(?!\w)"
                if re.search(pattern, text):  # re.IGNORECASE YOK -> tamamen case-sensitive!
                    matched.append(entry)
            else:
                if src in text:
                    matched.append(entry)
        return matched

    def format_prompt_block(self, matched_entries: list[GlossaryEntry]) -> str:
        """Gemini istemi (prompt) için sözlük kuralları bloğu oluşturur."""
        if not matched_entries:
            return ""
        lines = [
            "ZORUNLU SÖZLÜK (Büyük/küçük harf duyarlıdır — aşağıdaki terimleri TAM olarak belirtilen Türkçe karşılıkla kullan):"
        ]
        for entry in matched_entries:
            if entry.note:
                lines.append(f'- "{entry.source}" -> "{entry.target}" (Not: {entry.note})')
            else:
                lines.append(f'- "{entry.source}" -> "{entry.target}"')
        return "\n".join(lines)

    def signature_hash(self, entries: list[GlossaryEntry] | None = None) -> str:
        """Önbellek anahtarı için eşleşen sözlük kurallarının deterministik özetini üretir."""
        target_entries = entries if entries is not None else self.list_entries()
        canonical = "|".join(
            f"{e.source}=>{e.target}" for e in sorted(target_entries, key=lambda x: x.source)
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
