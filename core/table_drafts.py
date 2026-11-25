"""Proje Bazlı Otomatik Tablo Taslağı Yöneticisi (./_veri/taslaklar/).

Kullanıcı tabloda herhangi bir "Türkçe Dublaj Metni" veya "Oyunculuk / Duygu"
hücresini elle düzenlediğinde (veya iş hattı satırları ürettiğinde) ilgili proje
için `./_veri/taslaklar/<proje_adi>.json` dosyasına atomik olarak kaydedilir.
Uygulama kapatılıp tekrar açıldığında düzenlemeler otomatik geri yüklenir.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from core.backup_manager import sanitize_project_name
from core.models import AudioMetadata, DialogueLine, LineStatus
from core.portable_env import DATA_DIR

DRAFTS_DIR: Path = DATA_DIR / "taslaklar"


def _normalize_dir_key(path_str: str | Path) -> str:
    raw = str(path_str or "").strip()
    if not raw:
        return ""
    try:
        resolved = str(Path(raw).resolve())
    except OSError:
        resolved = str(Path(raw))
    return os.path.normcase(os.path.normpath(resolved))


class TableDraftManager:
    """Proje bazlı tablo düzenleme taslaklarını `./_veri/taslaklar/` içinde saklar."""

    def __init__(self, drafts_dir: Path = DRAFTS_DIR) -> None:
        self.drafts_dir = Path(drafts_dir)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)

    def draft_path(self, project_name: str) -> Path:
        return self.drafts_dir / f"{sanitize_project_name(project_name)}.json"

    def has_draft(self, project_name: str) -> bool:
        if not project_name or not project_name.strip():
            return False
        return self.draft_path(project_name).is_file()

    def save_lines(
        self,
        project_name: str,
        game_dir: str | Path,
        lines: Sequence[DialogueLine],
    ) -> Path:
        """Tablodaki tüm satırları atomik olarak proje taslağına kaydeder."""
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        safe_proj = sanitize_project_name(project_name)
        target = self.draft_path(safe_proj)
        serialized_lines: list[dict] = []
        for ln in lines:
            meta_dict = None
            if ln.metadata is not None:
                meta_dict = {
                    "duration_ms": int(ln.metadata.duration_ms),
                    "lufs": float(ln.metadata.lufs),
                    "sample_rate": int(ln.metadata.sample_rate),
                    "channels": int(ln.metadata.channels),
                    "codec": str(ln.metadata.codec),
                    "format_ext": str(ln.metadata.format_ext),
                    "bit_rate": ln.metadata.bit_rate,
                    "sample_fmt": ln.metadata.sample_fmt,
                }
            serialized_lines.append(
                {
                    "line_id": ln.line_id,
                    "rel_path": ln.rel_path,
                    "original_path": str(ln.original_path),
                    "adapter_name": ln.adapter_name,
                    "is_speech": bool(ln.is_speech),
                    "source_text": ln.source_text or "",
                    "source_lang": ln.source_lang or "en",
                    "translated_text": ln.translated_text or "",
                    "acting_prompt": (ln.acting_prompt or "").strip().strip("()"),
                    "status": ln.status.value,
                    "applied": bool(ln.applied),
                    "metadata": meta_dict,
                }
            )

        payload = {
            "schema_version": 1,
            "project_name": safe_proj,
            "game_dir": str(game_dir or ""),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "lines": serialized_lines,
        }
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(target)
        return target

    def load_raw(self, project_name: str) -> Optional[dict]:
        if not project_name or not project_name.strip():
            return None
        path = self.draft_path(project_name)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("lines"), list):
                return data
        except Exception:
            return None
        return None

    def load_lines(
        self,
        project_name: str,
        expected_game_dir: Optional[str | Path] = None,
    ) -> list[DialogueLine]:
        """Kaydedilmiş proje taslağından mevcut dosyalar için DialogueLine listesini geri yükler."""
        raw = self.load_raw(project_name)
        if not raw:
            return []

        if expected_game_dir:
            saved_dir = raw.get("game_dir", "")
            if saved_dir and _normalize_dir_key(saved_dir) != _normalize_dir_key(expected_game_dir):
                return []

        restored: list[DialogueLine] = []
        for idx, item in enumerate(raw.get("lines", []), start=1):
            if not isinstance(item, dict):
                continue
            orig_str = str(item.get("original_path") or "").strip()
            rel_path = str(item.get("rel_path") or "").strip()
            orig_path = Path(orig_str) if orig_str else (
                (Path(str(expected_game_dir)) / rel_path) if (expected_game_dir and rel_path) else Path()
            )
            if not orig_str or not orig_path.is_file():
                continue

            meta: Optional[AudioMetadata] = None
            raw_meta = item.get("metadata")
            if isinstance(raw_meta, dict):
                try:
                    meta = AudioMetadata(
                        duration_ms=int(raw_meta.get("duration_ms", 0)),
                        lufs=float(raw_meta.get("lufs", -23.0)),
                        sample_rate=int(raw_meta.get("sample_rate", 48000)),
                        channels=int(raw_meta.get("channels", 1)),
                        codec=str(raw_meta.get("codec", "pcm_s16le")),
                        format_ext=str(raw_meta.get("format_ext", orig_path.suffix.lower() or ".wav")),
                        bit_rate=raw_meta.get("bit_rate"),
                        sample_fmt=raw_meta.get("sample_fmt"),
                    )
                except Exception:
                    meta = None

            raw_status = str(item.get("status") or LineStatus.PENDING.value)
            try:
                status = LineStatus(raw_status)
            except ValueError:
                status = LineStatus.PENDING
            # Uygulama yeniden açıldığında ./_temp temizlenmiş olabileceğinden geçici durumları güvenli hale getir
            if status not in (
                LineStatus.SKIPPED_NON_SPEECH,
                LineStatus.SKIPPED_EMPTY,
                LineStatus.APPLIED,
            ):
                status = LineStatus.PENDING

            restored.append(
                DialogueLine(
                    line_id=str(item.get("line_id") or f"line_{idx:04d}"),
                    rel_path=rel_path or orig_path.name,
                    original_path=orig_path,
                    adapter_name=str(item.get("adapter_name") or "loose_files"),
                    metadata=meta,
                    is_speech=bool(item.get("is_speech", True)),
                    source_text=str(item.get("source_text") or ""),
                    source_lang=str(item.get("source_lang") or "en"),
                    translated_text=str(item.get("translated_text") or ""),
                    acting_prompt=str(item.get("acting_prompt") or "").strip().strip("()"),
                    status=status,
                    applied=bool(item.get("applied", False)),
                )
            )
        return restored

    def get_overrides_by_rel_path(
        self,
        project_name: str,
        expected_game_dir: Optional[str | Path] = None,
    ) -> dict[str, dict[str, str]]:
        """Kaydedilmiş taslaktaki metin ve oyunculuk/duygu değerlerini `rel_path` anahtarıyla döndürür."""
        raw = self.load_raw(project_name)
        if not raw:
            return {}
        if expected_game_dir:
            saved_dir = raw.get("game_dir", "")
            if saved_dir and _normalize_dir_key(saved_dir) != _normalize_dir_key(expected_game_dir):
                return {}
        overrides: dict[str, dict[str, str]] = {}
        for item in raw.get("lines", []):
            if not isinstance(item, dict):
                continue
            rel = str(item.get("rel_path") or "").strip().replace("\\", "/").lower()
            if not rel:
                continue
            tr_txt = str(item.get("translated_text") or "").strip()
            act_txt = str(item.get("acting_prompt") or "").strip().strip("()")
            if tr_txt or act_txt:
                overrides[rel] = {
                    "translated_text": tr_txt,
                    "acting_prompt": act_txt,
                }
        return overrides
