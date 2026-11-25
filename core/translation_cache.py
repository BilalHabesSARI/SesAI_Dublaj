"""Yerel SQLite Çeviri ve Oyunculuk Talimatı (`acting_prompt`) Önbelleği (Cache).

Aynı repliklerin Gemini API üzerinden tekrar tekrar çevrilmesini engeller;
hem Türkçe çeviriyi (`translated_text`) hem de Otomatik Dublaj Yönetmeni
oyunculuk talimatını (`acting_prompt`) `./_veri/ceviri_onbellek.db` içinde saklar.
Eski veritabanı şemalarıyla %100 geriye dönük uyumlu otomatik migration yapar.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.portable_env import DATA_DIR

DEFAULT_CACHE_DB_PATH: Path = DATA_DIR / "ceviri_onbellek.db"


class TranslationCache:
    """Thread-safe yerel SQLite çeviri ve oyunculuk talimatı önbelleği."""

    def __init__(self, db_path: Path = DEFAULT_CACHE_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.execute("PRAGMA journal_mode=DELETE;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    source_text TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    duration_bucket TEXT NOT NULL,
                    glossary_hash TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    acting_prompt TEXT NOT NULL DEFAULT '',
                    hits INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_text, source_lang, target_lang, duration_bucket, glossary_hash)
                )
                """
            )
            # Geriye dönük uyumlu migration: eski tabloda acting_prompt sütunu yoksa ekle
            cols = {
                str(r[1]).lower()
                for r in conn.execute("PRAGMA table_info(translations)").fetchall()
            }
            if "acting_prompt" not in cols:
                conn.execute(
                    "ALTER TABLE translations ADD COLUMN acting_prompt TEXT NOT NULL DEFAULT '';"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcriptions (
                    file_key TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    source_lang TEXT NOT NULL,
                    transcribed_text TEXT NOT NULL,
                    detected_lang TEXT NOT NULL DEFAULT 'en',
                    hits INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (file_key, file_size, mtime_ns, source_lang)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _file_signature(audio_path: Path) -> tuple[str, int, int]:
        """Dosya yolu + bayt boyutu + değiştirilme zamanı (mtime_ns) imzasını üretir."""
        resolved = str(audio_path.resolve()).replace("\\", "/").lower()
        st = audio_path.stat()
        size_b = int(st.st_size)
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000)))
        return resolved, size_b, mtime_ns

    def get_transcription(
        self,
        audio_path: Path,
        source_lang: str = "en",
    ) -> Optional[tuple[str, str]]:
        """Whisper deşifre önbelleğinden `(transcribed_text, detected_lang)` döndürür."""
        if not audio_path.is_file():
            return None
        try:
            file_key, file_size, mtime_ns = self._file_signature(audio_path)
        except OSError:
            return None

        lang_key = (source_lang or "en").strip().lower()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT transcribed_text, detected_lang FROM transcriptions
                WHERE file_key = ? AND file_size = ? AND mtime_ns = ? AND source_lang = ?
                """,
                (file_key, file_size, mtime_ns, lang_key),
            ).fetchone()
            if row is None:
                return None

            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE transcriptions SET hits = hits + 1, updated_at = ?
                WHERE file_key = ? AND file_size = ? AND mtime_ns = ? AND source_lang = ?
                """,
                (now_iso, file_key, file_size, mtime_ns, lang_key),
            )
            conn.commit()
            return str(row[0] or ""), str(row[1] or lang_key)

    def put_transcription(
        self,
        audio_path: Path,
        source_lang: str,
        transcribed_text: str,
        detected_lang: str = "en",
    ) -> None:
        """Whisper deşifre sonucunu `./_veri/ceviri_onbellek.db` içine kaydeder."""
        if not audio_path.is_file():
            return
        cleaned = (transcribed_text or "").strip()
        if not cleaned:
            return
        try:
            file_key, file_size, mtime_ns = self._file_signature(audio_path)
        except OSError:
            return

        lang_key = (source_lang or "en").strip().lower()
        det_lang = (detected_lang or lang_key).strip().lower()
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transcriptions (
                    file_key, file_size, mtime_ns, source_lang,
                    transcribed_text, detected_lang, hits, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(file_key, file_size, mtime_ns, source_lang)
                DO UPDATE SET
                    transcribed_text = excluded.transcribed_text,
                    detected_lang = excluded.detected_lang,
                    updated_at = excluded.updated_at
                """,
                (
                    file_key,
                    file_size,
                    mtime_ns,
                    lang_key,
                    cleaned,
                    det_lang,
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()

    def transcription_count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM transcriptions").fetchone()
            return int(row[0]) if row else 0

    @staticmethod
    def duration_bucket(duration_sec: float) -> str:
        """Süreyi 0.5 saniyelik kovalar halinde gruplar."""
        if duration_sec <= 0:
            return "0.0"
        bucket = max(0.5, round(duration_sec * 2.0) / 2.0)
        return f"{bucket:.1f}"

    def get_entry(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        duration_sec: float,
        glossary_hash: str,
    ) -> Optional[tuple[str, str]]:
        """Önbellekten `(translated_text, acting_prompt)` çiftini döndürür."""
        bucket = self.duration_bucket(duration_sec)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT translated_text, acting_prompt FROM translations
                WHERE source_text = ? AND source_lang = ? AND target_lang = ?
                  AND duration_bucket = ? AND glossary_hash = ?
                """,
                (source_text, source_lang, target_lang, bucket, glossary_hash),
            ).fetchone()

            if row is None:
                return None

            now_iso = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE translations SET hits = hits + 1, updated_at = ?
                WHERE source_text = ? AND source_lang = ? AND target_lang = ?
                  AND duration_bucket = ? AND glossary_hash = ?
                """,
                (now_iso, source_text, source_lang, target_lang, bucket, glossary_hash),
            )
            conn.commit()
            return str(row[0] or ""), str(row[1] or "")

    def get(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        duration_sec: float,
        glossary_hash: str,
    ) -> Optional[str]:
        """Geriye dönük uyumlu metin sorgusu."""
        entry = self.get_entry(source_text, source_lang, target_lang, duration_sec, glossary_hash)
        return entry[0] if entry is not None else None

    def put(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        duration_sec: float,
        glossary_hash: str,
        model_name: str,
        translated_text: str,
        acting_prompt: str = "",
    ) -> None:
        bucket = self.duration_bucket(duration_sec)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO translations (
                    source_text, source_lang, target_lang, duration_bucket,
                    glossary_hash, model_name, translated_text, acting_prompt,
                    hits, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(source_text, source_lang, target_lang, duration_bucket, glossary_hash)
                DO UPDATE SET
                    translated_text = excluded.translated_text,
                    acting_prompt = CASE
                        WHEN excluded.acting_prompt != '' THEN excluded.acting_prompt
                        ELSE translations.acting_prompt
                    END,
                    model_name = excluded.model_name,
                    updated_at = excluded.updated_at
                """,
                (
                    source_text,
                    source_lang,
                    target_lang,
                    bucket,
                    glossary_hash,
                    model_name,
                    translated_text,
                    (acting_prompt or "").strip(),
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM translations").fetchone()
            return int(row[0]) if row else 0

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM translations")
            conn.commit()
