"""SHA-256 Doğrulamalı Yedekleme ve Tek Tıkla Geri Alma Yöneticisi.

Kurallar:
- Oyunun orijinal ses dosyalarında değişiklik yapılmadan ÖNCE her dosyanın
  SHA-256 özeti hesaplanır, dosya `./_yedekler/<proje_adi>/dosyalar/...`
  altına kopyalanır ve `./_yedekler/<proje_adi>/manifest.json` dosyasına kaydedilir.
- Aynı dosya için mevcut bir orijinal yedek zaten varsa (örneğin kullanıcı ikinci
  kez dublaj çalıştırırsa), orijinal yedek KESİNLİKLE üzerine yazılmaz!
- "Orijinale Dön" çağrıldığında tüm orijinal dosyalar oyundaki yerlerine geri
  yüklenir, her dosyanın SHA-256 sağlamlık kontrolü yapılır ve YALNIZCA %100
  doğrulama sağlandığında ilgili yedek klasörü kendiliğinden silinir.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from core.models import BackupFileEntry, BackupManifest, RestoreResult
from core.portable_env import BACKUPS_DIR


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Bir dosyanın SHA-256 özetini akış (streaming) yöntemiyle hesaplar."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


GENERIC_AUDIO_SUBFOLDER_NAMES: frozenset[str] = frozenset(
    {
        "sesler",
        "ses",
        "audio",
        "audios",
        "sounds",
        "sound",
        "wav",
        "wavs",
        "voices",
        "voice",
        "dialog",
        "dialogue",
        "dialogues",
        "speech",
        "sfx",
        "media",
    }
)


def sanitize_project_name(name: str) -> str:
    """Klasör adı için güvenli proje adı üretir."""
    cleaned = re.sub(r"[^a-zA-Z0-9çğıöşüÇĞİÖŞÜ_\- ]+", "_", name.strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    return cleaned or "Varsayilan_Proje"


def is_generic_audio_subfolder(name: str) -> bool:
    """Klasör adının `sesler`, `audio`, `sounds`, `wav`, `voices` gibi genel alt klasör adı olup olmadığını döndürür."""
    return name.strip().lower() in GENERIC_AUDIO_SUBFOLDER_NAMES


def infer_smart_project_name(folder_path: str | Path) -> str:
    """Seçilen oyun/ses klasöründen akıllı proje / yedek adı üretir.

    Eğer seçilen klasör adı `sesler`, `audio`, `sounds`, `wav`, `voices` gibi genel bir
    alt klasör adıysa bir üst klasörün adını (ör. `Hitman_IlkSahne`) kullanır.
    """
    raw = str(folder_path or "").strip()
    if not raw:
        return "Varsayilan_Proje"
    p = Path(raw)
    leaf = p.name.strip()
    if leaf and is_generic_audio_subfolder(leaf):
        parent = p.parent
        parent_name = parent.name.strip()
        if parent_name and parent != p:
            return sanitize_project_name(parent_name)
    return sanitize_project_name(leaf)


class BackupManager:
    """./_yedekler/<proje_adi>/ altında SHA-256 doğrulamalı yedekleme yöneticisi."""

    def __init__(self, backups_root: Path = BACKUPS_DIR) -> None:
        self.backups_root = backups_root
        self.backups_root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_name: str) -> Path:
        return self.backups_root / sanitize_project_name(project_name)

    def manifest_path(self, project_name: str) -> Path:
        return self.project_dir(project_name) / "manifest.json"

    def list_projects(self) -> list[str]:
        """Mevcut yedekli projelerin adlarını döndürür."""
        if not self.backups_root.exists():
            return []
        projects: list[str] = []
        for entry in sorted(self.backups_root.iterdir()):
            if entry.is_dir() and (entry / "manifest.json").is_file():
                projects.append(entry.name)
        return projects

    def has_backup(self, project_name: str) -> bool:
        return self.manifest_path(project_name).is_file()

    def load_manifest(self, project_name: str) -> Optional[BackupManifest]:
        mpath = self.manifest_path(project_name)
        if not mpath.is_file():
            return None
        raw = json.loads(mpath.read_text(encoding="utf-8"))
        files: dict[str, BackupFileEntry] = {}
        for rel_key, item in (raw.get("files") or {}).items():
            files[rel_key] = BackupFileEntry(
                rel_path=item["rel_path"],
                original_abs_path=item["original_abs_path"],
                backup_rel_path=item["backup_rel_path"],
                sha256=item["sha256"],
                size_bytes=int(item["size_bytes"]),
            )
        return BackupManifest(
            project_name=raw["project_name"],
            game_root=raw["game_root"],
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
            files=files,
        )

    def _save_manifest(self, project_name: str, manifest: BackupManifest) -> Path:
        mpath = self.manifest_path(project_name)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "project_name": manifest.project_name,
            "game_root": manifest.game_root,
            "created_at": manifest.created_at,
            "updated_at": manifest.updated_at,
            "file_count": len(manifest.files),
            "files": {
                rel_key: {
                    "rel_path": entry.rel_path,
                    "original_abs_path": entry.original_abs_path,
                    "backup_rel_path": entry.backup_rel_path,
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                }
                for rel_key, entry in sorted(manifest.files.items())
            },
        }
        tmp = mpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(mpath)
        return mpath

    def backup_files(
        self,
        project_name: str,
        game_root: Path,
        file_paths: list[Path],
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> BackupManifest:
        """Verilen orijinal dosyaları SHA-256 doğrulamalı olarak yedekler.

        Daha önce yedeklenmiş bir dosya tekrar gelirse orijinal kopya korunur.
        """
        safe_name = sanitize_project_name(project_name)
        pdir = self.project_dir(safe_name)
        files_root = pdir / "dosyalar"
        files_root.mkdir(parents=True, exist_ok=True)

        now_iso = datetime.now(timezone.utc).isoformat()
        manifest = self.load_manifest(safe_name)
        if manifest is None:
            manifest = BackupManifest(
                project_name=safe_name,
                game_root=str(game_root.resolve()),
                created_at=now_iso,
                updated_at=now_iso,
                files={},
            )

        game_root_resolved = game_root.resolve()
        newly_backed_up = 0

        for src in file_paths:
            src_resolved = src.resolve()
            if not src_resolved.is_file():
                raise FileNotFoundError(f"Yedeklenecek kaynak dosya bulunamadı: {src_resolved}")

            try:
                rel_posix = src_resolved.relative_to(game_root_resolved).as_posix()
            except ValueError:
                rel_posix = src_resolved.name

            # Orijinal yedek zaten mevcut ve sağlamsa üzerine YAZMA!
            if rel_posix in manifest.files:
                existing = manifest.files[rel_posix]
                existing_backup = pdir / existing.backup_rel_path
                if existing_backup.is_file() and compute_sha256(existing_backup) == existing.sha256:
                    continue

            orig_hash = compute_sha256(src_resolved)
            orig_size = src_resolved.stat().st_size

            dst_backup = files_root / Path(rel_posix)
            dst_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_resolved, dst_backup)

            # Kopyalanan yedeğin SHA-256 bütünlüğünü anında doğrula
            copied_hash = compute_sha256(dst_backup)
            if copied_hash != orig_hash:
                dst_backup.unlink(missing_ok=True)
                raise IOError(
                    f"Yedek doğrulama hatası ({rel_posix}): kopya SHA-256 uyuşmuyor!"
                )

            manifest.files[rel_posix] = BackupFileEntry(
                rel_path=rel_posix,
                original_abs_path=str(src_resolved),
                backup_rel_path=dst_backup.relative_to(pdir).as_posix(),
                sha256=orig_hash,
                size_bytes=orig_size,
            )
            newly_backed_up += 1

        manifest.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_manifest(safe_name, manifest)

        if log_cb:
            log_cb(
                f"SHA-256 yedekleme tamamlandı: {newly_backed_up} yeni dosya yedeklendi "
                f"(toplam {len(manifest.files)} dosya) -> {self.manifest_path(safe_name)}"
            )
        return manifest

    def restore_project(
        self,
        project_name: str,
        log_cb: Optional[Callable[[str], None]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> RestoreResult:
        """Tüm orijinal dosyaları geri yükler, %100 SHA-256 doğrulama sonrası yedeği siler."""
        safe_name = sanitize_project_name(project_name)
        pdir = self.project_dir(safe_name)
        manifest = self.load_manifest(safe_name)
        if manifest is None:
            return RestoreResult(
                success=False,
                project_name=safe_name,
                total_files=0,
                restored_files=0,
                verified_files=0,
                backup_deleted=False,
                errors=[f"'{safe_name}' projesi için yedek manifestosu bulunamadı."],
            )

        entries = list(manifest.files.values())
        total = len(entries)
        restored = 0
        verified = 0
        errors: list[str] = []

        # 1. Aşama: Yedek klasöründeki tüm dosyaların bozulmadığını önceden doğrula
        for entry in entries:
            backup_file = pdir / entry.backup_rel_path
            if not backup_file.is_file():
                errors.append(f"Yedek dosyası eksik: {entry.rel_path}")
                continue
            b_hash = compute_sha256(backup_file)
            if b_hash != entry.sha256:
                errors.append(
                    f"Yedek dosyası bozulmuş (SHA-256 uyuşmazlığı): {entry.rel_path}"
                )

        if errors:
            if log_cb:
                log_cb("Yedek bütünlük kontrolü başarısız! Yedek klasörü korundu.")
            return RestoreResult(
                success=False,
                project_name=safe_name,
                total_files=total,
                restored_files=0,
                verified_files=0,
                backup_deleted=False,
                errors=errors,
            )

        # 2. Aşama: Orijinal konumlarına geri kopyala ve hedef dosyada SHA-256 doğrula
        for idx, entry in enumerate(entries, start=1):
            backup_file = pdir / entry.backup_rel_path
            target_file = Path(entry.original_abs_path)
            try:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup_file, target_file)
                restored += 1

                restored_hash = compute_sha256(target_file)
                if restored_hash == entry.sha256:
                    verified += 1
                else:
                    errors.append(
                        f"Geri yüklenen dosya SHA-256 doğrulamasından geçemedi: {entry.rel_path}"
                    )
            except Exception as exc:
                errors.append(f"Geri yükleme hatası ({entry.rel_path}): {exc}")

            if progress_cb:
                progress_cb(idx, total)

        # 3. Aşama: YALNIZCA %100 doğrulama başarılıysa yedek klasörünü otomatik sil
        backup_deleted = False
        all_ok = (len(errors) == 0) and (verified == total) and (restored == total)
        if all_ok:
            try:
                shutil.rmtree(pdir)
                backup_deleted = not pdir.exists()
                if log_cb:
                    log_cb(
                        f"Orijinale Dön başarılı: {verified}/{total} dosya SHA-256 ile "
                        "doğrulandı ve ilgili yedek klasörü otomatik olarak silindi."
                    )
            except Exception as exc:
                errors.append(f"Dosyalar doğrulandı ancak yedek klasörü silinemedi: {exc}")
        else:
            if log_cb:
                log_cb(
                    f"Geri yükleme kısmi/hatalı ({verified}/{total} doğrulandı). "
                    "Güvenlik amacıyla yedek klasörü SİLİNMEDİ."
                )

        return RestoreResult(
            success=all_ok and backup_deleted,
            project_name=safe_name,
            total_files=total,
            restored_files=restored,
            verified_files=verified,
            backup_deleted=backup_deleted,
            errors=errors,
        )
