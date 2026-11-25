"""SHA-256 doğrulamalı yedekleme ve tek tıkla Orijinale Dön testleri."""

from __future__ import annotations

from pathlib import Path

from core.backup_manager import BackupManager, compute_sha256


def test_backup_restore_and_auto_delete(tmp_path: Path) -> None:
    game_dir = tmp_path / "OyunSesleri"
    (game_dir / "dialog" / "hero").mkdir(parents=True)
    f1 = game_dir / "dialog" / "hero" / "line1.wav"
    f2 = game_dir / "dialog" / "line2.ogg"
    orig1 = b"ORIGINAL_AUDIO_BYTES_LINE_ONE_123456"
    orig2 = b"ORIGINAL_AUDIO_BYTES_LINE_TWO_ABCDEF"
    f1.write_bytes(orig1)
    f2.write_bytes(orig2)

    h1 = compute_sha256(f1)
    h2 = compute_sha256(f2)

    backups_root = tmp_path / "_yedekler"
    bm = BackupManager(backups_root=backups_root)

    # 1. Yedek al
    manifest = bm.backup_files("TestOyunu", game_dir, [f1, f2])
    assert len(manifest.files) == 2
    assert bm.manifest_path("TestOyunu").is_file()
    assert manifest.files["dialog/hero/line1.wav"].sha256 == h1
    assert manifest.files["dialog/line2.ogg"].sha256 == h2

    # 2. Oyundaki dosyaları dublajla değiştir
    f1.write_bytes(b"DUBBED_TURKISH_AUDIO_1")
    f2.write_bytes(b"DUBBED_TURKISH_AUDIO_2")

    # 3. İkinci kez backup_files çağrılsa bile orijinal yedeği ASLA üzerine yazmamalı!
    manifest2 = bm.backup_files("TestOyunu", game_dir, [f1, f2])
    assert manifest2.files["dialog/hero/line1.wav"].sha256 == h1
    assert manifest2.files["dialog/line2.ogg"].sha256 == h2

    # 4. Orijinale Dön -> %100 SHA-256 doğrulama + yedek klasörü otomatik silinmeli
    res = bm.restore_project("TestOyunu")
    assert res.success is True
    assert res.verified_files == 2
    assert res.restored_files == 2
    assert res.backup_deleted is True
    assert not bm.project_dir("TestOyunu").exists()
    assert f1.read_bytes() == orig1
    assert f2.read_bytes() == orig2
    assert compute_sha256(f1) == h1
    assert compute_sha256(f2) == h2


def test_restore_aborts_and_preserves_backup_if_corrupted(tmp_path: Path) -> None:
    game_dir = tmp_path / "Oyun2"
    game_dir.mkdir()
    f1 = game_dir / "voice.wav"
    f1.write_bytes(b"GENUINE_VOICE_DATA")

    bm = BackupManager(backups_root=tmp_path / "_yedekler")
    manifest = bm.backup_files("BozukYedekTest", game_dir, [f1])
    entry = manifest.files["voice.wav"]

    # Yedek kopyasını boz -> geri yükleme reddedilmeli ve yedek silinmemeli
    backup_copy = bm.project_dir("BozukYedekTest") / entry.backup_rel_path
    backup_copy.write_bytes(b"CORRUPTED_BYTES")

    res = bm.restore_project("BozukYedekTest")
    assert res.success is False
    assert res.backup_deleted is False
    assert bm.project_dir("BozukYedekTest").exists()
    assert len(res.errors) >= 1
