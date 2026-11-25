"""HITMAN 3 (RPKG) Oyun Modülü — Keşif, Çıkarma, Çoklu Paket Birleştirme, 0xFFFF Kurulum, Kaldırma ve Arayüz Testleri.

Tüm yazma testleri %100 izole `tmp_path` içinde çalışır; oyunun gerçek
`C:\\Program Files\\Epic Games\\HITMAN3\\Runtime` klasörüne KESİNLİKLE dokunmaz.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from core.backup_manager import compute_sha256
from core.config import ConfigManager
from core.gemini_models import GeminiQuotaManager
from core.hitman_manager import (
    DEFAULT_HITMAN_RUNTIME_DIR,
    HitmanExtractResult,
    allocate_next_package_dir,
    build_rpkgv2_meta_bytes,
    collect_existing_scene_hashes,
    collect_merged_approved_dubs,
    discover_hitman_scenes,
    extract_scene_voices,
    get_scenes_runtime_status,
    merge_repack_and_install_scene,
    uninstall_scene_patch,
)
from core.portable_env import APP_ROOT, TEMP_DIR, bootstrap_portable_environment
from tools.hitman_bridge import _inspect_packagedefinition_partitions
from tools.hitman_repack import CHUNK0_RPKGV2_META_BYTES, CHUNK29_RPKGV2_META_BYTES


def test_discover_hitman_scenes_presets_and_dynamic_chunks(tmp_path: Path) -> None:
    rt = tmp_path / "Runtime"
    rt.mkdir()
    for name in (
        "chunk0.rpkg",
        "chunk0patch1.rpkg",
        "chunk0patch4.rpkg",
        "chunk0patch5.rpkg",
        "chunk29.rpkg",
        "chunk29patch1.rpkg",
        "chunk29patch2.rpkg",
        "chunk29patch3.rpkg",
        "chunk22.rpkg",
        "chunk22patch4.rpkg",
        "chunk27.rpkg",
        "chunk27patch4.rpkg",
    ):
        (rt / name).write_bytes(b"2KPR")

    assert build_rpkgv2_meta_bytes(0) == CHUNK0_RPKGV2_META_BYTES
    assert build_rpkgv2_meta_bytes(29) == CHUNK29_RPKGV2_META_BYTES

    scenes = discover_hitman_scenes(rt, state_path=tmp_path / "state.json")
    keys = [s.scene_key for s in scenes]
    assert keys[:2] == ["prologue", "safehouse"]
    assert "chunk22" in keys
    assert "chunk27" in keys

    c22 = next(s for s in scenes if s.scene_key == "chunk22")
    assert c22.partition_name == "hokkaido"
    assert c22.patch_rpkg_stem == "chunk22patch5"

    c27 = next(s for s in scenes if s.scene_key == "chunk27")
    assert c27.partition_name == "paris"
    assert c27.patch_rpkg_stem == "chunk27patch5"


def test_collect_existing_hashes_and_multi_package_merge_newest_wins(tmp_path: Path) -> None:
    rt = tmp_path / "Runtime"
    rt.mkdir()
    scenes = discover_hitman_scenes(rt, state_path=tmp_path / "state.json")
    safehouse = next(s for s in scenes if s.scene_key == "safehouse")

    app_root = tmp_path / "app"
    pkg1 = app_root / "Hitman_SerbestEv"
    pkg2 = app_root / "Hitman_SerbestEv_Paket2"
    for p in (pkg1, pkg2):
        (p / "sesler").mkdir(parents=True)
        (p / "ham_wwes").mkdir(parents=True)

    # Paket 1: H1 (eski dublajlı), H2 (dublajlanmamış — orijinal SHA ile aynı!)
    h1_old_wav = pkg1 / "sesler" / "01_a_0000000000000001.wav"
    h2_orig_wav = pkg1 / "sesler" / "02_b_0000000000000002.wav"
    h1_old_wav.write_bytes(b"DUB_H1_OLD")
    h2_orig_wav.write_bytes(b"ORIG_H2")
    for h in ("0000000000000001", "0000000000000002"):
        (pkg1 / "ham_wwes" / f"{h}.WWES").write_bytes(b"WWES")
        (pkg1 / "ham_wwes" / f"{h}.WWES.meta").write_bytes(b"\x00" * 44)
    (pkg1 / "hitman_manifest.json").write_text(
        json.dumps(
            {
                "scene_key": "safehouse",
                "chunk_number": 29,
                "files": [
                    {
                        "hash_hex": "0000000000000001",
                        "wav_filename": h1_old_wav.name,
                        "wav_sha256": "orig_sha_different_from_dub",
                        "channels": 1,
                    },
                    {
                        "hash_hex": "0000000000000002",
                        "wav_filename": h2_orig_wav.name,
                        "wav_sha256": compute_sha256(h2_orig_wav),
                        "channels": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # Paket 2: H1 (daha yeni dublajlı — çakışmada kazanmalı!), H3 (yeni dublajlı)
    h1_new_wav = pkg2 / "sesler" / "01_a_0000000000000001.wav"
    h3_new_wav = pkg2 / "sesler" / "02_c_0000000000000003.wav"
    h1_new_wav.write_bytes(b"DUB_H1_NEW_VERSION")
    h3_new_wav.write_bytes(b"DUB_H3")
    now = time.time()
    os.utime(h1_old_wav, (now - 120, now - 120))
    os.utime(h1_new_wav, (now + 10, now + 10))
    for h in ("0000000000000001", "0000000000000003"):
        (pkg2 / "ham_wwes" / f"{h}.WWES").write_bytes(b"WWES")
        (pkg2 / "ham_wwes" / f"{h}.WWES.meta").write_bytes(b"\x00" * 44)
    (pkg2 / "hitman_manifest.json").write_text(
        json.dumps(
            {
                "scene_key": "safehouse",
                "chunk_number": 29,
                "files": [
                    {
                        "hash_hex": "0000000000000001",
                        "wav_filename": h1_new_wav.name,
                        "wav_sha256": "orig_sha_different_from_dub",
                        "channels": 1,
                    },
                    {
                        "hash_hex": "0000000000000003",
                        "wav_filename": h3_new_wav.name,
                        "wav_sha256": "orig_sha_h3",
                        "channels": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    existing = collect_existing_scene_hashes(app_root, safehouse)
    assert existing == {"0000000000000001", "0000000000000002", "0000000000000003"}

    next_dir = allocate_next_package_dir(app_root, safehouse)
    assert next_dir.name == "Hitman_SerbestEv_Paket3"

    merged, pkg_dirs, total_ext = collect_merged_approved_dubs(app_root, safehouse)
    assert len(pkg_dirs) == 2
    assert total_ext == 3
    by_hash = {m.hash_hex: m for m in merged}
    # H2 dublajlanmadığı için hariç tutulmalı; H1 Paket2'den (en yeni) alınmalı
    assert set(by_hash.keys()) == {"0000000000000001", "0000000000000003"}
    assert by_hash["0000000000000001"].source_package_dir == pkg2
    assert by_hash["0000000000000001"].wav_path == h1_new_wav


def test_read_only_extract_excludes_previous_hashes_and_cleans_temp(tmp_path: Path) -> None:
    bootstrap_portable_environment()
    real_rt = Path(DEFAULT_HITMAN_RUNTIME_DIR)
    if not (real_rt / "chunk0.rpkg").is_file():
        return

    scenes = discover_hitman_scenes(real_rt, state_path=tmp_path / "state.json")
    prologue = next(s for s in scenes if s.scene_key == "prologue")
    sandbox_root = tmp_path / "sandbox_app"
    sandbox_root.mkdir()

    # 1. Çıkarma (2 adet Diana sesi -> Hitman_IlkSahne)
    res1 = extract_scene_voices(
        runtime_dir=real_rt,
        scene=prologue,
        keyword="diana",
        min_duration_sec=1.0,
        max_duration_sec=12.0,
        limit_count=2,
        app_root=sandbox_root,
        state_path=tmp_path / "state.json",
    )
    assert res1.extracted_count == 2
    assert res1.package_dir.name == "Hitman_IlkSahne"
    assert res1.manifest_path.is_file()
    m1 = json.loads(res1.manifest_path.read_text(encoding="utf-8"))
    h1_set = {f["hash_hex"] for f in m1["files"]}
    assert len(h1_set) == 2
    for f in m1["files"]:
        assert f["sample_rate"] == 48000
        assert (res1.wav_dir / f["wav_filename"]).is_file()
        assert (res1.raw_wwes_dir / f"{f['hash_hex']}.WWES").is_file()
        assert (res1.raw_wwes_dir / f"{f['hash_hex']}.WWES.meta").is_file()

    # 2. Çıkarma -> aynı sahnede önceki 2 hash'i otomatik hariç tutmalı ve Paket2 oluşturmalı
    res2 = extract_scene_voices(
        runtime_dir=real_rt,
        scene=prologue,
        keyword="diana",
        min_duration_sec=1.0,
        max_duration_sec=12.0,
        limit_count=2,
        app_root=sandbox_root,
        state_path=tmp_path / "state.json",
    )
    assert res2.extracted_count == 2
    assert res2.package_dir.name == "Hitman_IlkSahne_Paket2"
    assert res2.skipped_existing_count == 2
    m2 = json.loads(res2.manifest_path.read_text(encoding="utf-8"))
    h2_set = {f["hash_hex"] for f in m2["files"]}
    assert h1_set.isdisjoint(h2_set)
    assert list(TEMP_DIR.iterdir()) == []


def test_merge_repack_0xffff_and_safe_uninstall_in_isolated_runtime(tmp_path: Path) -> None:
    bootstrap_portable_environment()
    orig_pkgdef = APP_ROOT / "Hitman_SerbestEv" / "oyun_yedek" / "packagedefinition.txt"
    ham_dir = APP_ROOT / "Hitman_SerbestEv" / "ham_wwes"
    if not orig_pkgdef.is_file() or not ham_dir.is_dir():
        return

    # İzole sahte Runtime (gerçek oyuna ASLA dokunmaz)
    fake_rt = tmp_path / "FakeRuntime"
    fake_rt.mkdir()
    shutil.copy2(orig_pkgdef, fake_rt / "packagedefinition.txt")

    sandbox_root = tmp_path / "sandbox_app"
    pkg1 = sandbox_root / "Hitman_SerbestEv"
    pkg2 = sandbox_root / "Hitman_SerbestEv_Paket2"
    for p in (pkg1, pkg2):
        (p / "sesler").mkdir(parents=True)
        (p / "ham_wwes").mkdir(parents=True)

    # Gerçek 2 mono WWES + meta yedeğini sahte 2 pakete kopyala ve 48 kHz dublajlı WAV üret
    h_a = "005B09B5A328F67F"
    h_b = "00B1D6EF6E47E684"
    for pkg, h, freq in ((pkg1, h_a, 320.0), (pkg2, h_b, 420.0)):
        shutil.copy2(ham_dir / f"{h}.WWES", pkg / "ham_wwes" / f"{h}.WWES")
        shutil.copy2(ham_dir / f"{h}.WWES.meta", pkg / "ham_wwes" / f"{h}.WWES.meta")
        t = np.linspace(0.0, 1.5, 48000 * 3 // 2, endpoint=False, dtype=np.float32)
        wav = (0.15 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        wname = f"01_test_{h}.wav"
        sf.write(str(pkg / "sesler" / wname), wav, 48000, subtype="PCM_16")
        (pkg / "hitman_manifest.json").write_text(
            json.dumps(
                {
                    "scene_key": "safehouse",
                    "chunk_number": 29,
                    "files": [
                        {
                            "hash_hex": h,
                            "wav_filename": wname,
                            "wav_sha256": "original_english_sha256_placeholder",
                            "channels": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    state_path = tmp_path / "hitman_state.json"
    scenes = discover_hitman_scenes(fake_rt, state_path=state_path)
    safehouse = next(s for s in scenes if s.scene_key == "safehouse")

    # 1. Birleştir + 0xFFFF Wwise Vorbis + ww2ogg/vgmstream/FFmpeg + Kur
    rep = merge_repack_and_install_scene(
        runtime_dir=fake_rt,
        scene=safehouse,
        app_root=sandbox_root,
        state_path=state_path,
    )
    assert rep["merged_approved_count"] == 2
    assert rep["rpkg_package"]["rpkg_magic"] == "2KPR"
    assert rep["rpkg_package"]["wwes_count"] == 2
    assert rep["deployment"]["installed_directly"] is True
    assert (fake_rt / "chunk29patch3.rpkg").is_file()
    for ent in rep["rpkg_package"]["entries"]:
        assert ent["format_tag_hex"] == "0xFFFF"
        assert ent["ww2ogg_verified"] is True
        assert ent["vgmstream_verified"] is True
        assert ent["ffmpeg_verified"] is True

    # Durum kartı 2 aktif dublajı raporlamalı
    statuses = get_scenes_runtime_status(
        fake_rt, scenes, app_root=sandbox_root, state_path=state_path
    )
    sh_st = next(st for st in statuses if st.profile.scene_key == "safehouse")
    assert sh_st.is_installed is True
    assert sh_st.installed_wwes_count == 2
    assert sh_st.local_approved_dubbed == 2

    # 2. Güvenli kaldırma: yalnızca chunk29patch3 silinmeli, snug patchlevel=2'ye dönmeli
    un_res = uninstall_scene_patch(
        runtime_dir=fake_rt,
        scene=safehouse,
        state_path=state_path,
    )
    assert un_res["removed_rpkg"] is True
    assert not (fake_rt / "chunk29patch3.rpkg").is_file()
    assert un_res["restored_patchlevel"] == 2

    # packagedefinition.txt doğrulaması
    chk_dir = tmp_path / "chk_pkg"
    chk_dir.mkdir()
    parts = _inspect_packagedefinition_partitions(fake_rt, chk_dir)
    snug = next(p for p in parts if p["codename"] == "snug")
    assert snug["patchlevel"] == 2
    assert list(TEMP_DIR.iterdir()) == []


def test_ui_hitman_rpkg_tab_controls_and_auto_load(tmp_path: Path, monkeypatch) -> None:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    bootstrap_portable_environment()

    from PySide6.QtWidgets import QApplication, QMessageBox
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    isolated_cfg = ConfigManager(path=tmp_path / "ayarlar.json")
    isolated_quota = GeminiQuotaManager(path=tmp_path / "kota.json")
    win = MainWindow(config_mgr=isolated_cfg, quota_mgr=isolated_quota)

    # 4. sekme 'HITMAN 3 (RPKG)' bulunmalı
    assert win.control_tabs.count() == 4
    assert "HITMAN 3 (RPKG)" in win.control_tabs.tabText(3)
    assert win.hitman_runtime_input.text().strip() == DEFAULT_HITMAN_RUNTIME_DIR
    assert win.hitman_scene_combo.count() >= 2
    assert "ICA Tesisi" in win.hitman_scene_combo.itemText(0)
    assert "Serbest Çalışan" in win.hitman_scene_combo.itemText(1)
    assert "Yeni Sesleri Çıkar" in win.hitman_extract_btn.text()
    assert "0xFFFF" in win.hitman_install_btn.text()
    assert "Kaldır" in win.hitman_uninstall_btn.text()

    # Çıkarma bitince Oyun Klasörü ve Proje/Yedek otomatik güncellenmeli
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    fake_pkg = tmp_path / "Hitman_IlkSahne_Paket2"
    fake_wavs = fake_pkg / "sesler"
    fake_wavs.mkdir(parents=True)
    fake_res = HitmanExtractResult(
        scene_key="prologue",
        scene_title="ICA Tesisi",
        chunk_number=0,
        package_dir=fake_pkg,
        wav_dir=fake_wavs,
        raw_wwes_dir=fake_pkg / "ham_wwes",
        manifest_path=fake_pkg / "hitman_manifest.json",
        project_name="Hitman_IlkSahne_Paket2",
        extracted_count=20,
        skipped_existing_count=15,
        matched_candidates_count=50,
        total_chunk_wwes=98937,
    )
    win._on_hitman_extract_done(fake_res)
    app.processEvents()
    assert win.game_dir_input.text() == str(fake_wavs)
    assert win.project_name_input.text() == "Hitman_IlkSahne_Paket2"
    win.close()
