"""HITMAN 3 / World of Assassination — Uygulama İçi Tam Otomatik RPKG Çıkarma, Birleştirme ve Kurulum Motoru.

Bu modül, SesAI Dublaj Stüdyosu'nun 'HITMAN 3 (RPKG)' sekmesini besler:
1. Oyun Runtime dizinindeki tüm `chunk*.rpkg` dosyalarını otomatik keşfeder;
   hazır sahne profilleri (`ICA Tesisi / chunk0 -> chunk0patch5.rpkg`,
   `Serbest Çalışan / chunk29 -> chunk29patch3.rpkg`) ve diğer tüm chunk'ları listeler.
2. Tek tıkla ses çıkarma:
   - Aynı sahnede daha önce çıkarılmış/dublajlanmış hash'leri otomatik hariç tutar.
   - Karakter/anahtar kelime, min/maks süre ve adet filtreleriyle yalnızca yeni
     orijinal `.WWES` + `.WWES.meta` dosyalarını yeni bir paket klasörüne çıkarır,
     `vgmstream-cli.exe` ile 48 kHz `.wav` oluşturur ve `hitman_manifest.json` yazar.
3. Tek tıkla birleştirme, `0xFFFF` Wwise Vorbis paketleme ve oyuna kurulum:
   - Aynı sahneye ait tüm paket klasörlerindeki onaylı (dublajlanmış) `.wav` dosyalarını
     otomatik birleştirir (çakışan hash varsa en güncel dosyayı alır).
   - `tools/hitman_repack.py` hattındaki gerçek `0xFFFF` Wwise Vorbis dönüştürücüsünü,
     `ww2ogg` + `vgmstream-cli` + FFmpeg üçlü doğrulamasını, canlı `packagedefinition.txt`
     korumasını ve SHA-256 doğrulamalı doğrudan kurulumu arka planda yürütür.
4. Güvenli sahne yaması kaldırma (diğer kurulu sahnelerin patchlevel ayarlarını bozmadan).

Taşınabilirlik garantisi: Tüm geçici dosyalar yalnızca `./_temp/` altında işlenir ve
işlem bitiminde 0 dosya bırakılarak temizlenir.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from core.backup_manager import compute_sha256, sanitize_project_name
from core.ffmpeg_manager import FFmpegManager
from core.portable_env import APP_ROOT, DATA_DIR, TEMP_DIR, cleanup_temp_dir
from tools.hitman_bridge import (
    WwesResourceEntry,
    _inspect_packagedefinition_partitions,
    _run_rpkg_cli,
    inspect_wwes_riff_header,
    list_chunk_rpkg_files,
    parse_rpkg_file,
    resolve_ioi_paths_from_hash_list,
)
from tools.hitman_repack import (
    VGMSTREAM_CLI_EXE,
    RepackedWwesEntry,
    backup_and_patch_packagedefinition,
    convert_wav_to_wwise_vorbis_wwes,
    try_direct_install_to_runtime,
    update_wwes_meta_file,
    verify_wwes_roundtrip,
    write_install_and_uninstall_bats,
)

DEFAULT_HITMAN_RUNTIME_DIR = r"C:\Program Files\Epic Games\HITMAN3\Runtime"
HITMAN_STATE_PATH: Path = DATA_DIR / "hitman_mod_durum.json"

# HITMAN 3 / WoA standart chunk -> IOI partition codename haritası (salt-okunur başlangıç listesi için)
HITMAN3_CHUNK_CODENAMES: dict[int, str] = {
    0: "super",
    1: "base",
    2: "season3",
    3: "ancestral",
    4: "edgy",
    5: "elegant",
    6: "wet",
    7: "trapped",
    8: "legacy",
    9: "season2",
    10: "opulent",
    11: "caged",
    12: "greedy",
    13: "salty",
    14: "hawk",
    15: "theark",
    16: "skunk",
    17: "mongoose",
    18: "colombia",
    19: "miami",
    20: "sheep",
    21: "season1",
    22: "hokkaido",
    23: "colorado",
    24: "bangkok",
    25: "marrakesh",
    26: "coastaltown",
    27: "paris",
    28: "dugong",
    29: "snug",
    30: "golden",
}

# Bilinen resmi/varsayılan maksimum patch seviyesi (canlı packagedefinition.txt okunamadığında yedek)
HITMAN3_OFFICIAL_BASE_PATCHLEVEL: dict[int, int] = {
    **{c: 4 for c in range(0, 28)},
    28: 1,
    29: 2,
    30: 0,
}


class HitmanCancelledError(RuntimeError):
    """Kullanıcı HITMAN 3 RPKG işlemini durdurduğunda fırlatılır."""


def build_rpkgv2_meta_bytes(chunk_number: int) -> bytes:
    """Belirtilen chunk numarası için 29 baytlık RPKGv2 (2KPR) güvenli paket meta başlığı üretir."""
    return b"2KPR\x01\x00\x00\x00" + bytes([int(chunk_number) & 0xFF]) + b"\x00\x00xx" + b"\x00" * 16


@dataclass
class HitmanSceneProfile:
    """Bir HITMAN 3 sahne / chunk yama profili."""

    scene_key: str  # "prologue", "safehouse", "chunk27" vb.
    label_tr: str  # Arayüz dropdown etiketi
    short_title: str  # Örn. "ICA Tesisi" veya "Serbest Çalışan"
    chunk_number: int
    partition_name: str  # Örn. "super", "snug", "paris"
    patch_number: int  # Hedef yama seviyesi (ör. 5 -> chunk0patch5.rpkg)
    patch_rpkg_stem: str  # Örn. "chunk0patch5"
    base_dir_prefix: str  # Örn. "Hitman_IlkSahne", "Hitman_SerbestEv"
    is_preset: bool = False
    rpkg_generate_flag: str = "-generate_rpkg_from"
    official_patchlevel: int = 0

    @property
    def patch_rpkg_filename(self) -> str:
        return f"{self.patch_rpkg_stem}.rpkg"

    @property
    def chunk_prefix(self) -> str:
        return f"chunk{self.chunk_number}"


@dataclass
class MergedDubItem:
    """Birleştirme havuzuna dahil edilen onaylı (dublajlanmış) tek bir WWES kaydı."""

    hash_hex: str
    wav_filename: str
    wav_path: Path
    orig_wwes_path: Path
    orig_meta_path: Path
    expected_channels: int
    source_package_dir: Path
    mtime: float
    original_wav_sha256: str
    current_wav_sha256: str


@dataclass
class HitmanExtractResult:
    """Tek tıkla ses çıkarma sonucu."""

    scene_key: str
    scene_title: str
    chunk_number: int
    package_dir: Path
    wav_dir: Path
    raw_wwes_dir: Path
    manifest_path: Path
    project_name: str
    extracted_count: int
    skipped_existing_count: int
    matched_candidates_count: int
    total_chunk_wwes: int


@dataclass
class SceneRuntimeStatus:
    """Bir sahnenin oyundaki kurulum ve yereldeki paket durumu."""

    profile: HitmanSceneProfile
    is_installed: bool
    installed_rpkg_path: Optional[Path]
    installed_wwes_count: int
    installed_rpkg_sha256: str
    local_package_dirs: list[Path] = field(default_factory=list)
    local_total_extracted: int = 0
    local_approved_dubbed: int = 0
    last_installed_utc: str = ""


def _load_mod_state(state_path: Path = HITMAN_STATE_PATH) -> dict[str, Any]:
    if not state_path.is_file():
        return {"scenes": {}}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("scenes"), dict):
            return data
    except Exception:
        pass
    return {"scenes": {}}


def _save_mod_state(state: dict[str, Any], state_path: Path = HITMAN_STATE_PATH) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state_path)


def _check_cancel(cancel_event: Optional[threading.Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise HitmanCancelledError("HITMAN 3 işlemi kullanıcı tarafından durduruldu.")


def discover_hitman_scenes(
    runtime_dir: Path,
    state_path: Path = HITMAN_STATE_PATH,
    live_partitions: Optional[list[dict[str, Any]]] = None,
) -> list[HitmanSceneProfile]:
    """Hazır sahne profillerini (ICA Tesisi / Serbest Çalışan) ve Runtime içindeki tüm chunk'ları listeler.

    Başlangıçta alt süreç (rpkg-cli) çalıştırmadan anında yanıt verir. `live_partitions`
    verilirse canlı `packagedefinition.txt` değerlerini kullanır.
    """
    state = _load_mod_state(state_path)
    saved_scenes = state.get("scenes") or {}

    part_by_chunk: dict[int, dict[str, Any]] = {}
    if live_partitions:
        for p in live_partitions:
            try:
                part_by_chunk[int(p["chunk_index"])] = p
            except Exception:
                continue

    # Hazır profiller her zaman listenin en üstünde yer alır
    prologue = HitmanSceneProfile(
        scene_key="prologue",
        label_tr="ICA Tesisi (Başlangıç) / chunk0 → chunk0patch5.rpkg",
        short_title="ICA Tesisi",
        chunk_number=0,
        partition_name="super",
        patch_number=5,
        patch_rpkg_stem="chunk0patch5",
        base_dir_prefix="Hitman_IlkSahne",
        is_preset=True,
        rpkg_generate_flag="-generate_rpkg_from",
        official_patchlevel=4,
    )
    safehouse = HitmanSceneProfile(
        scene_key="safehouse",
        label_tr="Serbest Çalışan (Serbest Ev) / chunk29 → chunk29patch3.rpkg",
        short_title="Serbest Çalışan",
        chunk_number=29,
        partition_name="snug",
        patch_number=3,
        patch_rpkg_stem="chunk29patch3",
        base_dir_prefix="Hitman_SerbestEv",
        is_preset=True,
        rpkg_generate_flag="-generate_rpkg_quickly_from",
        official_patchlevel=2,
    )

    profiles: list[HitmanSceneProfile] = [prologue, safehouse]
    if not runtime_dir.is_dir():
        return profiles

    # Runtime klasöründeki tüm chunk*.rpkg dosyalarını tara (salt-okunur, alt süreç yok)
    chunk_numbers: set[int] = set()
    rpkg_pat = re.compile(r"^chunk(\d+)(?:patch(\d+))?\.rpkg$", re.IGNORECASE)
    max_disk_patch: dict[int, int] = {}
    for rp in runtime_dir.glob("chunk*.rpkg"):
        m = rpkg_pat.match(rp.name)
        if not m:
            continue
        c_num = int(m.group(1))
        p_num = int(m.group(2)) if m.group(2) is not None else 0
        chunk_numbers.add(c_num)
        max_disk_patch[c_num] = max(max_disk_patch.get(c_num, 0), p_num)

    for c_num in sorted(chunk_numbers):
        if c_num in (0, 29):
            continue
        s_key = f"chunk{c_num}"
        saved = saved_scenes.get(s_key) if isinstance(saved_scenes.get(s_key), dict) else None
        p_info = part_by_chunk.get(c_num)

        codename = (
            (p_info.get("codename") if p_info else "")
            or (saved.get("partition_name") if saved else "")
            or HITMAN3_CHUNK_CODENAMES.get(c_num, f"chunk{c_num}")
        )
        if saved and saved.get("patch_number"):
            target_patch = int(saved["patch_number"])
            official_pl = int(saved.get("previous_patchlevel", max(0, target_patch - 1)))
        elif p_info is not None:
            official_pl = int(p_info.get("patchlevel", 0))
            target_patch = official_pl + 1
        else:
            disk_max = max_disk_patch.get(c_num, 0)
            known_pl = HITMAN3_OFFICIAL_BASE_PATCHLEVEL.get(c_num, disk_max)
            official_pl = min(disk_max, known_pl)
            target_patch = official_pl + 1

        patch_stem = f"chunk{c_num}patch{target_patch}"
        clean_code = re.sub(r"[^a-zA-Z0-9_]+", "", codename) or f"c{c_num}"
        base_prefix = f"Hitman_Chunk{c_num}_{clean_code}"
        profiles.append(
            HitmanSceneProfile(
                scene_key=s_key,
                label_tr=f"Genel Chunk {c_num} ({codename}) / chunk{c_num} → {patch_stem}.rpkg",
                short_title=f"Chunk {c_num} ({codename})",
                chunk_number=c_num,
                partition_name=codename,
                patch_number=target_patch,
                patch_rpkg_stem=patch_stem,
                base_dir_prefix=base_prefix,
                is_preset=False,
                rpkg_generate_flag="-generate_rpkg_from",
                official_patchlevel=official_pl,
            )
        )
    return profiles


def _manifest_matches_scene(manifest_path: Path, scene: HitmanSceneProfile) -> bool:
    """Bir hitman_manifest.json dosyasının verilen sahneye/chunk'a ait olup olmadığını kontrol eder."""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False

    if str(data.get("scene_key") or "").strip().lower() == scene.scene_key.lower():
        return True
    if data.get("chunk_number") is not None:
        try:
            if int(data["chunk_number"]) == scene.chunk_number:
                return True
        except (TypeError, ValueError):
            pass

    # Eski manifest şemaları için uyumluluk (Hitman_SerbestEv / Hitman_IlkSahne)
    if scene.scene_key == "safehouse" and "chunk29_safehouse_summary" in data:
        return True
    if scene.scene_key == "prologue" and (
        "prologue_summary" in data
        or "chunk0_prologue_summary" in data
        or "ica_facility" in str(data.get("scene_name", "")).lower()
    ):
        return True

    files = data.get("files") or []
    if files and isinstance(files, list):
        src_rpkg = str(files[0].get("source_rpkg") or "").lower()
        if re.match(rf"^chunk{scene.chunk_number}(?:patch\d+)?\.rpkg$", src_rpkg):
            return True
    return False


def find_scene_package_dirs(app_root: Path, scene: HitmanSceneProfile) -> list[Path]:
    """Belirtilen sahneye ait tüm yerel paket klasörlerini (eski + yeni Paket2/3...) döndürür."""
    if not app_root.is_dir():
        return []

    found: list[Path] = []
    prefix_low = scene.base_dir_prefix.lower()
    for child in sorted(app_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if child.name.startswith("_") or child.name.startswith("."):
            continue
        manifest = child / "hitman_manifest.json"
        if not manifest.is_file():
            continue
        name_low = child.name.lower()
        if (
            name_low == prefix_low
            or name_low.startswith(prefix_low + "_")
            or _manifest_matches_scene(manifest, scene)
        ):
            found.append(child)
    return found


def collect_existing_scene_hashes(app_root: Path, scene: HitmanSceneProfile) -> set[str]:
    """Aynı sahnede daha önce çıkarılmış veya dublajlanmış tüm WWES hash'lerini toplar."""
    hashes: set[str] = set()
    for pkg_dir in find_scene_package_dirs(app_root, scene):
        manifest = pkg_dir / "hitman_manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                for entry in data.get("files") or []:
                    h = str(entry.get("hash_hex") or "").strip().upper()
                    if h:
                        hashes.add(h)
            except Exception:
                pass
        raw_dir = pkg_dir / "ham_wwes"
        if raw_dir.is_dir():
            for wf in raw_dir.glob("*.WWES"):
                hashes.add(wf.stem.upper())
    return hashes


def allocate_next_package_dir(app_root: Path, scene: HitmanSceneProfile) -> Path:
    """Mevcut paket klasörlerinin üzerine yazmadan yeni bir paket klasörü adı tahsis eder."""
    base = app_root / scene.base_dir_prefix
    if not base.exists():
        return base
    idx = 2
    while True:
        cand = app_root / f"{scene.base_dir_prefix}_Paket{idx}"
        if not cand.exists():
            return cand
        idx += 1


def _parse_keyword_groups(keyword: str) -> list[list[str]]:
    """Kullanıcı filtresini virgülle ayrılmış VEYA (OR) gruplarına, boşlukla ayrılmış VE (AND) terimlerine ayırır."""
    raw = (keyword or "").strip().lower()
    if not raw:
        return []
    groups: list[list[str]] = []
    for part in raw.split(","):
        tokens = [t.strip() for t in part.split() if t.strip()]
        if tokens:
            groups.append(tokens)
    return groups


def _matches_keyword_groups(text_blob: str, groups: list[list[str]]) -> bool:
    if not groups:
        return True
    low = text_blob.lower()
    return any(all(tok in low for tok in grp) for grp in groups)


def _friendly_stem_from_ioi_path(ioi_path: str, hash_hex: str) -> str:
    m = re.search(r"/([^/\\]+?)\.wav\]\.pc_wes", ioi_path or "", re.IGNORECASE)
    if m:
        stem = re.sub(r"[^a-zA-Z0-9_\-]+", "_", m.group(1)).strip("_")
        if stem:
            return stem[:72]
    return f"wwes_{hash_hex.lower()}"


def _list_authentic_source_rpkgs(
    runtime_dir: Path,
    scene: HitmanSceneProfile,
    live_partitions: Optional[list[dict[str, Any]]] = None,
    state_path: Path = HITMAN_STATE_PATH,
) -> list[Path]:
    """Bir chunk'ın yalnızca orijinal oyun kaynak `.rpkg` dosyalarını döndürür.

    Kurulu SesAI yama dosyasını (ör. `chunk0patch5.rpkg`, `chunk29patch3.rpkg`) ve
    aktif `patchlevel` üzerindeki atıl dosyaları hariç tutar ki çıkarılan sesler her zaman
    orijinal oyun konuşması olsun.
    """
    state = _load_mod_state(state_path)
    saved = (state.get("scenes") or {}).get(scene.scene_key) or {}
    excluded_patch_num: Optional[int] = None
    if scene.is_preset:
        excluded_patch_num = scene.patch_number
    elif saved.get("installed") and saved.get("patch_number"):
        excluded_patch_num = int(saved["patch_number"])

    active_pl: Optional[int] = None
    if live_partitions:
        for p in live_partitions:
            if int(p.get("chunk_index", -1)) == scene.chunk_number:
                active_pl = int(p.get("patchlevel", 0))
                break

    # Eğer patchlevel bizim kurulu yamamızı içeriyorsa orijinal üst sınır patch_number - 1'dir
    if excluded_patch_num is not None:
        max_orig_patch = excluded_patch_num - 1
    elif active_pl is not None:
        max_orig_patch = active_pl
    else:
        max_orig_patch = 999

    rpkg_files = list_chunk_rpkg_files(runtime_dir, scene.chunk_number)
    pat = re.compile(rf"^chunk{scene.chunk_number}patch(\d+)\.rpkg$", re.IGNORECASE)
    authentic: list[Path] = []
    for rp in rpkg_files:
        m = pat.match(rp.name)
        if m:
            p_num = int(m.group(1))
            if excluded_patch_num is not None and p_num >= excluded_patch_num:
                continue
            if p_num > max_orig_patch:
                continue
        authentic.append(rp)
    return authentic


def _scan_authentic_chunk_wwes(
    runtime_dir: Path,
    scene: HitmanSceneProfile,
    live_partitions: Optional[list[dict[str, Any]]] = None,
    state_path: Path = HITMAN_STATE_PATH,
) -> dict[str, WwesResourceEntry]:
    """Orijinal oyun RPKG dosyalarındaki güncel (silinmemiş) WWES kaynaklarını toplar."""
    authentic_rpkgs = _list_authentic_source_rpkgs(
        runtime_dir, scene, live_partitions=live_partitions, state_path=state_path
    )
    active_map: dict[str, WwesResourceEntry] = {}
    for rp in authentic_rpkgs:
        _info, entries, deleted = parse_rpkg_file(rp, filter_type="WWES")
        for del_h in deleted:
            active_map.pop(del_h, None)
        for entry in entries:
            active_map[entry.hash_hex] = entry
    return active_map


def _decode_wwes_with_vgmstream_to_48k_wav(
    wwes_path: Path,
    dst_wav_path: Path,
    work_dir: Path,
    ffmpeg: FFmpegManager,
    expected_channels: int = 1,
) -> None:
    """Ham `.WWES` dosyasını taşınabilir `vgmstream-cli.exe` ile 48 kHz PCM_16 `.wav` dosyasına dönüştürür."""
    if not VGMSTREAM_CLI_EXE.is_file():
        raise FileNotFoundError(
            f"Taşınabilir vgmstream-cli.exe bulunamadı: {VGMSTREAM_CLI_EXE}"
        )
    dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    tmp_wem = work_dir / f"vgm_{wwes_path.stem}_{uid}.wem"
    tmp_raw_wav = work_dir / f"vgm_{wwes_path.stem}_{uid}.wav"
    try:
        shutil.copy2(wwes_path, tmp_wem)
        win_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0
        res = subprocess.run(
            [str(VGMSTREAM_CLI_EXE), "-o", str(tmp_raw_wav), str(tmp_wem)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            creationflags=win_flags,
        )
        if res.returncode != 0 or not tmp_raw_wav.is_file() or tmp_raw_wav.stat().st_size < 100:
            raise RuntimeError(
                f"vgmstream-cli ses çözme hatası ({wwes_path.name}): {res.stdout} {res.stderr}"
            )

        meta = ffmpeg.probe_audio(tmp_raw_wav, measure_lufs=False)
        target_ch = expected_channels if expected_channels > 0 else max(1, meta.channels)
        if meta.sample_rate == 48000 and meta.channels == target_ch and "pcm_s16" in (meta.codec or "").lower():
            shutil.copy2(tmp_raw_wav, dst_wav_path)
        else:
            ffmpeg.convert_to_wav(
                tmp_raw_wav,
                dst_wav_path,
                sample_rate=48000,
                channels=target_ch,
            )
    finally:
        tmp_wem.unlink(missing_ok=True)
        tmp_raw_wav.unlink(missing_ok=True)


def extract_scene_voices(
    runtime_dir: Path,
    scene: HitmanSceneProfile,
    keyword: str = "",
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 60.0,
    limit_count: int = 20,
    app_root: Path = APP_ROOT,
    state_path: Path = HITMAN_STATE_PATH,
    cancel_event: Optional[threading.Event] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> HitmanExtractResult:
    """Seçilen HITMAN 3 sahnesinden daha önce çıkarılmamış yeni `.WWES` konuşmalarını çıkarır,
    `vgmstream-cli.exe` ile 48 kHz `.wav` yapar ve `hitman_manifest.json` oluşturur.
    """

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def prog(cur: int, tot: int, label: str) -> None:
        if progress_cb:
            progress_cb(cur, tot, label)

    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"HITMAN 3 Runtime klasörü bulunamadı: {runtime_dir}")

    min_dur = max(0.1, float(min_duration_sec))
    max_dur = max(min_dur + 0.1, float(max_duration_sec))
    target_limit = max(1, min(500, int(limit_count)))

    work_dir = TEMP_DIR / f"hm_ext_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = FFmpegManager()

    try:
        _check_cancel(cancel_event)
        prog(5, 100, f"{scene.short_title} (chunk{scene.chunk_number}) taranıyor...")
        log(
            f"[HITMAN 3] {scene.short_title} (chunk{scene.chunk_number}) taranıyor | "
            f"Filtre: '{keyword or 'Tümü'}' | Süre: {min_dur:.1f}–{max_dur:.1f} sn | Adet: {target_limit}"
        )

        # 1. Canlı packagedefinition.txt bilgisini ./_temp içinde oku (salt-okunur)
        live_parts = _inspect_packagedefinition_partitions(runtime_dir, work_dir)
        active_wwes = _scan_authentic_chunk_wwes(
            runtime_dir, scene, live_partitions=live_parts, state_path=state_path
        )
        total_chunk_wwes = len(active_wwes)
        if total_chunk_wwes == 0:
            raise RuntimeError(
                f"chunk{scene.chunk_number} içinde çıkarılabilir orijinal WWES kaynağı bulunamadı."
            )

        _check_cancel(cancel_event)
        prog(15, 100, f"{total_chunk_wwes} WWES yolu çözülüyor (hash_list.txt)...")
        existing_hashes = collect_existing_scene_hashes(app_root, scene)
        remaining_entries = [
            ent for h, ent in active_wwes.items() if h.upper() not in existing_hashes
        ]
        skipped_existing = total_chunk_wwes - len(remaining_entries)
        log(
            f"[HITMAN 3] Toplam {total_chunk_wwes} benzersiz WWES bulundu | "
            f"Önceki paketlerden hariç tutulan: {skipped_existing} | Kalan: {len(remaining_entries)}"
        )

        if not remaining_entries:
            raise RuntimeError(
                f"{scene.short_title} (chunk{scene.chunk_number}) içindeki tüm WWES sesleri "
                f"({skipped_existing} adet) daha önce çıkarılmış!"
            )

        # IOI yollarını çöz
        resolved_paths = resolve_ioi_paths_from_hash_list({e.hash_id for e in remaining_entries})
        for e in remaining_entries:
            e.ioi_assembly_path = resolved_paths.get(e.hash_id, "")

        # 2. Anahtar kelime + kaba bayt boyutu ön filtresi
        kw_groups = _parse_keyword_groups(keyword)
        min_bytes_est = int(max(0.0, min_dur - 0.5) * 2_200)
        max_bytes_est = int((max_dur + 3.0) * 95_000)

        candidates: list[WwesResourceEntry] = []
        for e in remaining_entries:
            blob = f"{e.ioi_assembly_path} {_friendly_stem_from_ioi_path(e.ioi_assembly_path, e.hash_hex)} {e.hash_hex}"
            if not _matches_keyword_groups(blob, kw_groups):
                continue
            if e.final_size_bytes > 0 and (
                e.final_size_bytes < min_bytes_est or e.final_size_bytes > max_bytes_est
            ):
                continue
            candidates.append(e)

        # Deterministik sıralama: önce IOI yolu bulunanlar, alfabetik
        candidates.sort(
            key=lambda e: (
                0 if e.ioi_assembly_path else 1,
                e.ioi_assembly_path.lower(),
                e.hash_hex,
            )
        )
        matched_count = len(candidates)
        log(f"[HITMAN 3] Filtreyle eşleşen aday WWES sayısı: {matched_count}")

        if matched_count == 0:
            raise RuntimeError(
                f"'{keyword or 'Tümü'}' filtresi ve {min_dur:.1f}–{max_dur:.1f} sn aralığında "
                f"henüz çıkarılmamış yeni ses bulunamadı."
            )

        # 3. Adayları rpkg-cli -filter ile gruplar halinde ./_temp içine çıkar ve kesin süre filtresini uygula
        prog(25, 100, "Seçilen WWES adayları RPKG arşivinden çıkarılıyor...")
        accepted: list[tuple[WwesResourceEntry, Path, Path, dict[str, Any], float]] = []
        batch_size = max(25, min(80, target_limit * 2))
        raw_stage = work_dir / "r"
        raw_stage.mkdir(parents=True, exist_ok=True)

        cursor = 0
        while cursor < len(candidates) and len(accepted) < target_limit:
            _check_cancel(cancel_event)
            batch = candidates[cursor : cursor + batch_size]
            cursor += len(batch)

            by_rpkg: dict[str, list[WwesResourceEntry]] = {}
            for ent in batch:
                by_rpkg.setdefault(ent.source_rpkg, []).append(ent)

            for rpkg_name, grp in by_rpkg.items():
                _check_cancel(cancel_event)
                rp = runtime_dir / rpkg_name
                filt_str = ",".join(ent.hash_hex for ent in grp)
                _run_rpkg_cli(
                    [
                        "-output_path",
                        str(raw_stage),
                        "-filter",
                        filt_str,
                        "-extract_from_rpkg",
                        str(rp),
                    ]
                )

            for ent in batch:
                if len(accepted) >= target_limit:
                    break
                stem = Path(ent.source_rpkg).stem
                wwes_file = raw_stage / stem / "WWES" / f"{ent.hash_hex}.WWES"
                meta_file = raw_stage / stem / "WWES" / f"{ent.hash_hex}.WWES.meta"
                if not wwes_file.is_file() or not meta_file.is_file():
                    continue
                try:
                    hdr = inspect_wwes_riff_header(wwes_file)
                except Exception:
                    continue

                est_dur = float(hdr.vorbis_estimated_duration_sec or 0.0)
                if est_dur <= 0.0 and hdr.avg_bytes_per_sec > 0:
                    data_sz = int((hdr.chunks_present or {}).get("data", 0))
                    est_dur = data_sz / float(hdr.avg_bytes_per_sec)
                if est_dur > 0.0 and not (min_dur <= est_dur <= max_dur):
                    continue
                accepted.append((ent, wwes_file, meta_file, asdict(hdr), est_dur))

            pct = 25 + int((len(accepted) / float(target_limit)) * 30)
            prog(min(55, pct), 100, f"Süre filtresi: {len(accepted)}/{target_limit} ses seçildi...")

        if not accepted:
            raise RuntimeError(
                f"Adaylar incelendi ancak {min_dur:.1f}–{max_dur:.1f} sn aralığında "
                f"geçerli WWES sesi bulunamadı. Süre aralığını genişletmeyi deneyin."
            )

        # 4. Yeni paket klasörünü oluştur ve vgmstream-cli.exe ile 48 kHz WAV'a dönüştür
        package_dir = allocate_next_package_dir(app_root, scene)
        wav_dir = package_dir / "sesler"
        raw_wwes_dir = package_dir / "ham_wwes"
        wav_dir.mkdir(parents=True, exist_ok=True)
        raw_wwes_dir.mkdir(parents=True, exist_ok=True)

        extracted_records: list[dict[str, Any]] = []
        for idx, (ent, src_wwes, src_meta, hdr_dict, _est_dur) in enumerate(accepted, start=1):
            _check_cancel(cancel_event)
            friendly = _friendly_stem_from_ioi_path(ent.ioi_assembly_path, ent.hash_hex)
            wav_filename = f"{idx:02d}_{friendly}_{ent.hash_hex}.wav"
            dst_wav = wav_dir / wav_filename
            dst_raw_wwes = raw_wwes_dir / f"{ent.hash_hex}.WWES"
            dst_raw_meta = raw_wwes_dir / f"{ent.hash_hex}.WWES.meta"

            shutil.copy2(src_wwes, dst_raw_wwes)
            shutil.copy2(src_meta, dst_raw_meta)

            exp_ch = int(hdr_dict.get("channels") or 1)
            _decode_wwes_with_vgmstream_to_48k_wav(
                wwes_path=dst_raw_wwes,
                dst_wav_path=dst_wav,
                work_dir=work_dir,
                ffmpeg=ffmpeg,
                expected_channels=exp_ch,
            )

            audio_meta = ffmpeg.probe_audio(dst_wav, measure_lufs=True)
            # Kesin çözülmüş süre kontrolü (başlıkta süre yoksa)
            if not (min_dur - 0.15 <= audio_meta.duration_sec <= max_dur + 0.15):
                dst_wav.unlink(missing_ok=True)
                dst_raw_wwes.unlink(missing_ok=True)
                dst_raw_meta.unlink(missing_ok=True)
                continue

            rec_idx = len(extracted_records) + 1
            if rec_idx != idx:
                # Boşluksuz sıra numarası için yeniden adlandır
                new_wav_name = f"{rec_idx:02d}_{friendly}_{ent.hash_hex}.wav"
                new_dst_wav = wav_dir / new_wav_name
                dst_wav.replace(new_dst_wav)
                dst_wav = new_dst_wav
                wav_filename = new_wav_name

            extracted_records.append(
                {
                    "index": rec_idx,
                    "hash_hex": ent.hash_hex,
                    "hash_id": ent.hash_id,
                    "source_rpkg": ent.source_rpkg,
                    "ioi_assembly_path": ent.ioi_assembly_path,
                    "friendly_name": friendly,
                    "wav_filename": wav_filename,
                    "wav_rel_path": f"sesler/{wav_filename}",
                    "raw_wwes_rel_path": f"ham_wwes/{ent.hash_hex}.WWES",
                    "raw_meta_rel_path": f"ham_wwes/{ent.hash_hex}.WWES.meta",
                    "raw_meta_json_rel_path": "",
                    "raw_wwes_size_bytes": dst_raw_wwes.stat().st_size,
                    "raw_wwes_sha256": compute_sha256(dst_raw_wwes),
                    "wav_size_bytes": dst_wav.stat().st_size,
                    "wav_sha256": compute_sha256(dst_wav),
                    "duration_sec": round(audio_meta.duration_sec, 3),
                    "duration_ms": audio_meta.duration_ms,
                    "sample_rate": audio_meta.sample_rate,
                    "channels": audio_meta.channels,
                    "lufs": round(audio_meta.lufs, 2),
                    "codec_in_wwes": hdr_dict.get("codec_name", "Wwise Custom Vorbis (0xFFFF)"),
                    "format_tag_hex": hdr_dict.get("format_tag_hex", "0xFFFF"),
                    "dialogue_subtitle_en": "",
                    "dialogue_subtitle_tr": "",
                    "dialogue_subtitle_tr_mod": "",
                    "wem_header": hdr_dict,
                }
            )

            pct = 55 + int((idx / float(len(accepted))) * 40)
            prog(pct, 100, f"48 kHz WAV dönüştürülüyor ({idx}/{len(accepted)}): {wav_filename}")

        if not extracted_records:
            shutil.rmtree(package_dir, ignore_errors=True)
            raise RuntimeError(
                f"Çıkarılan adaylardan hiçbiri kesin süre aralığına ({min_dur:.1f}–{max_dur:.1f} sn) uymadı."
            )

        manifest_data: dict[str, Any] = {
            "schema_version": 2,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "SesAI Dublaj Stüdyosu — HITMAN 3 (RPKG) Modülü",
            "read_only_guarantee": True,
            "scene_key": scene.scene_key,
            "scene_title": scene.short_title,
            "chunk_number": scene.chunk_number,
            "partition_name": scene.partition_name,
            "patch_rpkg_stem": scene.patch_rpkg_stem,
            "hitman_runtime_dir": str(runtime_dir),
            "extraction_filters": {
                "keyword": keyword,
                "min_duration_sec": min_dur,
                "max_duration_sec": max_dur,
                "limit_count": target_limit,
            },
            "files": extracted_records,
        }
        manifest_path = package_dir / "hitman_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        prog(100, 100, f"✅ {len(extracted_records)} yeni ses çıkarıldı -> {package_dir.name}")
        log(
            f"[HITMAN 3] ✅ {len(extracted_records)} adet 48 kHz WAV çıkarıldı -> {wav_dir} "
            f"(manifest: {manifest_path.name})"
        )
        return HitmanExtractResult(
            scene_key=scene.scene_key,
            scene_title=scene.short_title,
            chunk_number=scene.chunk_number,
            package_dir=package_dir,
            wav_dir=wav_dir,
            raw_wwes_dir=raw_wwes_dir,
            manifest_path=manifest_path,
            project_name=sanitize_project_name(package_dir.name),
            extracted_count=len(extracted_records),
            skipped_existing_count=skipped_existing,
            matched_candidates_count=matched_count,
            total_chunk_wwes=total_chunk_wwes,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        cleanup_temp_dir(keep_root=True)


def collect_merged_approved_dubs(
    app_root: Path,
    scene: HitmanSceneProfile,
) -> tuple[list[MergedDubItem], list[Path], int]:
    """Aynı sahneye ait tüm paket klasörlerindeki onaylı (dublajlanmış) `.wav` dosyalarını birleştirir.

    Çakışan hash varsa dosya değişiklik zamanı (`st_mtime`) en yeni olanı seçer.

    Returns:
        (birlesik_onayli_liste, taranan_paket_klasorleri, toplam_cikarilmis_benzersiz_hash_sayisi)
    """
    pkg_dirs = find_scene_package_dirs(app_root, scene)
    all_extracted_hashes: set[str] = set()
    best_by_hash: dict[str, MergedDubItem] = {}

    for pkg_dir in pkg_dirs:
        manifest_path = pkg_dir / "hitman_manifest.json"
        wav_dir = pkg_dir / "sesler"
        raw_dir = pkg_dir / "ham_wwes"
        if not manifest_path.is_file() or not wav_dir.is_dir() or not raw_dir.is_dir():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in data.get("files") or []:
            h_hex = str(entry.get("hash_hex") or "").strip().upper()
            wav_name = str(entry.get("wav_filename") or "").strip()
            orig_sha = str(entry.get("wav_sha256") or "").strip().lower()
            if not h_hex or not wav_name:
                continue
            all_extracted_hashes.add(h_hex)

            wav_path = wav_dir / wav_name
            orig_wwes = raw_dir / f"{h_hex}.WWES"
            orig_meta = raw_dir / f"{h_hex}.WWES.meta"
            if not wav_path.is_file() or not orig_wwes.is_file() or not orig_meta.is_file():
                continue

            cur_sha = compute_sha256(wav_path).lower()
            # Yalnızca orijinal İngilizce SHA-256'sından farklı (dublajlanmış) dosyaları birleştir
            if orig_sha and cur_sha == orig_sha:
                continue

            mtime = float(wav_path.stat().st_mtime)
            exp_ch = int(entry.get("channels") or 1)
            item = MergedDubItem(
                hash_hex=h_hex,
                wav_filename=wav_name,
                wav_path=wav_path,
                orig_wwes_path=orig_wwes,
                orig_meta_path=orig_meta,
                expected_channels=exp_ch,
                source_package_dir=pkg_dir,
                mtime=mtime,
                original_wav_sha256=orig_sha,
                current_wav_sha256=cur_sha,
            )
            prev = best_by_hash.get(h_hex)
            if prev is None or item.mtime >= prev.mtime:
                best_by_hash[h_hex] = item

    merged = sorted(best_by_hash.values(), key=lambda x: (x.wav_filename.lower(), x.hash_hex))
    return merged, pkg_dirs, len(all_extracted_hashes)


def merge_repack_and_install_scene(
    runtime_dir: Path,
    scene: HitmanSceneProfile,
    app_root: Path = APP_ROOT,
    state_path: Path = HITMAN_STATE_PATH,
    cancel_event: Optional[threading.Event] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, Any]:
    """Aynı sahneye ait tüm paketlerdeki onaylı Türkçe WAV dosyalarını birleştirir,
    gerçek `0xFFFF` Wwise Vorbis `.WWES` + `.WWES.meta` üretip `ww2ogg` + `vgmstream-cli` + FFmpeg
    ile doğrular, bağımsız `.rpkg` yaması oluşturur ve `Runtime\\` klasörüne SHA-256 doğrulamasıyla kurar.
    """

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def prog(cur: int, tot: int, label: str) -> None:
        if progress_cb:
            progress_cb(cur, tot, label)

    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"HITMAN 3 Runtime klasörü bulunamadı: {runtime_dir}")

    _check_cancel(cancel_event)
    prog(3, 100, f"{scene.short_title} onaylı dublajları taranıyor ve birleştiriliyor...")
    merged_items, pkg_dirs, total_extracted = collect_merged_approved_dubs(app_root, scene)
    if not merged_items:
        raise RuntimeError(
            f"{scene.short_title} sahnesi için onaylanmış (Türkçe dublajla güncellenmiş) .wav dosyası bulunamadı.\n"
            "Önce ana stüdyoda 'Dublajı Başlat' ve 'Oyuna Uygula' adımlarını tamamlayın."
        )

    pkg_names = ", ".join(p.name for p in pkg_dirs)
    log(
        f"[HITMAN 3] {scene.short_title} birleştirme: {len(pkg_dirs)} paket ({pkg_names}) -> "
        f"{len(merged_items)} benzersiz onaylı Türkçe dublaj (toplam çıkarılan: {total_extracted})"
    )

    # Birincil sahne klasörü (paket/ ve oyun_yedek/ çıktıları için)
    primary_scene_dir = app_root / scene.base_dir_prefix
    if not primary_scene_dir.exists() and pkg_dirs:
        primary_scene_dir = pkg_dirs[0]
    package_dir = primary_scene_dir / "paket"
    backup_dir = primary_scene_dir / "oyun_yedek"
    package_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    work_dir = TEMP_DIR / f"hm_rep_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = FFmpegManager()

    try:
        # Canlı packagedefinition.txt içinden bu chunk'ın kesin partition adını ve patchlevel değerini doğrula
        live_parts = _inspect_packagedefinition_partitions(runtime_dir, work_dir)
        live_part = next(
            (p for p in live_parts if int(p.get("chunk_index", -1)) == scene.chunk_number),
            None,
        )
        partition_name = (
            str(live_part.get("codename") or "").strip()
            if live_part
            else scene.partition_name
        ) or scene.partition_name

        state = _load_mod_state(state_path)
        prev_saved = (state.get("scenes") or {}).get(scene.scene_key) or {}
        if scene.is_preset:
            target_patch_num = scene.patch_number
            previous_pl = scene.official_patchlevel
        elif prev_saved.get("installed") and prev_saved.get("patch_number"):
            target_patch_num = int(prev_saved["patch_number"])
            previous_pl = int(prev_saved.get("previous_patchlevel", max(0, target_patch_num - 1)))
        elif live_part is not None:
            previous_pl = int(live_part.get("patchlevel", 0))
            target_patch_num = previous_pl + 1
        else:
            target_patch_num = scene.patch_number
            previous_pl = max(0, target_patch_num - 1)

        patch_rpkg_stem = f"chunk{scene.chunk_number}patch{target_patch_num}"
        rpkg_filename = f"{patch_rpkg_stem}.rpkg"
        rpkgv2_meta = build_rpkgv2_meta_bytes(scene.chunk_number)

        stage_chunk_dir = package_dir / patch_rpkg_stem
        stage_wwes_dir = stage_chunk_dir / "WWES"
        if stage_chunk_dir.exists():
            shutil.rmtree(stage_chunk_dir)
        stage_wwes_dir.mkdir(parents=True, exist_ok=True)
        (stage_chunk_dir / f"{patch_rpkg_stem}.meta").write_bytes(rpkgv2_meta)

        repacked_entries: list[RepackedWwesEntry] = []
        total_items = len(merged_items)

        for idx, item in enumerate(merged_items, start=1):
            _check_cancel(cancel_event)
            pct = 8 + int(((idx - 1) / float(total_items)) * 72)
            prog(
                pct,
                100,
                f"0xFFFF Vorbis & 3'lü doğrulama ({idx}/{total_items}): {item.hash_hex}",
            )

            out_wwes_path = stage_wwes_dir / f"{item.hash_hex}.WWES"
            out_meta_path = stage_wwes_dir / f"{item.hash_hex}.WWES.meta"

            vorbis_info = convert_wav_to_wwise_vorbis_wwes(
                dubbed_wav_path=item.wav_path,
                orig_wwes_path=item.orig_wwes_path,
                output_wwes_path=out_wwes_path,
                target_channels=item.expected_channels,
                work_dir=work_dir,
                ffmpeg=ffmpeg,
                vorbis_quality=8,
            )

            wwes_size = out_wwes_path.stat().st_size
            updated_final_sz = update_wwes_meta_file(
                orig_meta_path=item.orig_meta_path,
                output_meta_path=out_meta_path,
                new_wwes_size_bytes=wwes_size,
            )
            if updated_final_sz != wwes_size:
                raise RuntimeError(f".WWES.meta boyut uyuşmazlığı: {item.hash_hex}")

            hdr_info = inspect_wwes_riff_header(out_wwes_path)
            if (
                hdr_info.riff_magic != "RIFF"
                or hdr_info.wave_magic != "WAVE"
                or hdr_info.format_tag_int != 0xFFFF
                or hdr_info.fmt_chunk_size != 66
                or hdr_info.sample_rate != 48000
                or hdr_info.channels != item.expected_channels
                or (hdr_info.riff_declared_size + 8) != wwes_size
            ):
                raise RuntimeError(
                    f"Üretilen 0xFFFF WWES başlık doğrulaması başarısız: {out_wwes_path.name}"
                )

            rt_info = verify_wwes_roundtrip(
                wwes_path=out_wwes_path,
                expected_channels=item.expected_channels,
                work_dir=work_dir,
                ffmpeg=ffmpeg,
                min_audible_dbfs=-40.0,
            )

            repacked_entries.append(
                RepackedWwesEntry(
                    index=idx,
                    hash_hex=item.hash_hex,
                    hash_id=f"{item.hash_hex}.WWES",
                    wav_filename=item.wav_filename,
                    wwes_size_bytes=wwes_size,
                    wwes_sha256=compute_sha256(out_wwes_path),
                    meta_size_bytes=out_meta_path.stat().st_size,
                    meta_sha256=compute_sha256(out_meta_path),
                    meta_hash_size_final=updated_final_sz,
                    format_tag_hex=hdr_info.format_tag_hex,
                    sample_rate=hdr_info.sample_rate,
                    channels=hdr_info.channels,
                    channel_config_hex=f"0x{vorbis_info['u_channel_config']:08X}",
                    pcm_data_bytes=vorbis_info["vorbis_data_bytes"],
                    duration_sec=rt_info["duration_sec"],
                    preserved_subchunks=vorbis_info["preserved_subchunks"],
                    ffmpeg_verified=rt_info["ffmpeg_verified"],
                    uid_hex=vorbis_info["uid_hex"],
                    setup_exact_match=vorbis_info["setup_exact_match"],
                    ww2ogg_verified=rt_info["ww2ogg_verified"],
                    vgmstream_verified=rt_info["vgmstream_verified"],
                    roundtrip_rms_dbfs=rt_info["rms_dbfs"],
                    roundtrip_lufs=rt_info["lufs"],
                )
            )
            log(
                f"[HITMAN 3] [{idx:02d}/{total_items:02d}] {item.hash_hex} ({item.source_package_dir.name}) "
                f"-> 0xFFFF OK | {rt_info['duration_sec']:.2f} sn | {rt_info['lufs']:.1f} LUFS"
            )

        _check_cancel(cancel_event)
        prog(84, 100, f"{rpkg_filename} paketi üretiliyor (rpkg-cli)...")
        out_rpkg_path = package_dir / rpkg_filename
        if out_rpkg_path.exists():
            out_rpkg_path.unlink()

        gen_res = _run_rpkg_cli(
            [
                "-output_path",
                str(package_dir),
                scene.rpkg_generate_flag,
                str(stage_chunk_dir),
            ]
        )
        if not out_rpkg_path.is_file():
            raise RuntimeError(
                f"{rpkg_filename} üretilemedi: {gen_res.stdout} {gen_res.stderr}"
            )

        rpkg_info, _rpkg_entries, _rpkg_deleted = parse_rpkg_file(out_rpkg_path, filter_type="WWES")
        if rpkg_info.magic != "2KPR" or rpkg_info.wwes_count != total_items:
            raise RuntimeError(f"Üretilen {rpkg_filename} başlık doğrulaması başarısız: {rpkg_info}")

        # rpkg-cli gidiş-dönüş açma doğrulaması
        verify_extract_dir = work_dir / "v_ext"
        verify_extract_dir.mkdir(parents=True, exist_ok=True)
        _run_rpkg_cli(
            [
                "-output_path",
                str(verify_extract_dir),
                "-filter",
                "WWES",
                "-extract_from_rpkg",
                str(out_rpkg_path),
            ]
        )
        extracted_back = list((verify_extract_dir / patch_rpkg_stem / "WWES").glob("*.WWES"))
        if len(extracted_back) != total_items:
            raise RuntimeError(
                f"rpkg-cli doğrulama çıkarması eksik: {len(extracted_back)}/{total_items}"
            )
        rpkg_sha256 = compute_sha256(out_rpkg_path)

        # Canlı packagedefinition.txt yedeği + yalnızca bu sahnenin partition patchlevel güncellemesi
        prog(91, 100, f"packagedefinition.txt ({partition_name} patchlevel={target_patch_num}) güncelleniyor...")
        pkgdef_info = backup_and_patch_packagedefinition(
            runtime_dir=runtime_dir,
            backup_dir=backup_dir,
            package_dir=package_dir,
            work_dir=work_dir,
            partition_name=partition_name,
            min_patchlevel=target_patch_num,
        )
        patched_pkgdef_path = Path(pkgdef_info["patched_package_path"])
        backup_pkgdef_path = Path(pkgdef_info["backup_path"])

        # Eğer genel chunk'ın hedef slotunda eski atıl bir dosya varsa (bizimki değilse) SHA-256 yedeğiyle koru
        dst_rpkg_live = runtime_dir / rpkg_filename
        if (
            not scene.is_preset
            and not prev_saved.get("installed")
            and dst_rpkg_live.is_file()
        ):
            q_dir = backup_dir / f"eski_mod_{patch_rpkg_stem}"
            q_dir.mkdir(parents=True, exist_ok=True)
            q_dst = q_dir / rpkg_filename
            if not q_dst.is_file():
                shutil.copy2(dst_rpkg_live, q_dst)
                log(f"[HITMAN 3] Hedefteki mevcut {rpkg_filename} güvenlik amaçlı yedeklendi -> {q_dst}")

        # .bat kurulum/kaldırma betikleri
        bat_prefix = sanitize_project_name(scene.scene_key).lower()
        install_bat_path = primary_scene_dir / f"{bat_prefix}_dublaj_kur.bat"
        uninstall_bat_path = primary_scene_dir / f"{bat_prefix}_dublaj_kaldir.bat"
        write_install_and_uninstall_bats(
            runtime_dir=runtime_dir,
            patched_rpkg_path=out_rpkg_path,
            patched_pkgdef_path=patched_pkgdef_path,
            backup_pkgdef_path=backup_pkgdef_path,
            patched_rpkg_sha256=rpkg_sha256,
            patched_pkgdef_sha256=pkgdef_info["patched_sha256"],
            original_pkgdef_sha256=pkgdef_info["original_sha256"],
            install_bat_path=install_bat_path,
            uninstall_bat_path=uninstall_bat_path,
            rpkg_filename=rpkg_filename,
            scene_label_tr=scene.short_title,
            partition_note=f"{partition_name} patchlevel={pkgdef_info['new_patchlevel']}",
            success_hint_tr=f"Oyuna girip {scene.short_title} sahnesinde Türkçe dublajı hemen test edebilirsiniz.",
            orig_chunk_label=scene.chunk_prefix,
        )

        # Doğrudan Runtime kurulumu + SHA-256 doğrulaması
        prog(96, 100, f"{rpkg_filename} HITMAN 3 Runtime klasörüne kuruluyor (SHA-256)...")
        install_result = try_direct_install_to_runtime(
            runtime_dir=runtime_dir,
            patched_rpkg_path=out_rpkg_path,
            patched_pkgdef_path=patched_pkgdef_path,
            expected_rpkg_sha256=rpkg_sha256,
            expected_pkgdef_sha256=pkgdef_info["patched_sha256"],
            rpkg_filename=rpkg_filename,
        )

        # Rapor ve kalıcı mod durum kaydı
        report_data: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scene_key": scene.scene_key,
            "scene_label_tr": scene.short_title,
            "chunk_number": scene.chunk_number,
            "partition_name": partition_name,
            "hitman_runtime_dir": str(runtime_dir),
            "merged_source_packages": [str(p) for p in pkg_dirs],
            "merged_approved_count": total_items,
            "original_rpkgs_untouched": True,
            "rpkg_package": {
                "rpkg_filename": rpkg_filename,
                "rpkg_path": str(out_rpkg_path),
                "rpkg_size_bytes": out_rpkg_path.stat().st_size,
                "rpkg_size_mb": round(out_rpkg_path.stat().st_size / (1024 * 1024), 2),
                "rpkg_sha256": rpkg_sha256,
                "rpkg_magic": rpkg_info.magic,
                "wwes_count": rpkg_info.wwes_count,
                "rpkg_cli_roundtrip_verified": True,
                "entries": [asdict(e) for e in repacked_entries],
            },
            "packagedefinition_patch": pkgdef_info,
            "deployment": {
                **install_result,
                "install_bat_path": str(install_bat_path),
                "uninstall_bat_path": str(uninstall_bat_path),
            },
        }
        (package_dir / "repack_raporu.json").write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if install_result.get("installed_directly"):
            state.setdefault("scenes", {})[scene.scene_key] = {
                "scene_key": scene.scene_key,
                "scene_title": scene.short_title,
                "chunk_number": scene.chunk_number,
                "partition_name": partition_name,
                "patch_number": target_patch_num,
                "patch_rpkg_stem": patch_rpkg_stem,
                "previous_patchlevel": previous_pl,
                "rpkg_sha256": rpkg_sha256,
                "wwes_count": total_items,
                "source_packages": [p.name for p in pkg_dirs],
                "installed": True,
                "installed_at_utc": report_data["generated_at_utc"],
            }
            _save_mod_state(state, state_path)
            log(
                f"[HITMAN 3] ✅ KURULUM TAMAMLANDI: {rpkg_filename} ({total_items} ses, "
                f"SHA-256: {rpkg_sha256[:16]}...)"
            )
            prog(100, 100, f"✅ {rpkg_filename} ({total_items} dublaj) oyuna kuruldu!")
        else:
            log(
                f"[HITMAN 3] ⚠️ Paket üretildi ancak doğrudan kopyalama için UAC gerekli: "
                f"{install_result.get('error', '')} -> {install_bat_path}"
            )
            prog(100, 100, f"⚠️ Paket hazır — kurulum için {install_bat_path.name} çalıştırın.")

        return report_data
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        cleanup_temp_dir(keep_root=True)


def _patch_packagedefinition_exact_level(
    runtime_dir: Path,
    partition_name: str,
    exact_patchlevel: int,
    work_dir: Path,
) -> dict[str, Any]:
    """Canlı `packagedefinition.txt` içinde yalnızca belirtilen partition'ın `patchlevel` değerini
    tam olarak `exact_patchlevel` değerine çeker; diğer tüm partition'lara dokunmaz.
    """
    live_pkg = runtime_dir / "packagedefinition.txt"
    if not live_pkg.is_file():
        raise FileNotFoundError(f"packagedefinition.txt bulunamadı: {live_pkg}")

    pkg_work = work_dir / "uninst_pkg"
    pkg_work.mkdir(parents=True, exist_ok=True)
    tmp_pkg = pkg_work / "packagedefinition.txt"
    shutil.copy2(live_pkg, tmp_pkg)

    res_dec = _run_rpkg_cli(
        ["-output_path", str(pkg_work), "-decrypt_packagedefinition_thumbs", str(tmp_pkg)]
    )
    dec_file = pkg_work / "packagedefinition.txt.decrypted"
    if not dec_file.is_file():
        raise RuntimeError(
            f"packagedefinition.txt şifresi çözülemedi: {res_dec.stdout} {res_dec.stderr}"
        )

    raw_text = dec_file.read_text(encoding="utf-8", errors="replace")
    part_pattern = re.compile(rf"(?:^|\s)name={re.escape(partition_name)}(?:\s|$)")
    old_line = ""
    new_line = ""
    out_lines: list[str] = []
    for line in raw_text.splitlines():
        s = line.strip()
        if s.startswith("@partition") and part_pattern.search(s):
            old_line = line
            if re.search(r"patchlevel=\d+", line):
                line = re.sub(r"patchlevel=\d+", f"patchlevel={int(exact_patchlevel)}", line)
            else:
                line = f"{line.rstrip()} patchlevel={int(exact_patchlevel)}"
            new_line = line
        out_lines.append(line)

    if not old_line:
        raise RuntimeError(f"packagedefinition.txt içinde '@partition name={partition_name}' bulunamadı.")

    edited_dec = pkg_work / "packagedefinition.txt.decrypted"
    edited_dec.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    enc_out = pkg_work / "enc"
    enc_out.mkdir(parents=True, exist_ok=True)
    res_enc = _run_rpkg_cli(
        ["-output_path", str(enc_out), "-encrypt_packagedefinition_thumbs", str(edited_dec)]
    )
    enc_candidates = [
        enc_out / "packagedefinition.txt.decrypted.encrypted",
        enc_out / "packagedefinition.txt.encrypted",
        enc_out / "packagedefinition.txt",
    ]
    enc_file = next((c for c in enc_candidates if c.is_file()), None)
    if enc_file is None:
        raise RuntimeError(
            f"packagedefinition.txt yeniden şifrelenemedi: {res_enc.stdout} {res_enc.stderr}"
        )

    expected_sha = compute_sha256(enc_file)
    shutil.copy2(enc_file, live_pkg)
    live_sha = compute_sha256(live_pkg)
    if live_sha.lower() != expected_sha.lower():
        raise RuntimeError("Güncellenen packagedefinition.txt SHA-256 doğrulaması başarısız!")

    return {
        "partition_name": partition_name,
        "old_line": old_line,
        "new_line": new_line,
        "live_sha256": live_sha,
    }


def uninstall_scene_patch(
    runtime_dir: Path,
    scene: HitmanSceneProfile,
    state_path: Path = HITMAN_STATE_PATH,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, Any]:
    """Seçili sahnenin bağımsız yama `.rpkg` dosyasını kaldırır ve yalnızca o sahnenin
    `packagedefinition.txt` `patchlevel` değerini diğer sahnelere dokunmadan orijinal seviyesine döndürür.
    """

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def prog(cur: int, tot: int, label: str) -> None:
        if progress_cb:
            progress_cb(cur, tot, label)

    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"HITMAN 3 Runtime klasörü bulunamadı: {runtime_dir}")

    state = _load_mod_state(state_path)
    saved = (state.get("scenes") or {}).get(scene.scene_key) or {}
    rpkg_filename = (
        f"{saved['patch_rpkg_stem']}.rpkg"
        if saved.get("patch_rpkg_stem")
        else scene.patch_rpkg_filename
    )
    partition_name = str(saved.get("partition_name") or scene.partition_name)
    restore_pl = int(
        saved.get("previous_patchlevel")
        if saved.get("previous_patchlevel") is not None
        else scene.official_patchlevel
    )

    dst_rpkg = runtime_dir / rpkg_filename
    work_dir = TEMP_DIR / f"hm_uninst_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        prog(20, 100, f"{rpkg_filename} kaldırılıyor...")
        removed_rpkg = False
        if dst_rpkg.is_file():
            dst_rpkg.unlink()
            removed_rpkg = True
            log(f"[HITMAN 3] {dst_rpkg} silindi.")
        else:
            log(f"[HITMAN 3] {rpkg_filename} Runtime klasöründe zaten mevcut değil.")

        prog(60, 100, f"packagedefinition.txt ({partition_name} -> patchlevel={restore_pl}) güncelleniyor...")
        pkg_res = _patch_packagedefinition_exact_level(
            runtime_dir=runtime_dir,
            partition_name=partition_name,
            exact_patchlevel=restore_pl,
            work_dir=work_dir,
        )
        log(
            f"[HITMAN 3] packagedefinition.txt güncellendi: "
            f"'{pkg_res['old_line']}' -> '{pkg_res['new_line']}' (SHA-256: {pkg_res['live_sha256'][:16]}...)"
        )

        if scene.scene_key in (state.get("scenes") or {}):
            state["scenes"][scene.scene_key]["installed"] = False
            state["scenes"][scene.scene_key]["uninstalled_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            _save_mod_state(state, state_path)

        prog(100, 100, f"✅ {scene.short_title} yaması kaldırıldı (diğer sahneler korundu).")
        return {
            "scene_key": scene.scene_key,
            "scene_title": scene.short_title,
            "removed_rpkg": removed_rpkg,
            "rpkg_filename": rpkg_filename,
            "partition_name": partition_name,
            "restored_patchlevel": restore_pl,
            "live_pkgdef_sha256": pkg_res["live_sha256"],
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        cleanup_temp_dir(keep_root=True)


def get_scenes_runtime_status(
    runtime_dir: Path,
    scenes: list[HitmanSceneProfile],
    app_root: Path = APP_ROOT,
    state_path: Path = HITMAN_STATE_PATH,
) -> list[SceneRuntimeStatus]:
    """Kurulu sahneleri, oyundaki aktif dublaj sayısını ve yerelde birleştirilmeye hazır
    onaylı dublaj sayısını salt-okunur (alt süreç açmadan, hızlı) hesaplar.
    """
    state = _load_mod_state(state_path)
    saved_scenes = state.get("scenes") or {}
    results: list[SceneRuntimeStatus] = []

    for sc in scenes:
        pkg_dirs = find_scene_package_dirs(app_root, sc)
        saved = saved_scenes.get(sc.scene_key) if isinstance(saved_scenes.get(sc.scene_key), dict) else None
        # Yalnızca hazır profiller, yerel paketi olanlar veya daha önce kurulmuş olanlar için tam tarama yap
        is_relevant = sc.is_preset or bool(pkg_dirs) or bool(saved and saved.get("installed"))

        installed_rpkg: Optional[Path] = None
        installed_wwes = 0
        installed_sha = ""
        is_installed = False

        if is_relevant and runtime_dir.is_dir():
            rpkg_name = (
                f"{saved['patch_rpkg_stem']}.rpkg"
                if (saved and saved.get("patch_rpkg_stem"))
                else sc.patch_rpkg_filename
            )
            cand_rpkg = runtime_dir / rpkg_name
            if cand_rpkg.is_file():
                try:
                    info, _entries, _del = parse_rpkg_file(cand_rpkg, filter_type="WWES")
                    # SesAI yama paketi mi kontrol et (kayıtlı durum, repack_raporu veya hazır profil eşleşmesi)
                    is_sesai_patch = bool(saved and saved.get("installed"))
                    if not is_sesai_patch:
                        rep_path = app_root / sc.base_dir_prefix / "paket" / "repack_raporu.json"
                        if rep_path.is_file():
                            try:
                                rep = json.loads(rep_path.read_text(encoding="utf-8"))
                                rep_sha = str(
                                    (rep.get("rpkg_package") or {}).get("rpkg_sha256") or ""
                                ).lower()
                                if rep_sha:
                                    live_sha = compute_sha256(cand_rpkg).lower()
                                    if live_sha == rep_sha:
                                        is_sesai_patch = True
                                        installed_sha = live_sha
                            except Exception:
                                pass
                    if is_sesai_patch:
                        is_installed = True
                        installed_rpkg = cand_rpkg
                        installed_wwes = info.wwes_count
                        if not installed_sha:
                            installed_sha = compute_sha256(cand_rpkg)
                except Exception:
                    pass

        merged_cnt = 0
        total_ext = 0
        if pkg_dirs:
            merged_items, _pdirs, total_ext = collect_merged_approved_dubs(app_root, sc)
            merged_cnt = len(merged_items)

        results.append(
            SceneRuntimeStatus(
                profile=sc,
                is_installed=is_installed,
                installed_rpkg_path=installed_rpkg,
                installed_wwes_count=installed_wwes,
                installed_rpkg_sha256=installed_sha,
                local_package_dirs=pkg_dirs,
                local_total_extracted=total_ext,
                local_approved_dubbed=merged_cnt,
                last_installed_utc=str((saved or {}).get("installed_at_utc") or ""),
            )
        )
    return results
