"""Hitman 3 / World of Assassination (Epic Games) Salt-Okunur Ses Çıkarma ve Güvenli Yama Analiz Köprüsü.

Özellikler:
1. Epic Games Launcher manifestlerini (`C:\\ProgramData\\Epic\\EpicGamesLauncher\\Data\\Manifests\\*.item`)
   ve standart kurulum dizinlerini salt-okunur tarayarak HITMAN 3 / World of Assassination
   kurulum klasörünü ve `Runtime\\` dizinini otomatik tespit eder.
2. `chunk29.rpkg` (+ `chunk29patch*.rpkg`) ve `chunk1.rpkg` (+ `chunk1patch*.rpkg`) dosyalarının
   RPKGv2 (`2KPR` / `GKPR`) başlıklarını ve silme (deletion) tablolarını okuyarak güncel (etkin)
   `WWES` (Wwise konuşma/diyalog) kaynaklarını listeler.
3. Taşınabilir `./_araclar/rpkg/rpkg-cli.exe` ve `./_araclar/ffmpeg` kullanarak `chunk29`
   (Serbest Ev / Freelancer Safehouse) içindeki tüm `WWES` konuşma seslerini:
   - Ham `.WWES` + `.WWES.meta` (+ `.WWES.meta.JSON`) olarak `./Hitman_SerbestEv/ham_wwes/` altına yedekler,
   - Temiz 48 kHz PCM_16 `.wav` formatına dönüştürüp `./Hitman_SerbestEv/sesler/` altına kaydeder,
   - Her dosyanın RPKG hash'ini, kaynak `.rpkg` adını, IOI assembly yolunu, varsa DLGE altyazı metnini,
     süresini, LUFS değerini ve RIFF/WEM hex başlık analizini `./Hitman_SerbestEv/hitman_manifest.json`
     dosyasına yazar.
4. Oyunun orijinal dosyalarına KESİNLİKLE dokunmaz ve proje klasörü dışına hiçbir dosya yazmaz.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backup_manager import compute_sha256
from core.ffmpeg_manager import FFmpegManager
from core.portable_env import (
    APP_ROOT,
    TEMP_DIR,
    TOOLS_DIR,
    bootstrap_portable_environment,
    cleanup_temp_dir,
    make_temp_subdir,
)

RPKG_TOOL_DIR = TOOLS_DIR / "rpkg"
RPKG_CLI_EXE = RPKG_TOOL_DIR / "rpkg-cli.exe"
RPKG_HASH_LIST = RPKG_TOOL_DIR / "hash_list.txt"

SAFEHOUSE_OUTPUT_ROOT = APP_ROOT / "Hitman_SerbestEv"
SAFEHOUSE_WAV_DIR = SAFEHOUSE_OUTPUT_ROOT / "sesler"
SAFEHOUSE_RAW_WWES_DIR = SAFEHOUSE_OUTPUT_ROOT / "ham_wwes"
SAFEHOUSE_MANIFEST_PATH = SAFEHOUSE_OUTPUT_ROOT / "hitman_manifest.json"

PROLOGUE_OUTPUT_ROOT = APP_ROOT / "Hitman_IlkSahne"
PROLOGUE_WAV_DIR = PROLOGUE_OUTPUT_ROOT / "sesler"
PROLOGUE_RAW_WWES_DIR = PROLOGUE_OUTPUT_ROOT / "ham_wwes"
PROLOGUE_MANIFEST_PATH = PROLOGUE_OUTPUT_ROOT / "hitman_manifest.json"

# ICA Tesisi (ICA Facility) İlk Eğitim Görevi — "Rehberli Eğitim (Guided Training)" & "Serbest Eğitim (Freeform Training)"
# Oyuna girer girmez 0. saniyeden itibaren sırayla tetiklenen ve sınırsız kez tekrarlanabilen 15 konuşma:
PROLOGUE_OPENING_15_TARGETS: tuple[dict[str, str], ...] = (
    {
        "wwes_hash": "00EF877A8B0484FC",
        "dlge_hash": "0055555E03438BC4",
        "friendly_name": "ds9268_arrival_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "ICA Tesisi -> 'Varış (Arrival)' görevi başlar başlamaz (0:00 sn, helikopter pistinde inince).",
    },
    {
        "wwes_hash": "00FD5831537234D0",
        "dlge_hash": "00CE162359A5EC5A",
        "friendly_name": "ds9215_flyin_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "ICA Tesisi -> 'Rehberli Eğitim (Guided Training)' görevi başlar başlamaz (0:00–0:18 sn, kamera yata doğru uçarken otomatik çalar).",
    },
    {
        "wwes_hash": "009BB25A2EF633C2",
        "dlge_hash": "003794C9492441AD",
        "friendly_name": "ds9216_instinct_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> Kamera uçuşu biter bitmez (0:18 sn, yat önü başlangıç noktasında kontrol size geçtiği an otomatik çalar).",
    },
    {
        "wwes_hash": "007523275A466473",
        "dlge_hash": "00D37DF2A854E01D",
        "friendly_name": "ds9216_instinct_young_diana_002",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> Başlangıç noktasında İçgüdü (Ctrl) tuşuna bastığınız anda (~0:25 sn) otomatik çalar.",
    },
    {
        "wwes_hash": "00BB586E5210CC86",
        "dlge_hash": "004A3736D6779CE3",
        "friendly_name": "ds9217_getstarted_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> İçgüdü tuşunu bıraktıktan 2 saniye sonra (~0:28 sn, yat iskelesine yönlendirirken) otomatik çalar.",
    },
    {
        "wwes_hash": "009D537EFD34FB38",
        "dlge_hash": "00F80D96602A8F38",
        "friendly_name": "ds9218_gangway_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> Başlangıç noktasından 5 metre ileriye, yat iskelesi (gangway) önündeki 2 korumaya yaklaştığınızda çalar.",
    },
    {
        "wwes_hash": "003A21A11C1F14B4",
        "dlge_hash": "00D58341360826CB",
        "friendly_name": "ds9219_firsttrespassing_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> İskelenin hemen sağındaki tamirci bölmesine (Trespassing / İzinsiz Giriş alanı) adım attığınız anda çalar.",
    },
    {
        "wwes_hash": "00AA4A59E3D0DA9B",
        "dlge_hash": "0060E66E97F3F2DB",
        "friendly_name": "ds9220_subdue_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> Başlangıç noktasının sağındaki kulübede sırtı dönük tamirciye yaklaştığınızda çalar.",
    },
    {
        "wwes_hash": "00F2898505D6B611",
        "dlge_hash": "000E430D283350D5",
        "friendly_name": "ds9221_concealbody_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> Tamirciyi bayılttığınız anda (cesedi dolaba saklama talimatı) çalar.",
    },
    {
        "wwes_hash": "00D1DE41CBDF6331",
        "dlge_hash": "00CF981446CC00D9",
        "friendly_name": "ds9222_bodydumpster_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> Tamirciyi dolaba sakladığınız anda çalar.",
    },
    {
        "wwes_hash": "00DEA49CD24BE83D",
        "dlge_hash": "00ADB4B00A0530AB",
        "friendly_name": "ds9223_mechanics_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> Tamirci kıyafetini (Mechanic Disguise) giydiğiniz anda çalar.",
    },
    {
        "wwes_hash": "0017B0FE07C356AF",
        "dlge_hash": "0062CBCBC335DC9B",
        "friendly_name": "ds9224_accessyachtcelebration_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "Rehberli Eğitim -> Tamirci kıyafetiyle iskeleden yata çıktığınız anda çalar.",
    },
    {
        "wwes_hash": "00525B83F0F4E91F",
        "dlge_hash": "000587C4FA7CE4CA",
        "friendly_name": "ds9006_yachtintro2_young_diana_001",
        "speaker": "Diana Burnwood",
        "trigger_moment": "ICA Tesisi -> 'Serbest Eğitim (Freeform Training)' görevi başlar başlamaz (0:00 sn, yat önünde doğduğunuz an) otomatik çalar.",
    },
    {
        "wwes_hash": "002C89DB6BC92D92",
        "dlge_hash": "00642256F306509D",
        "friendly_name": "dr9003_cargobay_civmale02_001",
        "speaker": "Yat Teknisyeni 1 (Cargo Bay Mechanic)",
        "trigger_moment": "Rehberli / Serbest Eğitim -> Yatın alt güverte girişindeki (kargo/garaj bölümü) iki teknisyenin yanına girdiğiniz anda başlayan diyalog (1. replik).",
    },
    {
        "wwes_hash": "0051196C5F3819F4",
        "dlge_hash": "0087DE43DFDA1788",
        "friendly_name": "dr9003_cargobay_civmale03_002",
        "speaker": "Yat Teknisyeni 2 (Cargo Bay Mechanic)",
        "trigger_moment": "Rehberli / Serbest Eğitim -> Yatın alt güverte girişindeki ikinci teknisyenin hemen ardından verdiği karşılık (2. replik).",
    },
)

WWISE_CODEC_NAMES: dict[int, str] = {
    0x0001: "PCM (Uncompressed Linear PCM)",
    0x0002: "ADPCM (Microsoft/Wwise IMA ADPCM)",
    0x0069: "Wwise IMA ADPCM (0x0069)",
    0x3039: "Wwise Opus (0x3039)",
    0x3040: "Wwise Opus (0x3040)",
    0x3041: "Wwise OpusWEM (0x3041)",
    0xFFFE: "WAVE_FORMAT_EXTENSIBLE / Wwise PCM (0xFFFE)",
    0xFFFF: "Wwise Custom Vorbis (0xFFFF)",
}


@dataclass
class RpkgFileInfo:
    name: str
    path: str
    size_bytes: int
    size_mb: float
    last_modified_iso: str
    magic: str
    is_patch_with_deletion_table: bool
    total_resources: int
    deleted_resources_count: int
    wwes_count: int


@dataclass
class WwesResourceEntry:
    hash_hex: str
    hash_id: str  # e.g. 0063350CB7FDD8A4.WWES
    source_rpkg: str
    data_offset: int
    raw_size_field: int
    final_size_bytes: int
    ioi_assembly_path: str = ""
    dialogue_subtitle_en: str = ""
    dialogue_subtitle_tr_mod: str = ""


@dataclass
class RiffWemHeaderAnalysis:
    file_name: str
    file_size_bytes: int
    riff_magic: str
    riff_declared_size: int
    wave_magic: str
    format_tag_int: int
    format_tag_hex: str
    codec_name: str
    channels: int
    sample_rate: int
    avg_bytes_per_sec: int
    approx_bitrate_kbps: float
    block_align: int
    bits_per_sample: int
    fmt_chunk_size: int
    extra_fmt_size: int
    vorbis_total_pcm_samples: int
    vorbis_estimated_duration_sec: float
    chunks_present: dict[str, int]
    header_hex_64: str


@dataclass
class ExtractedSafehouseVoice:
    index: int
    hash_hex: str
    hash_id: str
    source_rpkg: str
    ioi_assembly_path: str
    friendly_name: str
    wav_filename: str
    wav_rel_path: str
    raw_wwes_rel_path: str
    raw_meta_rel_path: str
    raw_meta_json_rel_path: str
    raw_wwes_size_bytes: int
    raw_wwes_sha256: str
    wav_size_bytes: int
    wav_sha256: str
    duration_sec: float
    duration_ms: int
    sample_rate: int
    channels: int
    lufs: float
    codec_in_wwes: str
    format_tag_hex: str
    dialogue_subtitle_en: str = ""
    dialogue_subtitle_tr: str = ""
    dialogue_subtitle_tr_mod: str = ""
    speaker_character: str = ""
    in_game_trigger_moment: str = ""
    meta_decoded: dict[str, Any] = field(default_factory=dict)
    wem_header: dict[str, Any] = field(default_factory=dict)


def _subprocess_win_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def find_epic_hitman_install() -> dict[str, Any]:
    """Epic Games manifestlerini ve standart klasörleri tarayarak Hitman 3 kurulumunu bulur."""
    manifest_dir = Path(r"C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests")
    matched_manifests: list[dict[str, str]] = []
    install_candidates: list[Path] = []

    if manifest_dir.is_dir():
        for item_file in sorted(manifest_dir.glob("*.item")):
            try:
                data = json.loads(item_file.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            disp = str(data.get("DisplayName", ""))
            loc = str(data.get("InstallLocation", ""))
            if "hitman" in disp.lower() or "hitman" in loc.lower():
                matched_manifests.append(
                    {
                        "manifest_file": item_file.name,
                        "display_name": disp,
                        "install_location": loc,
                        "app_name": str(data.get("AppName", "")),
                    }
                )
                if loc:
                    p = Path(loc)
                    if p not in install_candidates:
                        install_candidates.append(p)

    # Standart Epic Games yollarını da yedek olarak kontrol et
    for fallback in (
        Path(r"C:\Program Files\Epic Games\HITMAN3"),
        Path(r"D:\Program Files\Epic Games\HITMAN3"),
        Path(r"D:\Epic Games\HITMAN3"),
        Path(r"E:\Epic Games\HITMAN3"),
    ):
        if fallback not in install_candidates and fallback.is_dir():
            install_candidates.append(fallback)

    chosen_install: Optional[Path] = None
    for cand in install_candidates:
        if (cand / "Runtime").is_dir():
            chosen_install = cand
            break

    if chosen_install is None:
        raise FileNotFoundError(
            "HITMAN 3 / World of Assassination kurulum klasörü ve Runtime dizini bulunamadı."
        )

    runtime_dir = chosen_install / "Runtime"
    pkg_def = runtime_dir / "packagedefinition.txt"

    return {
        "install_dir": str(chosen_install),
        "runtime_dir": str(runtime_dir),
        "packagedefinition_path": str(pkg_def),
        "packagedefinition_exists": pkg_def.is_file(),
        "packagedefinition_size_bytes": pkg_def.stat().st_size if pkg_def.is_file() else 0,
        "epic_manifests_count": len(matched_manifests),
        "epic_manifests": matched_manifests,
    }


def parse_rpkg_file(
    rpkg_path: Path,
    filter_type: Optional[str] = "WWES",
) -> tuple[RpkgFileInfo, list[WwesResourceEntry], set[str]]:
    """RPKGv2 (2KPR) veya RPKGv1 (GKPR) başlığını salt-okunur tarar.

    Returns:
        (rpkg_file_info, matched_entries, deleted_hash_hex_set)
    """
    stat = rpkg_path.stat()
    mtime_iso = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

    with open(rpkg_path, "rb") as f:
        magic_bytes = f.read(4)
        magic = magic_bytes.decode("ascii", errors="replace")
        if magic_bytes == b"2KPR":
            f.read(9)
        elif magic_bytes != b"GKPR":
            raise ValueError(f"Desteklenmeyen RPKG başlığı: {magic!r} ({rpkg_path})")

        num_hashes, index_size, info_size = struct.unpack("<III", f.read(12))
        pos_after_counts = f.tell()

        has_deletion_table = False
        if num_hashes > 0:
            peek_idx = f.read(20)
            _, first_data_offset, _ = struct.unpack("<QQI", peek_idx)
            expected_non_patch_offset = pos_after_counts + index_size + info_size
            has_deletion_table = first_data_offset != expected_non_patch_offset

        deleted_hashes: set[str] = set()
        f.seek(pos_after_counts)
        if has_deletion_table:
            num_deleted = struct.unpack("<I", f.read(4))[0]
            if 0 < num_deleted < 10_000_000:
                del_raw = f.read(8 * num_deleted)
                deleted_hashes = {
                    f"{h:016X}" for h in struct.unpack(f"<{num_deleted}Q", del_raw)
                }

        index_raw = f.read(20 * num_hashes)
        matched_entries: list[WwesResourceEntry] = []
        wwes_count = 0

        for i in range(num_hashes):
            h_val, data_offset, data_size = struct.unpack_from("<QQI", index_raw, i * 20)
            res_type_rev = f.read(4)
            res_type = res_type_rev[::-1].decode("ascii", errors="replace")
            ref_table_size, _ref_dummy, final_size, _mem_sz, _vmem_sz = struct.unpack(
                "<IIIII", f.read(20)
            )
            if ref_table_size > 0:
                f.seek(ref_table_size, 1)

            if res_type == "WWES":
                wwes_count += 1

            if filter_type is None or res_type == filter_type:
                h_hex = f"{h_val:016X}"
                matched_entries.append(
                    WwesResourceEntry(
                        hash_hex=h_hex,
                        hash_id=f"{h_hex}.{res_type}",
                        source_rpkg=rpkg_path.name,
                        data_offset=data_offset,
                        raw_size_field=data_size,
                        final_size_bytes=final_size,
                    )
                )

    info = RpkgFileInfo(
        name=rpkg_path.name,
        path=str(rpkg_path),
        size_bytes=stat.st_size,
        size_mb=round(stat.st_size / (1024 * 1024), 2),
        last_modified_iso=mtime_iso,
        magic=magic,
        is_patch_with_deletion_table=has_deletion_table,
        total_resources=num_hashes,
        deleted_resources_count=len(deleted_hashes),
        wwes_count=wwes_count,
    )
    return info, matched_entries, deleted_hashes


def list_chunk_rpkg_files(runtime_dir: Path, chunk_number: int) -> list[Path]:
    """Belirtilen chunk (ör. 29 veya 1) için ana .rpkg ve sıralı patch*.rpkg dosyalarını döndürür."""
    base_file = runtime_dir / f"chunk{chunk_number}.rpkg"
    files: list[Path] = []
    if base_file.is_file():
        files.append(base_file)

    patch_pattern = re.compile(rf"^chunk{chunk_number}patch(\d+)\.rpkg$", re.IGNORECASE)
    patches: list[tuple[int, Path]] = []
    for p in runtime_dir.glob(f"chunk{chunk_number}patch*.rpkg"):
        m = patch_pattern.match(p.name)
        if m:
            patches.append((int(m.group(1)), p))
    patches.sort(key=lambda x: x[0])
    files.extend(p for _, p in patches)
    return files


def scan_chunk_wwes(
    runtime_dir: Path,
    chunk_number: int,
) -> tuple[list[RpkgFileInfo], dict[str, WwesResourceEntry], int]:
    """Bir chunk ve tüm patch dosyalarındaki WWES kayıtlarını tarar; güncel patch hash'lerini önceliklendirir.

    Returns:
        (rpkg_files_info, active_wwes_by_hash_hex, total_raw_wwes_entries_across_files)
    """
    rpkg_files = list_chunk_rpkg_files(runtime_dir, chunk_number)
    file_infos: list[RpkgFileInfo] = []
    active_map: dict[str, WwesResourceEntry] = {}
    raw_total = 0

    for rp in rpkg_files:
        info, entries, deleted = parse_rpkg_file(rp, filter_type="WWES")
        file_infos.append(info)
        raw_total += len(entries)
        for del_h in deleted:
            active_map.pop(del_h, None)
        for entry in entries:
            # Üstteki patch dosyası aynı hash'i içeriyorsa güncel sürüm olarak üzerine yazar
            active_map[entry.hash_hex] = entry

    return file_infos, active_map, raw_total


def resolve_ioi_paths_from_hash_list(hash_ids: set[str]) -> dict[str, str]:
    """`./_araclar/rpkg/hash_list.txt` dosyasından verilen hash_id (ör. `0063350CB7FDD8A4.WWES`) yollarını bulur."""
    if not hash_ids or not RPKG_HASH_LIST.is_file():
        return {}
    remaining = set(hash_ids)
    resolved: dict[str, str] = {}
    with open(RPKG_HASH_LIST, "r", encoding="utf-8", errors="replace") as hf:
        for line in hf:
            if not remaining:
                break
            comma_idx = line.find(",")
            if comma_idx <= 0:
                continue
            key = line[:comma_idx].strip().upper()
            if key in remaining:
                resolved[key] = line[comma_idx + 1 :].strip()
                remaining.remove(key)
    return resolved


def inspect_wwes_riff_header(wwes_path: Path) -> RiffWemHeaderAnalysis:
    """Ham `.WWES` (Wwise `.wem`) dosyasının RIFF/WAVE binary başlığını detaylı inceler."""
    data = wwes_path.read_bytes()
    if len(data) < 36:
        raise ValueError(f"WWES dosyası çok küçük ({len(data)} bayt): {wwes_path}")

    riff_magic = data[:4].decode("ascii", errors="replace")
    riff_size = struct.unpack_from("<I", data, 4)[0]
    wave_magic = data[8:12].decode("ascii", errors="replace")

    pos = 12
    chunks: dict[str, int] = {}
    fmt_tag = 0
    channels = 0
    sample_rate = 0
    avg_bps = 0
    block_align = 0
    bits_per_sample = 0
    fmt_chunk_size = 0
    ext_size = 0
    vorbis_total_pcm_samples = 0

    while pos + 8 <= len(data):
        cid = data[pos : pos + 4].decode("ascii", errors="replace")
        csz = struct.unpack_from("<I", data, pos + 4)[0]
        chunks[cid] = csz
        if cid == "fmt " and csz >= 16:
            fmt_chunk_size = csz
            fmt_tag, channels, sample_rate, avg_bps, block_align, bits_per_sample = (
                struct.unpack_from("<HHIIHH", data, pos + 8)
            )
            if csz >= 18:
                ext_size = struct.unpack_from("<H", data, pos + 24)[0]
            # Wwise Custom Vorbis (0xFFFF, fmt size 66 / ext 48) içinde fmt payload +24 (dosya başından +44) toplam PCM örnek sayısıdır
            if fmt_tag == 0xFFFF and csz >= 28:
                vorbis_total_pcm_samples = struct.unpack_from("<I", data, pos + 8 + 24)[0]
        pos += 8 + csz + (csz % 2)

    est_dur = (
        round(vorbis_total_pcm_samples / float(sample_rate), 3)
        if (sample_rate > 0 and vorbis_total_pcm_samples > 0)
        else 0.0
    )
    codec_name = WWISE_CODEC_NAMES.get(fmt_tag, f"Unknown Wwise Codec (0x{fmt_tag:04X})")

    return RiffWemHeaderAnalysis(
        file_name=wwes_path.name,
        file_size_bytes=len(data),
        riff_magic=riff_magic,
        riff_declared_size=riff_size,
        wave_magic=wave_magic,
        format_tag_int=fmt_tag,
        format_tag_hex=f"0x{fmt_tag:04X}",
        codec_name=codec_name,
        channels=channels,
        sample_rate=sample_rate,
        avg_bytes_per_sec=avg_bps,
        approx_bitrate_kbps=round((avg_bps * 8.0) / 1000.0, 2),
        block_align=block_align,
        bits_per_sample=bits_per_sample,
        fmt_chunk_size=fmt_chunk_size,
        extra_fmt_size=ext_size,
        vorbis_total_pcm_samples=vorbis_total_pcm_samples,
        vorbis_estimated_duration_sec=est_dur,
        chunks_present=chunks,
        header_hex_64=data[:64].hex(" "),
    )


def check_modding_tools(hitman_install_dir: Path) -> dict[str, Any]:
    """Bilgisayarda Simple Mod Framework (SMF) ve Audiokinetic Wwise kurulu olup olmadığını kontrol eder."""
    smf_candidates = [
        hitman_install_dir / "Simple Mod Framework",
        hitman_install_dir / "Simple Mod Framework" / "Mod Manager.exe",
        hitman_install_dir / "Simple Mod Framework" / "LoadOrderManager.exe",
    ]
    smf_installed = any(p.exists() for p in smf_candidates)

    wwise_root_env = os.environ.get("WWISEROOT", "") or os.environ.get("WWISE_SDK", "")
    wwise_dirs = [
        Path(r"C:\Program Files (x86)\Audiokinetic"),
        Path(r"C:\Program Files\Audiokinetic"),
        Path(r"D:\Program Files (x86)\Audiokinetic"),
    ]
    if wwise_root_env:
        wwise_dirs.insert(0, Path(wwise_root_env))

    wwise_found_path: Optional[str] = None
    for wd in wwise_dirs:
        if wd.exists():
            wwise_found_path = str(wd)
            break

    return {
        "rpkg_cli_portable_installed": RPKG_CLI_EXE.is_file(),
        "rpkg_cli_path": str(RPKG_CLI_EXE),
        "simple_mod_framework_installed": smf_installed,
        "simple_mod_framework_expected_dir": str(hitman_install_dir / "Simple Mod Framework"),
        "audiokinetic_wwise_installed": wwise_found_path is not None,
        "audiokinetic_wwise_path": wwise_found_path or "",
    }


def _run_rpkg_cli(args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    if not RPKG_CLI_EXE.is_file():
        raise FileNotFoundError(f"Taşınabilir rpkg-cli.exe bulunamadı: {RPKG_CLI_EXE}")
    cmd = [str(RPKG_CLI_EXE)] + args
    return subprocess.run(
        cmd,
        cwd=str(cwd or RPKG_TOOL_DIR),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        **_subprocess_win_kwargs(),
    )


def _extract_chunk29_subtitles_from_dlge(
    runtime_dir: Path,
    work_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """`chunk29` ve `chunk29patch*` içindeki DLGE dosyalarını geçici olarak JSON'a çözüp
    wavName -> İngilizce orijinal metin ve (varsa) mevcut Türkçe altyazı yamasını eşleştirir.
    """
    en_by_wav_stem: dict[str, str] = {}
    tr_mod_by_wav_stem: dict[str, str] = {}

    dlge_work = work_dir / "dlge_extract"
    dlge_work.mkdir(parents=True, exist_ok=True)

    for rpkg_name in ("chunk29.rpkg", "chunk29patch1.rpkg", "chunk29patch2.rpkg"):
        rp = runtime_dir / rpkg_name
        if not rp.is_file():
            continue
        _run_rpkg_cli(["-output_path", str(dlge_work), "-filter", "DLGE", "-extract_from_rpkg", str(rp)])
        chunk_sub = dlge_work / rp.stem
        if (chunk_sub / "DLGE").is_dir():
            _run_rpkg_cli(["-output_path", str(dlge_work), "-generate_rpkg_quickly_from", str(chunk_sub)])

    json_out = dlge_work / "json"
    _run_rpkg_cli(["-output_path", str(json_out), "-extract_dlge_to_json_from", str(dlge_work)])

    def _clean_dlge_text(raw: str) -> str:
        # //[DIANA]\\//(0,2.652)\\Welcome to your safehouse, 47. -> temiz metin
        cleaned = re.sub(r"//\[[^\]]*\]\\\\", "", raw)
        cleaned = re.sub(r"//\([0-9.,]+\)\\\\", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _walk_container(node: Any, source_rpkg_name: str) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "WavFile":
            wav_name = str(node.get("wavName", "")).strip().lower()
            langs = node.get("languages", {})
            if wav_name and isinstance(langs, dict):
                en_txt = _clean_dlge_text(str(langs.get("en", "")))
                if en_txt:
                    if source_rpkg_name == "chunk29patch2.rpkg":
                        tr_mod_by_wav_stem[wav_name] = en_txt
                    else:
                        en_by_wav_stem[wav_name] = en_txt
        for child in node.get("containers", []) or []:
            _walk_container(child, source_rpkg_name)

    for jf in sorted(json_out.rglob("*.dlge.json")):
        try:
            jdata = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
            src_rpkg = jf.parent.name  # e.g. chunk29.rpkg or chunk29patch2.rpkg
            _walk_container(jdata.get("rootContainer", {}), src_rpkg)
        except Exception:
            continue

    return en_by_wav_stem, tr_mod_by_wav_stem


def _inspect_packagedefinition_partitions(runtime_dir: Path, work_dir: Path) -> list[dict[str, Any]]:
    """`packagedefinition.txt` dosyasını `./_temp` içine kopyalayıp şifresini çözerek @partition bilgilerini okur."""
    pkg_src = runtime_dir / "packagedefinition.txt"
    if not pkg_src.is_file():
        return []
    pkg_copy = work_dir / "packagedefinition.txt"
    shutil.copy2(pkg_src, pkg_copy)
    _run_rpkg_cli(["-output_path", str(work_dir), "-decrypt_packagedefinition_thumbs", str(pkg_copy)])
    dec_path = work_dir / "packagedefinition.txt.decrypted"
    if not dec_path.is_file():
        return []

    partitions: list[dict[str, Any]] = []
    idx = 0
    for line in dec_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s.startswith("@partition"):
            attrs: dict[str, str] = {}
            for part in s.split()[1:]:
                if "=" in part:
                    k, v = part.split("=", 1)
                    attrs[k] = v
            partitions.append(
                {
                    "chunk_index": idx,
                    "chunk_name": f"chunk{idx}",
                    "codename": attrs.get("name", ""),
                    "parent": attrs.get("parent", ""),
                    "type": attrs.get("type", ""),
                    "patchlevel": int(attrs.get("patchlevel", "0") or 0),
                    "raw_line": s,
                }
            )
            idx += 1
    return partitions


def extract_chunk29_safehouse(
    output_root: Path = SAFEHOUSE_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Hitman Serbest Ev (`chunk29` + `chunk29patch*`) WWES konuşma seslerini çıkarır,
    48 kHz `.wav` formatına dönüştürür, `chunk1` istatistiklerini ve RIFF/WEM geri paketleme
    analizini `hitman_manifest.json` olarak kaydeder.
    """
    bootstrap_portable_environment()
    install_info = find_epic_hitman_install()
    hitman_install_dir = Path(install_info["install_dir"])
    runtime_dir = Path(install_info["runtime_dir"])

    # 1. chunk29 ve chunk1 WWES taraması (Salt-Okunur)
    c29_files_info, c29_active_wwes, c29_raw_total = scan_chunk_wwes(runtime_dir, 29)
    c1_files_info, c1_active_wwes, c1_raw_total = scan_chunk_wwes(runtime_dir, 1)

    # IOI assembly yollarını hash_list.txt içinden çöz
    c29_hash_ids = {entry.hash_id for entry in c29_active_wwes.values()}
    resolved_paths = resolve_ioi_paths_from_hash_list(c29_hash_ids)
    for entry in c29_active_wwes.values():
        entry.ioi_assembly_path = resolved_paths.get(entry.hash_id, "")

    wav_dir = output_root / "sesler"
    raw_wwes_dir = output_root / "ham_wwes"
    manifest_path = output_root / "hitman_manifest.json"
    if wav_dir.exists():
        shutil.rmtree(wav_dir)
    if raw_wwes_dir.exists():
        shutil.rmtree(raw_wwes_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    raw_wwes_dir.mkdir(parents=True, exist_ok=True)

    work_dir = TEMP_DIR / "hb"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = FFmpegManager()

    try:
        # 2. packagedefinition.txt ve DLGE altyazılarını ./_temp içinde oku
        partitions = _inspect_packagedefinition_partitions(runtime_dir, work_dir)
        en_subtitles, tr_mod_subtitles = _extract_chunk29_subtitles_from_dlge(runtime_dir, work_dir)

        # 3. chunk29 ve chunk29patch* içinden aktif WWES'leri ./_temp içine çıkar
        # Not: Windows MAX_PATH (260 karakter) sınırına takılmamak için kısa dizin adları (r / o) kullanılır.
        raw_extract_dir = work_dir / "r"
        raw_extract_dir.mkdir(parents=True, exist_ok=True)

        rpkgs_needed = sorted({entry.source_rpkg for entry in c29_active_wwes.values()})
        for rpkg_name in rpkgs_needed:
            rpkg_path = runtime_dir / rpkg_name
            _run_rpkg_cli(
                [
                    "-output_path",
                    str(raw_extract_dir),
                    "-filter",
                    "WWES",
                    "-extract_from_rpkg",
                    str(rpkg_path),
                ]
            )
            chunk_folder = raw_extract_dir / rpkg_path.stem
            if (chunk_folder / "WWES").is_dir():
                # Hızlı mini-RPKG oluştur ki -extract_wwes_to_ogg_from yalnızca bu 11 dosyayı <1 saniyede OGG'a çözsün
                _run_rpkg_cli(
                    [
                        "-output_path",
                        str(raw_extract_dir),
                        "-generate_rpkg_quickly_from",
                        str(chunk_folder),
                    ]
                )

        # 4. Mini-RPKG üzerinden rpkg-cli'nin yerleşik Wwise Vorbis -> OGG çözücüsünü çalıştır
        ogg_extract_dir = work_dir / "o"
        ogg_extract_dir.mkdir(parents=True, exist_ok=True)
        _run_rpkg_cli(
            [
                "-output_path",
                str(ogg_extract_dir),
                "-extract_wwes_to_ogg_from",
                str(raw_extract_dir),
            ]
        )

        # Üretilen .ogg dosyalarını dosya kök adına (stem) göre indeksle
        ogg_by_stem: dict[str, Path] = {}
        for ogg_file in ogg_extract_dir.rglob("*.ogg"):
            ogg_by_stem[ogg_file.stem.lower()] = ogg_file

        # 5. Sıralama: Önce chunk29.rpkg (Diana Safehouse Onboarding), ardından IOI dosya adına göre
        def _sort_key(e: WwesResourceEntry) -> tuple[int, str, str]:
            is_patch = 0 if e.source_rpkg == "chunk29.rpkg" else 1
            return (is_patch, e.ioi_assembly_path.lower(), e.hash_hex)

        sorted_entries = sorted(c29_active_wwes.values(), key=_sort_key)
        extracted_voices: list[ExtractedSafehouseVoice] = []

        for idx, entry in enumerate(sorted_entries, start=1):
            chunk_stem = Path(entry.source_rpkg).stem
            src_wwes = raw_extract_dir / chunk_stem / "WWES" / f"{entry.hash_hex}.WWES"
            src_meta = raw_extract_dir / chunk_stem / "WWES" / f"{entry.hash_hex}.WWES.meta"
            if not src_wwes.is_file() or not src_meta.is_file():
                raise FileNotFoundError(
                    f"Beklenen ham WWES veya meta dosyası çıkarılamadı: {src_wwes}"
                )

            # Ham .WWES ve .WWES.meta dosyalarını Hitman_SerbestEv/ham_wwes/ altına yedekle
            dst_raw_wwes = raw_wwes_dir / f"{entry.hash_hex}.WWES"
            dst_raw_meta = raw_wwes_dir / f"{entry.hash_hex}.WWES.meta"
            shutil.copy2(src_wwes, dst_raw_wwes)
            shutil.copy2(src_meta, dst_raw_meta)

            # .WWES.meta dosyasını JSON'a da çöz (.WWES.meta.JSON)
            _run_rpkg_cli(["-hash_meta_to_json", str(dst_raw_meta)])
            dst_raw_meta_json = raw_wwes_dir / f"{entry.hash_hex}.WWES.meta.JSON"
            meta_decoded: dict[str, Any] = {}
            if dst_raw_meta_json.is_file():
                try:
                    meta_decoded = json.loads(
                        dst_raw_meta_json.read_text(encoding="utf-8", errors="replace")
                    )
                except Exception:
                    meta_decoded = {}

            # RIFF/WEM hex başlığını analiz et
            wem_hdr = inspect_wwes_riff_header(dst_raw_wwes)

            # IOI yolundan anlamlı dosya adını çıkar (ör. ed002_onboarding_diana_001)
            m_name = re.search(r"/([^/\\]+?)\.wav\]\.pc_wes", entry.ioi_assembly_path, re.IGNORECASE)
            if m_name:
                friendly_stem = m_name.group(1)
            else:
                friendly_stem = f"wwes_{entry.hash_hex.lower()}"

            ogg_path = ogg_by_stem.get(friendly_stem.lower())
            if ogg_path is None or not ogg_path.is_file():
                # Hash ile eşleşen yedek kontrol
                ogg_path = ogg_by_stem.get(entry.hash_hex.lower())
            if ogg_path is None or not ogg_path.is_file():
                raise FileNotFoundError(
                    f"{entry.hash_id} ({friendly_stem}) için rpkg-cli OGG çıktısı bulunamadı."
                )

            # SesAI Dublaj Stüdyosu'nun doğrudan okuyabileceği temiz PCM_16 .wav dosyasına dönüştür
            # Orijinal WEM örnekleme hızını (48000 Hz) ve kanal sayısını (Stereo=2 / Mono=1) birebir koru!
            wav_filename = f"{idx:02d}_{friendly_stem}_{entry.hash_hex}.wav"
            dst_wav = wav_dir / wav_filename
            target_sr = wem_hdr.sample_rate if wem_hdr.sample_rate > 0 else 48000
            target_ch = wem_hdr.channels if wem_hdr.channels > 0 else 1
            ffmpeg.convert_to_wav(
                ogg_path,
                dst_wav,
                sample_rate=target_sr,
                channels=target_ch,
            )

            audio_meta = ffmpeg.probe_audio(dst_wav, measure_lufs=True)
            en_sub = en_subtitles.get(friendly_stem.lower(), "")
            tr_sub = tr_mod_subtitles.get(friendly_stem.lower(), "")

            extracted_voices.append(
                ExtractedSafehouseVoice(
                    index=idx,
                    hash_hex=entry.hash_hex,
                    hash_id=entry.hash_id,
                    source_rpkg=entry.source_rpkg,
                    ioi_assembly_path=entry.ioi_assembly_path,
                    friendly_name=friendly_stem,
                    wav_filename=wav_filename,
                    wav_rel_path=str(dst_wav.relative_to(output_root)).replace("\\", "/"),
                    raw_wwes_rel_path=str(dst_raw_wwes.relative_to(output_root)).replace("\\", "/"),
                    raw_meta_rel_path=str(dst_raw_meta.relative_to(output_root)).replace("\\", "/"),
                    raw_meta_json_rel_path=(
                        str(dst_raw_meta_json.relative_to(output_root)).replace("\\", "/")
                        if dst_raw_meta_json.is_file()
                        else ""
                    ),
                    raw_wwes_size_bytes=dst_raw_wwes.stat().st_size,
                    raw_wwes_sha256=compute_sha256(dst_raw_wwes),
                    wav_size_bytes=dst_wav.stat().st_size,
                    wav_sha256=compute_sha256(dst_wav),
                    duration_sec=round(audio_meta.duration_sec, 3),
                    duration_ms=audio_meta.duration_ms,
                    sample_rate=audio_meta.sample_rate,
                    channels=audio_meta.channels,
                    lufs=round(audio_meta.lufs, 2),
                    codec_in_wwes=wem_hdr.codec_name,
                    format_tag_hex=wem_hdr.format_tag_hex,
                    dialogue_subtitle_en=en_sub,
                    dialogue_subtitle_tr_mod=tr_sub,
                    meta_decoded=meta_decoded,
                    wem_header=asdict(wem_hdr),
                )
            )

        mod_tools = check_modding_tools(hitman_install_dir)
        sample_header = extracted_voices[0].wem_header if extracted_voices else {}

        chunk29_partition = next(
            (p for p in partitions if p["chunk_name"] == "chunk29"),
            {"chunk_name": "chunk29", "codename": "snug", "patchlevel": 1},
        )
        chunk1_partition = next(
            (p for p in partitions if p["chunk_name"] == "chunk1"),
            {"chunk_name": "chunk1", "codename": "base", "patchlevel": 4},
        )

        manifest_data: dict[str, Any] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "SesAI Dublaj Stüdyosu — Hitman RPKG/WWES Köprüsü (tools/hitman_bridge.py)",
            "read_only_guarantee": True,
            "hitman_installation": install_info,
            "modding_environment": mod_tools,
            "packagedefinition_summary": {
                "total_partitions": len(partitions),
                "chunk29_partition": chunk29_partition,
                "chunk1_partition": chunk1_partition,
            },
            "chunk29_safehouse_summary": {
                "chunk_name": "chunk29",
                "ioi_codename": "snug (Freelancer Safehouse / Serbest Ev)",
                "rpkg_files": [asdict(x) for x in c29_files_info],
                "total_raw_wwes_entries": c29_raw_total,
                "effective_unique_active_wwes": len(c29_active_wwes),
                "extracted_wav_count": len(extracted_voices),
                "extracted_wav_dir": str(wav_dir),
                "raw_wwes_backup_dir": str(raw_wwes_dir),
            },
            "chunk1_opening_scenes_summary": {
                "chunk_name": "chunk1",
                "ioi_codename": "base (Prologue / ICA Facility & Base Game Shared Voices)",
                "rpkg_files": [asdict(x) for x in c1_files_info],
                "total_raw_wwes_entries": c1_raw_total,
                "effective_unique_active_wwes": len(c1_active_wwes),
                "note": "Bilgi amaçlı sayılmıştır; şimdilik yalnızca chunk29 (Serbest Ev) sesleri çıkarılmıştır.",
            },
            "repack_technical_analysis": {
                "sample_file_analyzed": sample_header.get("file_name", ""),
                "container_format": "RIFF / WAVE (Wwise Encoded Media .wem / .WWES)",
                "codec_format_tag_hex": sample_header.get("format_tag_hex", "0xFFFF"),
                "codec_name": sample_header.get("codec_name", "Wwise Custom Vorbis (0xFFFF)"),
                "sample_rate_hz": sample_header.get("sample_rate", 48000),
                "fmt_chunk_size_bytes": sample_header.get("fmt_chunk_size", 66),
                "extra_fmt_size_bytes": sample_header.get("extra_fmt_size", 48),
                "subchunks_present": list((sample_header.get("chunks_present") or {}).keys()),
                "sample_header_hex_64": sample_header.get("header_hex_64", ""),
                "meta_file_structure": (
                    "44 bayt RPKGv2 meta: [0..8]=hash_u64, [8..16]=offset_u64, "
                    "[16..20]=size_flag_u32 (0x80000000 = sıkıştırmasız), [20..24]='WWES', "
                    "[24..28]=ref_table_size(0), [28..32]=ref_dummy(0), [32..36]=hash_size_final_u32 (dosya bayt boyutu), "
                    "[36..44]=0xFFFFFFFFFFFFFFFF"
                ),
                "recommended_safe_repack_workflow": [
                    "1. Orijinal chunk29.rpkg ve chunk29patch1.rpkg dosyalarına KESİNLİKLE dokunulmaz.",
                    "2. SesAI Dublaj Stüdyosu ile './Hitman_SerbestEv/sesler/' içindeki 48 kHz .wav dosyaları Türkçe dublajlanır.",
                    "3. Türkçe .wav dosyaları Wwise Vorbis (0xFFFF) veya Wwise PCM (.wem) formatına dönüştürülüp orijinal 16 haneli RPKG hash adı (<HASH>.WWES) ile adlandırılır.",
                    "4. Eşlik eden <HASH>.WWES.meta dosyasındaki 'hash_size_final' (offset 32..36, uint32) yeni .WWES dosyasının bayt uzunluğuyla güncellenir (veya rpkg-cli -json_to_hash_meta kullanılır).",
                    "5. './_araclar/rpkg/rpkg-cli.exe -generate_rpkg_from chunk29patch3' komutuyla bağımsız bir yama paketi (chunk29patch3.rpkg) üretilir ve packagedefinition.txt içindeki 'snug' patchlevel değeri şifresi çözülüp artırılarak (veya Simple Mod Framework RPKG mod klasörüyle) oyuna güvenli şekilde yüklenir.",
                ],
            },
            "files": [asdict(v) for v in extracted_voices],
        }

        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest_data
    finally:
        cleanup_temp_dir(keep_root=True)


def extract_prologue_opening_scene(
    output_root: Path = PROLOGUE_OUTPUT_ROOT,
) -> dict[str, Any]:
    """HITMAN 3 ICA Tesisi İlk Eğitim Görevi ('Rehberli Eğitim / Guided Training' ve
    'Serbest Eğitim / Freeform Training') başlangıcında ilk saniyelerde anında tetiklenen
    ve sınırsız kez tekrarlanabilen 15 konuşma (`WWES`) dosyasını çıkarır.
    """
    bootstrap_portable_environment()
    install_info = find_epic_hitman_install()
    hitman_install_dir = Path(install_info["install_dir"])
    runtime_dir = Path(install_info["runtime_dir"])

    # chunk0 ve chunk1 dosyalarını salt-okunur tara
    c0_files_info, c0_active_wwes, c0_raw_total = scan_chunk_wwes(runtime_dir, 0)
    c1_files_info, c1_active_wwes, c1_raw_total = scan_chunk_wwes(runtime_dir, 1)

    target_hash_hexes = {t["wwes_hash"].upper() for t in PROLOGUE_OPENING_15_TARGETS}
    target_hash_ids = {f"{h}.WWES" for h in target_hash_hexes}
    resolved_paths = resolve_ioi_paths_from_hash_list(target_hash_ids)

    selected_entries: dict[str, WwesResourceEntry] = {}
    for h_hex in target_hash_hexes:
        if h_hex in c0_active_wwes:
            ent = c0_active_wwes[h_hex]
        elif h_hex in c1_active_wwes:
            ent = c1_active_wwes[h_hex]
        else:
            raise KeyError(f"Hedef WWES hash bulunamadı: {h_hex}")
        ent.ioi_assembly_path = resolved_paths.get(ent.hash_id, "")
        selected_entries[h_hex] = ent

    wav_dir = output_root / "sesler"
    raw_wwes_dir = output_root / "ham_wwes"
    manifest_path = output_root / "hitman_manifest.json"
    if wav_dir.exists():
        shutil.rmtree(wav_dir)
    if raw_wwes_dir.exists():
        shutil.rmtree(raw_wwes_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)
    raw_wwes_dir.mkdir(parents=True, exist_ok=True)

    work_dir = TEMP_DIR / "hp0"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = FFmpegManager()

    try:
        partitions = _inspect_packagedefinition_partitions(runtime_dir, work_dir)

        # 1. 15 hedefe ait DLGE dosyalarından İngilizce (chunk0.rpkg) ve Türkçe (chunk0patch6.rpkg) altyazıları çöz
        dlge_work = work_dir / "d"
        dlge_work.mkdir(parents=True, exist_ok=True)
        dlge_filter = ",".join(t["dlge_hash"].upper() for t in PROLOGUE_OPENING_15_TARGETS)
        for rname in ("chunk0.rpkg", "chunk0patch2.rpkg", "chunk0patch6.rpkg"):
            rp = runtime_dir / rname
            if not rp.is_file():
                continue
            _run_rpkg_cli(
                ["-output_path", str(dlge_work), "-filter", dlge_filter, "-extract_from_rpkg", str(rp)]
            )
            stem = rp.stem
            if (dlge_work / stem / "DLGE").is_dir():
                _run_rpkg_cli(
                    ["-output_path", str(dlge_work), "-generate_rpkg_quickly_from", str(dlge_work / stem)]
                )

        dlge_json_dir = dlge_work / "j"
        _run_rpkg_cli(["-output_path", str(dlge_json_dir), "-extract_dlge_to_json_from", str(dlge_work)])

        def _clean_dlge_line(raw: str) -> str:
            cleaned = re.sub(r"//\[[^\]]*\]\\\\", "", raw)
            cleaned = re.sub(r"//\([0-9.,]+\)\\\\", " ", cleaned)
            return re.sub(r"\s+", " ", cleaned).strip()

        en_by_dlge: dict[str, str] = {}
        tr_by_dlge: dict[str, str] = {}
        for t in PROLOGUE_OPENING_15_TARGETS:
            dh = t["dlge_hash"].upper()
            for en_rpkg in ("chunk0patch2.rpkg", "chunk0.rpkg"):
                ef = dlge_json_dir / "DLGE" / en_rpkg / f"{dh}.dlge.json"
                if ef.is_file():
                    try:
                        j = json.loads(ef.read_text(encoding="utf-8", errors="replace"))
                        txt = _clean_dlge_line((j.get("rootContainer", {}).get("languages") or {}).get("en", ""))
                        if txt:
                            en_by_dlge[dh] = txt
                            break
                    except Exception:
                        pass
            tf = dlge_json_dir / "DLGE" / "chunk0patch6.rpkg" / f"{dh}.dlge.json"
            if tf.is_file():
                try:
                    j = json.loads(tf.read_text(encoding="utf-8", errors="replace"))
                    txt = _clean_dlge_line((j.get("rootContainer", {}).get("languages") or {}).get("en", ""))
                    if txt:
                        tr_by_dlge[dh] = txt
                except Exception:
                    pass

        # 2. Yalnızca seçilen 15 WWES dosyasını kaynak RPKG'den (-filter ile) ./_temp içine çıkar
        raw_extract_dir = work_dir / "r"
        raw_extract_dir.mkdir(parents=True, exist_ok=True)
        wwes_filter = ",".join(t["wwes_hash"].upper() for t in PROLOGUE_OPENING_15_TARGETS)
        rpkgs_needed = sorted({e.source_rpkg for e in selected_entries.values()})

        for rpkg_name in rpkgs_needed:
            rpkg_path = runtime_dir / rpkg_name
            _run_rpkg_cli(
                [
                    "-output_path",
                    str(raw_extract_dir),
                    "-filter",
                    wwes_filter,
                    "-extract_from_rpkg",
                    str(rpkg_path),
                ]
            )
            chunk_folder = raw_extract_dir / rpkg_path.stem
            if (chunk_folder / "WWES").is_dir():
                _run_rpkg_cli(
                    [
                        "-output_path",
                        str(raw_extract_dir),
                        "-generate_rpkg_quickly_from",
                        str(chunk_folder),
                    ]
                )

        # 3. Mini-RPKG üzerinden rpkg-cli'nin yerleşik Wwise Vorbis -> OGG çözücüsünü çalıştır
        ogg_extract_dir = work_dir / "o"
        ogg_extract_dir.mkdir(parents=True, exist_ok=True)
        _run_rpkg_cli(
            [
                "-output_path",
                str(ogg_extract_dir),
                "-extract_wwes_to_ogg_from",
                str(raw_extract_dir),
            ]
        )

        ogg_by_stem: dict[str, Path] = {}
        for ogg_file in ogg_extract_dir.rglob("*.ogg"):
            ogg_by_stem[ogg_file.stem.lower()] = ogg_file

        extracted_voices: list[ExtractedSafehouseVoice] = []

        for idx, target in enumerate(PROLOGUE_OPENING_15_TARGETS, start=1):
            h_hex = target["wwes_hash"].upper()
            dlge_hex = target["dlge_hash"].upper()
            friendly_stem = target["friendly_name"]
            entry = selected_entries[h_hex]
            chunk_stem = Path(entry.source_rpkg).stem

            src_wwes = raw_extract_dir / chunk_stem / "WWES" / f"{h_hex}.WWES"
            src_meta = raw_extract_dir / chunk_stem / "WWES" / f"{h_hex}.WWES.meta"
            if not src_wwes.is_file() or not src_meta.is_file():
                raise FileNotFoundError(f"Beklenen ham WWES veya meta çıkarılamadı: {src_wwes}")

            dst_raw_wwes = raw_wwes_dir / f"{h_hex}.WWES"
            dst_raw_meta = raw_wwes_dir / f"{h_hex}.WWES.meta"
            shutil.copy2(src_wwes, dst_raw_wwes)
            shutil.copy2(src_meta, dst_raw_meta)

            _run_rpkg_cli(["-hash_meta_to_json", str(dst_raw_meta)])
            dst_raw_meta_json = raw_wwes_dir / f"{h_hex}.WWES.meta.JSON"
            meta_decoded: dict[str, Any] = {}
            if dst_raw_meta_json.is_file():
                try:
                    meta_decoded = json.loads(
                        dst_raw_meta_json.read_text(encoding="utf-8", errors="replace")
                    )
                except Exception:
                    meta_decoded = {}

            wem_hdr = inspect_wwes_riff_header(dst_raw_wwes)
            ogg_path = ogg_by_stem.get(friendly_stem.lower()) or ogg_by_stem.get(h_hex.lower())
            if ogg_path is None or not ogg_path.is_file():
                raise FileNotFoundError(
                    f"{entry.hash_id} ({friendly_stem}) için rpkg-cli OGG çıktısı bulunamadı."
                )

            wav_filename = f"{idx:02d}_{friendly_stem}_{h_hex}.wav"
            dst_wav = wav_dir / wav_filename
            target_sr = wem_hdr.sample_rate if wem_hdr.sample_rate > 0 else 48000
            target_ch = wem_hdr.channels if wem_hdr.channels > 0 else 1
            ffmpeg.convert_to_wav(
                ogg_path,
                dst_wav,
                sample_rate=target_sr,
                channels=target_ch,
            )

            audio_meta = ffmpeg.probe_audio(dst_wav, measure_lufs=True)
            en_sub = en_by_dlge.get(dlge_hex, "")
            tr_sub = tr_by_dlge.get(dlge_hex, "")

            extracted_voices.append(
                ExtractedSafehouseVoice(
                    index=idx,
                    hash_hex=h_hex,
                    hash_id=entry.hash_id,
                    source_rpkg=entry.source_rpkg,
                    ioi_assembly_path=entry.ioi_assembly_path,
                    friendly_name=friendly_stem,
                    wav_filename=wav_filename,
                    wav_rel_path=str(dst_wav.relative_to(output_root)).replace("\\", "/"),
                    raw_wwes_rel_path=str(dst_raw_wwes.relative_to(output_root)).replace("\\", "/"),
                    raw_meta_rel_path=str(dst_raw_meta.relative_to(output_root)).replace("\\", "/"),
                    raw_meta_json_rel_path=(
                        str(dst_raw_meta_json.relative_to(output_root)).replace("\\", "/")
                        if dst_raw_meta_json.is_file()
                        else ""
                    ),
                    raw_wwes_size_bytes=dst_raw_wwes.stat().st_size,
                    raw_wwes_sha256=compute_sha256(dst_raw_wwes),
                    wav_size_bytes=dst_wav.stat().st_size,
                    wav_sha256=compute_sha256(dst_wav),
                    duration_sec=round(audio_meta.duration_sec, 3),
                    duration_ms=audio_meta.duration_ms,
                    sample_rate=audio_meta.sample_rate,
                    channels=audio_meta.channels,
                    lufs=round(audio_meta.lufs, 2),
                    codec_in_wwes=wem_hdr.codec_name,
                    format_tag_hex=wem_hdr.format_tag_hex,
                    dialogue_subtitle_en=en_sub,
                    dialogue_subtitle_tr=tr_sub,
                    dialogue_subtitle_tr_mod=tr_sub,
                    speaker_character=target["speaker"],
                    in_game_trigger_moment=target["trigger_moment"],
                    meta_decoded=meta_decoded,
                    wem_header=asdict(wem_hdr),
                )
            )

        existing_c0_patches = [f.name for f in c0_files_info if "patch" in f.name.lower()]
        max_patch_num = 0
        for pname in existing_c0_patches:
            m_p = re.search(r"patch(\d+)\.rpkg$", pname, re.IGNORECASE)
            if m_p:
                max_patch_num = max(max_patch_num, int(m_p.group(1)))
        next_free_patch_num = max_patch_num + 1
        next_free_patch_rpkg = f"chunk0patch{next_free_patch_num}.rpkg"

        chunk0_partition = next(
            (p for p in partitions if p["chunk_name"] == "chunk0"),
            {"chunk_name": "chunk0", "codename": "super", "patchlevel": 4},
        )

        manifest_data: dict[str, Any] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "SesAI Dublaj Stüdyosu — Hitman İlk Sahne (ICA Tesisi) Pilot Çıkarıcı",
            "read_only_guarantee": True,
            "scene_info": {
                "mission_group_tr": "Seferberlikler (Campaigns) -> Başlangıç (Prologue) -> ICA Tesisi (ICA Facility)",
                "missions_covered": [
                    "1. Rehberli Eğitim (Guided Training) — Yat önü başlangıç brifingi ve ilk 60 sn adım adım Diana yönlendirmeleri",
                    "2. Serbest Eğitim (Freeform Training) — Yat önü 0:00 sn başlangıç konuşması",
                    "3. Varış (Arrival) — Helikopter pisti 0:00 sn karşılaması",
                ],
                "repeatable": True,
                "source_chunk": "chunk0",
                "partition_codename": "super",
                "existing_chunk0_rpkg_files": [f.name for f in c0_files_info],
                "current_packagedefinition_patchlevel": chunk0_partition.get("patchlevel", 4),
                "next_available_patch_number": next_free_patch_num,
                "next_available_patch_rpkg": next_free_patch_rpkg,
                "chunk0_rpkgv2_13byte_header_hex": "32 4b 50 52 01 00 00 00 00 00 00 78 78",
            },
            "hitman_installation": install_info,
            "extracted_summary": {
                "selected_wwes_count": len(extracted_voices),
                "english_subtitles_matched": sum(1 for v in extracted_voices if v.dialogue_subtitle_en),
                "turkish_subtitles_matched": sum(1 for v in extracted_voices if v.dialogue_subtitle_tr),
                "extracted_wav_dir": str(wav_dir),
                "raw_wwes_backup_dir": str(raw_wwes_dir),
            },
            "files": [asdict(v) for v in extracted_voices],
        }

        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest_data
    finally:
        cleanup_temp_dir(keep_root=True)


def main() -> int:
    if "--scene" in sys.argv and "prologue" in sys.argv:
        print("=" * 78)
        print("SesAI Dublaj Stüdyosu — Hitman 3 İlk Sahne (ICA Tesisi / chunk0) Pilot Çıkarıcı")
        print("=" * 78)
        manifest = extract_prologue_opening_scene()
        sc = manifest["scene_info"]
        sm = manifest["extracted_summary"]
        print(f"[+] Menü Görev Yolu     : {sc['mission_group_tr']}")
        print(
            f"[+] Kaynak Chunk & Patch: {sc['source_chunk']} (@partition name={sc['partition_codename']}) "
            f"| Sıradaki Boş Patch: {sc['next_available_patch_rpkg']}"
        )
        print(
            f"[+] Çıkarılan Konuşma   : {sm['selected_wwes_count']} adet 48 kHz .wav "
            f"(EN Altyazı: {sm['english_subtitles_matched']}, TR Altyazı: {sm['turkish_subtitles_matched']})"
        )
        print(f"[+] Çıkarılan WAV Yolu  : {sm['extracted_wav_dir']}")
        print(f"[+] Ham WWES Yedek Yolu : {sm['raw_wwes_backup_dir']}")
        print(f"[+] Manifest Dosyası    : {PROLOGUE_MANIFEST_PATH}")
        print("=" * 78)
        return 0

    print("=" * 78)
    print("SesAI Dublaj Stüdyosu — Hitman 3 / WoA (chunk29 Serbest Ev) Ses Çıkarma Köprüsü")
    print("=" * 78)
    manifest = extract_chunk29_safehouse()
    inst = manifest["hitman_installation"]
    c29 = manifest["chunk29_safehouse_summary"]
    c1 = manifest["chunk1_opening_scenes_summary"]
    rep = manifest["repack_technical_analysis"]
    mod = manifest["modding_environment"]

    print(f"[+] Hitman Kurulum Yolu : {inst['install_dir']}")
    print(f"[+] Runtime Dizini      : {inst['runtime_dir']}")
    print(
        f"[+] Serbest Ev (chunk29): {c29['effective_unique_active_wwes']} etkin WWES konuşma dosyası "
        f"({c29['extracted_wav_count']} adet 48 kHz .wav çıkarıldı)"
    )
    print(f"[+] Çıkarılan WAV Yolu  : {c29['extracted_wav_dir']}")
    print(f"[+] Ham WWES Yedek Yolu : {c29['raw_wwes_backup_dir']}")
    print(f"[+] Manifest Dosyası    : {SAFEHOUSE_MANIFEST_PATH}")
    print(
        f"[+] İlk Sahneler(chunk1): {c1['effective_unique_active_wwes']} etkin tekil WWES "
        f"(toplam ham girdi: {c1['total_raw_wwes_entries']})"
    )
    print(
        f"[+] WWES Codec Başlığı  : {rep['codec_name']} | {rep['sample_rate_hz']} Hz | "
        f"fmt={rep['fmt_chunk_size_bytes']}B (ext={rep['extra_fmt_size_bytes']}B)"
    )
    print(
        f"[+] Mod Araçları Durumu : rpkg-cli={mod['rpkg_cli_portable_installed']} | "
        f"SMF={mod['simple_mod_framework_installed']} | Wwise={mod['audiokinetic_wwise_installed']}"
    )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
