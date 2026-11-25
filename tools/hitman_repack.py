"""Hitman 3 / World of Assassination — Serbest Ev (`chunk29patch3.rpkg`) Güvenli Geri Paketleme ve Kurulum Aracı.

Özellikler:
1. `Hitman_SerbestEv/sesler/` içindeki 11 `.wav` dosyasının `hitman_manifest.json` içindeki
   orijinal SHA-256 özetlerinden farklı (Türkçe dublajlanmış) olduğunu doğrular.
2. Her bir Türkçe `.wav` dosyasını Hitman 3 Glacier 2 / Wwise ses motorunun doğrudan oynatabildiği
   48.000 Hz, orijinal kanal sayısına birebir uyan (Diana için 2 kanal Stereo, diğer 10 dosya için
   1 kanal Mono), 16-bit Wwise PCM (`0xFFFE` `WAVE_FORMAT_EXTENSIBLE`, `fmt` = 24 bayt,
   orijinal `uChannelConfig`, `cue ` ve `LIST` `adtllabl` işaretçileri korunmuş) `.WWES` dosyasına dönüştürür.
3. Orijinal 44 baytlık `.WWES.meta` dosyasındaki `16..20` LZ4 sıkıştırmasız bayrağını (`0x80000000`)
   koruyarak `32..36` ofsetindeki `uint32 hash_size_final` alanını yeni `.WWES` boyutuyla günceller.
4. Taşınabilir `./_araclar/rpkg/rpkg-cli.exe -generate_rpkg_from` komutuyla yalnızca bu 11 dublajlı
   `.WWES` ve `.WWES.meta` dosyasını içeren RPKGv2 (`2KPR`) formatlı `Hitman_SerbestEv/paket/chunk29patch3.rpkg`
   yama paketini üretir.
5. `C:\\Program Files\\Epic Games\\HITMAN3\\Runtime\\packagedefinition.txt` dosyasını SHA-256 özetiyle
   birlikte `Hitman_SerbestEv/oyun_yedek/` altına yedekler (mevcut yedeğin üzerine asla yazmaz),
   ardından `@partition name=snug` satırındaki `patchlevel` değerini `3` yaparak
   `Hitman_SerbestEv/paket/packagedefinition.txt` olarak yeniden şifreler.
6. Tek tıkla UAC yönetici yetkili kurulum (`serbest_ev_dublaj_kur.bat`) ve %100 SHA-256 doğrulamalı
   orijinale dönüş (`serbest_ev_dublaj_kaldir.bat`) betiklerini oluşturur; `Runtime\\` dizinine doğrudan
   yazma izni varsa yamayı anında kurup SHA-256 ile doğrular.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
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
    TEMP_DIR,
    bootstrap_portable_environment,
    cleanup_temp_dir,
)
from tools.hitman_bridge import (
    RPKG_CLI_EXE,
    SAFEHOUSE_MANIFEST_PATH,
    SAFEHOUSE_OUTPUT_ROOT,
    SAFEHOUSE_RAW_WWES_DIR,
    SAFEHOUSE_WAV_DIR,
    _run_rpkg_cli,
    find_epic_hitman_install,
    inspect_wwes_riff_header,
    parse_rpkg_file,
)

SAFEHOUSE_PACKAGE_DIR = SAFEHOUSE_OUTPUT_ROOT / "paket"
SAFEHOUSE_PACKAGE_WWES_DIR = SAFEHOUSE_PACKAGE_DIR / "chunk29patch3" / "WWES"
SAFEHOUSE_GAME_BACKUP_DIR = SAFEHOUSE_OUTPUT_ROOT / "oyun_yedek"
INSTALL_BAT_PATH = SAFEHOUSE_OUTPUT_ROOT / "serbest_ev_dublaj_kur.bat"
UNINSTALL_BAT_PATH = SAFEHOUSE_OUTPUT_ROOT / "serbest_ev_dublaj_kaldir.bat"
REPACK_REPORT_PATH = SAFEHOUSE_PACKAGE_DIR / "repack_raporu.json"

PROLOGUE_OUTPUT_ROOT = PROJECT_ROOT / "Hitman_IlkSahne"
PROLOGUE_WAV_DIR = PROLOGUE_OUTPUT_ROOT / "sesler"
PROLOGUE_RAW_WWES_DIR = PROLOGUE_OUTPUT_ROOT / "ham_wwes"
PROLOGUE_MANIFEST_PATH = PROLOGUE_OUTPUT_ROOT / "hitman_manifest.json"
PROLOGUE_PACKAGE_DIR = PROLOGUE_OUTPUT_ROOT / "paket"
PROLOGUE_GAME_BACKUP_DIR = PROLOGUE_OUTPUT_ROOT / "oyun_yedek"
PROLOGUE_INSTALL_BAT_PATH = PROLOGUE_OUTPUT_ROOT / "ilk_sahne_dublaj_kur.bat"
PROLOGUE_UNINSTALL_BAT_PATH = PROLOGUE_OUTPUT_ROOT / "ilk_sahne_dublaj_kaldir.bat"

# Hitman 3 RPKGv2 (2KPR) 29 baytlık güvenli paket .meta başlıkları
# (13 bayt RPKGv2 başlığı + 16 bayt sıfır tablo/silme sayacı -> rpkg-cli EOF bellek hatasını önler)
CHUNK29_RPKGV2_META_BYTES = b"2KPR\x01\x00\x00\x00\x1d\x00\x00xx" + b"\x00" * 16
CHUNK0_RPKGV2_META_BYTES = b"2KPR\x01\x00\x00\x00\x00\x00\x00xx" + b"\x00" * 16

# Taşınabilir Wwise 0xFFFF Vorbis dönüştürücü ve gidiş-dönüş doğrulama araçları (./_araclar/wwise/)
WWISE_TOOLS_DIR = PROJECT_ROOT / "_araclar" / "wwise"
WAV2WEM_EXE = WWISE_TOOLS_DIR / "wav2wem.exe"
WW2OGG_EXE = WWISE_TOOLS_DIR / "ww2ogg" / "ww2ogg.exe"
WW2OGG_PCB = WWISE_TOOLS_DIR / "ww2ogg" / "packed_codebooks_aoTuV_603.bin"
VGMSTREAM_CLI_EXE = WWISE_TOOLS_DIR / "vgmstream" / "vgmstream-cli.exe"


@dataclass
class DubbedWavCheckResult:
    index: int
    hash_hex: str
    wav_filename: str
    wav_path: str
    exists: bool
    is_dubbed_modified: bool
    original_wav_sha256: str
    current_wav_sha256: str
    original_duration_sec: float
    current_duration_sec: float
    sample_rate: int
    channels: int
    expected_channels: int


@dataclass
class RepackedWwesEntry:
    index: int
    hash_hex: str
    hash_id: str
    wav_filename: str
    wwes_size_bytes: int
    wwes_sha256: str
    meta_size_bytes: int
    meta_sha256: str
    meta_hash_size_final: int
    format_tag_hex: str
    sample_rate: int
    channels: int
    channel_config_hex: str
    pcm_data_bytes: int
    duration_sec: float
    preserved_subchunks: list[str]
    ffmpeg_verified: bool
    uid_hex: str = ""
    setup_exact_match: bool = False
    ww2ogg_verified: bool = False
    vgmstream_verified: bool = False
    roundtrip_rms_dbfs: float = -100.0
    roundtrip_lufs: float = -100.0


def verify_dubbed_wavs(
    manifest_path: Path = SAFEHOUSE_MANIFEST_PATH,
    wav_dir: Path = SAFEHOUSE_WAV_DIR,
) -> tuple[list[DubbedWavCheckResult], list[str]]:
    """`Hitman_SerbestEv/sesler/` içindeki 11 WAV dosyasının Türkçe dublajla güncellendiğini doğrular."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Hitman manifest dosyası bulunamadı: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files_list = manifest.get("files", [])
    if not files_list:
        raise ValueError("hitman_manifest.json içinde 'files' listesi boş.")

    ff = FFmpegManager()
    results: list[DubbedWavCheckResult] = []
    warnings: list[str] = []

    for item in files_list:
        idx = int(item["index"])
        hash_hex = str(item["hash_hex"]).upper()
        wav_name = str(item["wav_filename"])
        wav_path = wav_dir / wav_name
        orig_sha = str(item["wav_sha256"]).lower()
        orig_dur = float(item["duration_sec"])
        expected_ch = int(item["channels"])

        if not wav_path.is_file():
            warnings.append(f"Eksik WAV dosyası: {wav_name}")
            results.append(
                DubbedWavCheckResult(
                    index=idx,
                    hash_hex=hash_hex,
                    wav_filename=wav_name,
                    wav_path=str(wav_path),
                    exists=False,
                    is_dubbed_modified=False,
                    original_wav_sha256=orig_sha,
                    current_wav_sha256="",
                    original_duration_sec=orig_dur,
                    current_duration_sec=0.0,
                    sample_rate=0,
                    channels=0,
                    expected_channels=expected_ch,
                )
            )
            continue

        cur_sha = compute_sha256(wav_path).lower()
        meta = ff.probe_audio(wav_path, measure_lufs=False)
        is_modified = cur_sha != orig_sha
        if not is_modified:
            warnings.append(
                f"UYARI: '{wav_name}' henüz Türkçe dublajla güncellenmemiş (orijinal İngilizce WAV ile aynı SHA-256)!"
            )

        results.append(
            DubbedWavCheckResult(
                index=idx,
                hash_hex=hash_hex,
                wav_filename=wav_name,
                wav_path=str(wav_path),
                exists=True,
                is_dubbed_modified=is_modified,
                original_wav_sha256=orig_sha,
                current_wav_sha256=cur_sha,
                original_duration_sec=orig_dur,
                current_duration_sec=round(meta.duration_sec, 3),
                sample_rate=meta.sample_rate,
                channels=meta.channels,
                expected_channels=expected_ch,
            )
        )

    return results, warnings


def _extract_original_wwes_subchunks(
    orig_wwes_path: Path,
    expected_channels: int,
) -> tuple[int, list[tuple[bytes, bytes]]]:
    """Orijinal ham `.WWES` dosyasından Wwise `uChannelConfig` değerini ve `cue ` / `LIST` alt bloklarını çıkarır."""
    default_ch_cfg = 0x00003102 if expected_channels == 2 else 0x00004101
    if not orig_wwes_path.is_file():
        return default_ch_cfg, []

    data = orig_wwes_path.read_bytes()
    if len(data) < 36 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return default_ch_cfg, []

    pos = 12
    ch_cfg = default_ch_cfg
    extra_chunks: list[tuple[bytes, bytes]] = []

    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        csz = struct.unpack_from("<I", data, pos + 4)[0]
        payload = data[pos + 8 : pos + 8 + csz]
        if cid == b"fmt " and csz >= 24:
            # Wwise Vorbis / PCM fmt bloğunda offset 20..24 serileştirilmiş AkChannelConfig değeridir
            cand_cfg = struct.unpack_from("<I", payload, 20)[0]
            if cand_cfg != 0:
                ch_cfg = cand_cfg
        elif cid in (b"cue ", b"LIST"):
            extra_chunks.append((cid, payload))
        pos += 8 + csz + (csz % 2)

    return ch_cfg, extra_chunks


def _read_pcm16_frames_48k(
    wav_path: Path,
    target_channels: int,
    work_dir: Path,
    ffmpeg: FFmpegManager,
) -> bytes:
    """WAV dosyasını 48.000 Hz, `target_channels` kanallı 16-bit signed little-endian PCM baytları olarak okur."""
    norm_wav = wav_path
    needs_convert = False
    try:
        with wave.open(str(wav_path), "rb") as wf:
            if (
                wf.getframerate() != 48000
                or wf.getnchannels() != target_channels
                or wf.getsampwidth() != 2
                or wf.getcomptype() != "NONE"
            ):
                needs_convert = True
            else:
                pcm_data = wf.readframes(wf.getnframes())
    except Exception:
        needs_convert = True

    if needs_convert:
        norm_wav = work_dir / f"norm_{target_channels}ch_{wav_path.name}"
        ffmpeg.convert_to_wav(
            wav_path,
            norm_wav,
            sample_rate=48000,
            channels=target_channels,
        )
        with wave.open(str(norm_wav), "rb") as wf:
            pcm_data = wf.readframes(wf.getnframes())

    if len(pcm_data) % 2 == 1:
        pcm_data += b"\x00"
    return pcm_data


def convert_wav_to_wwise_pcm_wwes(
    dubbed_wav_path: Path,
    orig_wwes_path: Path,
    output_wwes_path: Path,
    target_channels: int,
    work_dir: Path,
    ffmpeg: FFmpegManager,
) -> tuple[int, list[str], int]:
    """Türkçe `.wav` dosyasını Hitman 3 Wwise motorunun doğrudan oynattığı 48 kHz 16-bit Wwise PCM (`0xFFFE`)
    `.WWES` dosyasına dönüştürür.

    Wwise `WaveFormatExtensible` (`0xFFFE`, 24 bayt `fmt ` bloğu):
      - `0..2`  : `wFormatTag = 0xFFFE`
      - `2..4`  : `nChannels` (1 veya 2)
      - `4..8`  : `nSamplesPerSec = 48000`
      - `8..12` : `nAvgBytesPerSec = 48000 * nChannels * 2`
      - `12..14`: `nBlockAlign = nChannels * 2`
      - `14..16`: `wBitsPerSample = 16`
      - `16..18`: `cbSize = 6`
      - `18..20`: `wSamplesPerBlock = 0`
      - `20..24`: `uChannelConfig` (`0x00003102` Stereo / `0x00004101` Mono)
    """
    sample_rate = 48000
    bits_per_sample = 16
    block_align = target_channels * (bits_per_sample // 8)
    avg_bytes_per_sec = sample_rate * block_align

    u_channel_config, extra_chunks = _extract_original_wwes_subchunks(
        orig_wwes_path, expected_channels=target_channels
    )
    pcm_bytes = _read_pcm16_frames_48k(
        dubbed_wav_path,
        target_channels=target_channels,
        work_dir=work_dir,
        ffmpeg=ffmpeg,
    )

    fmt_payload = struct.pack(
        "<HHIIHHHHI",
        0xFFFE,
        target_channels,
        sample_rate,
        avg_bytes_per_sec,
        block_align,
        bits_per_sample,
        6,  # cbSize = 6 (Wwise WaveFormatExtensible)
        0,  # wSamplesPerBlock = 0
        u_channel_config,
    )
    assert len(fmt_payload) == 24

    subchunks_buf = bytearray()
    # 1. fmt chunk
    subchunks_buf.extend(b"fmt ")
    subchunks_buf.extend(struct.pack("<I", len(fmt_payload)))
    subchunks_buf.extend(fmt_payload)

    # 2. Orijinal cue ve LIST (adtllabl) alt blokları (yeni PCM örnek sayısını aşan cue noktalarını güvenli ayıkla)
    total_pcm_samples = len(pcm_bytes) // block_align
    preserved_names: list[str] = ["fmt "]
    for cid, payload in extra_chunks:
        if cid == b"cue " and len(payload) >= 4:
            cue_cnt = struct.unpack_from("<I", payload, 0)[0]
            valid_entries = bytearray()
            kept = 0
            for i in range(cue_cnt):
                off = 4 + i * 24
                if off + 24 <= len(payload):
                    _dw_id, dw_pos, _fcc, _cs, _bs, sample_off = struct.unpack_from(
                        "<II4sIII", payload, off
                    )
                    if dw_pos <= total_pcm_samples and sample_off <= total_pcm_samples:
                        valid_entries.extend(payload[off : off + 24])
                        kept += 1
            if kept == 0:
                continue
            payload = struct.pack("<I", kept) + bytes(valid_entries)
        subchunks_buf.extend(cid)
        subchunks_buf.extend(struct.pack("<I", len(payload)))
        subchunks_buf.extend(payload)
        if len(payload) % 2 == 1:
            subchunks_buf.append(0)
        preserved_names.append(cid.decode("ascii", errors="replace"))

    # 3. data chunk
    subchunks_buf.extend(b"data")
    subchunks_buf.extend(struct.pack("<I", len(pcm_bytes)))
    subchunks_buf.extend(pcm_bytes)
    if len(pcm_bytes) % 2 == 1:
        subchunks_buf.append(0)
    preserved_names.append("data")

    riff_size = 4 + len(subchunks_buf)
    output_wwes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_wwes_path, "wb") as out_f:
        out_f.write(b"RIFF")
        out_f.write(struct.pack("<I", riff_size))
        out_f.write(b"WAVE")
        out_f.write(subchunks_buf)

    return u_channel_config, preserved_names, len(pcm_bytes)


def _extract_original_vorbis_fmt_and_setup(
    orig_wwes_path: Path,
) -> tuple[bytes, bytes]:
    """Orijinal `0xFFFF` `.WWES` dosyasından 66 baytlık `fmt ` bloğunu ve Vorbis setup paketini çıkarır."""
    if not orig_wwes_path.is_file():
        return b"", b""
    data = orig_wwes_path.read_bytes()
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return b"", b""
    pos = 12
    fmt = b""
    dchunk = b""
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        csz = struct.unpack_from("<I", data, pos + 4)[0]
        payload = data[pos + 8 : pos + 8 + csz]
        if cid == b"fmt ":
            fmt = payload
        elif cid == b"data":
            dchunk = payload
        pos += 8 + csz + (csz % 2)
    if len(fmt) >= 66 and dchunk:
        setup_off, first_off = struct.unpack_from("<II", fmt, 40)
        if 0 <= setup_off < first_off <= len(dchunk):
            return fmt, dchunk[setup_off:first_off]
    return fmt, b""


def convert_wav_to_wwise_vorbis_wwes(
    dubbed_wav_path: Path,
    orig_wwes_path: Path,
    output_wwes_path: Path,
    target_channels: int,
    work_dir: Path,
    ffmpeg: FFmpegManager,
    vorbis_quality: int = 8,
) -> dict[str, Any]:
    """Türkçe `.wav` dosyasını HITMAN 3'ün orijinal `.WWES` dosyalarıyla %100 uyumlu gerçek
    `0xFFFF` Wwise Custom Vorbis (`fmt ` = 66 bayt, `aoTuV b6.03`, `no_granule`, `mod_packets`,
    orijinal `uChannelConfig` + `cue ` + `LIST` korunmuş) `.WWES` dosyasına dönüştürür.
    """
    if not WAV2WEM_EXE.is_file():
        raise FileNotFoundError(
            f"Taşınabilir wav2wem.exe bulunamadı: {WAV2WEM_EXE}"
        )

    # 1. Girdi WAV dosyasını 48.000 Hz, target_channels, 16-bit PCM olarak doğrula/normalize et
    norm_wav = dubbed_wav_path
    needs_norm = False
    try:
        with wave.open(str(dubbed_wav_path), "rb") as wf:
            if (
                wf.getframerate() != 48000
                or wf.getnchannels() != target_channels
                or wf.getsampwidth() != 2
                or wf.getcomptype() != "NONE"
            ):
                needs_norm = True
    except Exception:
        needs_norm = True

    if needs_norm:
        norm_wav = work_dir / f"vnorm_{target_channels}ch_{dubbed_wav_path.name}"
        ffmpeg.convert_to_wav(
            dubbed_wav_path,
            norm_wav,
            sample_rate=48000,
            channels=target_channels,
        )

    # 2. Taşınabilir wav2wem.exe -no-hash -q 8 ile gerçek 0xFFFF Wwise Vorbis (.wem) üret
    raw_wem_path = work_dir / f"{output_wwes_path.stem}_raw.wem"
    if raw_wem_path.exists():
        raw_wem_path.unlink()

    enc_res = subprocess.run(
        [
            str(WAV2WEM_EXE),
            "-no-hash",
            "-q",
            str(vorbis_quality),
            "-o",
            str(raw_wem_path),
            str(norm_wav),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if enc_res.returncode != 0 or not raw_wem_path.is_file():
        raise RuntimeError(
            f"wav2wem.exe dönüştürme hatası ({dubbed_wav_path.name}): "
            f"{enc_res.stdout} {enc_res.stderr}"
        )

    # 3. Üretilen .wem içinden 66 baytlık fmt ve tam data (tekil baytta sonda pad OLMADAN) bloklarını al
    wem_bytes = raw_wem_path.read_bytes()
    if len(wem_bytes) < 90 or wem_bytes[:4] != b"RIFF" or wem_bytes[8:12] != b"WAVE":
        raise RuntimeError(f"Geçersiz wav2wem RIFF/WAVE çıktısı: {raw_wem_path}")

    pos = 12
    new_fmt_raw = b""
    new_data_raw = b""
    while pos + 8 <= len(wem_bytes):
        cid = wem_bytes[pos : pos + 4]
        csz = struct.unpack_from("<I", wem_bytes, pos + 4)[0]
        payload = wem_bytes[pos + 8 : pos + 8 + csz]
        if cid == b"fmt ":
            new_fmt_raw = payload
        elif cid == b"data":
            new_data_raw = payload
        pos += 8 + csz + (csz % 2)

    if len(new_fmt_raw) != 66 or not new_data_raw:
        raise RuntimeError(
            f"Beklenmeyen wav2wem fmt/data yapısı (fmt={len(new_fmt_raw)}B, data={len(new_data_raw)}B)"
        )

    fmt_buf = bytearray(new_fmt_raw)

    # 4. Orijinal Hitman 3 .WWES'ten uChannelConfig, cue, LIST ve 228 baytlık setup eşleşmesini uygula
    u_channel_config, extra_chunks = _extract_original_wwes_subchunks(
        orig_wwes_path, expected_channels=target_channels
    )
    struct.pack_into("<I", fmt_buf, 20, u_channel_config)

    orig_fmt, orig_setup = _extract_original_vorbis_fmt_and_setup(orig_wwes_path)
    new_setup_off, new_first_off = struct.unpack_from("<II", fmt_buf, 40)
    new_setup = new_data_raw[new_setup_off:new_first_off]
    setup_exact_match = bool(orig_setup) and (new_setup == orig_setup)
    if setup_exact_match and len(orig_fmt) >= 66:
        # 228 baytlık setup paketi birebir aynı olduğundan orijinal Wwise alloc & uid (0x5921D930) alanlarını eşle!
        fmt_buf[52:64] = orig_fmt[52:64]

    total_pcm_frames = struct.unpack_from("<I", fmt_buf, 24)[0]
    uid_val = struct.unpack_from("<I", fmt_buf, 60)[0]

    # 5. Hitman 3 orijinal .WWES yapısıyla birebir RIFF/WAVE paketle (fmt -> cue -> LIST -> data, sonda pad yok!)
    subchunks_buf = bytearray()
    subchunks_buf.extend(b"fmt ")
    subchunks_buf.extend(struct.pack("<I", len(fmt_buf)))
    subchunks_buf.extend(fmt_buf)

    preserved_names: list[str] = ["fmt "]
    for cid, payload in extra_chunks:
        if cid == b"cue " and len(payload) >= 4:
            cue_cnt = struct.unpack_from("<I", payload, 0)[0]
            valid_entries = bytearray()
            kept = 0
            for i in range(cue_cnt):
                off = 4 + i * 24
                if off + 24 <= len(payload):
                    _dw_id, dw_pos, _fcc, _cs, _bs, sample_off = struct.unpack_from(
                        "<II4sIII", payload, off
                    )
                    if dw_pos <= total_pcm_frames and sample_off <= total_pcm_frames:
                        valid_entries.extend(payload[off : off + 24])
                        kept += 1
            if kept == 0:
                continue
            payload = struct.pack("<I", kept) + bytes(valid_entries)
        subchunks_buf.extend(cid)
        subchunks_buf.extend(struct.pack("<I", len(payload)))
        subchunks_buf.extend(payload)
        if len(payload) % 2 == 1:
            subchunks_buf.append(0)
        preserved_names.append(cid.decode("ascii", errors="replace"))

    # data bloğu (Hitman 3 orijinal .WWES dosyaları gibi sonda ekstra pad baytı eklenmez!)
    subchunks_buf.extend(b"data")
    subchunks_buf.extend(struct.pack("<I", len(new_data_raw)))
    subchunks_buf.extend(new_data_raw)
    preserved_names.append("data")

    riff_size = 4 + len(subchunks_buf)
    output_wwes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_wwes_path, "wb") as out_f:
        out_f.write(b"RIFF")
        out_f.write(struct.pack("<I", riff_size))
        out_f.write(b"WAVE")
        out_f.write(subchunks_buf)

    return {
        "u_channel_config": u_channel_config,
        "preserved_subchunks": preserved_names,
        "vorbis_data_bytes": len(new_data_raw),
        "total_pcm_frames": total_pcm_frames,
        "setup_exact_match": setup_exact_match,
        "uid_hex": f"0x{uid_val:08X}",
        "blocksize_pow": (fmt_buf[64], fmt_buf[65]),
    }


def verify_wwes_roundtrip(
    wwes_path: Path,
    expected_channels: int,
    work_dir: Path,
    ffmpeg: FFmpegManager,
    min_audible_dbfs: float = -40.0,
) -> dict[str, Any]:
    """Üretilen `0xFFFF` `.WWES` dosyasını `ww2ogg` + `vgmstream-cli` + `FFmpeg` ile gidiş-dönüş
    çözer ve gerçek RMS / LUFS ses seviyesinin > -40 dBFS (sıfır sessizlik) olduğunu doğrular.
    """
    if not WW2OGG_EXE.is_file() or not WW2OGG_PCB.is_file():
        raise FileNotFoundError(f"Taşınabilir ww2ogg bulunamadı: {WW2OGG_EXE}")
    if not VGMSTREAM_CLI_EXE.is_file():
        raise FileNotFoundError(f"Taşınabilir vgmstream-cli bulunamadı: {VGMSTREAM_CLI_EXE}")

    rt_ogg = work_dir / f"rt_{wwes_path.stem}.ogg"
    rt_wav = work_dir / f"rt_{wwes_path.stem}.wav"
    rt_ogg.unlink(missing_ok=True)
    rt_wav.unlink(missing_ok=True)

    # 1. ww2ogg (aoTuV 6.03 packed codebooks) doğrulaması
    res_ww2 = subprocess.run(
        [str(WW2OGG_EXE), str(wwes_path), "-o", str(rt_ogg), "--pcb", str(WW2OGG_PCB)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if res_ww2.returncode != 0 or not rt_ogg.is_file() or rt_ogg.stat().st_size == 0:
        raise RuntimeError(
            f"ww2ogg gidiş-dönüş doğrulaması başarısız ({wwes_path.name}): "
            f"{res_ww2.stdout} {res_ww2.stderr}"
        )

    # 2. vgmstream-cli ile doğrudan .wav dosyasına çözme doğrulaması (.wem uzantılı geçici kopya)
    rt_wem = work_dir / f"rt_{wwes_path.stem}.wem"
    shutil.copy2(wwes_path, rt_wem)
    res_vgm = subprocess.run(
        [str(VGMSTREAM_CLI_EXE), "-o", str(rt_wav), str(rt_wem)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    rt_wem.unlink(missing_ok=True)
    if res_vgm.returncode != 0 or not rt_wav.is_file() or rt_wav.stat().st_size < 100:
        raise RuntimeError(
            f"vgmstream-cli gidiş-dönüş doğrulaması başarısız ({wwes_path.name}): "
            f"{res_vgm.stdout} {res_vgm.stderr}"
        )

    # 3. Çözülen .wav üzerinde FFmpeg (süre, 48 kHz, kanal, LUFS) + gerçek RMS dBFS ölçümü
    ff_meta = ffmpeg.probe_audio(rt_wav, measure_lufs=True)
    if (
        ff_meta.sample_rate != 48000
        or ff_meta.channels != expected_channels
        or ff_meta.duration_ms <= 0
    ):
        raise RuntimeError(
            f"FFmpeg round-trip teknik doğrulaması başarısız ({wwes_path.name}): {ff_meta}"
        )

    import numpy as np
    import soundfile as sf

    samples, sr = sf.read(str(rt_wav), dtype="float32", always_2d=False)
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size > 0 else 0.0
    rms_dbfs = round(20.0 * math.log10(max(rms, 1e-12)), 2)
    lufs_val = round(float(ff_meta.lufs), 2)

    if rms_dbfs <= min_audible_dbfs or lufs_val <= min_audible_dbfs:
        raise RuntimeError(
            f"Round-trip ses seviyesi sessizlik eşiğinin altında ({wwes_path.name}): "
            f"RMS={rms_dbfs} dBFS, LUFS={lufs_val} LUFS (beklenen > {min_audible_dbfs} dBFS)"
        )

    rt_ogg.unlink(missing_ok=True)
    rt_wav.unlink(missing_ok=True)
    return {
        "ww2ogg_verified": True,
        "vgmstream_verified": True,
        "ffmpeg_verified": True,
        "duration_sec": round(ff_meta.duration_sec, 3),
        "sample_rate": ff_meta.sample_rate,
        "channels": ff_meta.channels,
        "rms_dbfs": rms_dbfs,
        "lufs": lufs_val,
    }


def update_wwes_meta_file(
    orig_meta_path: Path,
    output_meta_path: Path,
    new_wwes_size_bytes: int,
) -> int:
    """Orijinal 44 baytlık `.WWES.meta` dosyasını kopyalar; `16..20` ofsetindeki `0x80000000`
    bayrağını, `36..44` ofsetlerindeki `0xFFFFFFFF` akış bellek işaretçilerini koruyarak
    `32..36` ofsetindeki `uint32 hash_size_final` alanını yeni `.WWES` boyutuyla günceller.
    """
    raw_meta = bytearray(orig_meta_path.read_bytes())
    if len(raw_meta) != 44:
        raise ValueError(
            f"Geçersiz .WWES.meta boyutu ({len(raw_meta)} bayt, 44 bekleniyordu): {orig_meta_path}"
        )

    # Offset 16..20: LZ4 sıkıştırmasız + XOR bayrağı (0x80000000)
    struct.pack_into("<I", raw_meta, 16, 0x80000000)
    # Offset 32..36: hash_size_final (yeni .WWES dosyasının tam bayt boyutu)
    struct.pack_into("<I", raw_meta, 32, int(new_wwes_size_bytes))
    # Offset 36..40 ve 40..44: WWES akış kaynaklarında her zaman 0xFFFFFFFF
    struct.pack_into("<II", raw_meta, 36, 0xFFFFFFFF, 0xFFFFFFFF)

    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.write_bytes(bytes(raw_meta))
    updated_final_size = struct.unpack_from("<I", raw_meta, 32)[0]
    return updated_final_size


def backup_and_patch_packagedefinition(
    runtime_dir: Path,
    backup_dir: Path,
    package_dir: Path,
    work_dir: Path,
    partition_name: str = "snug",
    min_patchlevel: Optional[int] = None,
    min_snug_patchlevel: int = 3,
) -> dict[str, Any]:
    """`Runtime\\packagedefinition.txt` dosyasını SHA-256 özetiyle yedekler (mevcut yedeğin üzerine yazmaz),
    şifresini çözer, belirtilen `@partition name=<partition_name>` satırındaki `patchlevel` değerini
    diğer partition ayarlarına (ör. `snug patchlevel=3`) dokunmadan günceller ve yeniden şifreler.
    """
    target_min_pl = int(min_patchlevel if min_patchlevel is not None else min_snug_patchlevel)
    live_pkg_def = runtime_dir / "packagedefinition.txt"
    if not live_pkg_def.is_file():
        raise FileNotFoundError(f"Oyun packagedefinition.txt bulunamadı: {live_pkg_def}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_pkg_def = backup_dir / "packagedefinition.txt"
    backup_info_path = backup_dir / "packagedefinition_backup_info.json"

    backup_created_now = False
    if not backup_pkg_def.is_file():
        shutil.copy2(live_pkg_def, backup_pkg_def)
        backup_created_now = True
        orig_sha = compute_sha256(backup_pkg_def)
        backup_info = {
            "backed_up_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_path": str(live_pkg_def),
            "backup_path": str(backup_pkg_def),
            "original_size_bytes": backup_pkg_def.stat().st_size,
            "original_sha256": orig_sha,
        }
        backup_info_path.write_text(
            json.dumps(backup_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        orig_sha = compute_sha256(backup_pkg_def)
        if not backup_info_path.is_file():
            backup_info = {
                "backed_up_at_utc": datetime.now(timezone.utc).isoformat(),
                "original_path": str(live_pkg_def),
                "backup_path": str(backup_pkg_def),
                "original_size_bytes": backup_pkg_def.stat().st_size,
                "original_sha256": orig_sha,
            }
            backup_info_path.write_text(
                json.dumps(backup_info, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    # Canlı Runtime\packagedefinition.txt dosyasını ./_temp içine kopyalayıp çöz
    # (böylece diğer sahnelerin aktif patchlevel ayarları — ör. super=5 ve snug=3 — asla ezilmez!)
    pkg_work = work_dir / "pkg_def"
    pkg_work.mkdir(parents=True, exist_ok=True)
    temp_pkg = pkg_work / "packagedefinition.txt"
    shutil.copy2(live_pkg_def, temp_pkg)

    res_dec = _run_rpkg_cli(
        ["-output_path", str(pkg_work), "-decrypt_packagedefinition_thumbs", str(temp_pkg)]
    )
    dec_file = pkg_work / "packagedefinition.txt.decrypted"
    if not dec_file.is_file():
        raise RuntimeError(
            f"packagedefinition.txt şifresi çözülemedi: {res_dec.stdout} {res_dec.stderr}"
        )

    raw_text = dec_file.read_text(encoding="utf-8", errors="replace")
    part_pattern = re.compile(rf"(?:^|\s)name={re.escape(partition_name)}(?:\s|$)")
    old_part_line = ""
    new_part_line = ""
    preserved_snug_line = ""
    preserved_super_line = ""
    old_patchlevel = 0
    new_patchlevel = target_min_pl

    updated_lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("@partition") and re.search(r"(?:^|\s)name=snug(?:\s|$)", stripped):
            preserved_snug_line = stripped
        if stripped.startswith("@partition") and re.search(r"(?:^|\s)name=super(?:\s|$)", stripped):
            preserved_super_line = stripped
        if stripped.startswith("@partition") and part_pattern.search(stripped):
            old_part_line = stripped
            m_pl = re.search(r"patchlevel=(\d+)", line)
            if m_pl:
                old_patchlevel = int(m_pl.group(1))
                new_patchlevel = max(old_patchlevel, target_min_pl)
                newline = re.sub(r"patchlevel=\d+", f"patchlevel={new_patchlevel}", line)
            else:
                new_patchlevel = target_min_pl
                newline = f"{line.rstrip()} patchlevel={new_patchlevel}"
            new_part_line = newline.strip()
            updated_lines.append(newline)
        else:
            updated_lines.append(line)

    if not new_part_line:
        raise ValueError(
            f"packagedefinition.txt içinde '@partition name={partition_name}' satırı bulunamadı!"
        )

    # CRLF satır sonunu koruyarak güncellenmiş decrypted dosyayı yaz ve yeniden şifrele
    line_sep = "\r\n" if "\r\n" in raw_text else "\n"
    patched_text = line_sep.join(updated_lines)
    if raw_text.endswith(("\r\n", "\n")):
        patched_text += line_sep
    dec_file.write_text(patched_text, encoding="utf-8")

    res_enc = _run_rpkg_cli(
        ["-output_path", str(pkg_work), "-encrypt_packagedefinition_thumbs", str(dec_file)]
    )
    enc_candidates = [
        pkg_work / "packagedefinition.txt.decrypted.encrypted",
        pkg_work / "packagedefinition.txt.encrypted",
        pkg_work / "packagedefinition.txt",
    ]
    enc_file: Optional[Path] = None
    for cand in enc_candidates:
        if cand.is_file() and cand.stat().st_mtime >= dec_file.stat().st_mtime:
            enc_file = cand
            break
    if enc_file is None:
        raise RuntimeError(
            f"packagedefinition.txt yeniden şifrelenemedi: {res_enc.stdout} {res_enc.stderr}"
        )

    package_dir.mkdir(parents=True, exist_ok=True)
    out_pkg_def = package_dir / "packagedefinition.txt"
    shutil.copy2(enc_file, out_pkg_def)

    # Doğrulama: Şifrelenen dosyayı tekrar çözüp hedef partition + snug korunmasını teyit et
    verify_work = pkg_work / "verify"
    verify_work.mkdir(parents=True, exist_ok=True)
    verify_copy = verify_work / "packagedefinition.txt"
    shutil.copy2(out_pkg_def, verify_copy)
    _run_rpkg_cli(
        ["-output_path", str(verify_work), "-decrypt_packagedefinition_thumbs", str(verify_copy)]
    )
    verify_dec = verify_work / "packagedefinition.txt.decrypted"
    verify_txt = verify_dec.read_text(encoding="utf-8", errors="replace") if verify_dec.is_file() else ""
    verified_target_line = any(
        ln.strip() == new_part_line for ln in verify_txt.splitlines()
    )
    verified_snug_intact = (
        True
        if (partition_name == "snug" or not preserved_snug_line)
        else any(ln.strip() == preserved_snug_line for ln in verify_txt.splitlines())
    )
    verified_super_intact = (
        True
        if (partition_name == "super" or not preserved_super_line)
        else any(ln.strip() == preserved_super_line for ln in verify_txt.splitlines())
    )
    verified_ok = verified_target_line and verified_snug_intact and verified_super_intact
    if not verified_ok:
        raise RuntimeError("Şifrelenen packagedefinition.txt doğrulama testinden geçemedi!")

    return {
        "backup_path": str(backup_pkg_def),
        "backup_created_now": backup_created_now,
        "original_sha256": orig_sha,
        "patched_package_path": str(out_pkg_def),
        "patched_sha256": compute_sha256(out_pkg_def),
        "partition_name": partition_name,
        "old_partition_line": old_part_line,
        "new_partition_line": new_part_line,
        "preserved_snug_line": preserved_snug_line,
        "preserved_super_line": preserved_super_line,
        "old_snug_partition_line": old_part_line if partition_name == "snug" else preserved_snug_line,
        "new_snug_partition_line": new_part_line if partition_name == "snug" else preserved_snug_line,
        "old_patchlevel": old_patchlevel,
        "new_patchlevel": new_patchlevel,
        "encryption_verified": verified_ok,
    }


def quarantine_conflicting_mod_patches(
    runtime_dir: Path,
    quarantine_dir: Path,
    chunk_prefix: str,
    patch_numbers: tuple[int, ...],
    stale_rpkg_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Resmi oyun yamalarının (`patchlevel=4`) üzerinde kalan ve `patchlevel` yükseltildiğinde
    2026 oyun dosyalarıyla çakışarak CTD'ye yol açan eski atıl mod `.rpkg` dosyalarını
    SHA-256 manifestosuyla güvenli yedek klasörüne taşır (asla silmez, mevcut yedeğin üzerine yazmaz).
    """
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = quarantine_dir / "eski_modlar_manifest.json"
    existing_records: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in loaded.get("quarantined_files", []):
                existing_records[item["filename"]] = item
        except Exception:
            pass

    moved_now: list[str] = []
    for num in patch_numbers:
        fname = f"{chunk_prefix}patch{num}.rpkg"
        live_path = runtime_dir / fname
        bak_path = quarantine_dir / fname
        if live_path.is_file():
            # Eğer bu dosya daha önce bizim ürettiğimiz yama ise (yedekte orijinal eski mod zaten varsa) yedeği ezme!
            if not bak_path.is_file():
                sha = compute_sha256(live_path)
                size_b = live_path.stat().st_size
                shutil.copy2(live_path, bak_path)
                assert compute_sha256(bak_path) == sha, f"Yedek SHA-256 doğrulaması başarısız: {fname}"
                existing_records[fname] = {
                    "filename": fname,
                    "original_runtime_path": str(live_path),
                    "backup_path": str(bak_path),
                    "size_bytes": size_b,
                    "sha256": sha,
                    "backed_up_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            live_path.unlink()
            moved_now.append(fname)

    removed_stale: list[str] = []
    for stale_name in stale_rpkg_names:
        stale_path = runtime_dir / stale_name
        if stale_path.is_file():
            stale_path.unlink()
            removed_stale.append(stale_name)

    manifest_data = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quarantine_dir": str(quarantine_dir),
        "quarantined_files": [existing_records[k] for k in sorted(existing_records.keys())],
    }
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "quarantine_dir": str(quarantine_dir),
        "manifest_path": str(manifest_path),
        "moved_now": moved_now,
        "removed_stale": removed_stale,
        "all_backed_up_files": [existing_records[k] for k in sorted(existing_records.keys())],
    }


def write_install_and_uninstall_bats(
    runtime_dir: Path,
    patched_rpkg_path: Path,
    patched_pkgdef_path: Path,
    backup_pkgdef_path: Path,
    patched_rpkg_sha256: str,
    patched_pkgdef_sha256: str,
    original_pkgdef_sha256: str,
    install_bat_path: Path = INSTALL_BAT_PATH,
    uninstall_bat_path: Path = UNINSTALL_BAT_PATH,
    rpkg_filename: str = "chunk29patch3.rpkg",
    scene_label_tr: str = "Serbest Ev",
    partition_note: str = "snug patchlevel=3",
    success_hint_tr: str = "Oyuna girip Serbest Çalışan ^(Freelancer^) modunu açarak hemen test edebilirsiniz.",
    orig_chunk_label: str = "chunk29",
    quarantine_dir: Optional[Path] = None,
    quarantine_filenames: tuple[str, ...] = (),
    stale_rpkg_names: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    """Yönetici (UAC) yetkili tek tıkla kurulum ve kaldırma `.bat` betiklerini oluşturur."""
    uninstall_name = uninstall_bat_path.name
    scene_label_bat = scene_label_tr.replace("(", "^(").replace(")", "^)")

    quarantine_install_block = ""
    quarantine_restore_block = ""
    if quarantine_dir is not None and quarantine_filenames:
        q_lines = [
            f'set "QUARANTINE_DIR={quarantine_dir}"',
            'if not exist "%QUARANTINE_DIR%" mkdir "%QUARANTINE_DIR%"',
        ]
        for qf in quarantine_filenames:
            q_lines.append(
                f'if exist "%DST_RUNTIME%\\{qf}" (\n'
                f'    if not exist "%QUARANTINE_DIR%\\{qf}" copy /Y "%DST_RUNTIME%\\{qf}" "%QUARANTINE_DIR%\\{qf}" >nul\n'
                f'    del /F /Q "%DST_RUNTIME%\\{qf}"\n'
                f')'
            )
        for sf in stale_rpkg_names:
            q_lines.append(f'if exist "%DST_RUNTIME%\\{sf}" del /F /Q "%DST_RUNTIME%\\{sf}"')
        quarantine_install_block = "\n".join(q_lines) + "\n"

        r_lines = [
            f'set "QUARANTINE_DIR={quarantine_dir}"',
        ]
        for sf in stale_rpkg_names:
            r_lines.append(f'if exist "%DST_RUNTIME%\\{sf}" del /F /Q "%DST_RUNTIME%\\{sf}"')
        for qf in quarantine_filenames:
            r_lines.append(
                f'if exist "%QUARANTINE_DIR%\\{qf}" (\n'
                f'    echo [+] Yedeklenen eski dosya geri yükleniyor: {qf}...\n'
                f'    copy /Y "%QUARANTINE_DIR%\\{qf}" "%DST_RUNTIME%\\{qf}" >nul\n'
                f')'
            )
        quarantine_restore_block = "\n".join(r_lines) + "\n"

    install_bat_content = f"""@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title SesAI Dublaj Stüdyosu — Hitman 3 {scene_label_tr} ({rpkg_filename}) Türkçe Dublaj Kurulumu

:: Yönetici (UAC) yetkisi kontrolü — gerekirse otomatik yönetici onayı ister
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [BİLGİ] Program Files yazma izni için Yönetici ^(UAC^) onayı isteniyor...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo ============================================================================
echo  SesAI Dublaj Stüdyosu — HITMAN 3 {scene_label_bat} ^({rpkg_filename}^) Dublaj Kurulumu
echo ============================================================================
echo.

set "SRC_RPKG={patched_rpkg_path}"
set "SRC_PKGDEF={patched_pkgdef_path}"
set "BAK_PKGDEF={backup_pkgdef_path}"
set "DST_RUNTIME={runtime_dir}"
set "DST_RPKG=%DST_RUNTIME%\\{rpkg_filename}"
set "DST_PKGDEF=%DST_RUNTIME%\\packagedefinition.txt"

if not exist "%SRC_RPKG%" (
    echo [HATA] Kaynak yama paketi bulunamadı: "%SRC_RPKG%"
    pause
    exit /b 1
)
if not exist "%DST_RUNTIME%" (
    echo [HATA] Hitman 3 Runtime klasörü bulunamadı: "%DST_RUNTIME%"
    pause
    exit /b 1
)

:: İlk kez kuruluyorsa mevcut packagedefinition.txt yedeğini güvenceye al
if not exist "%BAK_PKGDEF%" (
    echo [+] Mevcut packagedefinition.txt yedekleniyor...
    copy /Y "%DST_PKGDEF%" "%BAK_PKGDEF%" >nul
)
{quarantine_install_block}
echo [+] {rpkg_filename} oyuna kopyalanıyor ^(resmi {orig_chunk_label} dosyalarına dokunulmaz^)...
copy /Y "%SRC_RPKG%" "%DST_RPKG%" >nul
if %errorLevel% neq 0 (
    echo [HATA] {rpkg_filename} kopyalanamadı! Oyun açıksa kapatıp tekrar deneyin.
    pause
    exit /b 1
)

echo [+] packagedefinition.txt ^({partition_note}^) güncelleniyor...
copy /Y "%SRC_PKGDEF%" "%DST_PKGDEF%" >nul
if %errorLevel% neq 0 (
    echo [HATA] packagedefinition.txt güncellenemedi! Oyun açıksa kapatıp tekrar deneyin.
    pause
    exit /b 1
)

echo [+] SHA-256 bütünlük doğrulaması yapılıyor...
powershell -NoProfile -Command ^
    "$h1 = (Get-FileHash -LiteralPath '%DST_RPKG%' -Algorithm SHA256).Hash.ToLower(); " ^
    "$h2 = (Get-FileHash -LiteralPath '%DST_PKGDEF%' -Algorithm SHA256).Hash.ToLower(); " ^
    "if ($h1 -ne '{patched_rpkg_sha256.lower()}') {{ Write-Host '[HATA] {rpkg_filename} SHA-256 uyuşmadı!' -ForegroundColor Red; exit 1 }}; " ^
    "if ($h2 -ne '{patched_pkgdef_sha256.lower()}') {{ Write-Host '[HATA] packagedefinition.txt SHA-256 uyuşmadı!' -ForegroundColor Red; exit 1 }}; " ^
    "Write-Host '[OK] SHA-256 Doğrulandı: {rpkg_filename} = ' + $h1 -ForegroundColor Green; " ^
    "Write-Host '[OK] SHA-256 Doğrulandı: packagedefinition.txt = ' + $h2 -ForegroundColor Green;"

if %errorLevel% neq 0 (
    echo [HATA] SHA-256 doğrulaması başarısız oldu!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo  [BAŞARILI] Hitman 3 {scene_label_bat} Türkçe Dublaj Kuruldu!
echo  {success_hint_tr}
echo  Kaldırmak için '{uninstall_name}' dosyasına çift tıklamanız yeterlidir.
echo ============================================================================
pause
exit /b 0
"""

    uninstall_bat_content = f"""@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
title SesAI Dublaj Stüdyosu — Hitman 3 {scene_label_tr} Türkçe Dublaj Kaldırma ^(Geri Al^)

:: Yönetici (UAC) yetkisi kontrolü — gerekirse otomatik yönetici onayı ister
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [BİLGİ] Program Files yazma izni için Yönetici ^(UAC^) onayı isteniyor...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo ============================================================================
echo  SesAI Dublaj Stüdyosu — HITMAN 3 {scene_label_bat} Dublaj Kaldırma ^(Geri Al^)
echo ============================================================================
echo.

set "BAK_PKGDEF={backup_pkgdef_path}"
set "DST_RUNTIME={runtime_dir}"
set "DST_RPKG=%DST_RUNTIME%\\{rpkg_filename}"
set "DST_PKGDEF=%DST_RUNTIME%\\packagedefinition.txt"

if exist "%DST_RPKG%" (
    echo [+] {rpkg_filename} siliniyor...
    del /F /Q "%DST_RPKG%"
    if exist "%DST_RPKG%" (
        echo [HATA] {rpkg_filename} silinemedi! Oyun açıksa kapatıp tekrar deneyin.
        pause
        exit /b 1
    )
    echo [OK] {rpkg_filename} silindi.
) else (
    echo [BİLGİ] {rpkg_filename} zaten silinmiş.
)
{quarantine_restore_block}
if not exist "%BAK_PKGDEF%" (
    echo [HATA] packagedefinition.txt yedeği bulunamadı: "%BAK_PKGDEF%"
    pause
    exit /b 1
)

echo [+] Yedek packagedefinition.txt geri yükleniyor...
copy /Y "%BAK_PKGDEF%" "%DST_PKGDEF%" >nul
if %errorLevel% neq 0 (
    echo [HATA] packagedefinition.txt geri yüklenemedi! Oyun açıksa kapatıp tekrar deneyin.
    pause
    exit /b 1
)

echo [+] packagedefinition.txt SHA-256 doğrulaması yapılıyor...
powershell -NoProfile -Command ^
    "$h = (Get-FileHash -LiteralPath '%DST_PKGDEF%' -Algorithm SHA256).Hash.ToLower(); " ^
    "if ($h -ne '{original_pkgdef_sha256.lower()}') {{ Write-Host '[HATA] Yedek SHA-256 uyuşmadı!' -ForegroundColor Red; exit 1 }}; " ^
    "Write-Host '[OK] packagedefinition.txt SHA-256 %%100 doğrulandı: ' + $h -ForegroundColor Green;"

if %errorLevel% neq 0 (
    echo [HATA] SHA-256 doğrulaması başarısız!
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo  [BAŞARILI] Hitman 3 {scene_label_bat} yaması %%100 geri alındı!
echo ============================================================================
pause
exit /b 0
"""

    install_bat_path.write_text(install_bat_content.replace("\n", "\r\n"), encoding="utf-8")
    uninstall_bat_path.write_text(uninstall_bat_content.replace("\n", "\r\n"), encoding="utf-8")
    return install_bat_path, uninstall_bat_path


def try_direct_install_to_runtime(
    runtime_dir: Path,
    patched_rpkg_path: Path,
    patched_pkgdef_path: Path,
    expected_rpkg_sha256: str,
    expected_pkgdef_sha256: str,
    rpkg_filename: str = "chunk29patch3.rpkg",
) -> dict[str, Any]:
    """`C:\\Program Files\\Epic Games\\HITMAN3\\Runtime\\` dizinine doğrudan yazma izni varsa
    yama `.rpkg` dosyasını ve güncellenmiş `packagedefinition.txt` dosyasını kopyalayıp SHA-256 ile doğrular.
    """
    dst_rpkg = runtime_dir / rpkg_filename
    dst_pkgdef = runtime_dir / "packagedefinition.txt"

    try:
        shutil.copy2(patched_rpkg_path, dst_rpkg)
        shutil.copy2(patched_pkgdef_path, dst_pkgdef)
        live_rpkg_sha = compute_sha256(dst_rpkg)
        live_pkgdef_sha = compute_sha256(dst_pkgdef)
        verified = (
            live_rpkg_sha.lower() == expected_rpkg_sha256.lower()
            and live_pkgdef_sha.lower() == expected_pkgdef_sha256.lower()
        )
        return {
            "installed_directly": verified,
            "requires_uac_bat": not verified,
            "dst_rpkg_path": str(dst_rpkg),
            "dst_pkgdef_path": str(dst_pkgdef),
            "live_rpkg_sha256": live_rpkg_sha,
            "live_pkgdef_sha256": live_pkgdef_sha,
            "error": "" if verified else "SHA-256 doğrulama uyuşmazlığı",
        }
    except PermissionError as exc:
        return {
            "installed_directly": False,
            "requires_uac_bat": True,
            "dst_rpkg_path": str(dst_rpkg),
            "dst_pkgdef_path": str(dst_pkgdef),
            "live_rpkg_sha256": "",
            "live_pkgdef_sha256": "",
            "error": f"Program Files yönetici (UAC) izni gerekiyor: {exc}",
        }
    except Exception as exc:
        return {
            "installed_directly": False,
            "requires_uac_bat": True,
            "dst_rpkg_path": str(dst_rpkg),
            "dst_pkgdef_path": str(dst_pkgdef),
            "live_rpkg_sha256": "",
            "live_pkgdef_sha256": "",
            "error": str(exc),
        }


def repack_scene_patch(
    scene_root: Path,
    patch_rpkg_stem: str,
    rpkgv2_meta_bytes: bytes,
    partition_name: str,
    min_patchlevel: int,
    scene_label_tr: str,
    install_bat_path: Path,
    uninstall_bat_path: Path,
    success_hint_tr: str,
    orig_chunk_label: str,
    require_all_modified: bool = True,
    rpkg_generate_flag: str = "-generate_rpkg_from",
    quarantine_patch_numbers: tuple[int, ...] = (),
    stale_rpkg_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Genel Hitman 3 sahne dublaj paketleyici ve kurulum aracı.

    Verilen sahne klasöründeki (`sesler/`, `ham_wwes/`, `hitman_manifest.json`) dublajlı `.wav`
    dosyalarını Wwise 16-bit PCM (`0xFFFE`) `.WWES` + `.WWES.meta` formatına dönüştürür,
    bağımsız `<patch_rpkg_stem>.rpkg` yaması üretir, `packagedefinition.txt` dosyasını yedekleyip
    günceller, kur/kaldır `.bat` betiklerini yazar ve `Runtime\\` klasörüne doğrudan kurar.
    """
    bootstrap_portable_environment()
    install_info = find_epic_hitman_install()
    runtime_dir = Path(install_info["runtime_dir"])

    manifest_path = scene_root / "hitman_manifest.json"
    wav_dir = scene_root / "sesler"
    raw_wwes_dir = scene_root / "ham_wwes"
    package_dir = scene_root / "paket"
    backup_dir = scene_root / "oyun_yedek"
    report_path = package_dir / "repack_raporu.json"
    rpkg_filename = f"{patch_rpkg_stem}.rpkg"

    # 1. Dublajlı WAV doğrulaması
    wav_checks, wav_warnings = verify_dubbed_wavs(
        manifest_path=manifest_path,
        wav_dir=wav_dir,
    )
    missing = [c.wav_filename for c in wav_checks if not c.exists]
    if missing:
        raise FileNotFoundError(f"Eksik dublaj WAV dosyaları: {missing}")

    unchanged = [c.wav_filename for c in wav_checks if not c.is_dubbed_modified]
    if require_all_modified and unchanged:
        raise RuntimeError(
            f"Henüz Türkçe dublajla güncellenmemiş {len(unchanged)} dosya var: {unchanged}"
        )

    work_dir = TEMP_DIR / f"hrp_{patch_rpkg_stem}"
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = FFmpegManager()

    try:
        # 2. Paketleme klasörünü hazırla (<scene>/paket/<patch_rpkg_stem>/WWES)
        stage_chunk_dir = package_dir / patch_rpkg_stem
        stage_wwes_dir = stage_chunk_dir / "WWES"
        if stage_chunk_dir.exists():
            shutil.rmtree(stage_chunk_dir)
        stage_wwes_dir.mkdir(parents=True, exist_ok=True)

        # RPKGv2 (2KPR) meta başlığını yaz ki rpkg-cli RPKGv2 formatında üretsin
        (stage_chunk_dir / f"{patch_rpkg_stem}.meta").write_bytes(rpkgv2_meta_bytes)

        repacked_entries: list[RepackedWwesEntry] = []

        for chk in wav_checks:
            wav_path = Path(chk.wav_path)
            orig_wwes_path = raw_wwes_dir / f"{chk.hash_hex}.WWES"
            orig_meta_path = raw_wwes_dir / f"{chk.hash_hex}.WWES.meta"
            if not orig_wwes_path.is_file() or not orig_meta_path.is_file():
                raise FileNotFoundError(
                    f"Ham WWES veya meta yedeği bulunamadı: {orig_wwes_path}"
                )

            out_wwes_path = stage_wwes_dir / f"{chk.hash_hex}.WWES"
            out_meta_path = stage_wwes_dir / f"{chk.hash_hex}.WWES.meta"

            vorbis_info = convert_wav_to_wwise_vorbis_wwes(
                dubbed_wav_path=wav_path,
                orig_wwes_path=orig_wwes_path,
                output_wwes_path=out_wwes_path,
                target_channels=chk.expected_channels,
                work_dir=work_dir,
                ffmpeg=ffmpeg,
                vorbis_quality=8,
            )

            wwes_size = out_wwes_path.stat().st_size
            updated_final_sz = update_wwes_meta_file(
                orig_meta_path=orig_meta_path,
                output_meta_path=out_meta_path,
                new_wwes_size_bytes=wwes_size,
            )
            assert updated_final_sz == wwes_size

            # RIFF/WAVE 0xFFFF başlık doğrulaması
            hdr_info = inspect_wwes_riff_header(out_wwes_path)
            if (
                hdr_info.riff_magic != "RIFF"
                or hdr_info.wave_magic != "WAVE"
                or hdr_info.format_tag_int != 0xFFFF
                or hdr_info.fmt_chunk_size != 66
                or hdr_info.sample_rate != 48000
                or hdr_info.channels != chk.expected_channels
                or (hdr_info.riff_declared_size + 8) != wwes_size
            ):
                raise RuntimeError(
                    f"Üretilen 0xFFFF WWES RIFF/WAVE doğrulaması başarısız: {out_wwes_path.name} ({hdr_info})"
                )

            # Üçlü gidiş-dönüş (ww2ogg + vgmstream-cli + FFmpeg RMS/LUFS > -40 dBFS) doğrulaması
            rt_info = verify_wwes_roundtrip(
                wwes_path=out_wwes_path,
                expected_channels=chk.expected_channels,
                work_dir=work_dir,
                ffmpeg=ffmpeg,
                min_audible_dbfs=-40.0,
            )

            repacked_entries.append(
                RepackedWwesEntry(
                    index=chk.index,
                    hash_hex=chk.hash_hex,
                    hash_id=f"{chk.hash_hex}.WWES",
                    wav_filename=chk.wav_filename,
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

        # 3. rpkg-cli.exe ile bağımsız yama .rpkg dosyasını üret
        out_rpkg_path = package_dir / rpkg_filename
        if out_rpkg_path.exists():
            out_rpkg_path.unlink()

        gen_res = _run_rpkg_cli(
            [
                "-output_path",
                str(package_dir),
                rpkg_generate_flag,
                str(stage_chunk_dir),
            ]
        )
        if not out_rpkg_path.is_file():
            raise RuntimeError(
                f"{rpkg_filename} üretilemedi: {gen_res.stdout} {gen_res.stderr}"
            )

        # Üretilen .rpkg dosyasını hem RPKGv2 parser hem de rpkg-cli -extract_from_rpkg ile doğrula
        rpkg_info, _rpkg_entries, _rpkg_deleted = parse_rpkg_file(out_rpkg_path, filter_type="WWES")
        if rpkg_info.magic != "2KPR" or rpkg_info.wwes_count != len(wav_checks):
            raise RuntimeError(
                f"Üretilen {rpkg_filename} başlık doğrulaması başarısız: {rpkg_info}"
            )

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
        if len(extracted_back) != len(wav_checks):
            raise RuntimeError(
                f"rpkg-cli doğrulama çıkarması eksik: {len(extracted_back)}/{len(wav_checks)}"
            )

        rpkg_sha256 = compute_sha256(out_rpkg_path)

        # 4. packagedefinition.txt yedekle ve hedef partition patchlevel değerini güvenli güncelle
        pkgdef_info = backup_and_patch_packagedefinition(
            runtime_dir=runtime_dir,
            backup_dir=backup_dir,
            package_dir=package_dir,
            work_dir=work_dir,
            partition_name=partition_name,
            min_patchlevel=min_patchlevel,
        )

        patched_pkgdef_path = Path(pkgdef_info["patched_package_path"])
        backup_pkgdef_path = Path(pkgdef_info["backup_path"])

        # 4B. Resmi patchlevel üzerinde kalan çakışan eski atıl mod dosyalarını SHA-256 yedeğiyle karantinaya al
        quarantine_dir: Optional[Path] = None
        quarantine_info: Optional[dict[str, Any]] = None
        quarantine_fnames: tuple[str, ...] = ()
        if quarantine_patch_numbers:
            q_range = f"{min(quarantine_patch_numbers)}_{max(quarantine_patch_numbers)}"
            quarantine_dir = backup_dir / f"eski_mod_{orig_chunk_label}patch{q_range}"
            quarantine_fnames = tuple(
                f"{orig_chunk_label}patch{n}.rpkg" for n in quarantine_patch_numbers
            )
            quarantine_info = quarantine_conflicting_mod_patches(
                runtime_dir=runtime_dir,
                quarantine_dir=quarantine_dir,
                chunk_prefix=orig_chunk_label,
                patch_numbers=quarantine_patch_numbers,
                stale_rpkg_names=stale_rpkg_names,
            )

        # 5. Yönetici (UAC) kurulum ve kaldırma .bat betiklerini oluştur
        install_bat, uninstall_bat = write_install_and_uninstall_bats(
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
            scene_label_tr=scene_label_tr,
            partition_note=f"{partition_name} patchlevel={pkgdef_info['new_patchlevel']}",
            success_hint_tr=success_hint_tr,
            orig_chunk_label=orig_chunk_label,
            quarantine_dir=quarantine_dir,
            quarantine_filenames=quarantine_fnames,
            stale_rpkg_names=stale_rpkg_names,
        )

        # 6. Runtime klasörüne doğrudan kurulum dene (yazma izni varsa anında kurar)
        install_result = try_direct_install_to_runtime(
            runtime_dir=runtime_dir,
            patched_rpkg_path=out_rpkg_path,
            patched_pkgdef_path=patched_pkgdef_path,
            expected_rpkg_sha256=rpkg_sha256,
            expected_pkgdef_sha256=pkgdef_info["patched_sha256"],
            rpkg_filename=rpkg_filename,
        )

        pkg_block = {
            "rpkg_filename": rpkg_filename,
            "rpkg_path": str(out_rpkg_path),
            "rpkg_size_bytes": out_rpkg_path.stat().st_size,
            "rpkg_size_mb": round(out_rpkg_path.stat().st_size / (1024 * 1024), 2),
            "rpkg_sha256": rpkg_sha256,
            "rpkg_magic": rpkg_info.magic,
            "wwes_count": rpkg_info.wwes_count,
            "rpkg_cli_roundtrip_verified": True,
            "entries": [asdict(e) for e in repacked_entries],
        }
        report_data: dict[str, Any] = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scene_label_tr": scene_label_tr,
            "hitman_runtime_dir": str(runtime_dir),
            "original_rpkgs_untouched": True,
            "dubbed_wav_verification": {
                "total_files": len(wav_checks),
                "dubbed_modified_count": sum(1 for c in wav_checks if c.is_dubbed_modified),
                "warnings": wav_warnings,
                "files": [asdict(c) for c in wav_checks],
            },
            "rpkg_package": pkg_block,
            f"{patch_rpkg_stem}_package": pkg_block,
            "packagedefinition_patch": pkgdef_info,
            "quarantined_legacy_mods": quarantine_info,
            "deployment": {
                **install_result,
                "install_bat_path": str(install_bat),
                "uninstall_bat_path": str(uninstall_bat),
            },
        }

        report_path.write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return report_data
    finally:
        cleanup_temp_dir(keep_root=True)


def repack_chunk29_safehouse() -> dict[str, Any]:
    """11 Türkçe dublajlı Serbest Ev `.wav` dosyasını `chunk29patch3.rpkg` olarak paketler ve kurar."""
    return repack_scene_patch(
        scene_root=SAFEHOUSE_OUTPUT_ROOT,
        patch_rpkg_stem="chunk29patch3",
        rpkgv2_meta_bytes=CHUNK29_RPKGV2_META_BYTES,
        partition_name="snug",
        min_patchlevel=3,
        scene_label_tr="Serbest Ev",
        install_bat_path=INSTALL_BAT_PATH,
        uninstall_bat_path=UNINSTALL_BAT_PATH,
        success_hint_tr="Oyuna girip Serbest Çalışan ^(Freelancer^) modunu açarak hemen test edebilirsiniz.",
        orig_chunk_label="chunk29",
        require_all_modified=False,
        rpkg_generate_flag="-generate_rpkg_quickly_from",
    )


def repack_chunk0_prologue() -> dict[str, Any]:
    """15 Türkçe dublajlı ICA Tesisi İlk Sahne `.wav` dosyasını güvenli ardışık `chunk0patch5.rpkg`
    (`super patchlevel=5`) olarak paketler; resmi `chunk0patch1..4.rpkg` dosyalarına dokunmadan,
    çökmeye yol açan eski atıl 2023–2024 mod dosyalarını (`chunk0patch5..9.rpkg`) SHA-256 yedeğiyle
    karantinaya alır ve kurar.
    """
    return repack_scene_patch(
        scene_root=PROLOGUE_OUTPUT_ROOT,
        patch_rpkg_stem="chunk0patch5",
        rpkgv2_meta_bytes=CHUNK0_RPKGV2_META_BYTES,
        partition_name="super",
        min_patchlevel=5,
        scene_label_tr="İlk Sahne (ICA Tesisi)",
        install_bat_path=PROLOGUE_INSTALL_BAT_PATH,
        uninstall_bat_path=PROLOGUE_UNINSTALL_BAT_PATH,
        success_hint_tr="Oyuna girip Seferberlikler -> Başlangıç -> ICA Tesisi ^(Rehberli Eğitim / Serbest Eğitim / Varış^) görevini açarak hemen test edebilirsiniz.",
        orig_chunk_label="chunk0",
        require_all_modified=True,
        rpkg_generate_flag="-generate_rpkg_from",
        quarantine_patch_numbers=(5, 6, 7, 8, 9),
        stale_rpkg_names=("chunk0patch10.rpkg",),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SesAI Dublaj Stüdyosu — Hitman 3 Bağımsız RPKG Yama Paketleyici ve Kurulum Aracı"
    )
    parser.add_argument(
        "--scene",
        choices=("safehouse", "prologue"),
        default="safehouse",
        help="Paketlenecek Hitman 3 sahnesi: 'safehouse' (chunk29patch3) veya 'prologue' (chunk0patch5)",
    )
    args = parser.parse_args(argv)

    print("=" * 78)
    if args.scene == "prologue":
        print("SesAI Dublaj Stüdyosu — Hitman 3 İlk Sahne / ICA Tesisi (chunk0patch5.rpkg) Paketleyici")
        print("=" * 78)
        report = repack_chunk0_prologue()
    else:
        print("SesAI Dublaj Stüdyosu — Hitman 3 Serbest Ev (chunk29patch3.rpkg) Paketleyici")
        print("=" * 78)
        report = repack_chunk29_safehouse()

    wv = report["dubbed_wav_verification"]
    pkg = report["rpkg_package"]
    pdef = report["packagedefinition_patch"]
    dep = report["deployment"]

    print(
        f"[+] Dublajlı WAV Kontrolü : {wv['dubbed_modified_count']}/{wv['total_files']} dosya Türkçe dublajlanmış"
    )
    if wv["warnings"]:
        for w in wv["warnings"]:
            print(f"    [!] {w}")
    print(
        f"[+] Üretilen RPKG Yaması  : {pkg['rpkg_path']} ({pkg['rpkg_size_mb']} MB | "
        f"Magic={pkg['rpkg_magic']} | WWES={pkg['wwes_count']})"
    )
    print(f"[+] RPKG SHA-256          : {pkg['rpkg_sha256']}")
    print(f"[+] Yedek packagedef      : {pdef['backup_path']} (SHA-256: {pdef['original_sha256']})")
    print(
        f"[+] Partition Güncelleme  : '{pdef['old_partition_line']}' -> '{pdef['new_partition_line']}'"
    )
    if pdef.get("preserved_snug_line") and pdef.get("partition_name") != "snug":
        print(f"[+] Korunan snug Ayarı    : '{pdef['preserved_snug_line']}'")
    if pdef.get("preserved_super_line") and pdef.get("partition_name") != "super":
        print(f"[+] Korunan super Ayarı   : '{pdef['preserved_super_line']}'")
    print(f"[+] Kurulum Betiği (.bat) : {dep['install_bat_path']}")
    print(f"[+] Kaldırma Betiği (.bat): {dep['uninstall_bat_path']}")
    if dep["installed_directly"]:
        print(
            f"[+] DOĞRUDAN KURULUM      : BAŞARILI! ({dep['dst_rpkg_path']} & packagedefinition.txt SHA-256 doğrulandı)"
        )
    else:
        print(
            f"[!] DOĞRUDAN KURULUM NOTU : {dep['error']}\n"
            f"    -> Yamayı oyuna kurmak için '{dep['install_bat_path']}' dosyasına çift tıklayıp Yönetici onayını verin."
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
