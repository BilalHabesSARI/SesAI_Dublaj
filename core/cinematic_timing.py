"""Sinematik Zamanlama Kilidi (>= 15 sn Uzun Monolog Eslerini Koruma).

Yalnızca kullanıcı isteğe bağlı olarak onay kutusunu açtığında ve dosya >= 15.0 sn
uzunluğunda olduğunda devreye girer; 15 saniyeden kısa tüm repliklerde standart akış
%100 aynı kalır.

İşleyiş:
1. Orijinal seste >= 0.65 sn sessizliklerle ayrılan konuşma bloklarını tespit eder.
2. Türkçe metni cümle sınırlarına göre bu konuşma bloklarına süreyle orantılı paylaştırır.
3. Her bloğu aynı referans sesle üretip orijinal bloğun başladığı saniyeye 15 ms
   yumuşak kosinüs fade-in / fade-out geçişiyle yerleştirir.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf

MIN_CINEMATIC_MONOLOGUE_SEC: float = 15.0
MIN_PAUSE_SEC: float = 0.65
MIN_BLOCK_MS: int = 150
EDGE_FADE_MS: float = 15.0


@dataclass(frozen=True)
class SpeechBlock:
    """Orijinal sesteki tek bir konuşma bloğu (milisaniye cinsinden)."""

    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


def _read_mono_audio(
    audio_path: Path,
    ffmpeg=None,
    work_dir: Optional[Path] = None,
) -> tuple[np.ndarray, int]:
    """Ses dosyasını float32 mono dizi ve örnekleme hızı olarak okur."""
    try:
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception:
        if ffmpeg is not None and work_dir is not None:
            tmp_wav = work_dir / f"_cin_src_{audio_path.stem}.wav"
            ffmpeg.convert_to_wav(audio_path, tmp_wav, sample_rate=48000, channels=1)
            data, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=False)
            tmp_wav.unlink(missing_ok=True)
        else:
            raise
    if data.ndim > 1:
        data = np.mean(data, axis=1, dtype=np.float32)
    return np.asarray(data, dtype=np.float32), int(sr)


def detect_speech_blocks(
    audio_path: Path,
    min_pause_sec: float = MIN_PAUSE_SEC,
    min_block_ms: int = MIN_BLOCK_MS,
    ffmpeg=None,
    work_dir: Optional[Path] = None,
) -> tuple[list[SpeechBlock], int]:
    """Orijinal sesteki konuşma bloklarını ve >= `min_pause_sec` uzun sessizlik aralıklarını tespit eder."""
    samples, sr = _read_mono_audio(audio_path, ffmpeg=ffmpeg, work_dir=work_dir)
    total_ms = int(round((len(samples) / max(1, sr)) * 1000.0))
    if samples.size == 0 or sr <= 0:
        return [], total_ms

    frame_len = max(1, int(round(sr * 0.020)))  # 20 ms pencere
    hop_len = max(1, int(round(sr * 0.010)))    # 10 ms kaydırma
    if len(samples) < frame_len:
        return [SpeechBlock(0, total_ms)] if total_ms >= min_block_ms else [], total_ms

    n_frames = 1 + (len(samples) - frame_len) // hop_len
    rms_db = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s0 = i * hop_len
        seg = samples[s0 : s0 + frame_len]
        rms = float(np.sqrt(np.mean(seg * seg, dtype=np.float64)))
        rms_db[i] = 20.0 * math.log10(max(rms, 1e-7))

    peak_db = float(np.percentile(rms_db, 98))
    if peak_db < -55.0:
        return [], total_ms

    # Gürültü tabanına uyumlu konuşma eşiği
    thresh_db = max(-48.0, min(-30.0, peak_db - 28.0))
    active = rms_db >= thresh_db

    # Ham aktif aralıkları çıkar
    raw_blocks: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for idx, is_on in enumerate(active):
        if is_on and not in_seg:
            seg_start = idx
            in_seg = True
        elif not is_on and in_seg:
            start_ms = int(round((seg_start * hop_len / sr) * 1000.0))
            end_ms = int(round(((idx * hop_len + frame_len) / sr) * 1000.0))
            raw_blocks.append((start_ms, min(total_ms, end_ms)))
            in_seg = False
    if in_seg:
        start_ms = int(round((seg_start * hop_len / sr) * 1000.0))
        raw_blocks.append((start_ms, total_ms))

    if not raw_blocks:
        return [], total_ms

    # `min_pause_sec` (0.65 sn) altındaki kısa nefes/duraklamaları aynı konuşma bloğunda birleştir
    min_pause_ms = int(round(min_pause_sec * 1000.0))
    merged: list[list[int]] = [[raw_blocks[0][0], raw_blocks[0][1]]]
    for s_ms, e_ms in raw_blocks[1:]:
        if s_ms - merged[-1][1] < min_pause_ms:
            merged[-1][1] = max(merged[-1][1], e_ms)
        else:
            merged.append([s_ms, e_ms])

    blocks = [
        SpeechBlock(start_ms=s, end_ms=e)
        for s, e in merged
        if (e - s) >= min_block_ms
    ]
    return blocks, total_ms


def _split_into_sentences(text: str) -> list[str]:
    """Türkçe metni cümle sınırlarına (., !, ?, ..., ;) göre böler."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    # Cümle sonu noktalama işaretleri (ve noktalı virgül) ardından gelen boşluklardan böl
    parts = re.split(r"(?<=[.!?;…])\s+", cleaned)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _expand_clauses_if_needed(units: list[str], target_count: int) -> list[str]:
    """Cümle sayısı blok sayısından azsa en uzun cümleleri virgül/tire sınırından yumuşakça ayırır."""
    out = list(units)
    while len(out) < target_count:
        # Bölünebilir en uzun parçayı bul
        best_idx = -1
        best_len = 0
        best_split: Optional[tuple[str, str]] = None
        for idx, u in enumerate(out):
            if len(u) <= best_len:
                continue
            # Ortaya en yakın ", " veya " — " veya " - " ayıracını bul
            candidates = [m.start() for m in re.finditer(r"(?:,\s+|\s+[—–-]\s+)", u)]
            if not candidates:
                continue
            mid = len(u) / 2.0
            split_at = min(candidates, key=lambda pos: abs(pos - mid))
            m = re.match(r"(?:,\s+|\s+[—–-]\s+)", u[split_at:])
            if not m:
                continue
            sep = m.group(0)
            left = (u[:split_at] + ("," if sep.startswith(",") else "")).strip()
            right = u[split_at + len(sep) :].strip()
            if len(left) >= 8 and len(right) >= 8:
                best_idx = idx
                best_len = len(u)
                best_split = (left, right)
        if best_idx < 0 or best_split is None:
            break
        out[best_idx : best_idx + 1] = [best_split[0], best_split[1]]
    return out


def split_turkish_text_to_blocks(
    text: str,
    blocks: Sequence[SpeechBlock],
) -> tuple[list[SpeechBlock], list[str]]:
    """Türkçe metni cümle sınırlarına göre orijinal konuşma bloklarına süreyle orantılı paylaştırır."""
    if not blocks:
        return [], []
    units = _split_into_sentences(text)
    if not units:
        return [], []

    cur_blocks = list(blocks)
    if len(units) < len(cur_blocks):
        units = _expand_clauses_if_needed(units, len(cur_blocks))

    # Hâlâ metin birimi blok sayısından azsa, aralarındaki sessizlik en kısa olan komşu blokları birleştir
    while len(cur_blocks) > len(units) and len(cur_blocks) > 1:
        min_gap_idx = min(
            range(len(cur_blocks) - 1),
            key=lambda i: cur_blocks[i + 1].start_ms - cur_blocks[i].end_ms,
        )
        merged_blk = SpeechBlock(
            start_ms=cur_blocks[min_gap_idx].start_ms,
            end_ms=cur_blocks[min_gap_idx + 1].end_ms,
        )
        cur_blocks[min_gap_idx : min_gap_idx + 2] = [merged_blk]

    k = len(cur_blocks)
    n = len(units)
    if k == 1:
        return cur_blocks, [" ".join(units)]
    if k == n:
        return cur_blocks, units

    # Dinamik programlama ile k bloğa n cümleyi süre oranlarına en yakın şekilde dağıt
    total_dur = float(sum(max(1, b.duration_ms) for b in cur_blocks))
    target_ratios = [max(1, b.duration_ms) / total_dur for b in cur_blocks]

    weights = [max(1, len(u)) for u in units]
    total_w = float(sum(weights))
    prefix = [0]
    for w in weights:
        prefix.append(prefix[-1] + w)

    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(k + 1)]
    parent = [[-1] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0

    for b_idx in range(1, k + 1):
        t_ratio = target_ratios[b_idx - 1]
        for end_u in range(b_idx, n - (k - b_idx) + 1):
            for start_u in range(b_idx - 1, end_u):
                if dp[b_idx - 1][start_u] == inf:
                    continue
                seg_ratio = (prefix[end_u] - prefix[start_u]) / total_w
                cost = dp[b_idx - 1][start_u] + (seg_ratio - t_ratio) ** 2
                if cost < dp[b_idx][end_u]:
                    dp[b_idx][end_u] = cost
                    parent[b_idx][end_u] = start_u

    assigned_chunks: list[str] = []
    cur_end = n
    for b_idx in range(k, 0, -1):
        start_u = parent[b_idx][cur_end]
        assigned_chunks.append(" ".join(units[start_u:cur_end]))
        cur_end = start_u
    assigned_chunks.reverse()
    return cur_blocks, assigned_chunks


def trim_block_edges_and_fade(
    audio: np.ndarray,
    sample_rate: int,
    fade_ms: float = EDGE_FADE_MS,
    margin_ms: float = 25.0,
    silence_db: float = -42.0,
) -> np.ndarray:
    """Bloğun yalnızca baş/son ölü sessizliğini temizler ve yumuşak 15 ms fade-in/fade-out uygular."""
    if audio.size == 0 or sample_rate <= 0:
        return audio.astype(np.float32, copy=True)

    mono = np.mean(audio, axis=1, dtype=np.float32) if audio.ndim > 1 else audio.astype(np.float32, copy=True)
    amp_thresh = 10.0 ** (silence_db / 20.0)
    nz = np.flatnonzero(np.abs(mono) >= amp_thresh)
    if nz.size > 0:
        margin_samples = int(round(sample_rate * (margin_ms / 1000.0)))
        start_idx = max(0, int(nz[0]) - margin_samples)
        end_idx = min(len(mono), int(nz[-1]) + 1 + margin_samples)
        mono = mono[start_idx:end_idx].copy()

    fade_samples = min(len(mono) // 2, int(round(sample_rate * (fade_ms / 1000.0))))
    if fade_samples > 1:
        t = np.linspace(0.0, math.pi, fade_samples, dtype=np.float32)
        ramp_in = 0.5 * (1.0 - np.cos(t))
        ramp_out = ramp_in[::-1]
        mono[:fade_samples] *= ramp_in
        mono[-fade_samples:] *= ramp_out

    return mono


def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio
    try:
        from scipy.signal import resample_poly

        g = math.gcd(src_sr, dst_sr)
        return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
    except Exception:
        new_len = max(1, int(round(len(audio) * (dst_sr / float(src_sr)))))
        old_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
        new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)
        return np.interp(new_x, old_x, audio).astype(np.float32)


def assemble_timed_blocks(
    block_wavs: Sequence[tuple[SpeechBlock, Path]],
    total_orig_duration_ms: int,
    output_wav_path: Path,
    fade_ms: float = EDGE_FADE_MS,
    min_inter_block_gap_ms: int = 40,
) -> tuple[Path, list[tuple[int, int]]]:
    """Üretilmiş her Türkçe bloğu orijinal bloğun başlangıç saniyesine 15 ms yumuşak geçişle yerleştirir."""
    if not block_wavs:
        raise ValueError("Birleştirilecek konuşma bloğu bulunamadı.")

    loaded: list[tuple[SpeechBlock, np.ndarray, int]] = []
    target_sr = 0
    for blk, wav_p in block_wavs:
        arr, sr = sf.read(str(wav_p), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1, dtype=np.float32)
        target_sr = max(target_sr, int(sr))
        loaded.append((blk, np.asarray(arr, dtype=np.float32), int(sr)))

    if target_sr <= 0:
        target_sr = 48000

    placements: list[tuple[int, np.ndarray]] = []
    placed_intervals_ms: list[tuple[int, int]] = []
    prev_end_sample = 0
    min_gap_samples = int(round(target_sr * (min_inter_block_gap_ms / 1000.0)))

    for idx, (blk, raw_arr, sr) in enumerate(loaded):
        resampled = _resample_if_needed(raw_arr, sr, target_sr)
        clean_blk = trim_block_edges_and_fade(resampled, target_sr, fade_ms=fade_ms)
        desired_start = int(round((blk.start_ms / 1000.0) * target_sr))
        if idx == 0:
            actual_start = max(0, desired_start)
        else:
            actual_start = max(desired_start, prev_end_sample + min_gap_samples)
        actual_end = actual_start + len(clean_blk)
        placements.append((actual_start, clean_blk))
        placed_intervals_ms.append(
            (
                int(round((actual_start / target_sr) * 1000.0)),
                int(round((actual_end / target_sr) * 1000.0)),
            )
        )
        prev_end_sample = actual_end

    orig_samples = int(round((max(0, total_orig_duration_ms) / 1000.0) * target_sr))
    tail_pad = int(round(target_sr * 0.080))
    total_samples = max(orig_samples, prev_end_sample + tail_pad, 1)

    canvas = np.zeros(total_samples, dtype=np.float32)
    for start_s, blk_arr in placements:
        end_s = min(total_samples, start_s + len(blk_arr))
        seg_len = end_s - start_s
        if seg_len > 0:
            canvas[start_s:end_s] += blk_arr[:seg_len]

    peak = float(np.max(np.abs(canvas))) if canvas.size > 0 else 0.0
    if peak > 0.99:
        canvas *= 0.98 / peak

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav_path), canvas, target_sr, subtype="PCM_16")
    return output_wav_path, placed_intervals_ms
