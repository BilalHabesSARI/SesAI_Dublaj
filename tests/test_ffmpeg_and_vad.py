"""FFmpeg sessizlik kırpma, doğal tempo (1.00x kapalı / 1.05x varsayılan), vokal ayırma & miks ve VAD testleri."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from core.ffmpeg_manager import FFmpegManager
from core.vad_filter import SileroVADFilter
from core.vocal_separator import VocalSeparator


def _make_sine_wav(
    path: Path,
    duration_sec: float,
    freq: float = 440.0,
    sr: int = 44100,
    channels: int = 2,
    amp: float = 0.25,
) -> None:
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    wave = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    if channels == 2:
        wave = np.stack([wave, wave], axis=1)
    sf.write(str(path), wave, sr, subtype="PCM_16")


def test_smart_silence_trimming_trims_edges_and_long_pauses(tmp_path: Path) -> None:
    ff = FFmpegManager()
    sr = 24000
    # 0.6s baş sessizlik + 0.6s ton + 1.0s aşırı uzun ara boşluk + 0.6s ton + 0.6s son sessizlik = 3.4s
    sil_edge = np.zeros(int(sr * 0.6), dtype=np.float32)
    sil_mid = np.zeros(int(sr * 1.0), dtype=np.float32)
    t = np.linspace(0.0, 0.6, int(sr * 0.6), endpoint=False, dtype=np.float32)
    tone = (0.3 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)

    full = np.concatenate([sil_edge, tone, sil_mid, tone, sil_edge])
    src = tmp_path / "padded_speech.wav"
    dst = tmp_path / "trimmed_speech.wav"
    sf.write(str(src), full, sr, subtype="PCM_16")

    out_path, saved_ms = ff.trim_silence_safely(src, dst)
    assert out_path.is_file()
    # Baş (~0.56s) + orta (~0.75s) + son (~0.56s) => en az 1400ms temizlenmiş olmalı!
    assert saved_ms >= 1400
    out_meta = ff.probe_audio(out_path, measure_lufs=False)
    assert 1350 <= out_meta.duration_ms <= 1750


def test_natural_tempo_disabled_at_1_00_and_capped_at_1_05(tmp_path: Path) -> None:
    ff = FFmpegManager()
    assert ff.is_available()

    orig_wav = tmp_path / "orig.wav"
    _make_sine_wav(orig_wav, duration_sec=2.0, freq=220.0, sr=44100, channels=2, amp=0.2)
    orig_ogg = tmp_path / "orig.ogg"
    ff._run([ff.ffmpeg_exe, "-y", "-i", str(orig_wav), "-c:a", "libvorbis", str(orig_ogg)])
    meta_ogg = ff.probe_audio(orig_ogg, measure_lufs=True)

    # 2.6s dublaj WAV
    dub_wav = tmp_path / "raw_dub_tr.wav"
    _make_sine_wav(dub_wav, duration_sec=2.6, freq=300.0, sr=24000, channels=1, amp=0.4)

    # 1. Maks Hız = 1.00x (Hızlandırma tamamen kapalı!)
    out_off = tmp_path / "natural_off.ogg"
    p_off, ms_off, tempo_off = ff.sync_and_package_dub(
        dub_wav_path=dub_wav,
        original_meta=meta_ogg,
        output_path=out_off,
        max_tempo_speedup=1.00,
        trim_silence=False,
    )
    assert p_off.is_file()
    assert tempo_off == 1.0
    assert 2450 <= ms_off <= 2750

    # 2. Maks Hız = 1.05x (Yeni doğal varsayılan üst sınır!)
    out_nat = tmp_path / "natural_105.ogg"
    p_nat, _, tempo_nat = ff.sync_and_package_dub(
        dub_wav_path=dub_wav,
        original_meta=meta_ogg,
        output_path=out_nat,
        max_tempo_speedup=1.05,
        trim_silence=False,
    )
    assert p_nat.is_file()
    assert 1.04 <= tempo_nat <= 1.055
    out_meta = ff.probe_audio(p_nat, measure_lufs=True)
    assert out_meta.sample_rate == 44100
    assert out_meta.channels == 2


def test_vocal_separator_ffmpeg_fallback_and_background_remix(tmp_path: Path) -> None:
    ff = FFmpegManager()
    sep = VocalSeparator(ffmpeg=ff)

    # Stereo karışım (merkez konuşma tonu + yanlarda arka plan efekti)
    sr = 44100
    t = np.linspace(0.0, 1.5, int(sr * 1.5), endpoint=False, dtype=np.float32)
    voice_center = 0.25 * np.sin(2.0 * np.pi * 600.0 * t)
    bg_left = 0.12 * np.sin(2.0 * np.pi * 90.0 * t)
    bg_right = -0.12 * np.sin(2.0 * np.pi * 90.0 * t)
    mix = np.stack([voice_center + bg_left, voice_center + bg_right], axis=1).astype(np.float32)

    orig = tmp_path / "scene_with_sfx.wav"
    sf.write(str(orig), mix, sr, subtype="PCM_16")
    orig_meta = ff.probe_audio(orig, measure_lufs=True)

    res = sep.separate(orig, tmp_path / "stems", stem_prefix="s1", prefer_local_demucs=False)
    assert res.backend_used == "ffmpeg_spectral"
    assert res.vocals_wav.is_file()
    assert res.background_wav.is_file()

    # Dublaj vokali arka planla tekrar miksle
    dub_voc = tmp_path / "dub_tr.wav"
    _make_sine_wav(dub_voc, duration_sec=1.4, freq=350.0, sr=24000, channels=1, amp=0.25)
    final_mix = tmp_path / "final_remixed.wav"

    out_path, final_ms, _ = ff.sync_and_package_dub(
        dub_wav_path=dub_voc,
        original_meta=orig_meta,
        output_path=final_mix,
        max_tempo_speedup=1.05,
        background_wav_path=res.background_wav,
        bg_volume_db=0.0,
    )
    assert out_path.is_file()
    assert final_ms >= 1300
    final_meta = ff.probe_audio(out_path, measure_lufs=False)
    assert final_meta.sample_rate == 44100
    assert final_meta.channels == 2


def test_silero_vad_onnx_rejects_silence_and_pure_sfx(tmp_path: Path) -> None:
    vad = SileroVADFilter()
    assert vad.ensure_model() is True

    silence = np.zeros(16000 * 2, dtype=np.float32)
    res_sil = vad.analyze_samples(silence, 16000)
    assert res_sil.is_speech is False

    t = np.linspace(0.0, 2.0, 32000, endpoint=False, dtype=np.float32)
    tone = (0.2 * np.sin(2.0 * np.pi * 1800.0 * t)).astype(np.float32)
    res_tone = vad.analyze_samples(tone, 16000)
    assert res_tone.engine_used == "silero_onnx"
    assert res_tone.is_speech is False


def test_active_speech_lufs_ignores_long_pauses_and_low_floor(tmp_path: Path) -> None:
    ff = FFmpegManager()
    sr = 48000
    t_sp = np.linspace(0.0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    speech_seg = (
        0.18 * np.sin(2.0 * np.pi * 320.0 * t_sp) + 0.08 * np.sin(2.0 * np.pi * 1200.0 * t_sp)
    ).astype(np.float32)

    pure_wav = tmp_path / "pure_speech.wav"
    sf.write(str(pure_wav), speech_seg, sr, subtype="PCM_16")
    pure_act_lufs, pure_act_rms = ff.measure_active_speech_lufs(pure_wav)

    # 20 saniyelik uzun monolog: 3 konuşma bloğu + aralarda uzun sessizlik / hafif oda tabanı (-52 dBFS)
    rng = np.random.default_rng(42)
    pause_4s = (0.0025 * rng.standard_normal(sr * 4)).astype(np.float32)
    long_mono = np.concatenate([pause_4s, speech_seg, pause_4s, speech_seg, pause_4s])
    long_wav = tmp_path / "long_monologue_with_pauses.wav"
    sf.write(str(long_wav), long_mono, sr, subtype="PCM_16")

    long_act_lufs, long_act_rms = ff.measure_active_speech_lufs(long_wav)
    assert abs(long_act_lufs - pure_act_lufs) <= 0.35
    assert abs(long_act_rms - pure_act_rms) <= 0.35


def test_match_active_speech_loudness_and_true_peak_limiter(tmp_path: Path) -> None:
    ff = FFmpegManager()
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    # Kısık konuşma + keskin tepe (transient)
    quiet = (0.04 * np.sin(2.0 * np.pi * 400.0 * t) + 0.02 * np.sin(2.0 * np.pi * 1500.0 * t)).astype(
        np.float32
    )
    quiet[sr : sr + 400] += 0.35 * np.sin(2.0 * np.pi * 2000.0 * t[:400])
    sil = np.zeros(sr * 2, dtype=np.float32)
    full = np.concatenate([sil, quiet, sil])

    src = tmp_path / "quiet_dub.wav"
    dst = tmp_path / "matched_dub.wav"
    sf.write(str(src), full, sr, subtype="PCM_16")

    # Hedef: -18.0 LUFS aktif konuşma ve -1.0 dBTP tavan
    res = ff.match_active_speech_loudness(src, target_active_lufs=-18.0, output_wav=dst, true_peak_dbtp=-1.0)
    assert dst.is_file()
    assert abs(res["new_active_lufs"] - (-18.0)) <= 0.25
    assert res["true_peak_dbtp"] <= -0.95

    # Agresif kazanç testinde dahi True Peak -1.0 dBTP üstüne taşmamalı
    hot_dst = tmp_path / "hot_limited.wav"
    res_hot = ff.match_active_speech_loudness(
        src, target_active_lufs=-8.0, output_wav=hot_dst, true_peak_dbtp=-1.0
    )
    assert res_hot["true_peak_dbtp"] <= -0.95


def test_sync_and_package_dub_matches_active_lufs_mono_to_stereo(tmp_path: Path) -> None:
    ff = FFmpegManager()
    sr = 48000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    orig_sp = (0.14 * np.sin(2.0 * np.pi * 350.0 * t) + 0.06 * np.sin(2.0 * np.pi * 1400.0 * t)).astype(
        np.float32
    )
    orig_sil = np.zeros(sr * 3, dtype=np.float32)
    orig_mono = np.concatenate([orig_sil, orig_sp, orig_sil, orig_sp])
    orig_stereo = np.stack([orig_mono, orig_mono], axis=1)
    orig_wav = tmp_path / "orig_stereo.wav"
    sf.write(str(orig_wav), orig_stereo, sr, subtype="PCM_16")
    orig_meta = ff.probe_audio(orig_wav, measure_lufs=True)

    # Kısık mono dublaj (24 kHz)
    t_d = np.linspace(0.0, 3.0, 24000 * 3, endpoint=False, dtype=np.float32)
    dub_mono = (0.03 * np.sin(2.0 * np.pi * 350.0 * t_d) + 0.015 * np.sin(2.0 * np.pi * 1400.0 * t_d)).astype(
        np.float32
    )
    dub_wav = tmp_path / "dub_mono.wav"
    sf.write(str(dub_wav), dub_mono, 24000, subtype="PCM_16")

    out_wav = tmp_path / "synced_stereo.wav"
    out_path, _, _ = ff.sync_and_package_dub(
        dub_wav_path=dub_wav,
        original_meta=orig_meta,
        output_path=out_wav,
        max_tempo_speedup=1.00,
        trim_silence=False,
    )
    assert out_path.is_file()
    out_meta = ff.probe_audio(out_path, measure_lufs=True)
    out_metrics = ff.measure_loudness_metrics(out_path)
    assert out_meta.channels == 2
    assert out_meta.sample_rate == 48000
    assert abs(out_meta.lufs - orig_meta.lufs) <= 0.35
    assert out_metrics["true_peak_dbtp"] <= -0.95
