"""Taşınabilir FFmpeg Yöneticisi, Ses Analizi, Akıllı Sessizlik Kırpma ve Senkronizasyon İşlemcisi.

Özellikler:
- FFmpeg çözümleme sırası (tamamen taşınabilir):
  1. ./_araclar/ffmpeg/ffmpeg.exe (veya ./_araclar/ffmpeg/bin/ffmpeg.exe)
  2. .venv içindeki imageio-ffmpeg ikili dosyası
  3. Sistem PATH (sadece salt-okunur tespit)
- Süre (ms), LUFS (ebur128/loudnorm), örnekleme hızı, kanal sayısı, codec ve format ölçümü
- Akıllı sessizlik kırpma: hızlandırmadan önce baştaki, sondaki ve cümle aralarındaki
  gereksiz uzun sessizlikleri doğal nefesi bozmadan kırpar.
- Doğal tempo (varsayılan üst sınır 1.05x, 1.00x = hızlandırma tamamen kapalı).
- Orijinal dosyanın ses şiddetine (LUFS) eşitleme.
- Arka plan koruma modunda dublajlı vokali orijinal arka plan efekti/müzik katmanıyla
  kırpma (clipping) olmadan yeniden miksleme.
- Orijinal format (.wav, .ogg, .mp3, .flac), kanal sayısı ve örnekleme hızına paketleme.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from core.models import AudioMetadata
from core.portable_env import TEMP_DIR, TOOLS_DIR

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class FFmpegNotFoundError(RuntimeError):
    """FFmpeg ikili dosyası bulunamadığında fırlatılır."""


class FFmpegManager:
    """Taşınabilir FFmpeg analiz, dönüştürme ve sinematik dublaj senkronizasyon motoru."""

    def __init__(self, ffmpeg_path: Optional[str] = None, ffprobe_path: Optional[str] = None) -> None:
        self.ffmpeg_exe = ffmpeg_path or self._discover_ffmpeg()
        self.ffprobe_exe = ffprobe_path or self._discover_ffprobe(self.ffmpeg_exe)

    @staticmethod
    def _discover_ffmpeg() -> Optional[str]:
        candidates = [
            TOOLS_DIR / "ffmpeg" / "ffmpeg.exe",
            TOOLS_DIR / "ffmpeg" / "bin" / "ffmpeg.exe",
            TOOLS_DIR / "ffmpeg" / "ffmpeg",
        ]
        for cand in candidates:
            if cand.is_file():
                return str(cand)

        try:
            import imageio_ffmpeg  # type: ignore

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and Path(exe).is_file():
                return exe
        except Exception:
            pass

        return shutil.which("ffmpeg")

    @staticmethod
    def _discover_ffprobe(ffmpeg_exe: Optional[str]) -> Optional[str]:
        candidates = [
            TOOLS_DIR / "ffmpeg" / "ffprobe.exe",
            TOOLS_DIR / "ffmpeg" / "bin" / "ffprobe.exe",
            TOOLS_DIR / "ffmpeg" / "ffprobe",
        ]
        if ffmpeg_exe:
            p = Path(ffmpeg_exe).parent
            candidates.insert(0, p / "ffprobe.exe")
            candidates.insert(1, p / "ffprobe")

        for cand in candidates:
            if cand.is_file():
                return str(cand)

        return shutil.which("ffprobe")

    def is_available(self) -> bool:
        return bool(self.ffmpeg_exe and Path(self.ffmpeg_exe).is_file())

    def get_version_string(self) -> str:
        if not self.is_available():
            return "FFmpeg bulunamadı"
        try:
            proc = self._run([self.ffmpeg_exe, "-version"], timeout=10)
            first_line = (proc.stdout or proc.stderr or "").splitlines()[0]
            return first_line.strip()
        except Exception as exc:
            return f"FFmpeg kontrol hatası: {exc}"

    def _require_ffmpeg(self) -> str:
        if not self.is_available() or not self.ffmpeg_exe:
            raise FFmpegNotFoundError(
                "FFmpeg bulunamadı. Lütfen taşınabilir ffmpeg.exe dosyasını "
                "'./_araclar/ffmpeg/' klasörüne yerleştirin veya imageio-ffmpeg paketini kurun."
            )
        return self.ffmpeg_exe

    def _run(self, cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["TEMP"] = str(TEMP_DIR)
        env["TMP"] = str(TEMP_DIR)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            creationflags=_CREATE_NO_WINDOW,
        )

    def probe_audio(self, file_path: Path, measure_lufs: bool = True) -> AudioMetadata:
        """Ses dosyasının süre (ms), LUFS, sample rate, kanal sayısı ve format bilgisini ölçer."""
        ext = file_path.suffix.lower()
        duration_ms = 0
        sample_rate = 44100
        channels = 1
        codec = ""
        bit_rate: Optional[int] = None
        sample_fmt: Optional[str] = None

        if self.ffprobe_exe and Path(self.ffprobe_exe).is_file():
            proc = self._run(
                [
                    self.ffprobe_exe,
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=codec_name,sample_rate,channels,bit_rate,sample_fmt,duration:format=duration,bit_rate",
                    "-of",
                    "json",
                    str(file_path),
                ],
                timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                streams = data.get("streams") or []
                fmt = data.get("format") or {}
                if streams:
                    st = streams[0]
                    codec = str(st.get("codec_name") or "")
                    sample_rate = int(st.get("sample_rate") or 44100)
                    channels = int(st.get("channels") or 1)
                    sample_fmt = st.get("sample_fmt")
                    br = st.get("bit_rate") or fmt.get("bit_rate")
                    bit_rate = int(br) if br and str(br).isdigit() else None
                    dur = st.get("duration") or fmt.get("duration")
                    if dur is not None:
                        duration_ms = max(1, int(round(float(dur) * 1000.0)))

        if duration_ms <= 0:
            sf_ok = False
            try:
                info = sf.info(str(file_path))
                if info.frames > 0 and info.samplerate > 0:
                    duration_ms = max(1, int(round((info.frames / info.samplerate) * 1000.0)))
                    sample_rate = int(info.samplerate)
                    channels = int(info.channels)
                    codec = (info.subtype or info.format or "").lower()
                    sf_ok = True
            except Exception:
                sf_ok = False

            if not sf_ok or not codec:
                ff = self._require_ffmpeg()
                proc = self._run([ff, "-hide_banner", "-i", str(file_path)], timeout=30)
                stderr = proc.stderr or ""
                parsed = self._parse_ffmpeg_info(stderr)
                if duration_ms <= 0 and parsed["duration_ms"] > 0:
                    duration_ms = parsed["duration_ms"]
                if not sf_ok:
                    sample_rate = parsed["sample_rate"] or sample_rate
                    channels = parsed["channels"] or channels
                codec = parsed["codec"] or codec
                bit_rate = parsed["bit_rate"] or bit_rate

        lufs = self.measure_active_speech_lufs(file_path)[0] if measure_lufs else -23.0

        return AudioMetadata(
            duration_ms=max(1, duration_ms),
            lufs=lufs,
            sample_rate=sample_rate,
            channels=channels,
            codec=codec or ext.lstrip("."),
            format_ext=ext,
            bit_rate=bit_rate,
            sample_fmt=sample_fmt,
        )

    @staticmethod
    def _parse_ffmpeg_info(stderr: str) -> dict:
        result = {
            "duration_ms": 0,
            "sample_rate": 44100,
            "channels": 1,
            "codec": "",
            "bit_rate": None,
        }
        m_dur = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
        if m_dur:
            h, m, s = int(m_dur.group(1)), int(m_dur.group(2)), float(m_dur.group(3))
            result["duration_ms"] = int(round((h * 3600 + m * 60 + s) * 1000.0))

        m_audio = re.search(
            r"Audio:\s*([a-zA-Z0-9_]+)(?:\s*\([^)]*\))?,\s*(\d+)\s*Hz,\s*([^,]+)",
            stderr,
        )
        if m_audio:
            result["codec"] = m_audio.group(1).lower()
            result["sample_rate"] = int(m_audio.group(2))
            ch_str = m_audio.group(3).lower().strip()
            if "mono" in ch_str:
                result["channels"] = 1
            elif "stereo" in ch_str:
                result["channels"] = 2
            elif "5.1" in ch_str:
                result["channels"] = 6
            elif "7.1" in ch_str:
                result["channels"] = 8
            else:
                m_ch = re.search(r"(\d+)\s*channels?", ch_str)
                if m_ch:
                    result["channels"] = int(m_ch.group(1))

        m_br = re.search(r"bitrate:\s*(\d+)\s*kb/s", stderr)
        if m_br:
            result["bit_rate"] = int(m_br.group(1)) * 1000

        return result

    @staticmethod
    def _build_true_peak_limiter_filter(
        target_sr: int,
        true_peak_dbtp: float = -1.0,
    ) -> str:
        """4x oversampling ile ITU-R BS.1770 uyumlu True Peak sınırlayıcı zinciri üretir."""
        safe_sr = max(8000, int(target_sr))
        os_sr = max(96000, min(192000, safe_sr * 4))
        # Yeniden örnekleme (downsample) taşmalarını önlemek için -0.05 dB güvenlik payı bırak
        ceiling_db = min(-0.1, float(true_peak_dbtp)) - 0.05
        limit_lin = 10.0 ** (ceiling_db / 20.0)
        return (
            f"aresample={os_sr},"
            f"alimiter=limit={limit_lin:.6f}:level=false:attack=3:release=40,"
            f"aresample={safe_sr}"
        )

    def measure_loudness_metrics(self, file_path: Path) -> dict[str, float]:
        """FFmpeg loudnorm filtresiyle entegre LUFS, True Peak (dBTP) ve LRA ölçer."""
        ff = self._require_ffmpeg()
        proc = self._run(
            [
                ff,
                "-hide_banner",
                "-nostats",
                "-i",
                str(file_path),
                "-af",
                "loudnorm=I=-16:TP=-1.0:LRA=11:print_format=json",
                "-f",
                "null",
                "-",
            ],
            timeout=60,
        )
        stderr = proc.stderr or ""
        match = re.search(r"\{\s*\"input_i\"\s*:.*?\}", stderr, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                val_i = float(payload.get("input_i", "-23.0"))
                val_tp = float(payload.get("input_tp", "-70.0"))
                val_lra = float(payload.get("input_lra", "0.0"))
                if math.isfinite(val_i) and -70.0 <= val_i <= 0.0:
                    return {
                        "lufs": round(val_i, 2),
                        "true_peak_dbtp": round(val_tp, 2) if math.isfinite(val_tp) else -70.0,
                        "lra": round(val_lra, 2) if math.isfinite(val_lra) else 0.0,
                    }
            except (ValueError, json.JSONDecodeError):
                pass

        try:
            wav_tmp = TEMP_DIR / f"_lufs_probe_{uuid.uuid4().hex[:8]}.wav"
            try:
                self.convert_to_wav(file_path, wav_tmp, sample_rate=16000, channels=1)
                data, _ = sf.read(str(wav_tmp), dtype="float32")
                rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
                peak = float(np.max(np.abs(data))) if data.size else 0.0
                tp_db = round(20.0 * math.log10(max(peak, 1e-12)), 2) if peak > 1e-12 else -70.0
                if rms > 1e-6:
                    approx = 20.0 * math.log10(rms) - 0.691
                    return {
                        "lufs": round(max(-50.0, min(-5.0, approx)), 2),
                        "true_peak_dbtp": tp_db,
                        "lra": 0.0,
                    }
            finally:
                wav_tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return {"lufs": -23.0, "true_peak_dbtp": -70.0, "lra": 0.0}

    def measure_lufs(self, file_path: Path) -> float:
        """FFmpeg loudnorm filtresiyle tüm dosyanın entegre ses yüksekliğini (LUFS) ölçer."""
        return self.measure_loudness_metrics(file_path)["lufs"]

    def _read_audio_float_2d(self, file_path: Path) -> tuple[np.ndarray, int]:
        """Ses dosyasını kanal sayısını ve örnekleme hızını koruyarak float32 2D dizi olarak okur."""
        try:
            data, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
            if data.size > 0 and sr > 0:
                return data, int(sr)
        except Exception:
            pass

        ff = self._require_ffmpeg()
        tmp_wav = TEMP_DIR / f"_read2d_{uuid.uuid4().hex[:8]}.wav"
        try:
            proc = self._run(
                [
                    ff,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(file_path),
                    "-vn",
                    "-c:a",
                    "pcm_f32le",
                    str(tmp_wav),
                ],
                timeout=60,
            )
            if proc.returncode != 0 or not tmp_wav.is_file():
                raise RuntimeError(f"Ses okuma hatası ({file_path.name}): {proc.stderr.strip()}")
            data, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=True)
            return data, int(sr)
        finally:
            tmp_wav.unlink(missing_ok=True)

    def measure_active_speech_lufs(self, file_path: Path) -> tuple[float, float]:
        """Uzun esleri/sessizlikleri hariç tutarak yalnızca aktif konuşma bölgesinin LUFS ve RMS değerini ölçer.

        Uzun monologlarda sessizliklerin entegre LUFS ortalamasını aşağı çekmesini önlemek için
        30 ms pencereli uyarlamalı enerji kapısı (speech gate) ile aktif konuşma örneklerini
        ayırır, kanal yapısını koruyarak ITU-R BS.1770 LUFS ölçer.

        Returns:
            (aktif_konusma_lufs, aktif_konusma_rms_dbfs)
        """
        act_tmp: Optional[Path] = None
        try:
            data, sr = self._read_audio_float_2d(file_path)
            if data.size == 0 or sr <= 0:
                return self.measure_lufs(file_path), -70.0

            peak = float(np.max(np.abs(data)))
            if peak <= 1e-6:
                return -70.0, -70.0

            mono = np.mean(data, axis=1, dtype=np.float32)
            n_samples = len(mono)
            win = max(1, int(sr * 0.030))
            hop = max(1, int(sr * 0.010))

            if n_samples <= win:
                active_data = data
            else:
                n_frames = 1 + (n_samples - win) // hop
                # Vektörize 30 ms çerçeve RMS hesabı (uzun dosyalarda < 5 ms sürer)
                sq = np.square(mono, dtype=np.float64)
                csum = np.concatenate(([0.0], np.cumsum(sq)))
                starts = np.arange(n_frames, dtype=np.int64) * hop
                ends = starts + win
                frame_ms = np.maximum((csum[ends] - csum[starts]) / float(win), 1e-12)
                frame_db = 10.0 * np.log10(frame_ms)

                p95_db = float(np.percentile(frame_db, 95))
                if p95_db <= -65.0:
                    return self.measure_lufs(file_path), round(p95_db, 2)

                # Normal konuşmada p95'in 18 dB altı (en az -42 dBFS); çok kısık kayıtlarda uyarlamalı
                if p95_db > -36.0:
                    thr_db = max(-42.0, p95_db - 18.0)
                else:
                    thr_db = max(-60.0, p95_db - 15.0)

                active_frames = np.flatnonzero(frame_db >= thr_db)
                if active_frames.size == 0:
                    active_data = data
                else:
                    mask = np.zeros(n_samples, dtype=bool)
                    for fi in active_frames:
                        s = int(fi) * hop
                        mask[s : s + win] = True
                    active_data = data[mask]
                    if active_data.shape[0] < max(1, int(sr * 0.05)):
                        active_data = data

            active_rms = float(np.sqrt(np.mean(np.square(active_data, dtype=np.float64))))
            active_rms_db = round(20.0 * math.log10(max(active_rms, 1e-12)), 2)

            # ITU-R BS.1770 400 ms kapı bloklarının kısa repliklerde -inf dönmemesi için en az 1.2 sn'ye katla
            min_len = int(sr * 1.2)
            meas_data = active_data
            if meas_data.shape[0] < min_len:
                reps = int(math.ceil(min_len / float(max(1, meas_data.shape[0]))))
                meas_data = np.tile(meas_data, (reps, 1))

            act_tmp = TEMP_DIR / f"_act_lufs_{uuid.uuid4().hex[:8]}.wav"
            sf.write(str(act_tmp), meas_data.astype(np.float32), sr, subtype="FLOAT")
            active_lufs = self.measure_lufs(act_tmp)
            return active_lufs, active_rms_db
        except Exception:
            fallback_lufs = self.measure_lufs(file_path)
            return fallback_lufs, fallback_lufs
        finally:
            if act_tmp is not None:
                act_tmp.unlink(missing_ok=True)

    def match_active_speech_loudness(
        self,
        input_wav: Path,
        target_active_lufs: float,
        output_wav: Path,
        true_peak_dbtp: float = -1.0,
        max_gain_db: float = 24.0,
    ) -> dict[str, float]:
        """Bir ses dosyasının aktif konuşma bölgesini hedef LUFS'a doğrusal kazanç + True Peak sınırlayıcıyla eşitler.

        Mevcut konuşma içeriğini, süresini, örnekleme hızını ve kanal sayısını korur;
        `output_wav == input_wav` durumunda atomik geçici dosya ile güvenle günceller.
        """
        ff = self._require_ffmpeg()
        output_wav.parent.mkdir(parents=True, exist_ok=True)

        meta = self.probe_audio(input_wav, measure_lufs=False)
        old_lufs, old_rms = self.measure_active_speech_lufs(input_wav)
        clamped_target = max(-40.0, min(-6.0, float(target_active_lufs)))
        gain_db = max(-max_gain_db, min(max_gain_db, clamped_target - old_lufs))

        tmp_out = TEMP_DIR / f"_lvl_{uuid.uuid4().hex[:8]}{output_wav.suffix.lower() or '.wav'}"
        ext = output_wav.suffix.lower() or meta.format_ext.lower() or ".wav"

        def _render(g_db: float) -> None:
            tp_filter = self._build_true_peak_limiter_filter(meta.sample_rate, true_peak_dbtp)
            af = f"volume={g_db:.3f}dB,{tp_filter}"
            cmd = [
                ff,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_wav),
                "-vn",
                "-af",
                af,
                "-ar",
                str(meta.sample_rate),
                "-ac",
                str(meta.channels),
            ]
            cmd.extend(self._encoder_args_for_format(ext, meta))
            cmd.append(str(tmp_out))
            proc = self._run(cmd, timeout=120)
            if proc.returncode != 0 or not tmp_out.is_file():
                raise RuntimeError(
                    f"Aktif konuşma seviye eşitleme hatası ({input_wav.name}): {proc.stderr.strip()}"
                )

        try:
            applied_gain = gain_db
            _render(applied_gain)
            new_lufs, new_rms = self.measure_active_speech_lufs(tmp_out)

            # Sınırlayıcı yoğun tepeleri kıstıysa tek seferlik temiz mikro-düzeltme uygula (çift sınırlama yapmadan)
            residual = clamped_target - new_lufs
            if abs(residual) > 0.15:
                corrected_gain = max(-max_gain_db, min(max_gain_db, applied_gain + residual))
                if abs(corrected_gain - applied_gain) >= 0.05:
                    applied_gain = corrected_gain
                    _render(applied_gain)
                    new_lufs, new_rms = self.measure_active_speech_lufs(tmp_out)

            final_metrics = self.measure_loudness_metrics(tmp_out)
            shutil.copy2(tmp_out, output_wav)
            return {
                "old_active_lufs": round(old_lufs, 2),
                "old_active_rms": round(old_rms, 2),
                "new_active_lufs": round(new_lufs, 2),
                "new_active_rms": round(new_rms, 2),
                "target_active_lufs": round(clamped_target, 2),
                "applied_gain_db": round(applied_gain, 2),
                "true_peak_dbtp": final_metrics["true_peak_dbtp"],
            }
        finally:
            tmp_out.unlink(missing_ok=True)

    def convert_to_wav(
        self,
        src_path: Path,
        dst_wav: Path,
        sample_rate: int = 16000,
        channels: int = 1,
        max_duration_sec: Optional[float] = None,
    ) -> Path:
        """Herhangi bir ses dosyasını temiz PCM 16-bit WAV dosyasına dönüştürür."""
        ff = self._require_ffmpeg()
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src_path),
            "-vn",
        ]
        if max_duration_sec is not None and max_duration_sec > 0:
            cmd.extend(["-t", f"{max_duration_sec:.3f}"])
        cmd.extend(
            [
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                str(dst_wav),
            ]
        )
        proc = self._run(cmd, timeout=90)
        if proc.returncode != 0 or not dst_wav.is_file():
            raise RuntimeError(f"WAV dönüştürme başarısız ({src_path.name}): {proc.stderr.strip()}")
        return dst_wav

    def prepare_reference_wav(
        self,
        src_path: Path,
        dst_wav: Path,
        sample_rate: int = 48000,
        max_duration_sec: float = 20.0,
        trim_edge_silence: bool = True,
    ) -> Path:
        """VoxCPM2 (48 kHz), Trendyol-TTS, Chatterbox, XTTS v2 ve ElevenLabs IVC için temiz referans WAV hazırlar.

        - Orijinal dosyaya (`src_path`) kesinlikle dokunmaz; `./_temp/` altında temiz kopya üretir.
        - Tek kanal (Mono) ve hedef örnekleme hızına (varsayılan 48000 Hz) dönüştürür.
        - `trim_edge_silence=True` ise baştaki, sondaki ve aralardaki aşırı uzun sessizlikleri kırpar.
        - Çok kısa (<0.65 sn) repliklerde klonlama kararlılığı için hafif sessizlik dolgusu ekler.
        """
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        raw_conv_tmp = TEMP_DIR / f"_ref_raw_{uuid.uuid4().hex[:8]}.wav"
        trimmed_tmp = TEMP_DIR / f"_ref_trim_{uuid.uuid4().hex[:8]}.wav"
        try:
            # Başta uzun sessizlik olma ihtimaline karşı önce biraz daha geniş dönüştür, sonra kırp ve sınırla
            pre_cap = max_duration_sec * 1.5 if (max_duration_sec and max_duration_sec > 0) else None
            self.convert_to_wav(
                src_path,
                raw_conv_tmp,
                sample_rate=sample_rate,
                channels=1,
                max_duration_sec=pre_cap,
            )
            working_wav = raw_conv_tmp
            if trim_edge_silence:
                trimmed_out, _ = self.trim_silence_safely(
                    raw_conv_tmp,
                    trimmed_tmp,
                    threshold_db=-40.0,
                    max_internal_pause_sec=0.25,
                    edge_margin_sec=0.04,
                )
                if trimmed_out.is_file():
                    working_wav = trimmed_out

            audio, sr = sf.read(str(working_wav), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1).astype(np.float32)

            if max_duration_sec and max_duration_sec > 0:
                max_samples = int(sr * max_duration_sec)
                if len(audio) > max_samples:
                    audio = audio[:max_samples]

            min_samples = int(sr * 0.65)
            if 0 < len(audio) < min_samples:
                pad_len = min_samples - len(audio)
                audio = np.concatenate([audio, np.zeros(pad_len, dtype=np.float32)])

            sf.write(str(dst_wav), audio, sr, subtype="PCM_16")
        except Exception:
            if not dst_wav.is_file():
                self.convert_to_wav(
                    src_path,
                    dst_wav,
                    sample_rate=sample_rate,
                    channels=1,
                    max_duration_sec=max_duration_sec,
                )
        finally:
            raw_conv_tmp.unlink(missing_ok=True)
            trimmed_tmp.unlink(missing_ok=True)
        return dst_wav

    def prepare_clean_fixed_reference(
        self,
        src_path: Path,
        dst_wav: Optional[Path] = None,
        sample_rate: int = 48000,
        max_duration_sec: float = 20.0,
    ) -> Path:
        """Kullanıcının seçtiği Sabit Referans (.wav) dosyasından ./_temp/ altında 48 kHz mono,
        baş/son uzun sessizlikleri kırpılmış temiz bir referans kopyası üretir (orijinal dosyaya dokunmaz).
        """
        if not src_path.is_file():
            raise FileNotFoundError(f"Sabit referans ses dosyası bulunamadı: {src_path}")
        if dst_wav is None:
            ref_dir = TEMP_DIR / "sabit_referans"
            ref_dir.mkdir(parents=True, exist_ok=True)
            safe_stem = re.sub(r"[^A-Za-z0-9_\-]", "_", src_path.stem)[:40] or "ref"
            dst_wav = ref_dir / f"{safe_stem}_48k_mono_clean.wav"
        return self.prepare_reference_wav(
            src_path=src_path,
            dst_wav=dst_wav,
            sample_rate=sample_rate,
            max_duration_sec=max_duration_sec,
            trim_edge_silence=True,
        )

    # Geriye dönük uyumluluk takma adı
    def prepare_xtts_reference_wav(self, src_path: Path, dst_wav: Path) -> Path:
        return self.prepare_reference_wav(src_path, dst_wav, sample_rate=22050, max_duration_sec=18.0)

    def trim_silence_safely(
        self,
        src_wav: Path,
        dst_wav: Path,
        threshold_db: float = -42.0,
        max_internal_pause_sec: float = 0.24,
        edge_margin_sec: float = 0.035,
    ) -> tuple[Path, int]:
        """Baştaki, sondaki ve cümle aralarındaki gereksiz sessizlikleri güvenli şekilde kırpar.

        Doğal konuşma ritmini ve nefes aralarını bozmadan yalnızca aşırı boşlukları
        kısaltır. Çok sessiz/fısıltı kliplerde sesin tamamı yanlışlıkla kırpılırsa
        otomatik olarak orijinal WAV'ı korur.

        Returns:
            (cikti_wav_yolu, kirpilan_sure_ms)
        """
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        try:
            audio, sr = sf.read(str(src_wav), dtype="float32")
        except Exception:
            shutil.copy2(src_wav, dst_wav)
            return dst_wav, 0

        if audio.size == 0 or sr <= 0:
            shutil.copy2(src_wav, dst_wav)
            return dst_wav, 0

        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
        orig_len = len(mono)
        orig_ms = int(round((orig_len / float(sr)) * 1000.0))

        # 20ms çerçevelerle RMS enerji zarfı çıkar
        frame_len = max(1, int(sr * 0.02))
        n_frames = math.ceil(orig_len / frame_len)
        padded = np.pad(mono, (0, n_frames * frame_len - orig_len), mode="constant")
        frames = padded.reshape(n_frames, frame_len)
        rms = np.sqrt(np.mean(frames * frames, axis=1))

        # Dinamik + mutlak eşik: fısıltılarda bile sesi yutmayan güvenli alt sınır
        peak_rms = float(np.max(rms)) if rms.size else 0.0
        if peak_rms < 1e-5:
            shutil.copy2(src_wav, dst_wav)
            return dst_wav, 0

        abs_thresh = 10.0 ** (threshold_db / 20.0)
        rel_thresh = peak_rms * 0.035
        thresh = min(abs_thresh, rel_thresh)
        active = rms >= thresh

        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            shutil.copy2(src_wav, dst_wav)
            return dst_wav, 0

        margin_frames = max(1, int(round(edge_margin_sec / 0.02)))
        first_f = max(0, int(active_indices[0]) - margin_frames)
        last_f = min(n_frames - 1, int(active_indices[-1]) + margin_frames)

        # Baştan ve sondan kırpma + içteki >0.36s uzun boşlukları max_internal_pause_sec'e indirgeme
        keep_pause_frames = max(2, int(round(max_internal_pause_sec / 0.02)))
        trigger_pause_frames = keep_pause_frames + 6  # >~0.36s sessizliklerde devreye gir

        segments: list[np.ndarray] = []
        f_idx = first_f
        while f_idx <= last_f:
            if active[f_idx]:
                seg_start = f_idx
                while f_idx <= last_f and active[f_idx]:
                    f_idx += 1
                s0 = seg_start * frame_len
                s1 = min(orig_len, f_idx * frame_len)
                segments.append(audio[s0:s1])
            else:
                sil_start = f_idx
                while f_idx <= last_f and not active[f_idx]:
                    f_idx += 1
                sil_len_frames = f_idx - sil_start
                # Aşırı uzun iç boşlukları doğal nefes aralığına (keep_pause_frames) düşür
                if sil_len_frames > trigger_pause_frames:
                    half = keep_pause_frames // 2
                    left_s0 = sil_start * frame_len
                    left_s1 = min(orig_len, (sil_start + half) * frame_len)
                    right_s0 = max(0, (f_idx - (keep_pause_frames - half)) * frame_len)
                    right_s1 = min(orig_len, f_idx * frame_len)
                    segments.append(audio[left_s0:left_s1])
                    segments.append(audio[right_s0:right_s1])
                else:
                    s0 = sil_start * frame_len
                    s1 = min(orig_len, f_idx * frame_len)
                    segments.append(audio[s0:s1])

        if not segments:
            shutil.copy2(src_wav, dst_wav)
            return dst_wav, 0

        trimmed = np.concatenate(segments, axis=0)
        new_ms = int(round((len(trimmed) / float(sr)) * 1000.0))

        # Güvenlik: sesin çok büyük kısmı (veya <180ms altına) kazara gitmişse orijinali koru
        if new_ms < 180 or (orig_ms > 500 and new_ms < orig_ms * 0.25):
            shutil.copy2(src_wav, dst_wav)
            return dst_wav, 0

        # Tıklama önleyici 8ms mikro fade-in / fade-out
        fade_n = min(int(sr * 0.008), len(trimmed) // 4)
        if fade_n > 1:
            ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
            if trimmed.ndim > 1:
                ramp = ramp[:, None]
            trimmed[:fade_n] *= ramp
            trimmed[-fade_n:] *= ramp[::-1]

        sf.write(str(dst_wav), trimmed, sr, subtype="PCM_16")
        saved_ms = max(0, orig_ms - new_ms)
        return dst_wav, saved_ms

    @staticmethod
    def build_atempo_chain(speed_ratio: float) -> str:
        """Ses perdesini (pitch) bozmadan FFmpeg atempo filtre zinciri üretir."""
        if speed_ratio <= 1.002:
            return ""
        factors: list[float] = []
        remaining = float(speed_ratio)
        while remaining > 1.5:
            factors.append(1.5)
            remaining /= 1.5
        if remaining > 1.001:
            factors.append(remaining)
        return ",".join(f"atempo={f:.5f}" for f in factors)

    def mix_vocals_with_background(
        self,
        vocals_wav: Path,
        background_wav: Path,
        original_meta: AudioMetadata,
        output_path: Path,
        bg_volume_db: float = 0.0,
    ) -> Path:
        """Dublajlı Türkçe vokal katmanını orijinal arka plan efekti/müziğiyle yeniden miksler."""
        ff = self._require_ffmpeg()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bg_db = max(-24.0, min(12.0, float(bg_volume_db)))

        tp_limiter = self._build_true_peak_limiter_filter(
            original_meta.sample_rate, true_peak_dbtp=-1.0
        )
        filter_complex = (
            f"[0:a]aformat=sample_rates={original_meta.sample_rate}:"
            f"channel_layouts={'stereo' if original_meta.channels >= 2 else 'mono'}[v];"
            f"[1:a]aformat=sample_rates={original_meta.sample_rate}:"
            f"channel_layouts={'stereo' if original_meta.channels >= 2 else 'mono'},"
            f"volume={bg_db:.2f}dB[bg];"
            f"[v][bg]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0,"
            f"{tp_limiter}[out]"
        )

        ext = output_path.suffix.lower() or original_meta.format_ext.lower()
        cmd = [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(vocals_wav),
            "-i",
            str(background_wav),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ar",
            str(original_meta.sample_rate),
            "-ac",
            str(original_meta.channels),
        ]
        cmd.extend(self._encoder_args_for_format(ext, original_meta))
        cmd.append(str(output_path))

        proc = self._run(cmd, timeout=120)
        if proc.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                f"Arka plan miksleme hatası ({output_path.name}): {proc.stderr.strip()}"
            )
        return output_path

    def sync_and_package_dub(
        self,
        dub_wav_path: Path,
        original_meta: AudioMetadata,
        output_path: Path,
        max_tempo_speedup: float = 1.05,
        trim_silence: bool = True,
        pad_shorter_to_original: bool = False,
        background_wav_path: Optional[Path] = None,
        bg_volume_db: float = 0.0,
        target_vocal_lufs: Optional[float] = None,
    ) -> tuple[Path, int, float]:
        """Üretilen Türkçe dublaj sesini sessizlik kırpma, doğal tempo, LUFS ve arka plan miksiyle paketler.

        İşlemler:
        1. `trim_silence=True` ise baştaki, sondaki ve cümle aralarındaki gereksiz
           sessizlikleri güvenli şekilde kırpar.
        2. Kırpılmış süre orijinalinden uzunsa VE `max_tempo_speedup > 1.00` ise
           sesin perdesini (pitch) bozmadan en fazla `max_tempo_speedup` (varsayılan 1.05x)
           hızlandırır. `1.00` seçiliyse hiç hızlandırma yapılmaz (tamamen doğal tempo).
        3. Ses yüksekliğini (LUFS) orijinal dosyayla (veya vokal katmanıyla) eşitler.
        4. `background_wav_path` verilmişse dublajlı vokali orijinal arka plan efekti/müzik
           katmanıyla kırpma olmadan yeniden miksler.
        5. Dosyayı orijinal format (.wav/.ogg/.mp3/.flac), kanal sayısı ve örnekleme
           hızına geri dönüştürür.

        Returns:
            (cikti_yolu, son_sure_ms, uygulanan_tempo_orani)
        """
        ff = self._require_ffmpeg()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trimmed_tmp: Optional[Path] = None
        processed_vocal_tmp: Optional[Path] = None
        working_in = dub_wav_path

        try:
            if trim_silence:
                trimmed_tmp = TEMP_DIR / f"_trim_{uuid.uuid4().hex[:8]}.wav"
                working_in, _ = self.trim_silence_safely(dub_wav_path, trimmed_tmp)

            dub_meta = self.probe_audio(working_in, measure_lufs=True)
            orig_ms = max(100, original_meta.duration_ms)
            dub_ms = max(50, dub_meta.duration_ms)

            tempo_factor = 1.0
            filters: list[str] = []

            # 1.00x = hızlandırma tamamen kapalı (doğal tempo)
            safe_cap = max(1.0, min(1.85, float(max_tempo_speedup)))
            if safe_cap > 1.002 and dub_ms > orig_ms * 1.02:
                raw_ratio = dub_ms / float(orig_ms)
                tempo_factor = min(raw_ratio, safe_cap)
                atempo_expr = self.build_atempo_chain(tempo_factor)
                if atempo_expr:
                    filters.append(atempo_expr)

            # Aktif konuşma LUFS eşitlemesi + 4x oversampled -1.0 dBTP True Peak sınırlayıcı
            ref_lufs = target_vocal_lufs if target_vocal_lufs is not None else original_meta.lufs
            target_lufs = max(-40.0, min(-6.0, float(ref_lufs)))
            current_lufs = float(dub_meta.lufs)
            apply_loudness = (
                math.isfinite(target_lufs)
                and math.isfinite(current_lufs)
                and current_lufs > -60.0
            )
            gain_db = (
                max(-24.0, min(24.0, target_lufs - current_lufs)) if apply_loudness else 0.0
            )
            tp_limiter = self._build_true_peak_limiter_filter(
                original_meta.sample_rate, true_peak_dbtp=-1.0
            )

            has_bg = bool(background_wav_path and background_wav_path.is_file())
            if has_bg:
                processed_vocal_tmp = TEMP_DIR / f"_voc_{uuid.uuid4().hex[:8]}.wav"
                stage_target = processed_vocal_tmp
            else:
                stage_target = output_path

            def _run_stage(stage_gain_db: float) -> None:
                stage_filters = list(filters)
                if original_meta.channels == 2:
                    stage_filters.append("aformat=channel_layouts=stereo")
                elif original_meta.channels == 1:
                    stage_filters.append("aformat=channel_layouts=mono")

                if apply_loudness:
                    stage_filters.append(f"volume={stage_gain_db:.3f}dB")
                stage_filters.append(tp_limiter)

                if pad_shorter_to_original and dub_ms < orig_ms:
                    stage_filters.append(f"apad=whole_dur={orig_ms / 1000.0:.4f}")

                cmd = [
                    ff,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(working_in),
                    "-vn",
                ]
                if stage_filters:
                    cmd.extend(["-af", ",".join(stage_filters)])

                cmd.extend(
                    [
                        "-ar",
                        str(original_meta.sample_rate),
                        "-ac",
                        str(original_meta.channels),
                    ]
                )
                if has_bg:
                    cmd.extend(["-c:a", "pcm_s16le"])
                else:
                    ext = output_path.suffix.lower() or original_meta.format_ext.lower()
                    cmd.extend(self._encoder_args_for_format(ext, original_meta))
                cmd.append(str(stage_target))

                proc = self._run(cmd, timeout=120)
                if proc.returncode != 0 or not stage_target.is_file():
                    raise RuntimeError(
                        f"FFmpeg senkronizasyon/paketleme hatası ({output_path.name}): {proc.stderr.strip()}"
                    )

            _run_stage(gain_db)

            # Sınırlayıcı veya atempo sonrası aktif konuşma LUFS'unda >0.25 dB sapma kaldıysa
            # ham `working_in` üzerinden tek seferlik temiz mikro-düzeltme yap.
            if apply_loudness:
                measured_out_lufs, _ = self.measure_active_speech_lufs(stage_target)
                residual_db = target_lufs - measured_out_lufs
                if math.isfinite(residual_db) and abs(residual_db) > 0.25:
                    corrected_gain = max(-24.0, min(24.0, gain_db + residual_db))
                    if abs(corrected_gain - gain_db) >= 0.1:
                        _run_stage(corrected_gain)

            if has_bg and processed_vocal_tmp is not None and background_wav_path is not None:
                self.mix_vocals_with_background(
                    vocals_wav=processed_vocal_tmp,
                    background_wav=background_wav_path,
                    original_meta=original_meta,
                    output_path=output_path,
                    bg_volume_db=bg_volume_db,
                )

            final_meta = self.probe_audio(output_path, measure_lufs=False)
            return output_path, final_meta.duration_ms, round(tempo_factor, 3)
        finally:
            if trimmed_tmp is not None:
                trimmed_tmp.unlink(missing_ok=True)
            if processed_vocal_tmp is not None:
                processed_vocal_tmp.unlink(missing_ok=True)

    @staticmethod
    def _encoder_args_for_format(ext: str, meta: AudioMetadata) -> list[str]:
        if ext == ".wav":
            if meta.codec == "pcm_s24le":
                return ["-c:a", "pcm_s24le"]
            if meta.codec == "pcm_f32le":
                return ["-c:a", "pcm_f32le"]
            return ["-c:a", "pcm_s16le"]
        if ext == ".ogg":
            if "opus" in (meta.codec or "").lower():
                return ["-c:a", "libopus", "-b:a", "128k"]
            return ["-c:a", "libvorbis", "-q:a", "6"]
        if ext == ".mp3":
            br_k = 192
            if meta.bit_rate and 64000 <= meta.bit_rate <= 320000:
                br_k = max(96, min(320, int(round(meta.bit_rate / 1000.0))))
            return ["-c:a", "libmp3lame", "-b:a", f"{br_k}k"]
        if ext == ".flac":
            return ["-c:a", "flac", "-compression_level", "5"]
        return ["-c:a", "pcm_s16le"]
