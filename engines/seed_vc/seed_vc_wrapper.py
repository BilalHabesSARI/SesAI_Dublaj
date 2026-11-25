"""Taşınabilir Seed-VC (Sıfır Eğitimli Speech-to-Speech Ses Dönüştürme) Motor Sarmalayıcısı.

Özellikler:
- Kullanıcının mikrofondan veya dış ses dosyasından (.wav/.mp3/.ogg/.flac) gelen gerçek
  insan oyunculuğunu (fısıltı, vurgu, alay, öfke, nefes ritmi) koruyarak doğrudan
  hedef karakterin ses rengine (timbre) ve ses kalınlığına (perde/F0) dönüştürür.
- RTX 4070 (12 GB VRAM) dostu tembel (lazy) yükleme: yalnızca ihtiyaç duyulan DiT +
  BigVGAN + RMVPE + CAMPPlus bileşenlerini yükler ve işlem sonunda VRAM'i tamamen
  boşaltabilir.
- Tüm model ağırlıkları yalnızca `./_modeller/huggingface/hub` içine indirilir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from transformers import AutoFeatureExtractor, WhisperModel

from core.portable_env import bootstrap_portable_environment
from engines.seed_vc.hf_utils import get_seed_vc_cache_dir, load_custom_model_from_hf
from engines.seed_vc.modules.audio import mel_spectrogram
from engines.seed_vc.modules.bigvgan import bigvgan
from engines.seed_vc.modules.campplus.DTDNN import CAMPPlus
from engines.seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch
from engines.seed_vc.modules.rmvpe import RMVPE


def _load_audio_mono(path: str | Path, target_sr: int) -> np.ndarray:
    """Ses dosyasını hedef örnekleme hızında 1D float32 NumPy dizisi olarak yükler."""
    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32)


class SeedVCWrapper:
    """RTX 4070 (12 GB VRAM) için optimize edilmiş taşınabilir Seed-VC ses dönüştürücü."""

    def __init__(self, device: Optional[str | torch.device] = None) -> None:
        bootstrap_portable_environment()
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.cache_dir = get_seed_vc_cache_dir()
        self.overlap_frame_len = 16

        # Ortak bileşenler (lazy)
        self.whisper_model = None
        self.whisper_feature_extractor = None
        self.campplus_model = None

        # F0-koşullu 44.1 kHz DiT modeli (otomatik perde eşleştirme + yarım ton ayarı destekli)
        self.model_f0 = None
        self.to_mel_f0 = None
        self.bigvgan_44k_model = None
        self.rmvpe = None
        self.sr_f0 = 44100
        self.hop_length_f0 = 512

        # Standart 22.05 kHz DiT modeli
        self.model = None
        self.to_mel = None
        self.bigvgan_model = None
        self.sr = 22050
        self.hop_length = 256

    def _ensure_common_encoders(self, whisper_name: str = "openai/whisper-small") -> None:
        if self.whisper_model is None or self.whisper_feature_extractor is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.whisper_model = WhisperModel.from_pretrained(
                whisper_name,
                torch_dtype=dtype,
                cache_dir=self.cache_dir,
            ).to(self.device)
            if hasattr(self.whisper_model, "decoder"):
                del self.whisper_model.decoder
            self.whisper_model.eval()
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
                whisper_name,
                cache_dir=self.cache_dir,
            )

        if self.campplus_model is None:
            campplus_ckpt_path = load_custom_model_from_hf(
                "funasr/campplus", "campplus_cn_common.bin", config_filename=None
            )
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus_model.load_state_dict(
                torch.load(campplus_ckpt_path, map_location="cpu")
            )
            self.campplus_model.eval()
            self.campplus_model.to(self.device)

    def _ensure_f0_model(self) -> None:
        """44.1 kHz F0-koşullu Seed-VC DiT + RMVPE + BigVGAN 44kHz modellerini yükler."""
        if self.model_f0 is not None:
            return
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        )
        with open(dit_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model_params = recursive_munch(config["model_params"])
        whisper_name = (
            model_params.speech_tokenizer.whisper_name
            if hasattr(model_params.speech_tokenizer, "whisper_name")
            else "openai/whisper-small"
        )
        self._ensure_common_encoders(whisper_name=whisper_name)

        self.model_f0 = build_model(model_params, stage="DiT")
        self.hop_length_f0 = int(config["preprocess_params"]["spect_params"]["hop_length"])
        self.sr_f0 = int(config["preprocess_params"]["sr"])

        self.model_f0, _, _, _ = load_checkpoint(
            self.model_f0,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model_f0:
            self.model_f0[key].eval()
            self.model_f0[key].to(self.device)
        self.model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        mel_fn_args_f0 = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr_f0,
            "fmin": 0,
            "fmax": None,
            "center": False,
        }
        self.to_mel_f0 = lambda x: mel_spectrogram(x, **mel_fn_args_f0)

        if self.bigvgan_44k_model is None:
            self.bigvgan_44k_model = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_44khz_128band_512x",
                use_cuda_kernel=False,
                cache_dir=self.cache_dir,
            )
            self.bigvgan_44k_model.remove_weight_norm()
            self.bigvgan_44k_model = self.bigvgan_44k_model.eval().to(self.device)

        if self.rmvpe is None:
            rmvpe_path = load_custom_model_from_hf(
                "lj1995/VoiceConversionWebUI", "rmvpe.pt", None
            )
            self.rmvpe = RMVPE(rmvpe_path, is_half=False, device=self.device)

    def _ensure_base_model(self) -> None:
        """22.05 kHz F0-koşulsuz Seed-VC DiT + BigVGAN 22kHz modellerini yükler."""
        if self.model is not None:
            return
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        )
        with open(dit_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        model_params = recursive_munch(config["model_params"])
        whisper_name = (
            model_params.speech_tokenizer.whisper_name
            if hasattr(model_params.speech_tokenizer, "whisper_name")
            else "openai/whisper-small"
        )
        self._ensure_common_encoders(whisper_name=whisper_name)

        self.model = build_model(model_params, stage="DiT")
        self.hop_length = int(config["preprocess_params"]["spect_params"]["hop_length"])
        self.sr = int(config["preprocess_params"]["sr"])

        self.model, _, _, _ = load_checkpoint(
            self.model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": 0,
            "fmax": None,
            "center": False,
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        if self.bigvgan_model is None:
            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_22khz_80band_256x",
                use_cuda_kernel=False,
                cache_dir=self.cache_dir,
            )
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval().to(self.device)

    @staticmethod
    def adjust_f0_semitones(f0_sequence: torch.Tensor, n_semitones: float) -> torch.Tensor:
        """F0 perde dizisini belirtilen yarım ton (semitone) kadar kaydırır."""
        if abs(float(n_semitones)) < 1e-4:
            return f0_sequence
        factor = float(2.0 ** (float(n_semitones) / 12.0))
        return f0_sequence * factor

    @staticmethod
    def crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
        """İki ses parçası arasında pürüzsüz kosinüs çapraz geçiş (crossfade) uygular."""
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = (
                chunk2[:overlap] * fade_in[: len(chunk2)]
                + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
            )
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    def _process_whisper_features(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """16 kHz ses tensöründen Whisper encoder içerik özniteliklerini çıkarır."""
        assert self.whisper_model is not None and self.whisper_feature_extractor is not None
        enc_dtype = self.whisper_model.encoder.dtype
        if audio_16k.size(-1) <= 16000 * 30:
            inputs = self.whisper_feature_extractor(
                [audio_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            outputs = self.whisper_model.encoder(
                input_features.to(enc_dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            features = outputs.last_hidden_state.to(torch.float32)
            features = features[:, : audio_16k.size(-1) // 320 + 1]
            return features

        overlapping_time = 5
        features_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < audio_16k.size(-1):
            if buffer is None:
                chunk = audio_16k[:, traversed_time : traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        audio_16k[
                            :,
                            traversed_time : traversed_time
                            + 16000 * (30 - overlapping_time),
                        ],
                    ],
                    dim=-1,
                )
            inputs = self.whisper_feature_extractor(
                [chunk.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            outputs = self.whisper_model.encoder(
                input_features.to(enc_dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            chunk_features = outputs.last_hidden_state.to(torch.float32)
            chunk_features = chunk_features[:, : chunk.size(-1) // 320 + 1]
            if traversed_time == 0:
                features_list.append(chunk_features)
            else:
                features_list.append(chunk_features[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += (
                30 * 16000
                if traversed_time == 0
                else chunk.size(-1) - 16000 * overlapping_time
            )
        return torch.cat(features_list, dim=1)

    @torch.no_grad()
    @torch.inference_mode()
    def convert_voice(
        self,
        source: str | Path,
        target: str | Path,
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        f0_condition: bool = True,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        stream_output: bool = False,
    ) -> tuple[np.ndarray, int]:
        """Kaynak insan oyunculuğu sesini hedef karakter sesine dönüştürür.

        Returns:
            (waveform_float32_1d, sample_rate)
        """
        if f0_condition:
            self._ensure_f0_model()
            inference_module = self.model_f0
            mel_fn = self.to_mel_f0
            bigvgan_fn = self.bigvgan_44k_model
            sr = self.sr_f0
            hop_length = self.hop_length_f0
        else:
            self._ensure_base_model()
            inference_module = self.model
            mel_fn = self.to_mel
            bigvgan_fn = self.bigvgan_model
            sr = self.sr
            hop_length = self.hop_length

        max_context_window = sr // hop_length * 30
        overlap_wave_len = self.overlap_frame_len * hop_length

        source_np = _load_audio_mono(source, target_sr=sr)
        ref_np = _load_audio_mono(target, target_sr=sr)

        source_audio = torch.tensor(source_np).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref_np[: sr * 25]).unsqueeze(0).float().to(self.device)

        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

        S_alt = self._process_whisper_features(converted_waves_16k)
        S_ori = self._process_whisper_features(ref_waves_16k)

        mel = mel_fn(source_audio.to(self.device).float())
        mel2 = mel_fn(ref_audio.to(self.device).float())

        target_lengths = torch.LongTensor([max(1, int(mel.size(2) * length_adjust))]).to(
            mel.device
        )
        target2_lengths = torch.LongTensor([max(1, mel2.size(2))]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        if f0_condition and self.rmvpe is not None:
            F0_ori_np = self.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
            F0_alt_np = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

            F0_ori = torch.from_numpy(F0_ori_np).float().to(self.device)[None]
            F0_alt = torch.from_numpy(F0_alt_np).float().to(self.device)[None]

            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            shifted_log_f0_alt = log_f0_alt.clone()
            if auto_f0_adjust and voiced_F0_ori.numel() > 0 and voiced_F0_alt.numel() > 0:
                voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
                voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
                median_log_f0_ori = torch.median(voiced_log_f0_ori)
                median_log_f0_alt = torch.median(voiced_log_f0_alt)
                shifted_log_f0_alt[F0_alt > 1] = (
                    log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
                )
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)
            if pitch_shift != 0:
                shifted_f0_alt[F0_alt > 1] = self.adjust_f0_semitones(
                    shifted_f0_alt[F0_alt > 1], pitch_shift
                )
        else:
            F0_ori = None
            shifted_f0_alt = None

        cond, _, _, _, _ = inference_module.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
        )
        prompt_condition, _, _, _, _ = inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
        )

        max_source_window = max(64, max_context_window - mel2.size(2))
        processed_frames = 0
        generated_wave_chunks: list[np.ndarray] = []
        previous_chunk: Optional[torch.Tensor] = None

        steps = max(4, min(100, int(diffusion_steps)))
        use_amp = self.device.type == "cuda"

        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    vc_target = inference_module.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                        mel2,
                        style2,
                        None,
                        steps,
                        inference_cfg_rate=float(inference_cfg_rate),
                    )
                    vc_target = vc_target[:, :, mel2.size(-1) :]
            else:
                vc_target = inference_module.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2,
                    style2,
                    None,
                    steps,
                    inference_cfg_rate=float(inference_cfg_rate),
                )
                vc_target = vc_target[:, :, mel2.size(-1) :]

            vc_wave = bigvgan_fn(vc_target.float())[0]

            if processed_frames == 0:
                if is_last_chunk:
                    generated_wave_chunks.append(vc_wave[0].detach().cpu().numpy())
                    break
                generated_wave_chunks.append(
                    vc_wave[0, :-overlap_wave_len].detach().cpu().numpy()
                )
                previous_chunk = vc_wave[0, -overlap_wave_len:].detach()
                processed_frames += vc_target.size(2) - self.overlap_frame_len
            elif is_last_chunk:
                assert previous_chunk is not None
                out_w = self.crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0].detach().cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(out_w)
                break
            else:
                assert previous_chunk is not None
                out_w = self.crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].detach().cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(out_w)
                previous_chunk = vc_wave[0, -overlap_wave_len:].detach()
                processed_frames += vc_target.size(2) - self.overlap_frame_len

        full_wav = (
            np.concatenate(generated_wave_chunks)
            if generated_wave_chunks
            else np.zeros(sr, dtype=np.float32)
        )
        return np.asarray(full_wav, dtype=np.float32), int(sr)

    def convert_voice_to_file(
        self,
        source_wav_path: str | Path,
        target_wav_path: str | Path,
        output_wav_path: str | Path,
        diffusion_steps: int = 25,
        auto_f0_adjust: bool = True,
        pitch_shift: int = 0,
        f0_condition: bool = True,
        inference_cfg_rate: float = 0.7,
    ) -> Path:
        """Kaynak oyunculuk sesini hedef karakter sesine dönüştürüp WAV dosyasına kaydeder."""
        out_p = Path(output_wav_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        wav_arr, sr = self.convert_voice(
            source=source_wav_path,
            target=target_wav_path,
            diffusion_steps=diffusion_steps,
            f0_condition=f0_condition,
            auto_f0_adjust=auto_f0_adjust,
            pitch_shift=pitch_shift,
            inference_cfg_rate=inference_cfg_rate,
            stream_output=False,
        )
        sf.write(str(out_p), wav_arr, sr, subtype="PCM_16")
        return out_p
