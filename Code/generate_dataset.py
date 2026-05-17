import os
import math
import random
import ast
import torch
from torch.utils.data import Dataset
import scipy.signal as signal
import numpy as np
import pandas as pd
import soundfile as sf
from rir_generator import generate as rir_generat


# ==============================================================================
# MAIN DATASET LOADER (Used in Model Training/Evaluation)
# ==============================================================================
class GeneratedData_Dynamic_Speakers_Babble(Dataset):
    """
    Loads clean multichannel speech, removes silences, and concatenates 
    different speech files to reach the target duration without repetition.
    
    Supports both 2-speaker (1 target + 1 null) and 3-speaker (2 targets + 1 null, or 1 target + 2 nulls) 
    modes via the 'use_3_speakers' and 'one_pass_two_nulls' flags in config, 
    WITH dynamic timelines for GEVD RTF estimation.
    """
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.use_3_speakers = getattr(cfg.params, 'use_3_speakers', 0) == 1
        self.one_pass_two_nulls = getattr(cfg.params, 'one_pass_two_nulls', 0) == 1
        
        # Load correct CSV based on mode and speaker count
        if self.use_3_speakers:
            csv_path = cfg.dataset.df_path_train_3spk if mode == "train" else cfg.dataset.df_path_test_3spk
        else:
            csv_path = cfg.dataset.df_path_train_2spk if mode == "train" else cfg.dataset.df_path_test_2spk

        self.df = pd.read_csv(csv_path)
        self.clean_dir = cfg.paths.train_path if mode == "train" else cfg.paths.test_path
        self.clean_prefix = getattr(cfg, "clean_prefix", "clean_example_")
        self.babble_dir = cfg.paths.babble_noise_train_path if mode == "train" else cfg.paths.babble_noise_test_path

        self.fs = int(getattr(cfg.params, "fs", 16000))
        self.T = float(getattr(cfg.params, "T", 4.0))
        self.N = int(self.fs * self.T)
        self.mic_ref = int(getattr(cfg.params, "mic_ref", 4))
        self.noise_only_time = float(getattr(cfg.params, "noise_only_time", 0.5))
        
        self.SNR_babble_db = random.uniform(3.0, 12.0) if mode == "train" else 6.0
        self.SNR_white_db = float(getattr(cfg.params, "SNR_white_db", 30.0))

    def __len__(self):
        return len(self.df)

    def _remove_silence(self, x, threshold_db=-40, frame_length_ms=20):
        return x
        # Silence removal is applied to the training set to maximize 
        # gradient density and RTF-learning efficiency. For the test/validation 
        # sets, silence is preserved to evaluate performance on natural, 
        # continuous speech as it would appear in real-world scenarios.
        '''frame_length = int(self.fs * frame_length_ms / 1000)
        if len(x) < frame_length:
            return x
        num_frames = len(x) // frame_length
        frames = x[:num_frames * frame_length].reshape(num_frames, frame_length)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        rms_db = 20 * np.log10(rms)
        active_frames = frames[rms_db > threshold_db]
        if len(active_frames) == 0:
            return x
        return active_frames.flatten()'''

    def _load_wav(self, path):
        OLD_PREFIX = "/dsi/gannot-lab1/datasets/LibriSpeech/LibriSpeech/"
        NEW_PREFIX = "/dsi/gannot-lab/gannot-lab1/datasets/LibriSpeech/LibriSpeech/"
        if isinstance(path, str) and path.startswith(OLD_PREFIX):
            path = path.replace(OLD_PREFIX, NEW_PREFIX)
        x, _ = sf.read(path)
        if x.ndim > 1:
            x = x[:, 0]
        return x.astype(np.float32)

    def _load_continuous_speech(self, start_idx, target_samples):
        combined_speech = np.array([], dtype=np.float32)
        current_offset = 0
        
        while len(combined_speech) < target_samples:
            idx = (start_idx + current_offset) % len(self.df)
            path = self.df.iloc[idx]["speaker_path"]
            
            x = self._load_wav(path)
            x_active = self._remove_silence(x)
            
            combined_speech = np.concatenate([combined_speech, x_active])
            current_offset += 1
            
        return combined_speech[:target_samples]
    
    def __getitem__(self, idx: int):
        C_K = self.cfg.params.ck
        row1 = self.df.iloc[idx]
        
        N = self.N
        fs = self.fs
        rev_env = self.cfg.paths.rev_env

        # --- Room Geometry ---
        L = [float(row1["room_x"]), float(row1["room_y"]), float(row1["room_z"])]
        beta_val = float(row1["beta"])
        n_taps_anechoic = int(row1["n"])
        mic_positions = np.array(ast.literal_eval(row1["mic_positions"]), dtype=np.float64)
        M = mic_positions.shape[0]

        spk1_pos = [float(row1["speaker_start_x"]), float(row1["speaker_start_y"]), float(row1["speaker_start_z"])]
        spk2_pos = [float(row1["noise1_x"]), float(row1["noise1_y"]), float(row1["noise1_z"])]
        
        spk_positions = [spk1_pos, spk2_pos]
        if self.use_3_speakers:
            spk3_pos = [float(row1["noise2_x"]), float(row1["noise2_y"]), float(row1["noise2_z"])]
            spk_positions.append(spk3_pos)

        # --- Build Continuous Speech Timelines ---
        idx_spk2 = idx + (len(self.df) // 3)
        idx_spk3 = idx + (2 * len(self.df) // 3)

        # Recording with/without partition 
        '''if False:  # self.cfg.modelParams.use_true_rirs and self.cfg.modelParams.DUAL_MODEL:
            noise_only_samples = int(self.noise_only_time * fs)
            needed = N - noise_only_samples
            
            x1_cut = self._load_continuous_speech(idx, needed)
            x2_cut = self._load_continuous_speech(idx_spk2, needed)
            
            x1_final = np.concatenate([np.zeros(noise_only_samples, dtype=np.float32), x1_cut])
            x2_final = np.concatenate([np.zeros(noise_only_samples, dtype=np.float32), x2_cut])
            
            if self.use_3_speakers:
                x3_cut = self._load_continuous_speech(idx_spk3, needed)
                x3_final = np.concatenate([np.zeros(noise_only_samples, dtype=np.float32), x3_cut])
        else:'''
        n_0_5, n_1_5, n_2_5 = int(0.5 * fs), int(1.5 * fs), int(2.5 * fs)
        # Timeline logic for GEVD/Blind tracking mode
        needed1 = (n_1_5 - n_0_5) + (N - n_2_5)
        x1_cont = self._load_continuous_speech(idx, needed1)
        x1_final = np.zeros(N, dtype=np.float32)
        x1_final[n_0_5:n_1_5], x1_final[n_2_5:N] = x1_cont[:(n_1_5 - n_0_5)], x1_cont[(n_1_5 - n_0_5):]

        needed2 = (n_2_5 - n_1_5) + (N - n_2_5)
        x2_cont = self._load_continuous_speech(idx_spk2, needed2)
        x2_final = np.zeros(N, dtype=np.float32)
        x2_final[n_1_5:n_2_5], x2_final[n_2_5:N] = x2_cont[:(n_2_5 - n_1_5)], x2_cont[(n_2_5 - n_1_5):]

        if self.use_3_speakers:
            if self.one_pass_two_nulls:
                x3_cont = self._load_continuous_speech(idx_spk3, needed2)
                x3_final = np.zeros(N, dtype=np.float32)
                x3_final[n_1_5:n_2_5], x3_final[n_2_5:N] = x3_cont[:(n_2_5 - n_1_5)], x3_cont[(n_2_5 - n_1_5):]
            else:
                needed3 = (n_1_5 - n_0_5) + (N - n_2_5)
                x3_cont = self._load_continuous_speech(idx_spk3, needed3)
                x3_final = np.zeros(N, dtype=np.float32)
                x3_final[n_0_5:n_1_5], x3_final[n_2_5:N] = x3_cont[:(n_1_5 - n_0_5)], x3_cont[(n_1_5 - n_0_5):]

        # --- RIR and Convolution ---
        if not rev_env:
            # ANECHOIC: Use original rir_generator
            beta_vec = [beta_val] * 6
            h1 = np.array(rir_generat(C_K, fs, mic_positions, spk1_pos, L, beta_vec, n_taps_anechoic)).T.astype(np.float32)
            h2 = np.array(rir_generat(C_K, fs, mic_positions, spk2_pos, L, beta_vec, n_taps_anechoic)).T.astype(np.float32)
            if self.use_3_speakers:
                h3 = np.array(rir_generat(C_K, fs, mic_positions, spk3_pos, L, beta_vec, n_taps_anechoic)).T.astype(np.float32)
            
            # Use lfilter for short anechoic taps
            d1 = np.stack([signal.lfilter(h1[m], [1.0], x1_final)[:N] for m in range(M)], axis=1)
            d2 = np.stack([signal.lfilter(h2[m], [1.0], x2_final)[:N] for m in range(M)], axis=1)
            if self.use_3_speakers:
                d3 = np.stack([signal.lfilter(h3[m], [1.0], x3_final)[:N] for m in range(M)], axis=1)
        else:
            # REVERBERANT: Use gpuRIR
            import gpuRIR
            target_taps = 9000
            beta_vec = gpuRIR.beta_SabineEstimation(L, beta_val).tolist()
            
            h_all = gpuRIR.simulateRIR(
                room_sz=L, beta=beta_vec, pos_src=np.stack(spk_positions),
                pos_rcv=mic_positions.astype(np.float32), nb_img=[35, 35, 35],
                Tdiff=target_taps / fs, Tmax=target_taps / fs, fs=float(fs),
                spkr_pattern="omni", mic_pattern="omni"
            )
            h1, h2 = h_all[0].astype(np.float32), h_all[1].astype(np.float32)
            
            # Use fftconvolve for long reverberant taps
            d1 = np.stack([signal.fftconvolve(x1_final, h1[m], mode='full')[:N] for m in range(M)], axis=1)
            d2 = np.stack([signal.fftconvolve(x2_final, h2[m], mode='full')[:N] for m in range(M)], axis=1)
            if self.use_3_speakers:
                h3 = h_all[2].astype(np.float32)
                d3 = np.stack([signal.fftconvolve(x3_final, h3[m], mode='full')[:N] for m in range(M)], axis=1)

        d = d1 + d2 + (d3 if self.use_3_speakers else 0)

        # --- Babble Noise ---
        babble_path = os.path.join(self.babble_dir, f"babble_{idx:07d}.wav")
        babble, _ = sf.read(babble_path, always_2d=True)
        if babble.shape[0] < N:
            babble = np.tile(babble, (math.ceil(N / babble.shape[0]), 1))
        babble = babble[:N, :M].astype(np.float32)

        # --- Noise Scaling ---
        ref = self.mic_ref - 1
        d_power = np.sum(d[:, ref] ** 2) + 1e-12
        b_power = np.sum(babble[:, ref] ** 2) + 1e-12
        babble *= np.sqrt(d_power * 10 ** (-self.SNR_babble_db / 10.0) / b_power)

        v = np.random.randn(N, M).astype(np.float32)
        v_power = np.sum(v[:, ref] ** 2) + 1e-12
        v *= np.sqrt(d_power * 10 ** (-self.SNR_white_db / 10.0) / v_power)

        y = d + babble + v
        peak = np.max(np.abs(y)) + 1e-12

        # Return standard tuples for 2 or 3 speakers
        if self.use_3_speakers:
            return (
                torch.from_numpy(y / peak).float(), torch.from_numpy(d1 / peak).float(),
                torch.from_numpy(d1 / peak).float(), torch.from_numpy(d2 / peak).float(),
                torch.from_numpy(d3 / peak).float(), torch.from_numpy(babble / peak).float(),
                torch.from_numpy(v / peak).float(), torch.from_numpy(h1).float(),
                torch.from_numpy(h2).float(), torch.from_numpy(h3).float()
            )
        else:
            return (
                torch.from_numpy(y / peak).float(), torch.from_numpy(d / peak).float(),
                torch.from_numpy(d1 / peak).float(), torch.from_numpy(d2 / peak).float(),
                torch.from_numpy(babble / peak).float(), torch.from_numpy(v / peak).float(),
                torch.from_numpy(h1).float(), torch.from_numpy(h2).float()
            )