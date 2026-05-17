"""
Master LCMV Beamformer Evaluation Suite.

This script benchmarks a traditional Linearly Constrained Minimum Variance (LCMV) 
beamformer against the deep learning model. It evaluates the SI-SDR, SNR, SIR, 
and SINR metrics across multiple acoustic scenarios.

It provides a master flag 'USE_TRUE_RIRS' to toggle between:
    1 (True RTFs): Calculates oracle spatial covariance matrices from room geometry.
    0 (Estimated RTFs): Uses Covariance Whitening (GEVD) to blindly estimate RTFs.
"""

import os
import ast
import glob
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import scipy.io
import soundfile as sf
from rir_generator import generate as rir_generate
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# FLAG: Set to 1 for True RTFs (Oracle RIRs), 0 for Estimated RTFs (Blind GEVD)
USE_TRUE_RIRS = 0

SAVE_AUDIO_RECORDS = True  # Toggle: True to save .wav files, False for metrics only

TIME = 8.0            
EVAL_START_TIME = 2.5 
SAMPLE_RATE = 16000
REF_MIC_IDX = 3  

DATA_FRAME_PATH  = "/home/dsi/engelba3/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
BASE_RESULT_PATH = "/home/dsi/engelba3/DNN_Based_Beamformer/Code/RESULTS_FINAL/"
AUDIO_BASE_DIR   = "/home/dsi/engelba3/DNN_Based_Beamformer/Code/LCMV_FINAL/AUDIO_OUT/"
SUMMARY_FILE     = f"/home/dsi/engelba3/DNN_Based_Beamformer/Code/Final_LCMV_Summary_{TIME}s.txt"

# Constants
RIR_ORDER = 4096
NOISE_ONLY_TIME = 0.5
NOISE_ONLY_SAMPLES_NUM = int(NOISE_ONLY_TIME * SAMPLE_RATE)
SOUND_SPEED = 343
EPSILON = 1e-12
REG_EPSILON = 1e-5  
WINDOW_LEN = 512
HOP_LEN = 128
DEVICE = torch.device("cpu")
START_IDX = int(EVAL_START_TIME * SAMPLE_RATE)


# ==============================================================================
# UTILITIES
# ==============================================================================

def dual_print(text):
    """
    Prints text to the console and simultaneously appends it to the summary text file.
    """
    print(text)
    with open(SUMMARY_FILE, "a") as f:
        f.write(text + "\n")


def get_pwr(sig):
    """
    Calculates the mean squared power of a signal along its last dimension safely.
    """
    return torch.mean(sig**2, dim=-1) + EPSILON


def compute_gevd_rtf(y_stft, ln_frames, mic_ref, eps=1e-8, num_eigenvectors=1):
    """
    Computes the static Covariance Whitening Relative Transfer Function (RTF).
    
    Vectorized over the frequency dimension using PyTorch batched operations. 
    Performs Generalized Eigenvalue Decomposition (GEVD) to estimate spatial 
    steering vectors without requiring oracle geometric data.
    
    Args:
        y_stft (torch.Tensor): Complex STFT tensor of shape [M, F, T].
        ln_frames (int): Number of noise-only frames at the start of the signal.
        mic_ref (int): Index of the reference microphone.
        eps (float): Small regularization constant for matrix stability.
        num_eigenvectors (int): 1 for target/single-interferer, 2 for dual-interferer.
        
    Returns:
        torch.Tensor or list: Tensor of shape [M, F], or a list of such Tensors.
    """
    m_mics, f_bins, t_frames = y_stft.shape

    # Permute to [F, M, Time] for batched frequency processing
    y_n = y_stft[:, :, :ln_frames].permute(1, 0, 2)  
    y_s = y_stft[:, :, ln_frames:].permute(1, 0, 2)  

    phi_nn = torch.matmul(y_n, y_n.mH) / ln_frames
    phi_yy = torch.matmul(y_s, y_s.mH) / (t_frames - ln_frames)

    identity = torch.eye(m_mics, device=y_stft.device, dtype=y_stft.dtype).view(1, m_mics, m_mics)
    phi_nn = phi_nn + eps * identity

    # EVD of the Noise Covariance Matrix
    d_nn, v_nn = torch.linalg.eigh(phi_nn) 

    d_nn_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(d_nn, min=eps))).to(v_nn.dtype)
    phi_nn_inv_sqrt = torch.matmul(v_nn, torch.matmul(d_nn_inv_sqrt, v_nn.mH))

    d_nn_sqrt = torch.diag_embed(torch.sqrt(torch.clamp(d_nn, min=eps))).to(v_nn.dtype)
    phi_nn_sqrt = torch.matmul(v_nn, torch.matmul(d_nn_sqrt, v_nn.mH))

    # Whiten the noisy speech covariance and perform GEVD
    phi_yw = torch.matmul(phi_nn_inv_sqrt, torch.matmul(phi_yy, phi_nn_inv_sqrt))
    d_yw, v_yw = torch.linalg.eigh(phi_yw)

    rtfs = []
    for i in range(num_eigenvectors):
        psi = v_yw[..., -(i + 1)]  # Extract highest eigenvectors
        h_tilde = torch.matmul(phi_nn_sqrt, psi.unsqueeze(-1)).squeeze(-1) 

        # Normalize relative to the reference microphone
        denom = h_tilde[:, mic_ref] + eps
        a_cw = h_tilde / denom.unsqueeze(-1) 
        rtfs.append(a_cw.T) 

    if num_eigenvectors == 1:
        return rtfs[0]
    return rtfs


def process_with_lcmv(time_signal, weights, length, device):
    """
    Applies the computed mathematical LCMV spatial weights to a time-domain signal.
    
    Converts the signal to the STFT domain, applies the complex spatial filters 
    via tensor contraction (einsum), and returns it to the time domain.
    """
    if time_signal.ndim == 1:
        sig_torch = torch.from_numpy(time_signal).to(device).unsqueeze(0)
    else:
        sig_torch = torch.from_numpy(time_signal).to(device).transpose(0, 1)
        
    window = torch.hamming_window(WINDOW_LEN, device=device)
    stft = torch.stft(
        sig_torch, n_fft=WINDOW_LEN, hop_length=HOP_LEN, win_length=WINDOW_LEN, 
        window=window, center=False, return_complex=True
    )
    
    # Cast weights to cfloat to avoid dtype mismatch with PyTorch STFT
    w_weights = torch.from_numpy(weights).to(device).conj().cfloat()
    
    # Apply weights across the microphone array (dim 'm')
    out_stft = torch.einsum('bm,mbf->bf', w_weights, stft)
    
    out_time = torch.istft(
        out_stft, n_fft=WINDOW_LEN, hop_length=HOP_LEN, win_length=WINDOW_LEN, 
        window=window, center=False, length=length
    )
    return out_time


# ==============================================================================
# MAIN EVALUATION LOOP
# ==============================================================================

def main():
    """
    Main evaluation loop for the LCMV beamformer.
    
    Iterates through the testing scenarios, processes the noisy mixtures,
    acquires the RTFs (either via Oracle or GEVD estimation), applies the 
    beamformer, and logs the objective evaluation metrics.
    """
    if SAVE_AUDIO_RECORDS:
        os.makedirs(AUDIO_BASE_DIR, exist_ok=True)
        
    mode_str = "TRUE_RTFS" if USE_TRUE_RIRS == 1 else "ESTIMATED_GEVD"
    
    with open(SUMMARY_FILE, "w") as f:
        f.write(f"{'='*70}\nMASTER LCMV EVALUATION SUITE (TIME = {TIME}s | MODE = {mode_str})\n"
                f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}\n\n")

    lcmv_scenarios = [
        {"name": "2 Spk | No Rev", "use_3_speakers": False, "rev_env": False, "folder": "2spkEstWithout"},
        {"name": "2 Spk | Rev",    "use_3_speakers": False, "rev_env": True,  "folder": "2spkEstWith"},
        {"name": "3 Spk | No Rev", "use_3_speakers": True,  "rev_env": False, "folder": "3spkEstWithout"},
        #{"name": "3 Spk | Rev",    "use_3_speakers": True,  "rev_env": True,  "folder": "3spkEstWith"}
    ]

    dual_print(f"\n{'='*70}\n[START] Executing Master LCMV Suite\n{'='*70}\n")

    for scenario in lcmv_scenarios:
        dual_print(f"\n{'*'*60}\n>>> RUNNING LCMV SCENARIO: {scenario['name']}\n{'*'*60}")
        
        if SAVE_AUDIO_RECORDS:
            scenario_audio_dir = os.path.join(AUDIO_BASE_DIR, scenario["folder"])
            os.makedirs(scenario_audio_dir, exist_ok=True)

        mat_dir = os.path.join(BASE_RESULT_PATH, scenario["folder"])
        if not os.path.exists(mat_dir):
            dual_print(f"[WARNING] Path {mat_dir} not found. Skipping...\n")
            continue

        mat_files = sorted(glob.glob(os.path.join(mat_dir, "TEST_time_domain_results_*.mat")))
        df = pd.read_csv(DATA_FRAME_PATH)
        si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(DEVICE)
        
        # Initialize metric accumulators
        accum = {k: 0.0 for k in [
            'in_sisdr', 'out_sisdr', 'in_snr', 'out_snr', 'in_sir', 'out_sir', 
            'in_sinr', 'out_sinr', 'in_sisdr_i1', 'out_sisdr_i1', 'pwr_ratio_i1', 
            'in_sisdr_i2', 'out_sisdr_i2', 'pwr_ratio_i2', 'pwr_ratio_noise'
        ]}
        
        num_processed, max_examples = 0, 100

        for f_path in mat_files:
            data = scipy.io.loadmat(f_path)
            y_batch = data['y']
            s1_batch = data['first_speaker']
            s2_batch = data['second_speaker']
            s3_batch = data.get('third_speaker', None)
            
            for b_idx in range(y_batch.shape[0]):
                if num_processed >= max_examples: 
                    break
                
                row = df.iloc[num_processed]
                mix = y_batch[b_idx].astype(np.float32)
                f_f = s1_batch[b_idx].astype(np.float32)
                s_f = s2_batch[b_idx].astype(np.float32)
                
                samples_num = mix.shape[0]
                num_mics = mix.shape[1]
                
                # --- 1. Extract Room Geometries ---
                mic_pos = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)
                room = np.array([row[f"room_{d}"] for d in "xyz"], dtype=np.float64)
                pos_src = [
                    np.array([row["speaker_start_x"], row["speaker_start_y"], row["speaker_start_z"]]), 
                    np.array([row["noise1_x"], row["noise1_y"], row["noise1_z"]])
                ]
                
                if scenario["use_3_speakers"]: 
                    pos_src.append(np.array([row["noise2_x"], row["noise2_y"], row["noise2_z"]]))

                # --- 2. Compute Full Mixture STFT ---
                mix_torch = torch.from_numpy(mix).transpose(0, 1).to(DEVICE)
                window = torch.hamming_window(WINDOW_LEN, device=DEVICE)
                mix_stft = torch.stft(
                    mix_torch, n_fft=WINDOW_LEN, hop_length=HOP_LEN, win_length=WINDOW_LEN, 
                    window=window, center=False, return_complex=True
                )
                num_bins = mix_stft.shape[1]

                # --- 3. RTF Acquisition (True vs Estimated) ---
                if USE_TRUE_RIRS == 1:
                    # RIR Generation logic (Oracle)
                    rir_dfts = [
                        np.fft.rfft(
                            np.array(rir_generate(
                                SOUND_SPEED, SAMPLE_RATE, mic_pos, p, room, [float(row["beta"])]*6, RIR_ORDER
                            )).T.astype(np.float32), 
                            n=WINDOW_LEN, axis=1
                        ).astype(np.complex128) for p in pos_src
                    ]
                    # Normalize DFTs to obtain relative transfer functions
                    rtfs_np = [rir_dfts[s] / (rir_dfts[s][REF_MIC_IDX, :] + EPSILON) for s in range(len(pos_src))]
                    
                    # Compute STFT only for the isolated noise segment
                    ln_frames = int(NOISE_ONLY_SAMPLES_NUM // HOP_LEN)
                    stft_n = mix_stft[:, :, :ln_frames].cpu().numpy()

                else:
                    # Blind GEVD Estimation logic
                    idx_0_5 = int(0.5 * SAMPLE_RATE // HOP_LEN)
                    idx_1_5 = int(1.5 * SAMPLE_RATE // HOP_LEN)
                    idx_2_5 = int(2.5 * SAMPLE_RATE // HOP_LEN)
                    ln_frames = idx_0_5

                    if not scenario["use_3_speakers"]:
                        y_spk1 = mix_stft[:, :, :idx_1_5] 
                        y_spk2 = torch.cat([mix_stft[:, :, :idx_0_5], mix_stft[:, :, idx_1_5:idx_2_5]], dim=-1) 

                        rtf_target = compute_gevd_rtf(y_spk1, ln_frames, REF_MIC_IDX)
                        rtf_interf1 = compute_gevd_rtf(y_spk2, ln_frames, REF_MIC_IDX)
                        rtfs = [rtf_target, rtf_interf1]
                    else:
                        y_target = torch.cat([mix_stft[:, :, :idx_0_5], mix_stft[:, :, idx_0_5:idx_1_5]], dim=-1)
                        y_interf = torch.cat([mix_stft[:, :, :idx_0_5], mix_stft[:, :, idx_1_5:idx_2_5]], dim=-1) 

                        rtf_target = compute_gevd_rtf(y_target, ln_frames, REF_MIC_IDX, num_eigenvectors=1)
                        rtf_interfs = compute_gevd_rtf(y_interf, ln_frames, REF_MIC_IDX, num_eigenvectors=2)
                        rtfs = [rtf_target, rtf_interfs[0], rtf_interfs[1]]

                    rtfs_np = [rtf.cpu().numpy() for rtf in rtfs]
                    stft_n = mix_stft[:, :, :ln_frames].cpu().numpy()

                # --- 4. LCMV Solver Logic ---
                raw_weights = np.zeros((num_bins, num_mics), dtype=np.complex128)
                
                # Constrain Target to 1.0, Nulls to 0.0
                gain_vec = np.array([1.0] + [0.0] * (len(rtfs_np) - 1), dtype=np.complex128)

                # Solve spatial covariance per frequency bin
                for b in range(num_bins):
                    c_matrix = np.stack([rtfs_np[s][:, b] for s in range(len(rtfs_np))], axis=1)
                    
                    r_n = stft_n[:, b, :]
                    r_nn = (r_n @ r_n.conj().T) / r_n.shape[1] + REG_EPSILON * np.eye(num_mics)
                    
                    inv_r_c = np.linalg.solve(r_nn, c_matrix)
                    raw_weights[b, :] = inv_r_c @ np.linalg.solve(c_matrix.conj().T @ inv_r_c, gain_vec)

                # --- 5. Target Power Normalization ---
                raw_out_t = process_with_lcmv(f_f, raw_weights, samples_num, DEVICE)
                in_t_ref = torch.from_numpy(f_f[:, REF_MIC_IDX]).to(DEVICE)
                g_factor = torch.sqrt(get_pwr(in_t_ref[START_IDX:]) / get_pwr(raw_out_t[START_IDX:]))
                norm_weights = raw_weights * g_factor.item()

                # --- 6. Process All Signal Components ---
                out_m = process_with_lcmv(mix, norm_weights, samples_num, DEVICE)
                out_t = process_with_lcmv(f_f, norm_weights, samples_num, DEVICE)
                out_i1 = process_with_lcmv(s_f, norm_weights, samples_num, DEVICE)
                
                t_f = s3_batch[b_idx].astype(np.float32) if s3_batch is not None else np.zeros_like(f_f)
                noise_only = mix - f_f - s_f - (t_f if scenario["use_3_speakers"] else 0)
                out_n = process_with_lcmv(noise_only, norm_weights, samples_num, DEVICE)

                # ==========================================
                # AUDIO RECORDING CREATION (IF ENABLED)
                # ==========================================
                if SAVE_AUDIO_RECORDS and num_processed < max_examples:
                    # Save Mixture as Mono using the Reference Microphone
                    mix_norm = mix / (np.max(np.abs(mix)) + EPSILON)
                    sf.write(os.path.join(scenario_audio_dir, f"LCMV_mixture_INDEX_{num_processed}.wav"), 
                             mix_norm[:, 0], SAMPLE_RATE)
                    
                    # Save Beamformed Output (Mono)
                    y_out = out_m.detach().cpu().numpy()
                    y_out_norm = y_out / (np.max(np.abs(y_out)) + EPSILON)
                    sf.write(os.path.join(scenario_audio_dir, f"LCMV_beamformed_INDEX_{num_processed}.wav"), 
                             y_out_norm, SAMPLE_RATE)

                # ==========================================
                # EVALUATION SLICES & METRICS
                # ==========================================
                in_m_e = torch.from_numpy(mix[START_IDX:, REF_MIC_IDX]).to(DEVICE)
                in_t_e = torch.from_numpy(f_f[START_IDX:, REF_MIC_IDX]).to(DEVICE)
                in_i1_e = torch.from_numpy(s_f[START_IDX:, REF_MIC_IDX]).to(DEVICE)
                in_n_e = torch.from_numpy(noise_only[START_IDX:, REF_MIC_IDX]).to(DEVICE)
                
                out_m_e = out_m[START_IDX:]
                out_t_e = out_t[START_IDX:]
                out_i1_e = out_i1[START_IDX:]
                out_n_e = out_n[START_IDX:]

                p_in_t, p_out_t = get_pwr(in_t_e), get_pwr(out_t_e)
                p_in_i1, p_out_i1 = get_pwr(in_i1_e), get_pwr(out_i1_e)
                p_in_n, p_out_n = get_pwr(in_n_e), get_pwr(out_n_e)

                accum['in_sisdr'] += si_sdr_metric(in_m_e, in_t_e).item()
                accum['out_sisdr'] += si_sdr_metric(out_m_e, in_t_e).item()
                accum['in_snr'] += 10 * torch.log10(p_in_t / p_in_n).item()
                accum['out_snr'] += 10 * torch.log10(p_out_t / p_out_n).item()
                
                denom_in, denom_out = p_in_i1 + p_in_n, p_out_i1 + p_out_n
                
                if scenario["use_3_speakers"]:
                    in_i2_e = torch.from_numpy(t_f[START_IDX:, REF_MIC_IDX]).to(DEVICE)
                    out_i2_e = process_with_lcmv(t_f, norm_weights, samples_num, DEVICE)[START_IDX:]
                    p_in_i2, p_out_i2 = get_pwr(in_i2_e), get_pwr(out_i2_e)
                    
                    denom_in += p_in_i2
                    denom_out += p_out_i2
                    
                    # 3-Speaker SIR
                    accum['in_sir'] += 10 * torch.log10(p_in_t / (p_in_i1 + p_in_i2)).item()
                    accum['out_sir'] += 10 * torch.log10(p_out_t / (p_out_i1 + p_out_i2)).item()
                    
                    accum['in_sisdr_i2'] += si_sdr_metric(in_m_e, in_i2_e).item()
                    accum['out_sisdr_i2'] += si_sdr_metric(out_m_e, in_i2_e).item()
                    accum['pwr_ratio_i2'] += 10 * torch.log10(p_out_i2 / p_in_i2).item()
                else:
                    # 2-Speaker SIR
                    accum['in_sir'] += 10 * torch.log10(p_in_t / p_in_i1).item()
                    accum['out_sir'] += 10 * torch.log10(p_out_t / p_out_i1).item()

                accum['in_sinr'] += 10 * torch.log10(p_in_t / denom_in).item()
                accum['out_sinr'] += 10 * torch.log10(p_out_t / denom_out).item()
                accum['in_sisdr_i1'] += si_sdr_metric(in_m_e, in_i1_e).item()
                accum['out_sisdr_i1'] += si_sdr_metric(out_m_e, in_i1_e).item()
                accum['pwr_ratio_i1'] += 10 * torch.log10(p_out_i1 / p_in_i1).item()
                accum['pwr_ratio_noise'] += 10 * torch.log10(p_out_n / p_in_n).item()

                num_processed += 1
                if num_processed % 10 == 0: 
                    print(f"Progress: {num_processed}/{max_examples}")

        # --- SUMMARY PRINTING ---
        if num_processed == 0: 
            continue

        def avg(val): 
            return val / num_processed
            
        res = f"\n{'='*85}\nLCMV BASELINE RESULTS (dB) | Eval Window: {EVAL_START_TIME}s-8s | Scenario: {scenario['name']}\n{'='*85}\n"
        res += f"{'Source':<15} | {'Metric':<10} | {'Input':<12} | {'Output':<12} | {'Gain/Loss':<12}\n{'-'*85}\n"
        res += f"{'Target':<15} | SI-SDR     | {avg(accum['in_sisdr']):10.2f} | {avg(accum['out_sisdr']):10.2f} | {avg(accum['out_sisdr']-accum['in_sisdr']):+10.2f}\n"
        res += f"{'':<15} | SNR        | {avg(accum['in_snr']):10.2f} | {avg(accum['out_snr']):10.2f} | {avg(accum['out_snr']-accum['in_snr']):+10.2f}\n"
        res += f"{'':<15} | SIR        | {avg(accum['in_sir']):10.2f} | {avg(accum['out_sir']):10.2f} | {avg(accum['out_sir']-accum['in_sir']):+10.2f}\n"
        res += f"{'':<15} | SINR       | {avg(accum['in_sinr']):10.2f} | {avg(accum['out_sinr']):10.2f} | {avg(accum['out_sinr']-accum['in_sinr']):+10.2f}\n{'-'*85}\n"
        res += f"{'Interferer 1':<15} | SI-SDR     | {avg(accum['in_sisdr_i1']):10.2f} | {avg(accum['out_sisdr_i1']):10.2f} | {avg(accum['out_sisdr_i1']-accum['in_sisdr_i1']):+10.2f}\n"
        res += f"{'':<15} | Pwr Ratio  | {'-':>12} | {'-':>12} | {avg(accum['pwr_ratio_i1']):10.2f} dB\n"
        
        if scenario["use_3_speakers"]:
            res += f"{'Interferer 2':<15} | SI-SDR     | {avg(accum['in_sisdr_i2']):10.2f} | {avg(accum['out_sisdr_i2']):10.2f} | {avg(accum['out_sisdr_i2']-accum['in_sisdr_i2']):+10.2f}\n"
            res += f"{'':<15} | Pwr Ratio  | {'-':>12} | {'-':>12} | {avg(accum['pwr_ratio_i2']):10.2f} dB\n"
            
        res += f"{'-'*85}\n{'Noise (NR)':<15} | Pwr Ratio  | {'-':>12} | {'-':>12} | {avg(accum['pwr_ratio_noise']):10.2f} dB\n{'='*85}\n"
        dual_print(res)

if __name__ == "__main__":
    main()