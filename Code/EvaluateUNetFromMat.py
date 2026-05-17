"""
Master U-Net Offline Evaluation Suite.

This script performs rapid offline metric calculation (SI-SDR, SNR, SIR, SINR) 
by loading pre-calculated spatial weights and audio tensors from .mat files. 
It bypasses the need to rerun the heavy PyTorch Neural Network inference, 
reducing evaluation time from hours to seconds while maintaining 100% 
mathematical parity with the online test loop.
"""

import os
import glob
from datetime import datetime
import torch
import numpy as np
import scipy.io
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# Local imports
from utils import Preprocesing, Postprocessing, return_as_complex

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TIME = 8.0            
EVAL_START_TIME = 2.5    # Matches eval_start in test.py
SAMPLE_RATE = 16000
REF_MIC_IDX = 4          # Matches args.mic_ref

BASE_RESULT_PATH = "/home/dsi/engelba3/DNN_Based_Beamformer/Code/RESULTS_FINAL/"
SUMMARY_FILE = f"/home/dsi/engelba3/DNN_Based_Beamformer/Code/Final_UNET_Offline_Summary_8.0s.txt"

# Constants
WIN_LEN = 512
R = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fs = SAMPLE_RATE


# ==============================================================================
# UTILITIES
# ==============================================================================

def dual_print(text):
    """Prints text to the console and appends it to the summary text file."""
    print(text)
    with open(SUMMARY_FILE, "a") as f:
        f.write(text + "\n")


def apply_complex_weights_batch(y_batch_tensor, W_complex_batch, device):
    """
    Applies the spatial weights to the entire batch simultaneously.
    
    This function exactly mimics the math inside beamformingOperationStage1,
    ensuring offline metrics perfectly match online metrics.

    Args:
        y_batch_tensor (torch.Tensor): Time-domain batch signal.
        W_complex_batch (torch.Tensor): Pre-calculated complex spatial weights.
        device (torch.device): Compute device.

    Returns:
        torch.Tensor: Time-domain output signal after beamforming.
    """
    # 1. Real-Valued STFT
    Y_stft_real = Preprocesing(y_batch_tensor, WIN_LEN, fs, 8, R, device)
    
    # 2. Convert to Complex STFT
    Y_stft = return_as_complex(Y_stft_real)
    
    # 3. Apply complex conjugate weights (W^H * Y)
    X_hat_stft = torch.sum(torch.mul(torch.conj(W_complex_batch), Y_stft), dim=1) 
    
    # 4. ISTFT back to time domain
    out_time = Postprocessing(X_hat_stft, R, WIN_LEN, device)
    return out_time


# ==============================================================================
# MAIN EVALUATION LOOP
# ==============================================================================

def main():
    """Main execution loop for offline metric extraction."""
    
    with open(SUMMARY_FILE, "w") as f:
        f.write(f"{'='*85}\nMASTER UNET OFFLINE EVALUATION SUITE\n"
                f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*85}\n\n")

    test_configurations = [
        {"name": "2 Spk | No RTFs | No Rev",   "folder": "2spkNoWithout",   "use_3_speakers": False},
        {"name": "2 Spk | True RTFs | No Rev", "folder": "2spkTrueWithout", "use_3_speakers": False},
        {"name": "2 Spk | Est RTFs | No Rev",  "folder": "2spkEstWithout",  "use_3_speakers": False},
        {"name": "2 Spk | No RTFs | Rev",      "folder": "2spkNoWith",      "use_3_speakers": False},
        {"name": "2 Spk | True RTFs | Rev",    "folder": "2spkTrueWith",    "use_3_speakers": False},
        {"name": "2 Spk | Est RTFs | Rev",     "folder": "2spkEstWith",     "use_3_speakers": False},
        {"name": "3 Spk | No RTFs | No Rev",   "folder": "3spkNoWithout",   "use_3_speakers": True},
        {"name": "3 Spk | True RTFs | No Rev", "folder": "3spkTrueWithout", "use_3_speakers": True},
        {"name": "3 Spk | Est RTFs | No Rev",  "folder": "3spkEstWithout",  "use_3_speakers": True},
        {"name": "3 Spk | No RTFs | Rev",      "folder": "3spkNoWith",      "use_3_speakers": True},
        {"name": "3 Spk | True RTFs | Rev",    "folder": "3spkTrueWith",    "use_3_speakers": True},
        {"name": "3 Spk | Est RTFs | Rev",     "folder": "3spkEstWith",     "use_3_speakers": True},
        {"name": "3 Spk | No RTFs | No Rev | No Partition",    "folder": "3spkNoWithoutNOP",    "use_3_speakers": True},
        {"name": "3 Spk | True RTFs | No Rev | No Partition",  "folder": "3spkTrueWithoutNOP",  "use_3_speakers": True}
    ]

    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(DEVICE)

    dual_print(f"\n{'='*85}\n[START] Executing Fast Offline Extraction\n{'='*85}\n")

    for scenario in test_configurations:
        dual_print(f"\n{'*'*60}\n>>> EXTRACTING SCENARIO: {scenario['name']}\n{'*'*60}")
        
        MAT_DIR = os.path.join(BASE_RESULT_PATH, scenario["folder"])
        if not os.path.exists(MAT_DIR):
            dual_print(f"[WARNING] Path {MAT_DIR} not found. Skipping...\n")
            continue

        mat_files = sorted(glob.glob(os.path.join(MAT_DIR, "TEST_STFT_domain_results_*.mat")))
        time_files = sorted(glob.glob(os.path.join(MAT_DIR, "TEST_time_domain_results_*.mat")))
        
        if len(mat_files) == 0 or len(time_files) == 0:
            dual_print(f"[WARNING] .mat files missing in {MAT_DIR}. Skipping...\n")
            continue

        # Exact Trackers from test.py (Added SIR trackers here)
        tot_in_sisdr, tot_out_sisdr = 0.0, 0.0
        tot_in_snr, tot_out_snr = 0.0, 0.0
        tot_in_sir, tot_out_sir = 0.0, 0.0
        tot_in_sinr, tot_out_sinr = 0.0, 0.0
        tot_in_sisdr_i1, tot_out_sisdr_i1 = 0.0, 0.0
        tot_pwr_ratio_i1 = 0.0
        tot_pwr_ratio_noise = 0.0
        tot_in_sisdr_i2, tot_out_sisdr_i2 = 0.0, 0.0
        tot_pwr_ratio_i2 = 0.0

        num_batches = len(mat_files)
        eval_start = int(EVAL_START_TIME * fs)
        mic_ref_idx = REF_MIC_IDX - 1  # Ensures we use index 3 for 0-based arrays

        for f_stft_path, f_time_path in zip(mat_files, time_files):
            data_stft = scipy.io.loadmat(f_stft_path)
            data_time = scipy.io.loadmat(f_time_path)
            
            # Load entire batch to GPU
            y_f = torch.from_numpy(data_time['y']).float().to(DEVICE)
            f_f = torch.from_numpy(data_time['first_speaker']).float().to(DEVICE)
            s_f = torch.from_numpy(data_time['second_speaker']).float().to(DEVICE)
            
            if scenario["use_3_speakers"]:
                t_f = torch.from_numpy(data_time['third_speaker']).float().to(DEVICE)
            else:
                t_f = torch.zeros_like(f_f)

            # Reconstruct Pure Background Noise
            noise_f = y_f - f_f - s_f
            if scenario["use_3_speakers"]:
                noise_f = noise_f - t_f
                
            # Extract weights (ALREADY MULTIPLIED BY target gain 'g' in test.py)
            W_full_complex = torch.from_numpy(data_stft['W_Stage1_left']).to(torch.complex64).to(DEVICE)
            
            # Load the exact mixture output saved by test.py to guarantee 100% parity
            x_hat_full = torch.from_numpy(data_time['x_hat_stage1_left']).float().to(DEVICE)
            
            # Process individual components mathematically to calculate SIR/SNR
            out_t = apply_complex_weights_batch(f_f, W_full_complex, DEVICE)
            out_i1 = apply_complex_weights_batch(s_f, W_full_complex, DEVICE)
            out_n = apply_complex_weights_batch(noise_f, W_full_complex, DEVICE)

            # --- SLICING (2.5s to 8.0s) exactly like test.py ---
            in_m_e = y_f[:, eval_start:, mic_ref_idx]
            in_t_e = f_f[:, eval_start:, mic_ref_idx]
            in_i1_e = s_f[:, eval_start:, mic_ref_idx]
            in_n_e = noise_f[:, eval_start:, mic_ref_idx]

            out_m_e = x_hat_full[:, eval_start:]
            out_t_e = out_t[:, eval_start:]
            out_i1_e = out_i1[:, eval_start:]
            out_n_e = out_n[:, eval_start:]

            # Ensure lengths match exactly
            min_l = min(in_m_e.shape[1], out_m_e.shape[1])
            def clean_slice(sig): 
                return sig[:, :min_l]
            
            in_m_e, in_t_e, in_i1_e, in_n_e = map(clean_slice, [in_m_e, in_t_e, in_i1_e, in_n_e])
            out_m_e, out_t_e, out_i1_e, out_n_e = map(clean_slice, [out_m_e, out_t_e, out_i1_e, out_n_e])

            def get_pwr(sig): 
                return torch.mean(sig**2, dim=-1) + 1e-12

            p_in_t, p_out_t = get_pwr(in_t_e), get_pwr(out_t_e)
            p_in_i1, p_out_i1 = get_pwr(in_i1_e), get_pwr(out_i1_e)
            p_in_n, p_out_n = get_pwr(in_n_e), get_pwr(out_n_e)
            
            # --- METRICS ACCUMULATION ---
            # Using the exact same torchmetrics class over the [B, T] arrays
            tot_in_sisdr += si_sdr_metric(in_m_e, in_t_e).item()
            tot_out_sisdr += si_sdr_metric(out_m_e, in_t_e).item()

            if scenario["use_3_speakers"]:
                out_i2_time = apply_complex_weights_batch(t_f, W_full_complex, DEVICE)
                in_i2_e = clean_slice(t_f[:, eval_start:, mic_ref_idx])
                out_i2_e = clean_slice(out_i2_time[:, eval_start:])
                p_in_i2, p_out_i2 = get_pwr(in_i2_e), get_pwr(out_i2_e)
                
                tot_in_sinr += torch.mean(10 * torch.log10(p_in_t / (p_in_i1 + p_in_i2 + p_in_n))).item()
                tot_out_sinr += torch.mean(10 * torch.log10(p_out_t / (p_out_i1 + p_out_i2 + p_out_n))).item()
                
                # SIR accumulation for 3-speakers
                tot_in_sir += torch.mean(10 * torch.log10(p_in_t / (p_in_i1 + p_in_i2))).item()
                tot_out_sir += torch.mean(10 * torch.log10(p_out_t / (p_out_i1 + p_out_i2))).item()
                
                tot_pwr_ratio_i2 += torch.mean(10 * torch.log10(p_out_i2 / p_in_i2)).item()
                tot_in_sisdr_i2 += si_sdr_metric(in_m_e, in_i2_e).item()
                tot_out_sisdr_i2 += si_sdr_metric(out_m_e, in_i2_e).item()
            else:
                tot_in_sinr += torch.mean(10 * torch.log10(p_in_t / (p_in_i1 + p_in_n))).item()
                tot_out_sinr += torch.mean(10 * torch.log10(p_out_t / (p_out_i1 + p_out_n))).item()
                
                # SIR accumulation for 2-speakers
                tot_in_sir += torch.mean(10 * torch.log10(p_in_t / p_in_i1)).item()
                tot_out_sir += torch.mean(10 * torch.log10(p_out_t / p_out_i1)).item()

            tot_in_snr += torch.mean(10 * torch.log10(p_in_t / p_in_n)).item()
            tot_out_snr += torch.mean(10 * torch.log10(p_out_t / p_out_n)).item()
            
            tot_in_sisdr_i1 += si_sdr_metric(in_m_e, in_i1_e).item()
            tot_out_sisdr_i1 += si_sdr_metric(out_m_e, in_i1_e).item()
            
            tot_pwr_ratio_i1 += torch.mean(10 * torch.log10(p_out_i1 / p_in_i1)).item()
            tot_pwr_ratio_noise += torch.mean(10 * torch.log10(p_out_n / p_in_n)).item()

        # ==============================================================================
        # FINAL PRINT (100% Matches test.py Output Table)
        # ==============================================================================
        if num_batches == 0:
            continue
            
        def avg(val): 
            return val / num_batches
        
        res = f"\n{'='*85}\n"
        res += f"TEST RESULTS (dB) | Eval Window: Last 5.5s | Target Gain Normalized\n"
        res += f"{'='*85}\n"
        res += f"{'Source':<15} | {'Metric':<10} | {'Input':<12} | {'Output':<12} | {'Gain/Loss':<12}\n"
        res += f"{'-'*85}\n"
        res += f"{'Target':<15} | SI-SDR     | {avg(tot_in_sisdr):10.2f} | {avg(tot_out_sisdr):10.2f} | {avg(tot_out_sisdr-tot_in_sisdr):+10.2f}\n"
        res += f"{'':<15} | SNR        | {avg(tot_in_snr):10.2f} | {avg(tot_out_snr):10.2f} | {avg(tot_out_snr-tot_in_snr):+10.2f}\n"
        res += f"{'':<15} | SIR        | {avg(tot_in_sir):10.2f} | {avg(tot_out_sir):10.2f} | {avg(tot_out_sir-tot_in_sir):+10.2f}\n"
        res += f"{'':<15} | SINR       | {avg(tot_in_sinr):10.2f} | {avg(tot_out_sinr):10.2f} | {avg(tot_out_sinr-tot_in_sinr):+10.2f}\n"
        res += f"{'-'*85}\n"
        res += f"{'Interferer 1':<15} | SI-SDR     | {avg(tot_in_sisdr_i1):10.2f} | {avg(tot_out_sisdr_i1):10.2f} | {avg(tot_out_sisdr_i1-tot_in_sisdr_i1):+10.2f}\n"
        res += f"{'':<15} | Pwr Ratio  | {'-':>10} | {'-':>10} | {avg(tot_pwr_ratio_i1):10.2f} dB\n"
        
        if scenario["use_3_speakers"]:
            res += f"{'Interferer 2':<15} | SI-SDR     | {avg(tot_in_sisdr_i2):10.2f} | {avg(tot_out_sisdr_i2):10.2f} | {avg(tot_out_sisdr_i2-tot_in_sisdr_i2):+10.2f}\n"
            res += f"{'':<15} | Pwr Ratio  | {'-':>10} | {'-':>10} | {avg(tot_pwr_ratio_i2):10.2f} dB\n"
            
        res += f"{'-'*85}\n"
        res += f"{'Noise (NR)':<15} | Pwr Ratio  | {'-':>10} | {'-':>10} | {avg(tot_pwr_ratio_noise):10.2f} dB\n"
        res += f"{'='*85}\n"
        
        dual_print(res)

if __name__ == "__main__":
    main()