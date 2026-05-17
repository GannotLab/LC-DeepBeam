"""
LCMV Beamforming Pipeline

Implements Linearly Constrained Minimum Variance (LCMV) beamforming for speech 
enhancement. Supports both Oracle mode (using ground-truth RIRs) and Blind mode 
(estimating Relative Transfer Functions via GEVD). Designed for multi-speaker 
scenarios in both reverberant and non-reverberant environments.
"""


import gpuRIR  # Must be imported before PyTorch
import os
import ast
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from rir_generator import generate as rir_generate

matplotlib.use("Agg")  # Headless mode for server runs

# ==========================================
# CONFIGURATION & GLOBAL STYLES
# ==========================================
# FLAG: Set to 1 to use True RIRs (Oracle), or 0 to use Estimated RTFs (GEVD)
USE_TRUE_RIRS = 0

# Synced with All_Static_Beampatterns.py for visual consistency across the paper
plt.rcParams.update({
    'axes.titlesize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 17,
})

OUT_ROOT = "/home/dsi/engelor1/DNN_Based_Beamformer/Code/LCMV_final"
DATA_FRAME_PATH = "/home/dsi/engelor1/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
SURROUND_BASE = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround"

# Format: "Case_Name": ([STFT_Paths], [Time_Paths], USE_3_SPEAKERS_FLAG, REV_ENV_FLAG, BATCH_SIZE)
CASES = {
    "2SPK_NonRev": (
        ["/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Two_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_28_04_2026__07_42_11_0.mat"],
        ["/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Two_Speakers_No_RTFs_Fixed/TEST_time_domain_results_28_04_2026__07_42_11_0.mat"],
        False, False, 8
    ),
    "3SPK_NonRev": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Three_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_28_04_2026__06_55_43_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Three_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_28_04_2026__06_55_49_1.mat"
        ],
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Three_Speakers_No_RTFs_Fixed/TEST_time_domain_results_28_04_2026__06_55_43_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Three_Speakers_No_RTFs_Fixed/TEST_time_domain_results_28_04_2026__06_55_49_1.mat"
        ],
        True, False, 4
    ),
}

TARGET_INDICES = range(8)
SAMPLE_RATE, TIME_SEC = 16000, 4.0
SAMPLES_NUM = int(SAMPLE_RATE * TIME_SEC)
NOISE_ONLY_SAMPLES_NUM = int(0.5 * SAMPLE_RATE)
REF_MIC_IDX, SOUND_SPEED = 3, 343
WINDOW_LEN, HOP_LEN = 512, 128
REG_EPSILON, EPSILON = 1e-5, 1e-12

# Use CPU for standalone processing to avoid CUDA conflicts
DEVICE = torch.device("cpu") 

# Limits for visualization scales
BP_DB_MIN, BP_DB_MAX = -15.0, 0.0
SPEC_DB_MIN, SPEC_DB_MAX = -60.0, 0.0


# ==========================================
# UTILITIES
# ==========================================

def batch_static_covariance_whitening(Y_stft, Ln, mic_ref, eps=1e-8, num_eigenvectors=1):
    """
    Blindly estimates RTFs using GEVD from ExNetBFPFModel.py logic.
    """
    if Y_stft.ndim == 3: 
        Y_stft = Y_stft.unsqueeze(0)

    B, M, F, T = Y_stft.shape
    Y_n = Y_stft[..., :Ln].permute(0, 2, 1, 3)  
    Y_s = Y_stft[..., Ln:].permute(0, 2, 1, 3)  
    
    Phi_nn = torch.matmul(Y_n, Y_n.mH) / Ln      
    Phi_yy = torch.matmul(Y_s, Y_s.mH) / (T - Ln)  
    
    Phi_nn = Phi_nn + eps * torch.eye(M, device=Y_stft.device).view(1, 1, M, M)
    D_nn, V_nn = torch.linalg.eigh(Phi_nn) 
    
    # Casting to V_nn.dtype (ComplexFloat) to avoid matmul error
    D_nn_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(D_nn, min=eps))).to(V_nn.dtype)
    Phi_nn_inv_sqrt = torch.matmul(V_nn, torch.matmul(D_nn_inv_sqrt, V_nn.mH))
    
    D_nn_sqrt = torch.diag_embed(torch.sqrt(torch.clamp(D_nn, min=eps))).to(V_nn.dtype)
    Phi_nn_sqrt = torch.matmul(V_nn, torch.matmul(D_nn_sqrt, V_nn.mH))
    
    Phi_yw = torch.matmul(Phi_nn_inv_sqrt, torch.matmul(Phi_yy, Phi_nn_inv_sqrt))
    D_yw, V_yw = torch.linalg.eigh(Phi_yw)
    
    res = []
    for i in range(num_eigenvectors):
        psi = V_yw[..., -(i + 1)]  
        h_tilde = torch.matmul(Phi_nn_sqrt, psi.unsqueeze(-1)).squeeze(-1) 
        a_cw = h_tilde / (h_tilde[:, :, mic_ref].unsqueeze(-1) + eps) 
        res.append(a_cw.permute(0, 2, 1).squeeze(0)) # [M, F]
        
    return res[0] if num_eigenvectors == 1 else res


def doa_local_deg(x, y, cx, cy, orientation_deg):
    """
    Calculates the relative Direction of Arrival (DOA) angle.
    
    Transforms the global room coordinates of a speaker into an angle relative 
    to the linear microphone array's center and orientation.

    Args:
        x, y (float): Coordinates of the speaker.
        cx, cy (float): Coordinates of the array center.
        orientation_deg (float): Array rotation in the room.

    Returns:
        float: The local DOA in degrees [0, 180].
    """
    ang_global = (np.degrees(np.arctan2(y - cy, x - cx)) + 360.0) % 360.0
    ang_local = (ang_global - orientation_deg) % 360.0
    return 360.0 - ang_local if ang_local > 180.0 else ang_local


def save_audio_formatted(signal, folder, key, index):
    """
    Saves a NumPy array signal as a 16kHz WAV file, automatically handling 
    stereo/mono conversion based on array dimensions.
    """
    if np.iscomplexobj(signal):
        signal = np.real(signal)
    signal = signal.astype(np.float32)
    if signal.ndim == 2 and signal.shape[1] >= 2:
        stereo_sig = np.stack([signal[:, 0], signal[:, -1]], axis=-1)
        out_path = os.path.join(folder, f"{key}_stereo_idx_{index}.wav")
        sf.write(out_path, stereo_sig, SAMPLE_RATE)
    else:
        out_path = os.path.join(folder, f"{key}_mono_idx_{index}.wav")
        sf.write(out_path, signal, SAMPLE_RATE)


def stft_multichannel(time_signal):
    """Computes STFT on a multichannel signal using PyTorch."""
    signal_torch = torch.from_numpy(time_signal).to(DEVICE).transpose(0, 1) 
    window = torch.hamming_window(WINDOW_LEN, device=DEVICE)
    return torch.stft(
        signal_torch, n_fft=WINDOW_LEN, hop_length=HOP_LEN, win_length=WINDOW_LEN, 
        window=window, center=False, return_complex=True
    )


def istft_singlechannel(stft_signal, length):
    """Computes Inverse-STFT to return to the time domain."""
    window = torch.hamming_window(WINDOW_LEN, device=DEVICE)
    return torch.istft(
        stft_signal, n_fft=WINDOW_LEN, hop_length=HOP_LEN, win_length=WINDOW_LEN, 
        window=window, center=False, length=length
    )


# ==========================================
# PROCESSING ENGINE
# ==========================================

def run_lcmv():
    """
    Master execution function.
    
    Loops through predefined test cases, computes mathematical LCMV spatial filters
    based on room geometries, applies them to generate enhanced audio, and plots
    wideband and narrowband spatial beampatterns.
    """
    df = pd.read_csv(DATA_FRAME_PATH)
    
    for case_name, (stft_paths, time_paths, USE_3_SPEAKERS, REV_ENV, BATCH_SIZE) in CASES.items():
        print(f"\n" + "="*60)
        mode_str = "TRUE RIRs" if USE_TRUE_RIRS else "BLIND RTF ESTIMATION"
        print(f" STARTING CASE: {case_name} ({mode_str})")
        print("="*60)
        
        # Setup output directories
        case_root = os.path.join(OUT_ROOT, case_name)
        plot_dir = os.path.join(case_root, "plots")
        y_audio_dir = os.path.join(case_root, "y")
        x_audio_dir = os.path.join(case_root, "x_hat_stage1_left")
        
        for d in [plot_dir, y_audio_dir, x_audio_dir]: 
            os.makedirs(d, exist_ok=True)
        
        # Pre-load data tensors
        loaded_stft = {p: sio.loadmat(p) for p in stft_paths if p and os.path.exists(p)}
        loaded_time = {p: sio.loadmat(p) for p in time_paths if p and os.path.exists(p)}
        
        # Define LCMV constraints (Target=1, Interferers=0)
        gain_vec = np.array([1.0, 0.0, 0.0] if USE_3_SPEAKERS else [1.0, 0.0], dtype=np.complex128)

        for INDEX in TARGET_INDICES:
            # Handle batch splitting logic
            f_idx = INDEX // BATCH_SIZE if len(stft_paths) > 1 else 0
            internal_idx = INDEX % BATCH_SIZE if len(stft_paths) > 1 else INDEX

            p_stft, p_time = stft_paths[f_idx], time_paths[f_idx]
            if not p_stft or p_stft not in loaded_stft or p_time not in loaded_time:
                print(f" [SKIP] Index {INDEX}: File for path index {f_idx} not found.")
                continue

            # Load room geometry from CSV
            row = df.iloc[INDEX]
            mix_raw = loaded_time[p_time]['y'][internal_idx]
            mix = (np.real(mix_raw) if np.iscomplexobj(mix_raw) else mix_raw)[:SAMPLES_NUM, :].astype(np.float32)
            
            mic_pos = np.array(ast.literal_eval(row["mic_positions"]), dtype=np.float64)
            mic_center = mic_pos.mean(axis=0)
            orientation = float(row["angleOrientation"])

            pos = [np.array([row["speaker_start_x"], row["speaker_start_y"], row["speaker_start_z"]]),
                   np.array([row["noise1_x"], row["noise1_y"], row["noise1_z"]])]
            if USE_3_SPEAKERS: 
                pos.append(np.array([row["noise2_x"], row["noise2_y"], row["noise2_z"]]))
            
            # Calculate local angles for plotting
            rel_angles = [doa_local_deg(p[0], p[1], mic_center[0], mic_center[1], orientation) for p in pos]
            print(f"     Processing Index: {INDEX}")

            # Pre-compute STFT of the mixture for logic branching
            stft_m_tensor = stft_multichannel(mix)
            stft_m_np = stft_m_tensor.cpu().numpy()
            
            # --- RTF EXTRACTION LOGIC ---
            if USE_TRUE_RIRS == 1:
                rir_dfts = []
                room = np.array([row[f"room_{d}"] for d in "xyz"], dtype=np.float64)
                if not REV_ENV:
                    for p in pos:
                        rir = np.array(rir_generate(SOUND_SPEED, SAMPLE_RATE, mic_pos, p, room, [float(row["beta"])]*6, 4096)).T
                        rir_dfts.append(np.fft.rfft(rir.astype(np.float32), n=WINDOW_LEN, axis=1).astype(np.complex128))
                else:
                    beta_vec = gpuRIR.beta_SabineEstimation(room, float(row["beta"])).tolist()
                    h_all = gpuRIR.simulateRIR(room_sz=room, beta=beta_vec, pos_src=np.stack(pos), pos_rcv=mic_pos.astype(np.float32), 
                                                nb_img=[35, 35, 35], Tdiff=0.5, Tmax=0.5, fs=float(SAMPLE_RATE))
                    for h in h_all: 
                        rir_dfts.append(np.fft.rfft(h.astype(np.float32), n=WINDOW_LEN, axis=1).astype(np.complex128))
                
                num_bins = rir_dfts[0].shape[1]
                
            else:
                num_bins = stft_m_np.shape[1]
                idx_0_5 = int(0.5 * SAMPLE_RATE // HOP_LEN)
                idx_1_5 = int(1.5 * SAMPLE_RATE // HOP_LEN)
                idx_2_5 = int(2.5 * SAMPLE_RATE // HOP_LEN)

                if not USE_3_SPEAKERS:
                    Y_spk1 = stft_m_tensor[..., :idx_1_5]
                    Y_spk2 = torch.cat([stft_m_tensor[..., :idx_0_5], stft_m_tensor[..., idx_1_5:idx_2_5]], dim=-1)
                    est_rtfs = [
                        batch_static_covariance_whitening(Y_spk1, idx_0_5, REF_MIC_IDX).cpu().numpy(),
                        batch_static_covariance_whitening(Y_spk2, idx_0_5, REF_MIC_IDX).cpu().numpy()
                    ]
                else:
                    Y_target = stft_m_tensor[..., :idx_1_5]
                    Y_interf = torch.cat([stft_m_tensor[..., :idx_0_5], stft_m_tensor[..., idx_1_5:idx_2_5]], dim=-1)
                    rtf_t = batch_static_covariance_whitening(Y_target, idx_0_5, REF_MIC_IDX)
                    rtf_i1, rtf_i2 = batch_static_covariance_whitening(Y_interf, idx_0_5, REF_MIC_IDX, num_eigenvectors=2)
                    est_rtfs = [rtf_t.cpu().numpy(), rtf_i1.cpu().numpy(), rtf_i2.cpu().numpy()]

            # --- LCMV WEIGHTS CALCULATION ---
            stft_n_np = stft_m_np[..., :int(NOISE_ONLY_SAMPLES_NUM // HOP_LEN)]
            weights = np.zeros((num_bins, mic_pos.shape[0]), dtype=np.complex128)
            
            for b in range(num_bins):
                if USE_TRUE_RIRS == 1:
                    C = np.stack([rir_dfts[s][:, b] / (rir_dfts[s][REF_MIC_IDX, b] + EPSILON) for s in range(len(pos))], axis=1)
                else:
                    C = np.stack([est_rtfs[s][:, b] for s in range(len(est_rtfs))], axis=1)
                    
                Rnn = (stft_n_np[:, b, :] @ stft_n_np[:, b, :].conj().T) / stft_n_np.shape[2] + REG_EPSILON * np.eye(mic_pos.shape[0])
                weights[b, :] = np.linalg.solve(Rnn, C) @ np.linalg.solve(C.conj().T @ np.linalg.solve(Rnn, C), gain_vec)
            
            # --- APPLY BEAMFORMER ---
            out_stft = np.zeros((num_bins, stft_m_np.shape[2]), dtype=np.complex128)
            for b in range(num_bins): 
                out_stft[b, :] = np.conj(weights[b, :]) @ stft_m_np[:, b, :]
            y_out = istft_singlechannel(torch.from_numpy(out_stft), SAMPLES_NUM).cpu().numpy().astype(np.float32)

            # --- BEAMPATTERN CALCULATION ---
            bp_avg, bp_spec = np.zeros(181), np.zeros((num_bins, 181))
            for ang in range(181):
                surr = sio.loadmat(os.path.join(SURROUND_BASE, f"my_surround_feature_vector_angle_{ang}.mat"))
                s_stft = stft_multichannel(surr["feature"].astype(np.float32)).cpu().numpy()
                for f in range(num_bins): 
                    b_stft_f = np.conj(weights[f, :]) @ s_stft[:, f, :] 
                    pwr = np.mean(np.abs(b_stft_f) ** 2)
                    bp_avg[ang] += pwr
                    bp_spec[f, ang] = pwr
            
            a_db = np.clip(10.0 * np.log10(bp_avg / (np.max(bp_avg) + EPSILON) + EPSILON), BP_DB_MIN, BP_DB_MAX)
            r_val = a_db - BP_DB_MIN
            spec_db = np.clip(10.0 * np.log10(bp_spec + EPSILON), SPEC_DB_MIN, SPEC_DB_MAX)
            mapped_theta = np.pi/2 - np.deg2rad(np.arange(181))

            # ==========================================
            # PLOTTING: WIDEBAND POLAR BEAMPATTERN
            # ==========================================
            fig_bp, ax_bp = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6.25, 5.0))
            ax_bp.plot(mapped_theta, r_val, label="Beampower", linewidth=3)
            r_max = BP_DB_MAX - BP_DB_MIN
            
            spk_styles = [("red", "Speaker 1 (Target)"), ("mediumblue", "Speaker 2 (Null)")]
            if USE_3_SPEAKERS: 
                spk_styles.append(("mediumblue", "Speaker 3 (Null)"))
            
            for ang, (color, label) in zip(rel_angles, spk_styles):
                phi = np.pi/2 - np.deg2rad(ang)
                ax_bp.plot([phi, phi], [0, r_max], linestyle="--", color=color, linewidth=2.0)
                ax_bp.plot(phi, r_max, "o", color=color, label=label, markersize=8)

            ax_bp.set_theta_zero_location("N")
            ax_bp.set_theta_direction(-1)
            ax_bp.set_thetamin(-90)
            ax_bp.set_thetamax(90)
            ax_bp.set_rlim(0, r_max)
            
            rticks = np.arange(0, r_max + 0.1, 5)
            ax_bp.set_rticks(rticks)
            ax_bp.set_yticklabels([f"{int(BP_DB_MIN + t)}" for t in rticks])
            ax_bp.set_rlabel_position(135)
            ax_bp.grid(True)
            ax_bp.set_title("Wideband Beampattern [dB]", y=0.87)
            
            ax_bp.legend(
                loc="lower center", bbox_to_anchor=(0.5, -0.05),
                ncol=2, fontsize=16, columnspacing=0.2, handletextpad=0.2,
                handlelength=1.3, labelspacing=1.0, borderaxespad=0.1, frameon=True
            )
            ax_bp.tick_params(axis='x', pad=7)
            fig_bp.tight_layout()
            fig_bp.savefig(os.path.join(plot_dir, f"static_beampattern_INDEX_{INDEX}.png"), bbox_inches="tight", dpi=300)

            # ==========================================
            # PLOTTING: NARROWBAND SPECTROGRAM
            # ==========================================
            fig_sp, ax_sp = plt.subplots(figsize=(6.25, 5.5))
            im = ax_sp.imshow(
                spec_db, aspect="auto", origin="lower", 
                extent=[90, -90, 0, SAMPLE_RATE/2], cmap="viridis",
                vmin=SPEC_DB_MIN, vmax=SPEC_DB_MAX
            )
            ax_sp.set_xlim(-90, 90)
            
            cbar = fig_sp.colorbar(im, ax=ax_sp)
            cbar.set_label("Power (dB)", fontsize=20, labelpad=12)
            cbar.ax.tick_params(labelsize=14)
            
            ax_sp.set_xlabel("DOA (deg)", fontsize=18, labelpad=12)
            ax_sp.set_ylabel("Frequency (Hz)", fontsize=20, labelpad=12)
            ax_sp.set_title("Narrowband Beampattern", fontsize=21, pad=20)
            
            for ang, (color, label) in zip(rel_angles, spk_styles):
                doa_adj = 90 - ang 
                ax_sp.axvline(x=doa_adj, color=color, linestyle="--", linewidth=2.5)
                ax_sp.plot(doa_adj, SAMPLE_RATE/4, "o", color=color, label=label, markersize=10)
            
            ax_sp.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=2, fontsize=17, columnspacing=0.8, handletextpad=0.4, frameon=True
            )
            ax_sp.tick_params(axis='both', which='major', labelsize=16, pad=10)
            fig_sp.savefig(os.path.join(plot_dir, f"static_spectrogram_INDEX_{INDEX}.png"), bbox_inches="tight", dpi=300)
            
            # --- CLEANUP & AUDIO ---
            plt.close('all') 
            save_audio_formatted(mix, y_audio_dir, "y", INDEX)
            save_audio_formatted(y_out, x_audio_dir, "x_hat_stage1_left", INDEX)
            print(f"     Finished Index {INDEX}.\n")

if __name__ == "__main__":
    run_lcmv()
