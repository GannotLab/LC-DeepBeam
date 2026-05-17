"""
Static Beampattern Evaluation Script.

This script evaluates the spatial performance of the trained U-Net beamformer. 
It loads pre-calculated spatial weights (from .mat files), averages them over 
the active speech segment, and tests them against pre-recorded white noise 
sources placed at 1-degree intervals around the microphone array. 

It generates both wideband polar plots and narrowband spectrograms to visualize 
how well the network steered spatial nulls toward the interferers.
"""

import os
import ast
import torch
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt

# Local imports
from utils import Preprocesing, Postprocessing, return_as_complex

matplotlib.use("Agg")  # Run headless to prevent display issues on servers

# ==============================================================================
# GLOBAL STYLES & MATPLOTLIB CONFIGURATION
# ==============================================================================
# Increase global font sizes for paper-ready figures
plt.rcParams.update({
    'axes.titlesize': 18,      # Title size
    'xtick.labelsize': 16,     # X-axis tick labels
    'ytick.labelsize': 17,     # Y-axis tick labels
})

# ==============================================================================
# USER SETTINGS & CASE MAPPING
# ==============================================================================
CSV_FILE     = "/home/dsi/engelor1/DNN_Based_Beamformer/Code/create_dataset_python/room_parameters_tracking_test.csv"
SURROUND_DIR = "/dsi/gannot-lab/gannot-lab1/datasets/Ilai_data/Correct_White_Beampattern_Surround"
OUT_ROOT     = "/home/dsi/engelor1/DNN_Based_Beamformer/Code/beampattern_pngs_final"
ONE_PASS_TWO_NULLS = True

# Format: "Case_Name": ([STFT_Paths], USE_3_SPEAKERS_FLAG)
CASES = {
    "2SPK_NonRev_True": (
        ["/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/30_04_Two_Speakers_True_RTFs_Fixed/TEST_STFT_domain_results_02_05_2026__13_07_41_0.mat"], 
        False
    ),
    "2SPK_NonRev_Estimated": (
        ["/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/11_04_FINAL_Estimated_RTF/TEST_STFT_domain_results_13_04_2026__05_37_32_0.mat"], 
        False
    ),
    "2SPK_NonRev_No": (
        ["/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Two_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_28_04_2026__07_42_11_0.mat"], 
        False
    ),
    "3SPK_NonRev_True": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/30_04_Three_Speakers_True_RTFs_Fixed/TEST_STFT_domain_results_02_05_2026__13_13_44_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/30_04_Three_Speakers_True_RTFs_Fixed/TEST_STFT_domain_results_02_05_2026__13_13_48_1.mat"
        ], 
        True
    ),
    "3SPK_NonRev_True_NoPartition": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/24_04_Three_Speakers_True_RTFs/STFT_0_3spk_True_withoutRev.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/24_04_Three_Speakers_True_RTFs/STFT_1_3spk_True_withoutRev.mat"
        ], 
        True
    ),
    "3SPK_NonRev_Estimated": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/14_04_Three_Speakers_Improved_GEVD/TEST_STFT_domain_results_15_04_2026__18_50_47_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/14_04_Three_Speakers_Improved_GEVD/TEST_STFT_domain_results_15_04_2026__18_50_51_1.mat"
        ], 
        True
    ),
    "3SPK_NonRev_No": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Three_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_28_04_2026__06_55_43_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/26_04_Three_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_28_04_2026__06_55_49_1.mat"
        ], 
        True
    ),
    "3SPK_NonRev_No_NoPartition": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/23_04_Three_Speakers_No_RTFs/TEST_STFT_domain_results_24_04_2026_18_39_12_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Non_Reverberant_Environment/23_04_Three_Speakers_No_RTFs/TEST_STFT_domain_results_24_04_2026_18_39_16_1.mat"
        ], 
        True
    ),
    "2SPK_Rev_True": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/02_05_Rev_Two_Speakers_True_RTFs_Fixed/TEST_STFT_domain_results_04_05_2026__18_31_24_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/02_05_Rev_Two_Speakers_True_RTFs_Fixed/TEST_STFT_domain_results_04_05_2026__18_31_25_1.mat"
        ], 
        False
    ),
    "2SPK_Rev_Estimated": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/21_04_Rev_Two_Speakers_Estimated_RTFs/TEST_STFT_domain_results_23_04_2026__12_56_37_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/21_04_Rev_Two_Speakers_Estimated_RTFs/TEST_STFT_domain_results_23_04_2026__12_56_38_1.mat"
        ], 
        False
    ),
    "2SPK_Rev_No": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/28_04_Rev_Two_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_30_04_2026__11_46_09_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/28_04_Rev_Two_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_30_04_2026__11_46_11_1.mat"
        ], 
        False
    ),
    "3SPK_Rev_True": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/02_05_Rev_Three_Speakers_True_RTFs_Fixed/TEST_STFT_domain_results_04_05_2026__18_39_08_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/02_05_Rev_Three_Speakers_True_RTFs_Fixed/TEST_STFT_domain_results_04_05_2026__18_39_10_1.mat"
        ], 
        True
    ),
    "3SPK_Rev_Estimated": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/24_04_Rev_Three_Speakers_Estimated_RTFs/TEST_STFT_domain_results_28_04_2026__08_17_17_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/24_04_Rev_Three_Speakers_Estimated_RTFs/TEST_STFT_domain_results_28_04_2026__08_17_19_1.mat"
        ], 
        True
    ),
    "3SPK_Rev_No": (
        [
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/28_04_Rev_Three_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_30_04_2026__11_44_43_0.mat", 
            "/home/dsi/engelor1/DNN_Based_Beamformer/Code/Results_Reverberant_Environment/28_04_Rev_Three_Speakers_No_RTFs_Fixed/TEST_STFT_domain_results_30_04_2026__11_44_45_1.mat"
        ], 
        True
    ),
}

TARGET_INDICES   = range(8)
FS               = 16000
WIN_LENGTH       = 512
HOP              = WIN_LENGTH // 4
NOISE_HEAD_SEC   = 0.5     
BP_DB_MIN        = -15.0
BP_DB_MAX        = 0.0
SPEC_DB_MIN      = -60.0
SPEC_DB_MAX      = -10.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# UTILITIES
# ==============================================================================

def doa_local_deg(x, y, cx, cy, orientation_deg):
    """
    Transforms the global room coordinates of a speaker into an angle relative 
    to the linear microphone array's center and orientation.
    """
    ang_global = (np.degrees(np.arctan2(y - cy, x - cx)) + 360.0) % 360.0
    ang_local  = (ang_global - orientation_deg) % 360.0
    return 360.0 - ang_local if ang_local > 180.0 else ang_local


def main():
    """Main execution loop for generating spatial beampattern plots."""
    
    # ---------------------------------------------------
    # Preload surround feature vectors (White Noise Probes)
    # ---------------------------------------------------
    print("Preloading surround feature vectors...")
    surround = []
    for doa in range(181):
        # Load a pre-simulated white noise burst arriving from the specific degree 'doa'
        m = sio.loadmat(os.path.join(SURROUND_DIR, f"my_surround_feature_vector_angle_{doa}.mat"))
        feat = torch.from_numpy(np.asarray(m["feature"]).astype(np.float32)).to(device)  
        surround.append(feat.unsqueeze(0))

    df = pd.read_csv(CSV_FILE)

    for case_name, (mat_paths, USE_3_SPEAKERS) in CASES.items():
        print(f"\nProcessing Case: {case_name}")
        out_dir = os.path.join(OUT_ROOT, case_name)
        os.makedirs(out_dir, exist_ok=True)

        # Pre-load spatial weights from the saved PyTorch output
        loaded_mats = {}
        for p in mat_paths:
            if os.path.exists(p):
                mat_data = sio.loadmat(p)
                loaded_mats[p] = torch.from_numpy(mat_data["W_Stage1_left"]).to(device)

        for INDEX in TARGET_INDICES:
            # File/Index selection logic
            if len(mat_paths) == 1:
                current_mat_path = mat_paths[0]
                internal_idx = INDEX 
            else:
                file_idx = 0 if INDEX < 4 else 1
                current_mat_path = mat_paths[file_idx]
                internal_idx = INDEX % 4 

            if current_mat_path not in loaded_mats:
                continue

            W_all = loaded_mats[current_mat_path]
            B, M, F, L = W_all.shape
            
            if internal_idx >= B:
                print(f"Skipping INDEX {INDEX}: Out of bounds")
                continue

            print(f"  Generating plots for INDEX {INDEX}...")
            
            # --- Extract Geometries ---
            row1 = df.iloc[INDEX]
            mic_positions = np.array(ast.literal_eval(row1["mic_positions"]), dtype=np.float64)  
            mic_center    = mic_positions.mean(axis=0)
            cx, cy        = mic_center[0], mic_center[1]
            orientation   = float(row1["angleOrientation"])

            # Map the true room coordinates to array-relative angles
            doa_spk1 = doa_local_deg(float(row1["speaker_start_x"]), float(row1["speaker_start_y"]), cx, cy, orientation)
            doa_spk2 = doa_local_deg(float(row1["noise1_x"]), float(row1["noise1_y"]), cx, cy, orientation)
            if USE_3_SPEAKERS:
                doa_spk3 = doa_local_deg(float(row1["noise2_x"]), float(row1["noise2_y"]), cx, cy, orientation)

            # Average the temporal weights over the active speech segment (ignoring initial noise)
            noise_frames = int(NOISE_HEAD_SEC * FS // HOP)
            W = W_all[internal_idx]  
            W_avg = W[:, :, min(noise_frames, L-1):].mean(dim=-1, keepdim=True).unsqueeze(0)  

            # --- Probe Spatial Response ---
            pow_list = []
            spec_acc = torch.zeros((F, 181), dtype=torch.float32, device=device)

            for doa in range(181):
                y_in = surround[doa]  
                Y = Preprocesing(y_in, WIN_LENGTH, FS, y_in.shape[1]/FS, HOP, device)   
                Yc = return_as_complex(Y)                                   
                
                # Apply the averaged neural network weights to the test signal
                Z = torch.sum(torch.conj(W_avg) * Yc, dim=1)  
                z_t = Postprocessing(Z, HOP, WIN_LENGTH, device)  
                
                # Calculate total wideband power and per-frequency power
                pow_list.append(torch.sum(torch.abs(z_t)**2))  
                spec_acc[:, doa] = torch.mean(torch.abs(Z)**2, dim=-1).squeeze(0)

            # Power scaling for plots
            a = torch.stack(pow_list)  
            a_db = 10.0 * torch.log10(torch.clamp(a / (torch.max(a) + 1e-12), min=1e-12)).detach().cpu().numpy()
            a_db = np.clip(a_db, BP_DB_MIN, BP_DB_MAX)
            r = a_db - BP_DB_MIN  
            
            spec_db = 10.0 * torch.log10(torch.clamp(spec_acc, min=1e-12)).detach().cpu().numpy()
            spec_db = np.clip(spec_db, SPEC_DB_MIN, SPEC_DB_MAX)
            freqs = np.linspace(0.0, FS / 2, F)


            # ===================================================
            # PLOT 1: WIDEBAND POLAR BEAMPATTERN
            # ===================================================
            # Map standard [0, 180] azimuth to Matplotlib's top-half polar projection
            mapped = np.pi/2 - np.deg2rad(np.arange(181))
            fig_bp, ax_bp = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6.25, 5.0))
            
            ax_bp.plot(mapped, r, label="Beampower", linewidth=3)
            r_max = BP_DB_MAX - BP_DB_MIN
            
            # Plot Oracle Speaker Locations
            phi1 = np.pi/2 - np.deg2rad(doa_spk1)
            phi2 = np.pi/2 - np.deg2rad(doa_spk2)

            ax_bp.plot([phi1, phi1], [0, r_max], linestyle="--", color="red", linewidth=2.0)
            ax_bp.plot(phi1, r_max, "ro", label="Speaker 1 (Target)", markersize=8)
            ax_bp.plot([phi2, phi2], [0, r_max], linestyle="--", color="mediumblue", linewidth=2.0)
            ax_bp.plot(phi2, r_max, "o", color="mediumblue", label="Speaker 2 (Null)", markersize=8)

            if USE_3_SPEAKERS:
                phi3 = np.pi/2 - np.deg2rad(doa_spk3)
                ax_bp.plot([phi3, phi3], [0, r_max], linestyle="--", color="mediumblue", linewidth=2.0)
                ax_bp.plot(phi3, r_max, "o", color="mediumblue", label="Speaker 3 (Null)", markersize=8)

            # Polar formatting
            ax_bp.set_theta_zero_location("N")
            ax_bp.set_theta_direction(-1)
            ax_bp.set_thetamin(-90)
            ax_bp.set_thetamax(90)
            ax_bp.set_rlim(0, r_max)
            rticks = np.arange(0, r_max + 0.1, 5)
            ax_bp.set_rticks(rticks)
            ax_bp.set_yticklabels([f"{BP_DB_MIN + t:.0f}" for t in rticks])
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
            fig_bp.savefig(os.path.join(out_dir, f"static_beampattern_INDEX_{INDEX}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_bp)


            # ===================================================
            # PLOT 2: DOA–FREQ SPECTROGRAM
            # ===================================================
            fig_sp, ax_sp = plt.subplots(figsize=(6.25, 5.5))
            
            # extent=[90, -90, ...] flips the array so 0 deg is at +90 and 180 deg is at -90
            im = ax_sp.imshow(
                spec_db, aspect="auto", origin="lower",
                extent=[90, -90, freqs[0], freqs[-1]], 
                vmin=SPEC_DB_MIN, vmax=SPEC_DB_MAX, cmap="viridis"
            )
            
            # Ensures left-to-right is -90 to +90
            ax_sp.set_xlim(-90, 90)

            cbar = fig_sp.colorbar(im, ax=ax_sp)
            cbar.set_label("Power (dB)", fontsize=20, labelpad=12)
            cbar.ax.tick_params(labelsize=14)

            ax_sp.set_xlabel("DOA (deg)", fontsize=18, labelpad=12)
            ax_sp.set_ylabel("Frequency (Hz)", fontsize=20, labelpad=12)
            ax_sp.set_title("Narrowband Beampattern", fontsize=21, pad=20)

            # Transformation matches the polar coordinate shift: 90 - doa
            speakers_to_plot = [
                (90 - doa_spk1, "red", "Speaker 1 (Target)"), 
                (90 - doa_spk2, "mediumblue", "Speaker 2 (Null)")
            ]
            if USE_3_SPEAKERS:
                speakers_to_plot.append((90 - doa_spk3, "mediumblue", "Speaker 3 (Null)"))

            for doa_adj, color, label in speakers_to_plot:
                ax_sp.axvline(x=doa_adj, color=color, linestyle="--", linewidth=2.5) 
                mid_f = freqs[len(freqs) // 2]
                ax_sp.plot(doa_adj, mid_f, marker="o", color=color, label=label, markersize=10)

            ax_sp.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.22),
                ncol=2, fontsize=17, columnspacing=0.8, handletextpad=0.4, frameon=True
            )

            ax_sp.tick_params(axis='both', which='major', labelsize=16, pad=10)
            fig_sp.savefig(os.path.join(out_dir, f"static_spectrogram_INDEX_{INDEX}.png"), bbox_inches="tight", dpi=300)
            plt.close(fig_sp)

if __name__ == "__main__":
    main()