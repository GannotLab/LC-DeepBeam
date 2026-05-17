"""
Master Audio Extraction Script.

This script iterates through the evaluation results (.mat files) for all 14 acoustic 
scenarios. It extracts the raw time-domain tensors for the noisy mixture, the 
clean target reference, and the U-Net enhanced output, and saves them as standard 
16kHz WAV mono files.
"""

import os
import glob

import numpy as np
import scipy.io
import soundfile as sf

# ====================================================================
# CONFIGURATION
# ====================================================================
OUT_ROOT = "/home/dsi/engelba3/DNN_Based_Beamformer/Code/Recordings_Final"
BASE_RESULTS_DIR = "/home/dsi/engelba3/DNN_Based_Beamformer/Code/RESULTS_FINAL"
SAMPLE_RATE = 16000
KEYS = ['y', 'x_hat_stage1_left', 'first_speaker']
TARGET_INDICES = range(100)

# Mapping the case names directly to their subfolder names
CASE_FOLDERS = {
    "2SPK_NonRev_True": "2spkTrueWithout",
    "2SPK_NonRev_Estimated": "2spkEstWithout",
    "2SPK_NonRev_No": "2spkNoWithout",
    "3SPK_NonRev_True": "3spkTrueWithout",
    "3SPK_NonRev_Estimated": "3spkEstWithout",
    "3SPK_NonRev_No": "3spkNoWithout",
    "2SPK_Rev_True": "2spkTrueWith",
    "2SPK_Rev_Estimated": "2spkEstWith",
    "2SPK_Rev_No": "2spkNoWith",
    "3SPK_Rev_True": "3spkTrueWith",
    "3SPK_Rev_Estimated": "3spkEstWith",
    "3SPK_Rev_No": "3spkNoWith",
    "3SPK_NonRev_No_NOPartition": "3spkNoWithoutNOP",
    "3SPK_NonRev_True_NOPartition": "3spkTrueWithoutNOP"
}

# Dynamically building the file lists
CASES = {}
for case_name, folder_name in CASE_FOLDERS.items():
    folder_path = os.path.join(BASE_RESULTS_DIR, folder_name)
    
    # Automatically find and sort every time-domain .mat file in the folder
    search_pattern = os.path.join(folder_path, "TEST_time_domain_results_*.mat")
    mat_files = sorted(glob.glob(search_pattern))
    
    # Only add the case if it actually found files inside the folder
    if mat_files:
        CASES[case_name] = mat_files

# ====================================================================


def save_audio(signal, folder, key, index):
    """
    Saves a raw numpy array as a standardized 16kHz WAV file.
    
    If the signal is multichannel, it safely extracts a single microphone 
    channel (index 0) to ensure a Mono output. If the array contains 
    complex numbers, it safely isolates the real component.
    """
    # If complex, take the real part
    if np.iscomplexobj(signal):
        signal = np.real(signal)
    
    signal = signal.astype(np.float32)

    # Multichannel -> Extract single mic for Mono
    if signal.ndim == 2 and signal.shape[1] >= 1:
        mono_sig = signal[:, 0]  # Taking the first microphone
        output_file = os.path.join(folder, f"{key}_mono_idx_{index}.wav")
        sf.write(output_file, mono_sig, SAMPLE_RATE)
        
    # Mono -> Save single channel directly
    else:
        output_file = os.path.join(folder, f"{key}_mono_idx_{index}.wav")
        sf.write(output_file, signal, SAMPLE_RATE)


def main():
    """
    Main execution loop.
    Iterates through all defined test cases, pre-loads the .mat data, and triggers 
    the audio extraction and saving process dynamically across all file batches.
    """
    for case_name, mat_paths in CASES.items():
        print(f"\n[STARTING CASE] {case_name}")
        
        # Create dedicated subfolders for each signal key ('y', 'x_hat', etc.)
        case_paths = {}
        for key in KEYS:
            path = os.path.join(OUT_ROOT, case_name, key)
            os.makedirs(path, exist_ok=True)
            case_paths[key] = path

        # Pre-load data to handle the batch splitting logic efficiently
        loaded_data = {}
        for p in mat_paths:
            if os.path.exists(p):
                loaded_data[p] = scipy.io.loadmat(p)
            else:
                print(f"!!! Warning: File not found {p}")

        # Iterate through the target test examples
        for INDEX in TARGET_INDICES:
            current_global_idx = 0
            found = False

            # Loop through files sequentially to dynamically find the correct batch
            for current_path in mat_paths:
                if current_path not in loaded_data:
                    continue

                mat_data = loaded_data[current_path]
                
                # Check how many samples are in this specific file batch
                if 'y' not in mat_data:
                    continue
                batch_size = mat_data['y'].shape[0]

                # If the target index falls within this specific file's batch size
                if current_global_idx <= INDEX < current_global_idx + batch_size:
                    internal_idx = INDEX - current_global_idx

                    # Extract and save each requested signal component
                    for key in KEYS:
                        if key not in mat_data:
                            print(f"  [SKIP] Key '{key}' not in {os.path.basename(current_path)}")
                            continue
                        
                        sig = mat_data[key][internal_idx]
                        save_audio(sig, case_paths[key], key, INDEX)

                    found = True
                    break

                # Add the batch size and check the next file
                current_global_idx += batch_size

            if not found:
                print(f"  [ERR] Target Index {INDEX} exceeds available samples or files are missing.")
                
        print(f"[FINISHED CASE] {case_name}")

    print("\nAll audio sets have been successfully saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()