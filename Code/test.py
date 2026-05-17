import torch
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

# Local imports
from utils import Preprocesing, Postprocessing, return_as_complex
from saveResults import saveResults
from Beamformer import beamformingOperationStage1


def test(model, args, results_path, test_loader, device, cfg_loss, iftest):
    """
    Evaluates the beamformer model on a test dataset.

    Calculates spatial weights based on the initial signal segment, applies 
    target gain normalization, and computes evaluation metrics (SI-SDR, SNR, 
    SIR, SINR) strictly on the active speech window (e.g., 2.5s to 8.0s).

    Args:
        model (nn.Module): The trained U-Net spatial weight estimator.
        args (object): Configuration arguments (sample rate, window length, etc.).
        results_path (str): Directory path to save the resulting .mat files.
        test_loader (DataLoader): PyTorch DataLoader for the test dataset.
        device (torch.device): Compute device (CPU or CUDA).
        cfg_loss (object): Loss configuration parameters (unused in inference but kept for signature).
        iftest (int): Flag (1 or 0) indicating whether to save .mat output files.

    Returns:
        float: The average negative SI-SDR across the test set (used as test loss).
    """

    # Parameters
    fs, win_len, mic_ref = args.fs, args.win_length, args.mic_ref
    R = eval(args.R)
    
    # Evaluate only on the last 5.5 seconds (skipping the initial 2.5s noise-only profile)
    eval_start = int(2.5 * fs) 
 
    num_batches = len(test_loader)
    epoch_test_loss = 0
    
    # Trackers (All stored as linear sums to be averaged and converted to dB at the end)
    
    # Target Trackers
    tot_in_sisdr, tot_out_sisdr = 0.0, 0.0
    tot_in_snr, tot_out_snr = 0.0, 0.0
    tot_in_sir, tot_out_sir = 0.0, 0.0
    tot_in_sinr, tot_out_sinr = 0.0, 0.0
    
    # Interferer 1 Trackers
    tot_in_sisdr_i1, tot_out_sisdr_i1 = 0.0, 0.0
    tot_pwr_ratio_i1 = 0.0
    
    # Noise Trackers
    tot_pwr_ratio_noise = 0.0
    
    # Optional Interferer 2 Trackers (Used only if 3 speakers are present)
    tot_in_sisdr_i2, tot_out_sisdr_i2 = 0.0, 0.0
    tot_pwr_ratio_i2 = 0.0
        
    model.eval()   
    si_sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
       
    with torch.no_grad():
        for i, batch in enumerate(test_loader): 
            
            # --- 1. UNPACKING ---
            # Dynamically handle either 2-speaker or 3-speaker batches
            if len(batch) == 10:
                use_3 = True
                (y_f, _, f_f, s_f, t_f, bab, wht, rir_f, rir_s, rir_t) = batch
                t_f = t_f.to(device)
            else:
                use_3 = False
                (y_f, _, f_f, s_f, bab, wht, rir_f, rir_s) = batch
            
            # Push signals to the designated device (GPU/CPU)
            y_f, f_f, s_f = y_f.to(device), f_f.to(device), s_f.to(device)
            noise_f = (bab + wht).to(device)

            # --- 2. GET WEIGHTS (0-4s) ---
            # Extract weights using the first 4 seconds of the mixture
            mid = int(4 * fs)
            Y_1st = Preprocesing(y_f[:, :mid, :], win_len, fs, 4, R, device)
            
            if use_3:
                _, W_f_complex, _, _ = model(Y_1st, rir_f, rir_s, rir_t, device, mode="test")
            else:
                _, W_f_complex, _, _ = model(Y_1st, rir_f, rir_s, device=device, mode="test")

            # Static Weight Reconstruction: Take the first temporal frame and expand
            W_static_complex = W_f_complex[:, :, :, 0]
            W_frozen_514 = torch.view_as_real(W_static_complex).reshape(W_f_complex.shape[0], W_f_complex.shape[1], -1)

            # --- 3. TARGET POWER NORMALIZATION (g) ---
            # Push target speech through the unnormalized weights to calculate power difference
            Y_target_stft = Preprocesing(f_f, win_len, fs, 8, R, device)
            X_hat_target_temp, _, _, _ = beamformingOperationStage1(Y_target_stft, W_frozen_514)
            out_target_temp = Postprocessing(X_hat_target_temp, R, win_len, device)
            
            in_t_pwr_slice = f_f[:, eval_start:, mic_ref-1]
            out_t_pwr_slice = out_target_temp[:, eval_start:]
            min_l_pwr = min(in_t_pwr_slice.shape[1], out_t_pwr_slice.shape[1])
            
            # Calculate Gain (g) to ensure the target speaker's volume is perfectly preserved
            P_target_in = torch.mean(in_t_pwr_slice[:, :min_l_pwr]**2, dim=-1) + 1e-12
            P_target_out = torch.mean(out_t_pwr_slice[:, :min_l_pwr]**2, dim=-1) + 1e-12
            
            g = torch.sqrt(P_target_in / P_target_out).view(-1, 1, 1)
            W_frozen_514 = W_frozen_514 * g

            # --- 4. COMPONENT PROCESSING ---
            def process_component(audio):
                """Helper function to apply spatial weights to a specific audio component."""
                stft = Preprocesing(audio, win_len, fs, 8, R, device)
                out_stft, _, _, W_c = beamformingOperationStage1(stft, W_frozen_514)
                return Postprocessing(out_stft, R, win_len, device), out_stft, W_c

            # Process the full mixture and isolate each individual component
            x_hat_full, X_hat_C_full, W_full_complex = process_component(y_f)
            out_t, _, _ = process_component(f_f)
            out_i1, _, _ = process_component(s_f)
            out_n, _, _ = process_component(noise_f)

            # --- 5. SLICING & METRICS ---
            # Slice arrays to evaluate metrics ONLY during the active speech segment (2.5s - 8s)
            in_m_e = y_f[:, eval_start:, mic_ref-1]
            in_t_e = f_f[:, eval_start:, mic_ref-1]
            in_i1_e = s_f[:, eval_start:, mic_ref-1]
            in_n_e = noise_f[:, eval_start:, mic_ref-1]

            out_m_e = x_hat_full[:, eval_start:]
            out_t_e = out_t[:, eval_start:]
            out_i1_e = out_i1[:, eval_start:]
            out_n_e = out_n[:, eval_start:]

            # Ensure all arrays match in length before math operations
            min_l = min(in_m_e.shape[1], out_m_e.shape[1])
            def clean_slice(sig): return sig[:, :min_l]
            
            in_m_e, in_t_e, in_i1_e, in_n_e = map(clean_slice, [in_m_e, in_t_e, in_i1_e, in_n_e])
            out_m_e, out_t_e, out_i1_e, out_n_e = map(clean_slice, [out_m_e, out_t_e, out_i1_e, out_n_e])

            def get_pwr(sig): 
                """Helper to calculate signal power safely."""
                return torch.mean(sig**2, dim=-1) + 1e-12

            # Target & Combined Calculations
            p_in_t, p_out_t = get_pwr(in_t_e), get_pwr(out_t_e)
            p_in_i1, p_out_i1 = get_pwr(in_i1_e), get_pwr(out_i1_e)
            p_in_n, p_out_n = get_pwr(in_n_e), get_pwr(out_n_e)
            
            # --- METRIC ACCUMULATION ---
            
            # Target SI-SDR
            tot_in_sisdr += si_sdr_metric(in_m_e, in_t_e).item()
            out_sisdr = si_sdr_metric(out_m_e, in_t_e).item()
            tot_out_sisdr += out_sisdr
            epoch_test_loss += -out_sisdr

            # SINR & SIR Logic
            if use_3:
                # Process the second interferer
                out_i2_time, _, _ = process_component(t_f)
                in_i2_e = clean_slice(t_f[:, eval_start:, mic_ref-1])
                out_i2_e = clean_slice(out_i2_time[:, eval_start:])
                p_in_i2, p_out_i2 = get_pwr(in_i2_e), get_pwr(out_i2_e)
                
                # 3-Speaker SINR
                tot_in_sinr += torch.mean(10 * torch.log10(p_in_t / (p_in_i1 + p_in_i2 + p_in_n))).item()
                tot_out_sinr += torch.mean(10 * torch.log10(p_out_t / (p_out_i1 + p_out_i2 + p_out_n))).item()
                
                # 3-Speaker SIR
                tot_in_sir += torch.mean(10 * torch.log10(p_in_t / (p_in_i1 + p_in_i2))).item()
                tot_out_sir += torch.mean(10 * torch.log10(p_out_t / (p_out_i1 + p_out_i2))).item()
                
                # Interferer 2 specific metrics
                tot_pwr_ratio_i2 += torch.mean(10 * torch.log10(p_out_i2 / p_in_i2)).item()
                tot_in_sisdr_i2 += si_sdr_metric(in_m_e, in_i2_e).item()
                tot_out_sisdr_i2 += si_sdr_metric(out_m_e, in_i2_e).item()
            else:
                # 2-Speaker SINR
                tot_in_sinr += torch.mean(10 * torch.log10(p_in_t / (p_in_i1 + p_in_n))).item()
                tot_out_sinr += torch.mean(10 * torch.log10(p_out_t / (p_out_i1 + p_out_n))).item()
                
                # 2-Speaker SIR
                tot_in_sir += torch.mean(10 * torch.log10(p_in_t / p_in_i1)).item()
                tot_out_sir += torch.mean(10 * torch.log10(p_out_t / p_out_i1)).item()

            # Background Noise & Interferer 1 metrics
            tot_in_snr += torch.mean(10 * torch.log10(p_in_t / p_in_n)).item()
            tot_out_snr += torch.mean(10 * torch.log10(p_out_t / p_out_n)).item()
            
            tot_in_sisdr_i1 += si_sdr_metric(in_m_e, in_i1_e).item()
            tot_out_sisdr_i1 += si_sdr_metric(out_m_e, in_i1_e).item()
            
            tot_pwr_ratio_i1 += torch.mean(10 * torch.log10(p_out_i1 / p_in_i1)).item()
            tot_pwr_ratio_noise += torch.mean(10 * torch.log10(p_out_n / p_in_n)).item()

            # Save generated arrays to .mat files if requested
            if iftest == 1:
                # --- CREATE STFTS FOR SAVING ---
                Y_save = Preprocesing(y_f, win_len, fs, 8, R, device)

                FIRST_SPEAKER = Y_target_stft
                SECOND_SPEAKER = Preprocesing(s_f, win_len, fs, 8, R, device)

                FIRST_SPEAKER_stft = return_as_complex(FIRST_SPEAKER)
                SECOND_SPEAKER_stft = return_as_complex(SECOND_SPEAKER)

                if use_3:
                    THIRD_SPEAKER = Preprocesing(t_f, win_len, fs, 8, R, device)
                    THIRD_SPEAKER_stft = return_as_complex(THIRD_SPEAKER)
                
                save_kwargs = {
                    "Y": Y_save,
                    "FIRST_SPEAKER_stft": FIRST_SPEAKER_stft,
                    "SECOND_SPEAKER_stft": SECOND_SPEAKER_stft,
                    "W_Stage1_left": W_full_complex,
                    "X_hat_Stage1_C_left": X_hat_C_full,
                    "y": y_f,
                    "first_speaker": f_f,
                    "second_speaker": s_f,
                    "x_hat_stage1_left": x_hat_full,
                    "results_path": results_path,
                    "i": i,
                    "fs": fs,
                }
                
                # Pass third speaker data to saveResults only if operating in 3-speaker mode
                if use_3:
                    save_kwargs["third_speaker"] = t_f
                    save_kwargs["THIRD_SPEAKER_stft"] = THIRD_SPEAKER_stft

                saveResults(**save_kwargs)

    # --- FINAL PRINT ---
    def avg(val): 
        """Helper to calculate batch average."""
        return val / num_batches
    
    print(f"\n{'='*85}")
    print(f"TEST RESULTS (dB) | Eval Window: Last 5.5s | Target Gain Normalized")
    print(f"{'='*85}")
    print(f"{'Source':<15} | {'Metric':<10} | {'Input':<12} | {'Output':<12} | {'Gain/Loss':<12}")
    print(f"{'-'*85}")
    print(f"{'Target':<15} | SI-SDR     | {avg(tot_in_sisdr):10.2f} | {avg(tot_out_sisdr):10.2f} | {avg(tot_out_sisdr-tot_in_sisdr):+10.2f}")
    print(f"{'':<15} | SNR        | {avg(tot_in_snr):10.2f} | {avg(tot_out_snr):10.2f} | {avg(tot_out_snr-tot_in_snr):+10.2f}")
    print(f"{'':<15} | SIR        | {avg(tot_in_sir):10.2f} | {avg(tot_out_sir):10.2f} | {avg(tot_out_sir-tot_in_sir):+10.2f}")
    print(f"{'':<15} | SINR       | {avg(tot_in_sinr):10.2f} | {avg(tot_out_sinr):10.2f} | {avg(tot_out_sinr-tot_in_sinr):+10.2f}")
    print(f"{'-'*85}")
    print(f"{'Interferer 1':<15} | SI-SDR     | {avg(tot_in_sisdr_i1):10.2f} | {avg(tot_out_sisdr_i1):10.2f} | {avg(tot_out_sisdr_i1-tot_in_sisdr_i1):+10.2f}")
    print(f"{'':<15} | Pwr Ratio  | {'-':>10} | {'-':>10} | {avg(tot_pwr_ratio_i1):10.2f} dB")
    
    if use_3:
        print(f"{'Interferer 2':<15} | SI-SDR     | {avg(tot_in_sisdr_i2):10.2f} | {avg(tot_out_sisdr_i2):10.2f} | {avg(tot_out_sisdr_i2-tot_in_sisdr_i2):+10.2f}")
        print(f"{'':<15} | Pwr Ratio  | {'-':>10} | {'-':>10} | {avg(tot_pwr_ratio_i2):10.2f} dB")
        
    print(f"{'-'*85}")
    print(f"{'Noise (NR)':<15} | Pwr Ratio  | {'-':>10} | {'-':>10} | {avg(tot_pwr_ratio_noise):10.2f} dB")
    print(f"{'='*85}\n")

    return epoch_test_loss