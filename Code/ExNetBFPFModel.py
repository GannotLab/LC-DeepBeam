import torch
import torch.nn as nn

# Local imports
from UnetModel import UNETDualInput_Two_Speakers, UNETDualInput_Three_Speakers
from Beamformer import beamformingOperationStage1
from utils import true_rtf_from_rirs_bmk, return_as_complex


def batch_static_covariance_whitening(Y_stft, Ln, mic_ref, eps=1e-8, num_eigenvectors=1):
    """
    Computes the static Covariance Whitening Relative Transfer Function (RTF) across a batch.
    
    Uses Generalized Eigenvalue Decomposition (GEVD) to blindly estimate the spatial 
    steering vectors of the active speakers without needing True Room Impulse Responses.
    Extracts multiple principal eigenvectors when multiple speakers overlap.

    Args:
        Y_stft (torch.Tensor): Multi-channel STFT tensor of the mixed signal.
        Ln (int): Number of noise-only frames at the start of the signal.
        mic_ref (int): Index of the reference microphone (0-based).
        eps (float): Small regularization constant to ensure matrix invertibility.
        num_eigenvectors (int): Number of principal eigenvectors to extract (1 per speaker).

    Returns:
        torch.Tensor or list: A single estimated RTF tensor if num_eigenvectors=1, 
                              otherwise a list of estimated RTF tensors.
    """
    # 1. Ensure the tensor is in true complex format
    if len(Y_stft.shape) == 5 and Y_stft.shape[-1] == 2:
        Y_stft = torch.complex(Y_stft[..., 0], Y_stft[..., 1])
    elif len(Y_stft.shape) == 4 and not Y_stft.is_complex():
        B, M, F514, T = Y_stft.shape
        Y_reshaped = Y_stft.view(B, M, F514 // 2, 2, T)
        Y_stft = torch.complex(Y_reshaped[..., 0, :], Y_reshaped[..., 1, :])

    B, M, F, T = Y_stft.shape
    
    # 2. Separate noise-only frames (Y_n) and speech+noise frames (Y_s)
    Y_n = Y_stft[..., :Ln].permute(0, 2, 1, 3)  
    Y_s = Y_stft[..., Ln:].permute(0, 2, 1, 3)  
    
    # 3. Calculate spatial covariance matrices
    Phi_nn = torch.matmul(Y_n, Y_n.mH) / Ln      
    Phi_yy = torch.matmul(Y_s, Y_s.mH) / (T - Ln)  
    
    # Regularize noise covariance matrix to prevent singularity
    I = torch.eye(M, device=Y_stft.device, dtype=Y_stft.dtype).view(1, 1, M, M)
    Phi_nn = Phi_nn + eps * I
    
    # 4. Eigenvalue Decomposition (EVD) of Noise
    D_nn, V_nn = torch.linalg.eigh(Phi_nn) 
    
    D_nn_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.clamp(D_nn, min=eps))).to(V_nn.dtype)
    Phi_nn_inv_sqrt = torch.matmul(V_nn, torch.matmul(D_nn_inv_sqrt, V_nn.mH))
    
    D_nn_sqrt = torch.diag_embed(torch.sqrt(torch.clamp(D_nn, min=eps))).to(V_nn.dtype)
    Phi_nn_sqrt = torch.matmul(V_nn, torch.matmul(D_nn_sqrt, V_nn.mH))
    
    # 5. Whiten the noisy speech covariance and perform GEVD
    Phi_yw = torch.matmul(Phi_nn_inv_sqrt, torch.matmul(Phi_yy, Phi_nn_inv_sqrt))
    D_yw, V_yw = torch.linalg.eigh(Phi_yw)
    
    # 6. Extract the requested number of highest eigenvectors
    rtfs = []
    for i in range(num_eigenvectors):
        # -1 is highest eigenvalue (principal speaker), -2 is second highest, etc.
        psi = V_yw[..., -(i + 1)]  
        h_tilde = torch.matmul(Phi_nn_sqrt, psi.unsqueeze(-1)).squeeze(-1) 
        
        # Normalize relative to the reference microphone
        denom = h_tilde[:, :, mic_ref] + eps
        a_cw = h_tilde / denom.unsqueeze(-1) 
        rtfs.append(a_cw.permute(0, 2, 1))
        
    if num_eigenvectors == 1:
        return rtfs[0]
    return rtfs  # Returns a list of separate RTFs


class ExNetBFPF(nn.Module):
    """
    ExNet-BF+PF Model: A deep learning spatial filter.
    
    Dynamically initializes and routes either a 2-Speaker or 3-Speaker U-Net 
    architecture based on the provided parameters.
    """
    def __init__(self, modelParams, params):
        super(ExNetBFPF, self).__init__()
        self.modelParams = modelParams
        self.params = params
        
        # Read the flag directly from modelParams
        self.use_3_speakers = getattr(params, 'use_3_speakers', 0) == 1

        # Dynamically instantiate the correct U-Net architecture
        if self.use_3_speakers:
            self.unet_multiChannel_left = UNETDualInput_Three_Speakers(
                rtf_in_ch=modelParams.channelsStage1,    
                mix_in_ch=modelParams.channelsStage1,    
                out_channels=modelParams.channelsStage1, 
                activation=modelParams.activationStage1,
                EnableSkipAttention=modelParams.EnableSkipAttention,
                stem_each=8  
            )
        else:
            self.unet_multiChannel_left = UNETDualInput_Two_Speakers(
                rtf_in_ch=modelParams.channelsStage1,    
                mix_in_ch=modelParams.channelsStage1,    
                out_channels=modelParams.channelsStage1, 
                activation=modelParams.activationStage1,
                EnableSkipAttention=modelParams.EnableSkipAttention,
                stem_each=8  
            )

    def forward(self, Y, rir_first, rir_second, rir_third=None, device="cuda", mode="test"):
        """
        Executes the forward pass of the ExNet model.
        
        Extracts spatial steering vectors (via Oracle RIRs or Blind GEVD), 
        formats them into tracking features, processes them through the U-Net 
        to estimate complex spatial weights, and applies static beamforming.
        """
        noise_only_time = self.params.noise_only_time
        mic_REF = self.params.mic_ref - 1
        DUAL_MODEL = self.modelParams.DUAL_MODEL
        USE_TRUE_RIRS = getattr(self.modelParams, 'use_true_rirs', 1)
        
        Y_stft = return_as_complex(Y)  # [B, M, 257, T]
        R = 128
        win_len = 512
        fs = 16000
        B_full, M, F_pos, T_full = Y_stft.shape

        Ln = int(noise_only_time * fs // R)
        T_eff = T_full - Ln
        F_514 = F_pos * 2 

        # =======================================================
        # 1. RTF Extraction (True vs Estimated GEVD)
        # =======================================================
        if USE_TRUE_RIRS:
            # Use true geometric Room Impulse Responses (Oracle Mode)
            true_RTF_c_target = true_rtf_from_rirs_bmk(rir_first, win_len=win_len, ref_mic=mic_REF)
            true_RTF_c_interf1 = true_rtf_from_rirs_bmk(rir_second, win_len=win_len, ref_mic=mic_REF)
            if self.use_3_speakers and rir_third is not None:
                true_RTF_c_interf2 = true_rtf_from_rirs_bmk(rir_third, win_len=win_len, ref_mic=mic_REF)
        else:
            # Blind Estimation Mode using GEVD Timeline Segments
            idx_0_5 = int(0.5 * fs // R)   
            idx_1_5 = int(1.5 * fs // R)   
            idx_2_5 = int(2.5 * fs // R)   
            Ln_frames = idx_0_5

            if not self.use_3_speakers:
                # --- TWO SPEAKER GEVD ---
                Y_spk1_segment = Y_stft[..., :idx_1_5] 
                Y_spk2_segment = torch.cat([Y_stft[..., :idx_0_5], Y_stft[..., idx_1_5:idx_2_5]], dim=-1) 
                
                true_RTF_c_target = batch_static_covariance_whitening(Y_spk1_segment, Ln_frames, mic_REF)
                true_RTF_c_interf1 = batch_static_covariance_whitening(Y_spk2_segment, Ln_frames, mic_REF)
            else:
                # --- THREE SPEAKER GEVD ---
                Y_target_segment = torch.cat([Y_stft[..., :idx_0_5], Y_stft[..., idx_0_5:idx_1_5]], dim=-1)
                Y_interf_segment = torch.cat([Y_stft[..., :idx_0_5], Y_stft[..., idx_1_5:idx_2_5]], dim=-1) 

                true_RTF_c_target = batch_static_covariance_whitening(Y_target_segment, Ln_frames, mic_REF, num_eigenvectors=1)
                true_RTF_c_interf1, true_RTF_c_interf2 = batch_static_covariance_whitening(Y_interf_segment, Ln_frames, mic_REF, num_eigenvectors=2)

        # =======================================================
        # 2. Format RTFs to Real/Imag features [B, M, 514]
        # =======================================================
        def format_rtf(rtf_c):
            """Splits complex RTFs into real and imaginary halves."""
            rtf_c_pos = rtf_c[:, :, :F_pos]
            return torch.view_as_real(rtf_c_pos).reshape(B_full, M, F_514)

        rtf_feat_t = format_rtf(true_RTF_c_target)
        rtf_feat_i1 = format_rtf(true_RTF_c_interf1)
        if self.use_3_speakers:
            rtf_feat_i2 = format_rtf(true_RTF_c_interf2)

        # =======================================================
        # 3. Expand over effective time and add noise-only padding
        # =======================================================
        def expand_and_pad(feat):
            """Broadcasts the static RTF across time and pads the initial noise segment."""
            expanded = feat.to(device).unsqueeze(-1).expand(-1, -1, -1, T_eff)
            zeros_pad = torch.zeros((B_full, M, F_514, Ln), device=device)
            return torch.cat([zeros_pad, expanded], dim=-1)

        true_rtf_t = expand_and_pad(rtf_feat_t)
        true_rtf_i1 = expand_and_pad(rtf_feat_i1)
        if self.use_3_speakers:
            true_rtf_i2 = expand_and_pad(rtf_feat_i2)

        # =======================================================
        # 4. Pass through dynamically selected U-Net
        # =======================================================
        if self.use_3_speakers:
            W_time_normalized, W_time_gained, skip_Stage1 = self.unet_multiChannel_left(
                Y.float(), true_rtf_t, true_rtf_i1, true_rtf_i2, DUAL_MODEL
            )
        else:
            W_time_normalized, W_time_gained, skip_Stage1 = self.unet_multiChannel_left(
                Y.float(), true_rtf_t, true_rtf_i1, DUAL_MODEL
            )

        # =======================================================
        # 5. Temporal Averaging & Static Beamforming
        # =======================================================
        W_fixed_normalized = torch.mean(W_time_normalized, dim=3)
        W_fixed_gained = torch.mean(W_time_gained, dim=3)
        
        # Apply the estimated spatial weights to the input mixture
        Y_copy = Y.clone()
        Y_copy_gained = Y.clone()

        X_hat_Stage1_C_left, _, _, W_Stage1_norm = beamformingOperationStage1(Y_copy, W_fixed_normalized)
        _, _, _, W_Stage1_gained = beamformingOperationStage1(Y_copy_gained, W_fixed_gained)

        return W_Stage1_norm, W_Stage1_gained, X_hat_Stage1_C_left, Y