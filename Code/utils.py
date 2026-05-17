import torch


def Preprocesing(y, win_len, fs, T, R, device):
    """
    Converts a multi-channel time-domain signal into the STFT domain.

    Truncates the signal to length T (seconds) and applies a Hamming windowed
    Short-Time Fourier Transform (STFT). The output is stored as a real tensor 
    representing complex numbers (real, imag).

    Args:
        y (torch.Tensor): Multi-channel time-domain signal, shape (B, samples, M).
        win_len (int): Length of the STFT window.
        fs (int): Sample rate.
        T (float): Duration in seconds to truncate to.
        R (int): Hop length.
        device (torch.device): Compute device (CPU or CUDA).

    Returns:
        torch.Tensor: Real-valued STFT tensor, shape (B, M, F*2, L).
    """
    B, samples, M = y.size()
    
    # Truncate to maximum T seconds
    if samples > fs * T:
        y = y[:, 0:fs * T, :]                                  
        
    # Reshape to (B*M, samples) for batch STFT processing
    y = y.permute(0, 2, 1).contiguous().view(B * M, -1)         

    w_analysis = torch.hamming_window(win_len, device=device)
    
    # Perform STFT (returns real tensor representation of complex by default in older PyTorch)
    Y = torch.stft(
        y, n_fft=win_len, hop_length=int(R), win_length=win_len, 
        window=w_analysis, center=False, return_complex=False
    )
    B_M, F, L, C = Y.size()      

    # Reshape back to (B, M, F*C, L) where C=2 (real, imag)
    Y = Y.permute(0, 1, 3, 2).contiguous().view(B, M, F * C, L)    
    return Y


def Postprocessing(X_hat, R, win_len, device):
    """
    Converts a signal from the STFT domain back to the time domain.

    Args:
        X_hat (torch.Tensor): STFT domain signal. Can be a pure complex tensor 
                              or a stacked real tensor (real, imag).
        R (int): Hop length.
        win_len (int): Window length.
        device (torch.device): Compute device.

    Returns:
        torch.Tensor: Reconstructed time-domain signal.
    """
    # Ensure that X_hat is a true complex tensor for ISTFT
    if X_hat.is_complex():
        X_hat_complex = X_hat
    else:
        # Create complex tensor from real array split along last dimension
        X_hat_complex = torch.complex(X_hat[..., 0], X_hat[..., 1]) 

    w_analysis = torch.hamming_window(win_len).to(device)
    
    # Inverse STFT transformation
    x_hat = torch.istft(
        X_hat_complex, n_fft=win_len, hop_length=int(R), win_length=win_len, 
        window=w_analysis, center=False, return_complex=False
    )

    return x_hat


def return_as_complex(Y_stft):
    """
    Helper function to convert a stacked real STFT tensor back to true complex format.
    """
    B, M, F, L = Y_stft.size()
    Y_STFT = Y_stft.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()
    Y = torch.view_as_complex(Y_STFT)
    return Y


def true_rtf_from_rirs_bmk(rirs_bmk, win_len, ref_mic, use_window=True, eps=1e-12):
    """
    Calculates the Oracle Relative Transfer Function (RTF) safely to prevent NaN errors.
    
    Mathematically safe complex division algorithm: 
    Instead of simply dividing H_target by H_ref (which causes NaN if H_ref is close to 0),
    we multiply numerator and denominator by the complex conjugate of H_ref:
    RTF = (H_target * conj(H_ref)) / (|H_ref|^2 + eps)

    Args:
        rirs_bmk (torch.Tensor): Room Impulse Responses, shape (Batch, Mics, Time).
        win_len (int): STFT window length.
        ref_mic (int): Index of the reference microphone.
        use_window (bool): Apply a Hamming window to the RIR before STFT.
        eps (float): Small constant to prevent division by zero.

    Returns:
        torch.Tensor: Safe Oracle RTFs, shape (B, M, F).
    """
    B, M, K = rirs_bmk.shape
    device = rirs_bmk.device
    
    window = torch.hamming_window(win_len, periodic=True, device=device) if use_window else None
    h_flat = rirs_bmk.view(B * M, K)
    
    H_stft = torch.stft(
        h_flat, n_fft=win_len, hop_length=win_len, win_length=win_len,
        window=window, center=False, pad_mode='constant', return_complex=True
    ) 
    
    # We only care about the first time frame of the RIR STFT
    H = H_stft[:, :, 0].view(B, M, 257)
    
    # Extract the response at the reference microphone
    H_ref = H[:, ref_mic, :].unsqueeze(1)
    
    # Safe Complex Division: (a * conj(b)) / (|b|^2 + eps)
    power_ref = torch.real(H_ref * torch.conj(H_ref)) + eps
    A_true = (H * torch.conj(H_ref)) / power_ref

    return A_true