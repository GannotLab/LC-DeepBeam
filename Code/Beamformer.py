"""
Mathematical Beamforming & Masking Operations.

This module contains the core PyTorch operations for applying spatial weights 
and spectral masks to the audio signals in the STFT domain. It handles the 
necessary tensor reshaping to convert between real-stacked arrays and true 
complex arrays.
"""

import torch


def beamformingOperationStage1(Y, W):
    """
    Applies spatial beamforming weights to the input mixture signal.
    
    Supports both time-invariant weights (static across the sequence) and 
    time-varying weights (dynamic tracking).

    Args:
        Y (torch.Tensor): The noisy input signal in the stacked real STFT domain.
                          Shape: (B, M, F, L)
        W (torch.Tensor): The estimated spatial weights.
                          Can be Time-invariant: (B, M, F)
                          or Time-varying:       (B, M, F, L)

    Returns:
        tuple: 
            - X_hat_complex (torch.Tensor): Estimated signal (complex), shape (B, F/2, L).
            - X_hat (torch.Tensor): Estimated signal (real-valued stacked), shape (B, 1, F, L).
            - Y (torch.Tensor): Noisy input in complex form, shape (B, M, F/2, L).
            - W (torch.Tensor): Weights in complex form, shape (B, M, F/2, L).
    """
    # Reshape mixture from stacked real/imag to true complex
    B, M, F, L = Y.size()  
    Y = Y.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()  
    Y = torch.view_as_complex(Y)                                     

    # Handle temporal expansion depending on whether weights are static or dynamic
    if W.dim() == 3:  
        # Time-invariant weights: (B, M, F)
        W = W.view(B, M, F // 2, 2)                                
        W = torch.view_as_complex(W)                             
        W = W.unsqueeze(-1).expand(-1, -1, -1, L)                
    elif W.dim() == 4:  
        # Time-varying weights: (B, M, F, L)
        W = W.view(B, M, F // 2, 2, L).permute(0, 1, 2, 4, 3).contiguous()  
        W = torch.view_as_complex(W)                                     
    else:
        raise ValueError("W must have shape (B, M, F) or (B, M, F, L)")

    # Execute Beamforming: W^H * Y
    X_hat = torch.conj(W) * Y                                     
    X_hat_complex = torch.sum(X_hat, dim=1)                       
    
    # Convert the complex result back to real-stacked format
    X_hat = torch.view_as_real(X_hat_complex)                    

    # Reformat back to (B, 1, F, L) real-valued form
    B, Fh, L, C = X_hat.size()
    X_hat = X_hat.permute(0, 1, 3, 2).contiguous().view(B, 1, Fh * C, L)  

    return X_hat_complex, X_hat, Y, W
