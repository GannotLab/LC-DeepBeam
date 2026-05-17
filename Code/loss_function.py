import torch

EPS = 1e-8


def si_sdr(est, ref, eps=EPS):
    """
    Computes the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).

    SI-SDR measures the quality of separated/enhanced speech by projecting the 
    estimated signal onto the reference signal, making it invariant to volume 
    (scale) differences between the two.

    Args:
        est (torch.Tensor): Estimated speech signal, shape (B, T).
        ref (torch.Tensor): Ground truth reference signal, shape (B, T).
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: SI-SDR values in decibels (dB), shape (B,).
    """
    # 1. Zero-mean center the signals to remove DC bias
    ref = ref - torch.mean(ref, dim=-1, keepdim=True)
    est = est - torch.mean(est, dim=-1, keepdim=True)

    # 2. Calculate scaling factor alpha: <est, ref> / <ref, ref>
    dot_product = torch.sum(est * ref, dim=-1, keepdim=True)
    ref_energy = torch.sum(ref**2, dim=-1, keepdim=True) + eps
    
    # 3. Project the estimated signal onto the reference signal
    proj = (dot_product / ref_energy) * ref
    
    # 4. Calculate the residual (the distortion/noise)
    noise = est - proj

    # 5. Calculate power ratio between the projection and the noise
    proj_power = torch.sum(proj**2, dim=-1)
    noise_power = torch.sum(noise**2, dim=-1)

    # 6. Log-scale with clamping for numerical stability
    ratio = (proj_power + eps) / (noise_power + eps)
    
    return 10.0 * torch.log10(ratio)


# =========================
# Beamformer Constraints
# =========================

def pass_constraint(W, steer):
    """
    Calculates the physical beamformer pass-constraint penalty.
    
    Encourages the spatial weights to perfectly preserve the target direction
    by aiming for a spatial response (W^H * steer) of 1.0.

    Args:
        W (torch.Tensor): Complex spatial weights, shape (B, M, F, T).
        steer (torch.Tensor): Relative Transfer Function (RTF) of the target, 
                              shape (B, M, F).

    Returns:
        torch.Tensor: Mean Squared Error (MSE) against a perfect 1.0 response.
    """
    # Expand RTF to match the time dimension of W
    steer = steer.unsqueeze(-1).to(W.device)
    
    # Calculate the spatial response: H = W^H * a
    inner = torch.sum(torch.conj(W) * steer, dim=1)  # Sum across M (microphones)
    
    # Penalty: How far the response is from a perfect 1.0
    return torch.mean(torch.abs(inner - 1.0) ** 2)


def null_constraint(W, steer):
    """
    Calculates the physical beamformer null-constraint penalty (Linear Scale).
    
    Encourages the spatial weights to suppress an interference direction
    by aiming for a spatial response (W^H * steer) of 0.0.

    Args:
        W (torch.Tensor): Complex spatial weights, shape (B, M, F, T).
        steer (torch.Tensor): RTF of the interferer, shape (B, M, F).

    Returns:
        torch.Tensor: Mean Squared Error (MSE) against a perfect 0.0 response.
    """
    # Expand RTF to match the time dimension of W
    steer = steer.unsqueeze(-1).to(W.device)  # (B, M, F, 1)
    
    # Calculate the spatial response: H = W^H * a
    inner = torch.sum(torch.conj(W) * steer, dim=1)
    
    # Penalty: Absolute power of the residual response
    return torch.mean(torch.abs(inner) ** 2)


def null_constraint_db(W, steer, eps=1e-10):
    """
    Calculates the physical beamformer null-constraint penalty (Log Scale).
    
    Optimizes the null constraint directly in decibels (dB), which often yields
    deeper mathematical nulls during gradient descent compared to linear scale.

    Args:
        W (torch.Tensor): Complex spatial weights, shape (B, M, F, T).
        steer (torch.Tensor): RTF of the interferer, shape (B, M, F).
        eps (float): Small constant to prevent log10(0).

    Returns:
        torch.Tensor: Average suppression level in dB.
    """
    # Expand RTF to match the time dimension of W
    steer = steer.unsqueeze(-1).to(W.device) 
    
    # Calculate the spatial response: H = W^H * a
    inner = torch.sum(torch.conj(W) * steer, dim=1) 
    power_response = torch.abs(inner)**2
    
    # Calculate mean response in dB-like log scale
    log_null = 10.0 * torch.log10(torch.mean(power_response, dim=-1) + eps)
    
    return torch.mean(log_null)


def get_scheduled_value(schedule, epoch):
    """
    Retrieves the active hyperparameter value for a given epoch based on a schedule.

    Args:
        schedule (list): List of dictionaries, e.g., [{'epoch': int, 'value': float}, ...].
        epoch (int): Current training epoch.

    Returns:
        float: The scheduled value for the current epoch.
    """
    value = schedule[0]["value"]
    for s in schedule:
        if epoch >= s["epoch"]:
            value = s["value"]
        else:
            # Schedule lists are assumed to be sorted by epoch; break early if exceeded
            break
            
    return value


# =========================
# Final Unified Loss
# =========================

def compute_loss(
    epoch,
    X_hat_time,
    X_target_time,
    W=None,
    W_GAINED=None,
    steer_pass=None,
    steer_null_1=None, 
    steer_null_2=None, 
    cfg=None,
):
    """
    Computes the unified loss function combining SI-SDR and spatial constraints.
    
    Dynamically routes hyperparameters based on whether the architecture is 
    configured for 2 speakers (1 null) or 3 speakers (2 nulls).

    Args:
        epoch (int): Current training epoch for schedule lookups.
        X_hat_time (torch.Tensor): Network's enhanced output in time-domain.
        X_target_time (torch.Tensor): Ground truth target speech in time-domain.
        W (torch.Tensor): Raw, unnormalized spatial weights from the U-Net.
        W_GAINED (torch.Tensor): Gain-normalized spatial weights.
        steer_pass (torch.Tensor): True/Estimated RTF of the target speaker.
        steer_null_1 (torch.Tensor): RTF of the first interferer.
        steer_null_2 (torch.Tensor, optional): RTF of the second interferer (None for 2-speaker).
        cfg (object): Loss configuration object containing hyperparameter schedules.

    Returns:
        tuple: (Total combined loss scalar, Dictionary of individual logging metrics)
    """
    # Base metric weight
    lambda_sisdr = get_scheduled_value(cfg.lambda_sisdr_schedule, epoch)
    
    # --- DYNAMIC SCHEDULE ROUTING ---
    if steer_null_2 is not None:
        # 3-SPEAKER mode (1 Target, 2 Nulls)
        lambda_pass = get_scheduled_value(cfg.lambda_pass_schedule_3spk, epoch)
        lambda_null_1 = get_scheduled_value(cfg.lambda_null1_schedule_3spk, epoch)
        lambda_null_2 = get_scheduled_value(cfg.lambda_null2_schedule_3spk, epoch)
    else:
        # 2-SPEAKER mode (1 Target, 1 Null)
        lambda_pass = get_scheduled_value(cfg.lambda_pass_schedule_2spk, epoch)
        lambda_null_1 = get_scheduled_value(cfg.lambda_null_schedule_2spk, epoch)

    logs = {}

    # --- 1. SI-SDR Objective (Primary Loss) ---
    sdr_target = si_sdr(X_hat_time, X_target_time)
    
    # PyTorch minimizes loss, so we negate SI-SDR (higher SI-SDR = lower loss)
    loss = -lambda_sisdr * sdr_target.mean()
    logs["si_sdr_target"] = sdr_target.mean().item()
    
    # --- 2. Pass Constraint (Target preservation) ---
    # Applied using the GAINED weights to ensure volume preservation
    if W_GAINED is not None and steer_pass is not None:
        c_pass = pass_constraint(W_GAINED, steer_pass)
        loss = loss + lambda_pass * c_pass
        logs["c_pass"] = c_pass.item()

    # --- 3. Null Constraints (Interference suppression) ---
    # Applied using RAW weights to allow deep mathematical cancellation
    if W is not None:
        if steer_null_1 is not None:
            cn1 = null_constraint_db(W, steer_null_1)
            loss = loss + lambda_null_1 * cn1
            
            # Log key changes dynamically based on the number of speakers
            log_name = "c_null_1_db" if steer_null_2 is not None else "c_null_db"
            logs[log_name] = cn1.item()

        if steer_null_2 is not None:
            cn2 = null_constraint_db(W, steer_null_2)
            loss = loss + lambda_null_2 * cn2
            logs["c_null_2_db"] = cn2.item()

        # Track the raw magnitude of the weights for stability monitoring
        logs["weight_abs"] = W.abs().mean().item()

    # Save final composite loss for wandb tracking
    logs["train_loss"] = loss.item()
    
    return loss, logs