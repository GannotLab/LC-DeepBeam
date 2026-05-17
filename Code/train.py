import torch
import wandb
from tqdm import tqdm

# Local imports
from utils import Preprocesing, Postprocessing, true_rtf_from_rirs_bmk
from test import test
from loss_function import compute_loss


def train(
    model,
    args,
    results_path,
    train_loader,
    val_loader,
    optimizer,
    device,
    cfg_loss,
    epoch,
):
    """
    Executes a single training epoch for the U-Net spatial weight estimator.

    Handles dynamic batch unpacking (2-speaker vs. 3-speaker scenarios), 
    computes STFT representations, calculates true Oracle RTFs from Room 
    Impulse Responses (RIRs), executes the forward pass, and performs 
    backpropagation with gradient clipping.

    Args:
        model (nn.Module): The U-Net neural network model.
        args (object): Configuration arguments (sample rate, window length, etc.).
        results_path (str): Path to save evaluation output (passed to validation).
        train_loader (DataLoader): PyTorch DataLoader for the training dataset.
        val_loader (DataLoader): PyTorch DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): The optimizer (e.g., Adam).
        device (torch.device): Compute device (CPU or CUDA).
        cfg_loss (object): Configuration object containing loss schedules and weights.
        epoch (int): The current training epoch index.

    Returns:
        tuple: (Total accumulated training loss for the epoch, Validation loss).
    """
    fs = args.fs
    win_len = args.win_length
    T = args.T
    R = eval(args.R)
    mic_ref = args.mic_ref

    epoch_train_loss = 0.0
    
    # Set the model to training mode (enables dropout, batch norm updates, etc.)
    model.train()

    # Catch the batch as a single tuple to dynamically unpack it based on size
    for _, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

        # ==========================================
        # 1. DYNAMIC UNPACKING & DATA TRANSFER
        # ==========================================
        if len(batch) == 10:
            use_3_speakers = True
            (y, labels_x, first_speaker, second_speaker, third_speaker, 
             babble, white_noise, rir_first, rir_second, rir_third) = batch
        elif len(batch) == 8:
            use_3_speakers = False
            (y, labels_x, first_speaker, second_speaker, 
             babble, white_noise, rir_first, rir_second) = batch
            third_speaker = None
            rir_third = None
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}. Check dataset __getitem__.")

        # Push arrays to the target compute device
        y = y.to(device)
        first_speaker = first_speaker.to(device)
        second_speaker = second_speaker.to(device)
        if use_3_speakers:
            third_speaker = third_speaker.to(device)

        # ==========================================
        # 2. PREPROCESSING & RTF CALCULATION
        # ==========================================
        
        # Convert the time-domain mixture into the complex STFT domain
        Y = Preprocesing(y, win_len, fs, T, R, device)

        # Calculate the True Oracle Relative Transfer Functions (RTFs) from the RIRs
        rtf_target = true_rtf_from_rirs_bmk(rir_first, win_len=win_len, ref_mic=mic_ref-1).to(device)
        rtf_null1 = true_rtf_from_rirs_bmk(rir_second, win_len=win_len, ref_mic=mic_ref-1).to(device)
        
        if use_3_speakers:
            rtf_null2 = true_rtf_from_rirs_bmk(rir_third, win_len=win_len, ref_mic=mic_ref-1).to(device)
        else:
            rtf_null2 = None

        # ==========================================
        # 3. FORWARD PASS
        # ==========================================
        
        if use_3_speakers:
            W, W_GAINED, X_hat_stft, _ = model(Y, rir_first, rir_second, rir_third, device, mode="train")
        else:
            # We don't pass rir_third to the 2-speaker model architecture
            W, W_GAINED, X_hat_stft, _ = model(Y, rir_first, rir_second, device=device, mode="train")

        # Convert the estimated STFT back to the time domain for SI-SDR calculation
        X_hat_time = Postprocessing(X_hat_stft, R, win_len, device)

        # Extract the ground truth reference signal at the reference microphone
        x_target = first_speaker[:, :, mic_ref - 1]

        # ==========================================
        # 4. LOSS COMPUTATION & BACKPROPAGATION
        # ==========================================
        
        # Compute loss (Dynamically handles 2 or 3 nulls inside the function)
        loss, logs = compute_loss(
            epoch=epoch,
            X_hat_time=X_hat_time,
            X_target_time=x_target,
            W=W,
            W_GAINED=W_GAINED,
            steer_pass=rtf_target,
            steer_null_1=rtf_null1,
            steer_null_2=rtf_null2,
            cfg=cfg_loss,
        )

        # Clear old gradients and backpropagate the new error
        optimizer.zero_grad()
        loss.backward()
        
        # --- Calculate Total Gradient Norm (For tracking instability) ---
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        # --- Apply Gradient Clipping ---
        # Prevents exploding gradients by capping the max norm at 15.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0)

        # --- Get Applied Target Normalization Gain ---
        # Handle DataParallel wrapping if training on multiple GPUs
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        if hasattr(base_model, 'unet_multiChannel_left'):
            current_gain = base_model.unet_multiChannel_left.get_applied_gain()
        else:
            current_gain = 1.0

        # --- Log Metrics to Weights & Biases ---
        wandb.log({
            "train/total_loss": loss.item(),
            "train/grad_norm": total_grad_norm,
            "train/applied_gain": current_gain,
            **logs
        })

        # Update weights
        optimizer.step()
        epoch_train_loss += loss.item()

    # ==========================================
    # 5. VALIDATION PHASE
    # ==========================================
    # Run the testing script on the validation dataset (save output flag set to 0)
    epoch_val_loss = test(
        model, args, results_path, val_loader, device, cfg_loss, 0
    )

    return epoch_train_loss, epoch_val_loss