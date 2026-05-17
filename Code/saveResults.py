import os
from datetime import datetime
import scipy.io as sio


def saveResults(
    Y,
    FIRST_SPEAKER_stft,
    SECOND_SPEAKER_stft,
    W_Stage1_left,
    X_hat_Stage1_C_left,
    y,
    first_speaker,
    second_speaker,
    x_hat_stage1_left,
    results_path,
    i,
    fs,
    THIRD_SPEAKER_stft=None,  # Optional (Only passed if 3 speakers)
    third_speaker=None,       # Optional (Only passed if 3 speakers)
):
    """
    Saves STFT-domain and time-domain neural network outputs as .mat files.
    Dynamically scales to support either 2 or 3 speaker acoustic scenarios.

    This function safely detaches PyTorch tensors from the GPU computation graph, 
    moves them to the CPU, converts them to NumPy arrays, and exports them.

    Args:
        Y (torch.Tensor): Mixture signal in the STFT domain.
        FIRST_SPEAKER_stft (torch.Tensor): Clean first speaker in the STFT domain.
        SECOND_SPEAKER_stft (torch.Tensor): Clean second speaker in the STFT domain.
        W_Stage1_left (torch.Tensor): Complex spatial beamformer weights.
        X_hat_Stage1_C_left (torch.Tensor): Enhanced signal in the STFT domain.
        y (torch.Tensor): Mixture signal in the time domain.
        first_speaker (torch.Tensor): Clean first speaker in the time domain.
        second_speaker (torch.Tensor): Clean second speaker in the time domain.
        x_hat_stage1_left (torch.Tensor): Enhanced signal in the time domain.
        results_path (str): Directory path where the .mat files will be saved.
        i (int): Current batch or iteration index for unique file naming.
        fs (int): Sample rate of the audio (e.g., 16000).
        THIRD_SPEAKER_stft (torch.Tensor, optional): Clean third speaker (STFT). Defaults to None.
        third_speaker (torch.Tensor, optional): Clean third speaker (Time). Defaults to None.
    """

    # Ensure the target directory exists; create it with full read/write permissions if missing
    os.makedirs(results_path, mode=0o777, exist_ok=True)
    
    # Generate a unique timestamp string to prevent accidental file overwriting
    now = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    # ==========================================
    # 1. Export STFT Domain Data
    # ==========================================
    
    # Detach from graph -> move to CPU -> convert to NumPy array
    stft_dict = {
        "Y_STFT": Y.cpu().detach().numpy(),
        "FIRST_SPEAKER_STFT": FIRST_SPEAKER_stft.cpu().detach().numpy(),
        "SECOND_SPEAKER_STFT": SECOND_SPEAKER_stft.cpu().detach().numpy(),
        "W_Stage1_left": W_Stage1_left.cpu().detach().numpy(),
        "X_hat_Stage1_C_left": X_hat_Stage1_C_left.cpu().detach().numpy(),
        "fs": fs,
        "index": i,
        "timestamp_str": now,
    }
    
    # Dynamically append the 3rd speaker data only if it was provided
    if THIRD_SPEAKER_stft is not None:
        stft_dict["THIRD_SPEAKER_STFT"] = THIRD_SPEAKER_stft.cpu().detach().numpy()

    # Construct the file path and save the STFT dictionary
    stft_filename = os.path.join(results_path, f"TEST_STFT_domain_results_{now}_{i}.mat")
    sio.savemat(stft_filename, stft_dict)

    # ==========================================
    # 2. Export Time Domain Data
    # ==========================================
    
    time_dict = {
        "y": y.cpu().detach().numpy(),
        "first_speaker": first_speaker.cpu().detach().numpy(),
        "second_speaker": second_speaker.cpu().detach().numpy(),
        "x_hat_stage1_left": x_hat_stage1_left.cpu().detach().numpy(),
        "fs": fs,
        "index": i,
        "timestamp_str": now,
    }
    
    # Dynamically append the 3rd speaker data only if it was provided
    if third_speaker is not None:
        time_dict["third_speaker"] = third_speaker.cpu().detach().numpy()

    # Construct the file path and save the Time domain dictionary
    time_filename = os.path.join(results_path, f"TEST_time_domain_results_{now}_{i}.mat")
    sio.savemat(time_filename, time_dict)