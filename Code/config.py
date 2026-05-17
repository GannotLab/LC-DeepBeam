"""
Configuration Schema Definition.

This module uses Python dataclasses to define a strictly-typed configuration 
schema for the DNN-Based Beamformer project. This schema is consumed by Hydra 
to parse the 'config.yaml' file, ensuring all hyperparameters and paths are 
type-checked before the training or evaluation scripts run.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Paths:
    """
    Defines all file system paths and environment flags required for data 
    loading and model saving.
    """
    rev_env: bool
    train_path: str
    test_path: str
    babble_noise_train_path: str
    babble_noise_test_path: str
    results_path: str
    modelData_path: str
    log_path: str


@dataclass
class Params:
    """
    Defines the core acoustic and signal processing parameters.
    """
    use_3_speakers: int          # Flag (1 or 0) indicating 3-speaker vs 2-speaker mode
    one_pass_two_nulls: int      # Flag (1 or 0) for beamformer constraint logic
    win_length: int              # STFT window length
    R: int                       # STFT hop length
    T: int                       # Audio truncation length in seconds
    M: int                       # Number of microphones
    fs: int                      # Sample rate (Hz)
    ck: int                      # Channel/Feature dimensions
    mic_ref: int                 # Index of the reference microphone (1-based)
    noise_only_time: float       # Duration (seconds) of initial noise-only segment


@dataclass
class ModelParams:
    """
    Defines the architectural hyperparameters for the U-Net spatial estimator.
    """
    EnableSkipAttention: int     # Flag to toggle attention mechanisms in skip connections
    activationStage1: str        # Activation function (e.g., 'relu', 'leaky_relu')
    channelsStage1: int          # Base channel multiplier for the U-Net feature maps
    use_true_rirs: int           # Flag to use Oracle True RTFs vs Estimated RTFs
    DUAL_MODEL: int              # Flag indicating if dual-input architecture is active


@dataclass
class Device:
    """
    Defines the hardware target for PyTorch computation.
    """
    device_num: int              # Index of the CUDA GPU (e.g., 0, 1)


@dataclass
class Loss:
    """
    Defines the schedules and weights for the custom multi-objective loss function.
    These use 'Any' typing to allow lists of dictionaries representing schedules.
    """
    lambda_sisdr_schedule: Any
    lambda_pass_schedule_2spk: Any
    lambda_null_schedule_2spk: Any
    lambda_pass_schedule_3spk: Any
    lambda_null1_schedule_3spk: Any
    lambda_null2_schedule_3spk: Any


@dataclass
class Model_HP:
    """
    Defines the high-level training loop hyperparameters.
    """
    train_size_spilt: float      # Proportion of dataset used for training
    val_size_spilt: float        # Proportion of dataset used for validation
    epochs: int                  # Total number of training epochs
    data_loader_shuffle: bool    # Toggle shuffling in training dataloader
    test_loader_shuffle: bool    # Toggle shuffling in testing dataloader
    batchSize_2spk: int          # Batch size for 2-speaker architectures
    batchSize_3spk: int          # Batch size for 3-speaker architectures


@dataclass
class Optimizer:   
    """
    Defines the optimizer settings and learning rates.
    """
    optimizer: str               # Name of the optimizer (e.g., 'Adam')
    weight_decay: float          # L2 regularization factor
    learning_rate_2spk: float    # Learning rate for 2-speaker architectures
    learning_rate_3spk: float    # Learning rate for 3-speaker architectures


@dataclass 
class Dataset:
    """
    Defines the paths to the CSV files containing the acoustic room geometries 
    and RIR parameter tracking.
    """
    df_path_train_2spk: str   
    df_path_test_2spk: str 
    df_path_train_3spk: str   
    df_path_test_3spk: str 


@dataclass
class Wandb:
    """
    Defines the Weights & Biases (wandb) configuration for experiment tracking.
    """
    project_name: str
    entity_2spk: str             # WandB entity/team name for 2-speaker runs
    entity_3spk: str             # WandB entity/team name for 3-speaker runs


@dataclass 
class CUNETConfig:
    """
    The Master Configuration Node.
    Composes all sub-schemas into a single hierarchical structure.
    """
    SavedModel: int              # 0 = Train mode, 1 = Evaluation mode
    params: Params
    modelParams: ModelParams
    device: Device
    loss: Loss
    data_set_path: str           # Root path to audio dataset
    model_path_2spk: str         # Checkpoint save path for 2-speaker model
    folder_name_2spk: str        # Output directory name for 2-speaker results
    model_path_3spk: str         # Checkpoint save path for 3-speaker model
    folder_name_3spk: str        # Output directory name for 3-speaker results
    paths: Paths
    wandb: Wandb
    model_hp: Model_HP
    optimizer: Optimizer
    dataset: Dataset