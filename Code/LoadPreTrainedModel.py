"""
Model Loading Utility.

This module handles safely loading a pre-trained PyTorch model from disk.
It includes logic to safely decouple models that were trained and saved
across multiple GPUs (DataParallel) so they can be evaluated on a single GPU
or CPU without raising tensor mapping errors.
"""

import torch

# Local imports
from ExNetBFPFModel import ExNetBFPF


def loadPreTrainedModel(cfg):
    """
    Loads a pre-trained ExNetBFPF model from disk.

    Instantiates a fresh model architecture based on the configuration parameters,
    locates the saved weights file, strips any multi-GPU (DataParallel) wrappers
    from the saved state dictionary, and applies the weights to the new model.

    Args:
        cfg (object): Hydra configuration object containing paths and model parameters.

    Returns:
        nn.Module: The fully loaded PyTorch model ready for inference.
    """
    # Instantiate the raw architecture dynamically based on 2-speaker vs 3-speaker config
    model = ExNetBFPF(cfg.modelParams, cfg.params)

    # The master script dynamically sets modelData_path to the correct sub-folder
    PATH = cfg.paths.modelData_path + 'trained_model_latest.pt'

    # Default loading to CPU to prevent CUDA memory spikes or mapping mismatches
    device = torch.device("cpu")
    
    # 1. Load the checkpoint directly to the CPU to wipe its memory of old GPU assignments
    checkpoint = torch.load(PATH, map_location='cpu', weights_only=False)
    
    # 2. Extract the raw state_dict based on how the model was saved
    
    # If it was saved wrapped in DataParallel (multi-GPU)
    if isinstance(checkpoint, torch.nn.DataParallel):
        state_dict = checkpoint.module.state_dict()
        
    # If it was saved purely as a state_dict dictionary
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        
    # If it was saved as a standard single-GPU nn.Module object
    else:
        state_dict = checkpoint.state_dict()
    
    # Load the cleaned weights into the instantiated architecture
    model.load_state_dict(state_dict)
    
    # Move the model to the correct device after loading
    model.to(device)  
    
    return model