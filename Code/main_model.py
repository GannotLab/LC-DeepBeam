# import torch.multiprocessing as mp
# if __name__ == "__main__":
#     try:
#         mp.set_start_method("spawn", force=True)
#     except RuntimeError:
#         pass
# (Use this for reverberant because of gpuRIR)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
from generate_dataset import GeneratedData_Dynamic_Speakers_Babble 
import hydra
import os
from datetime import datetime
from config import CUNETConfig 
from hydra.core.config_store import ConfigStore
from ExNetBFPFModel import ExNetBFPF
from LoadPreTrainedModel import loadPreTrainedModel
from train import train
from test import test
import wandb
from tqdm import tqdm
import copy
import sys
from io import StringIO

# Register the structured configuration schema with Hydra
cs = ConfigStore.instance()
cs.store(name="cunet_config", node=CUNETConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: CUNETConfig):
    """
    Main entry point for the DNN-Based Beamformer project.

    This script handles two primary modes of operation based on 'cfg.SavedModel':
        0: Training Mode - Initializes the U-Net model, configures optimizers, 
           runs the epoch loop, logs to WandB, and saves model checkpoints.
        1: Evaluation Mode - Iterates through a predefined suite of acoustic 
           scenarios, loads specific trained weights, runs inference, and 
           generates a master summary text file.

    Args:
        cfg (CUNETConfig): The strongly-typed Hydra configuration object.
    """
    device_ids = [cfg.device.device_num]
    
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    if cfg.SavedModel == 0:
        # ====================================================================
        # TRAINING MODE
        # ====================================================================
        print("\n[START] Executing Training Mode...\n")
        use_3_speakers = getattr(cfg.params, 'use_3_speakers', 0) == 1
        
        # Route hyperparameters and paths based on the acoustic architecture
        if use_3_speakers:
            batchSize = cfg.model_hp.batchSize_3spk
            learning_rate = cfg.optimizer.learning_rate_3spk
            active_entity = cfg.wandb.entity_3spk
            cfg.paths.results_path = f"{cfg.folder_name_3spk}/"
            cfg.paths.modelData_path = f"{cfg.model_path_3spk}/"
            cfg.paths.log_path = f"/logs/{cfg.folder_name_3spk}/"
        else:
            batchSize = cfg.model_hp.batchSize_2spk
            learning_rate = cfg.optimizer.learning_rate_2spk
            active_entity = cfg.wandb.entity_2spk
            cfg.paths.results_path = f"{cfg.folder_name_2spk}/"
            cfg.paths.modelData_path = f"{cfg.model_path_2spk}/"
            cfg.paths.log_path = f"/logs/{cfg.folder_name_2spk}/"

        # Initialize Weights & Biases for experiment tracking
        wandb.init(
            entity=active_entity, 
            project="DNN Based Beamformer", 
            config={
                "learning_rate": learning_rate, 
                "epochs": cfg.model_hp.epochs, 
                "batch_size": batchSize, 
                "use_3_speakers": use_3_speakers
            }
        )

        # Prepare datasets and dataloaders
        train_data = GeneratedData_Dynamic_Speakers_Babble(cfg, mode='train')
        train_set, val_set = torch.utils.data.random_split(
            train_data, 
            [cfg.model_hp.train_size_spilt, cfg.model_hp.val_size_spilt]
        )  
        train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=cfg.model_hp.data_loader_shuffle, num_workers=16)  # 4 for reverberant, 16 for non-reverberant
        val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=cfg.model_hp.data_loader_shuffle, num_workers=16)  # 4 for reverberant, 16 for non-reverberant
        
        # Initialize model and push to target device
        model = ExNetBFPF(cfg.modelParams, cfg.params)  
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg.optimizer.weight_decay)
        
        train_loss, val_loss = [], []
        
        # Epoch loop
        for epoch in tqdm(range(cfg.model_hp.epochs)):
            # Execute one epoch of training and validation
            epoch_train_loss, epoch_val_loss = train(
                model, cfg.params, cfg.paths.results_path, train_loader, 
                val_loader, optimizer, device, cfg.loss, epoch
            ) 
            
            # Average losses over the batches
            train_loss.append(epoch_train_loss / len(train_loader)) 
            val_loss.append(epoch_val_loss / len(val_loader)) 
            
            print(f"Epoch {epoch+1}/{cfg.model_hp.epochs}, train_loss: {train_loss[epoch]}, val_loss: {val_loss[epoch]}")
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss[epoch], "val_loss": val_loss[epoch]})

            # Save the latest model checkpoint
            os.makedirs(cfg.paths.modelData_path, mode=0o777, exist_ok=True)
            torch.save(model, os.path.join(cfg.paths.modelData_path, 'trained_model_latest.pt'))
            
            # Save periodic historical checkpoints every 15 epochs
            if (epoch + 1) % 15 == 0:
                torch.save(model, os.path.join(cfg.paths.modelData_path, f'trained_model_epoch_{epoch+1}.pt'))
                print(f"--> Saved checkpoint at Epoch {epoch+1}")

    else:
        # ====================================================================
        # ONE-CLICK MASTER EVALUATION MODE
        # ====================================================================
        
        SUMMARY_FILE = f"/home/dsi/engelba3/DNN_Based_Beamformer/Code/Final_Results_Summary_{cfg.params.T}s.txt"
        
        def dual_print(text):
            """Helper function to print to terminal and append to summary text file."""
            print(text)
            with open(SUMMARY_FILE, "a") as f:
                f.write(text + "\n")

        # Initialize the summary file header
        with open(SUMMARY_FILE, "w") as f:
            f.write(f"{'='*70}\nMASTER EVALUATION SUITE (T = {cfg.params.T}s)\n"
                    f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}\n\n")

        dual_print(f"\n{'='*70}\n[START] Executing Master Evaluation Suite\n{'='*70}\n")
        
        # Define the testing matrix (14 acoustic scenarios)
        test_configurations = [
            {"name": "2spkNoWithout", "use_3_speakers": 0, "rev_env": False, "DUAL_MODEL": 0, "use_true_rirs": 1},
            {"name": "2spkTrueWithout", "use_3_speakers": 0, "rev_env": False, "DUAL_MODEL": 1, "use_true_rirs": 1},
            {"name": "2spkEstWithout", "use_3_speakers": 0, "rev_env": False, "DUAL_MODEL": 1, "use_true_rirs": 0},
            {"name": "2spkNoWith", "use_3_speakers": 0, "rev_env": True,  "DUAL_MODEL": 0, "use_true_rirs": 1},
            {"name": "2spkTrueWith", "use_3_speakers": 0, "rev_env": True,  "DUAL_MODEL": 1, "use_true_rirs": 1},
            {"name": "2spkEstWith", "use_3_speakers": 0, "rev_env": True,  "DUAL_MODEL": 1, "use_true_rirs": 0},
            {"name": "3spkNoWithout", "use_3_speakers": 1, "rev_env": False, "DUAL_MODEL": 0, "use_true_rirs": 1},
            {"name": "3spkTrueWithout", "use_3_speakers": 1, "rev_env": False, "DUAL_MODEL": 1, "use_true_rirs": 1},
            {"name": "3spkEstWithout", "use_3_speakers": 1, "rev_env": False, "DUAL_MODEL": 1, "use_true_rirs": 0},
            {"name": "3spkNoWith", "use_3_speakers": 1, "rev_env": True,  "DUAL_MODEL": 0, "use_true_rirs": 1},
            {"name": "3spkTrueWith", "use_3_speakers": 1, "rev_env": True,  "DUAL_MODEL": 1, "use_true_rirs": 1},
            {"name": "3spkEstWith", "use_3_speakers": 1, "rev_env": True,  "DUAL_MODEL": 1, "use_true_rirs": 0},
            {"name": "3spkNoWithoutNOP", "use_3_speakers": 1, "rev_env": False,  "DUAL_MODEL": 0, "use_true_rirs": 1},
            {"name": "3spkTrueWithoutNOP", "use_3_speakers": 1, "rev_env": False,  "DUAL_MODEL": 1, "use_true_rirs": 1}
        ]

        BASE_MODEL_PATH = "/home/dsi/engelba3/DNN_Based_Beamformer/Code/Trained_Models/forthepaper/"
        BASE_RESULT_PATH = "/home/dsi/engelba3/DNN_Based_Beamformer/Code/RESULTS_FINAL/"

        # Iterate over all defined acoustic configurations
        for idx, t_cfg in enumerate(test_configurations):
            scenario_header = f"\n{'*'*60}\n>>> {idx+1}. RUNNING TEST SCENARIO: {t_cfg['name']}\n{'*'*60}"
            dual_print(scenario_header)
            
            # Deepcopy config to safely alter properties for the current scenario
            run_cfg = copy.deepcopy(cfg)
            run_cfg.params.use_3_speakers = t_cfg["use_3_speakers"]
            run_cfg.paths.rev_env = t_cfg["rev_env"]
            run_cfg.modelParams.DUAL_MODEL = t_cfg["DUAL_MODEL"]
            run_cfg.modelParams.use_true_rirs = t_cfg["use_true_rirs"]
            
            # Setup input/output paths for the specific configuration
            run_cfg.paths.modelData_path = os.path.join(BASE_MODEL_PATH, t_cfg["name"], "")
            run_cfg.paths.results_path = os.path.join(BASE_RESULT_PATH, t_cfg["name"], "")
            os.makedirs(run_cfg.paths.results_path, exist_ok=True)
            
            model_file = os.path.join(run_cfg.paths.modelData_path, 'trained_model_latest.pt')
            if not os.path.exists(model_file):
                dual_print(f"[WARNING] Model not found at {model_file}. Skipping {t_cfg['name']}...\n")
                continue

            # Load the corresponding pre-trained model and push to GPU
            model = loadPreTrainedModel(run_cfg)
            model.to(device)
            
            # Adjust batch size based on memory requirements of the scenario
            if run_cfg.params.use_3_speakers == 0 and run_cfg.paths.rev_env == False:
                batchSize = 8 
            else:
                batchSize = 4 

            dual_print(f"[*] Set Batch Size to: {batchSize}")

            # Instantiate dataloader
            test_data = GeneratedData_Dynamic_Speakers_Babble(run_cfg, mode='test')
            test_loader = DataLoader(test_data, batch_size=batchSize, shuffle=run_cfg.model_hp.test_loader_shuffle)
            
            # Capture standard output from the test function to route into our text file
            old_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            # Assuming you want to generate .mat files, pass 1. Pass 0 if you want to skip saving .mat files.
            test_loss = test(model, run_cfg.params, run_cfg.paths.results_path, test_loader, device, run_cfg.loss, 1) 
            
            # Restore standard output and log captured text
            sys.stdout = old_stdout
            dual_print(captured_output.getvalue())
            dual_print(f"Test Loss ({t_cfg['name']}): {test_loss / len(test_loader)}")

        dual_print(f"\n{'='*70}\n[COMPLETE] Master Evaluation Suite Finished!\n{'='*70}\n")

if __name__ == '__main__':
    main()