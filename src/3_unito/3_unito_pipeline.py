import os
import warnings
import ssl
import urllib3
import pandas as pd
import time

# Removal of unnessecary error msgs caused by shit UNITO code.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 

# Standard Imports
from pathlib import Path
import concurrent.futures
import random
import shutil
import fcsparser
import torch
import subprocess
import numpy as np
import psutil
from contextlib import contextmanager

# UNITO Imports
from UNITO_Train_Predict.hyperparameter_tunning import tune
from UNITO_Train_Predict.Train import train
from UNITO_Train_Predict.Validation_Recon_Plot_Single import plot_all
from UNITO_Train_Predict.Data_Preprocessing import process_table, train_test_val_split
from UNITO_Train_Predict.Predict import UNITO_gating, evaluation

# Own Imports
from generate_gating_strategy import parse_fcs_add_gate_label, extract_gating_strategy, clean_gating_strategy, add_gate_labels_to_test_files
from apply_unito_to_fcs import apply_predictions_to_csv

# RAM DISK mount and functions for quicker train ─────────────────────────────────
def mount_ramdisk(ram_disk):
    if not ram_disk:
        return
    
    print("Starting RAM disk creation...")
    # Scan & detach any existing RAMDisk mounts
    try:
        info = subprocess.check_output(["hdiutil", "info"], text=True)
        print("Checked existing mounts")
    except Exception as e:
        print(f"Error checking existing mounts: {e}")
        return
    
    for line in info.splitlines():
        if "/Volumes/RAMDisk" in line or line.strip().startswith("/dev/ram"):
            dev = line.split()[0]
            try:
                subprocess.check_call(
                    ["hdiutil", "detach", dev, "-force"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                print(f"Detached existing RAM disk: {dev}")
            except subprocess.CalledProcessError:
                pass

    # Create RAMDisk
    try:
        vm = psutil.virtual_memory()
        usable_bytes = vm.available * 0.9
        blocks = int(usable_bytes // 512)
        print(f"Creating RAM disk: {blocks} blocks ({usable_bytes/1024/1024/1024:.1f} GB)")
        
        dev = subprocess.check_output(
            ["hdiutil", "attach", "-nomount", f"ram://{blocks}"],
            text=True
        ).strip()
        print(f"RAM device created: {dev}")
        
        subprocess.check_call(
            ["diskutil", "eraseVolume", "HFS+", "RAMDisk", dev],
            stdout=subprocess.DEVNULL
        )
        print("RAM disk formatted")
        
        os.environ["UNITO_DEST"] = "/Volumes/RAMDisk/UNITO_train_data"
        print(f"UNITO_DEST set to: {os.environ['UNITO_DEST']}")
        
        # Verify it was created
        if os.path.exists("/Volumes/RAMDisk"):
            print("✅ RAM disk successfully mounted at /Volumes/RAMDisk")
        else:
            print("❌ RAM disk mount failed")
            
    except Exception as e:
        print(f"Error creating RAM disk: {e}")
    
def cleanup_ramdisk():
    """ Unmount the RAMDisk so diskimages-helper exits and frees the RAM. """
    if os.path.ismount("/Volumes/RAMDisk"):
        subprocess.check_call(
            ["hdiutil","detach","/Volumes/RAMDisk"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

def flush_ramdisk_to_disk(disk_dest):
    """ Copy the four UNITO output subfolders plus strategy & hyperparam CSVs from the RAM disk ($UNITO_DEST) into disk_dest.
     Empties RAM disk on completion. """
    ram_dest = os.getenv("UNITO_DEST")
    subdirs = ["figures", "model", "prediction", "Data"]

    for sub in subdirs:
        src = os.path.join(ram_dest, sub)
        dst = os.path.join(disk_dest, sub)
        if os.path.exists(src):
            os.makedirs(dst, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Clear RAM
    for sub in ["figures", "model", "prediction", "Data"]:
        ram_sub = os.path.join(ram_dest, sub)
        if os.path.exists(ram_sub):
            shutil.rmtree(ram_sub)
            os.makedirs(ram_sub, exist_ok=True)
    
    # Copy summary .csv if possible
    for fn in ["gating_strategy.csv", "hyperparameter_tunning.csv"]:
        if os.path.exists(fn):
            shutil.copy(fn, os.path.join(disk_dest, fn))

    print(f"Flushed RAMDisk contents from {ram_dest} to {disk_dest}")

@contextmanager
def cd(newdir):
    """ For plot_all to work with RAM disk: Temporarily chdir into newdir for the duration of a with-block."""
    prev = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prev)
# ────────────────────────────────────────────────────────────────────────────────

# Torch settings 
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Setting Directories ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Prediction .fcs files 
fcs_dir = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/altered_fcs_files/post_flowai/'
fcs_files = [f for f in os.listdir(fcs_dir) if f.endswith('.fcs')]

# Train wsp and .fcs file dir
wsp_path = '/Users/user/Documents/UNITO_train_wsp/WSP_22052025.wsp'
wsp_files_dir = '/Users/user/Documents/UNITO_train_wsp/' #these are the fcs files that get moved to csv_train_dir

# Prediction .csv files (converted)
csv_conversion_dir = '/Users/user/Documents/UNITO_csv_conversion/'
csv_train_dir = os.path.join(csv_conversion_dir, 'train')
if not os.path.exists(csv_train_dir):
    os.mkdir(csv_train_dir)

# UNITO train/prediciton save dir
disk_dest = '/Users/user/Documents/UNITO_train_data'
disk_prediction_dir = os.path.join(disk_dest, 'prediction/')
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# UNITO requires .csv so convert files
def _convert_fcs_to_csv(fcs_file, csv_conversion_dir):
    """ Takes a .fcs file, generates .csv of that fcs_file required for UNITO processing """
    fcs_filename = os.path.basename(fcs_file)
    meta, data = fcsparser.parse(fcs_file, reformat_meta=True)
    df = pd.DataFrame(data)
    csv_filename = fcs_filename.replace('.fcs', '.csv')  # Convert extension
    df_output = Path(csv_conversion_dir, csv_filename)
    df.to_csv(df_output, index=False)
    print(f'{fcs_filename} converted to csv')

def convert_all_fcs():
    """ Convert all FCS files to CSV """
    print("Converting FCS files to CSV...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(_convert_fcs_to_csv, os.path.join(fcs_dir, fcs_file), csv_conversion_dir) for fcs_file in fcs_files]
        for future in concurrent.futures.as_completed(results):
            try:
                future.result()  # Raise any exceptions that occurred
            except Exception as e:
                print(f"Error processing file: {e}")
    print("FCS to CSV conversion complete!")

def parse_gates():
    """ Parse gates from WSP file and add to CSV files """
    print("Parsing gates from WSP file...")
    parse_fcs_add_gate_label(wsp_path, wsp_files_dir, csv_conversion_dir)
    print("Gate parsing complete!")

def generate_gating_strategy():
    """ Generate and save gating strategy """
    print("Generating gating strategy...")
    gating_strategy = extract_gating_strategy(wsp_path, wsp_files_dir)
    panel_meta = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/panel_metadata_all_batches.csv'
    final_gating_strategy = clean_gating_strategy(panel_meta, gating_strategy)
    out_file = os.path.join('./', "gating_strategy.csv")
    final_gating_strategy.to_csv(out_file, index=False)
    print("Gating strategy saved!")
    return final_gating_strategy

def downsample_csv(in_csv, max_rows=200_000, out_dir=None):
    """
    Reads `in_csv`, samples up to `max_rows`, and writes:
      - back to `in_csv` if out_dir is None
      - into `out_dir/<basename(in_csv)>` otherwise
    """
    df = pd.read_csv(in_csv)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(in_csv))
    else:
        out_path = in_csv

    df.to_csv(out_path, index=False)
    return out_path

def main(ram_disk):
    """ Main pipeline execution """
    # Step 1: Convert all FCS to CSV in fcs_dir - function sends them to the csv_conversion_dir following conversion.
    #convert_all_fcs()
    
    # Step 2: Parse gates from WSP and add binary classification to CSV files
    #parse_gates()

    # Step 4: Generate gating strategy for UNITO
    final_gating_strategy = generate_gating_strategy()
    
    # Step 5: UNITO steps
    gating = final_gating_strategy
    
    gate_pre_list = list(gating.Parent_Gate)
    gate_pre_list[0] = None # the first gate does not have parent gate
    gate_list = list(gating.Gate)
    x_axis_list = list(gating.X_axis)
    y_axis_list = list(gating.Y_axis)

    device = 'mps'
    n_worker = 30
    epoches = 7

    hyperparameter_set = [
    [1e-3,  128],   
    [1e-4,  256],   
    [5e-4,  512],   
    ]

    # Step 6. Define paths and build dirs for UNITO
    if ram_disk == True:
        dest                 = os.getenv("UNITO_DEST")
        save_data_img_path   = f"{dest}/Data/"
        save_figure_path     = f"{dest}/figures/"
        save_model_path      = f"{dest}/model/"
        save_prediction_path = f"{dest}/prediction/"
        downsample_path      = f"{dest}/downsample/"

        for path in [
            save_data_img_path,
            save_figure_path,
            save_model_path,
            save_prediction_path,
            downsample_path
        ]:
            os.makedirs(path, exist_ok=True)
    else:
        dest                  = '/Users/user/Documents/UNITO_train_data'
        save_data_img_path    = f'{dest}/Data/'
        save_figure_path      = f'{dest}/figures/'
        save_model_path       = f'{dest}/model/'
        save_prediction_path  = f'{dest}/prediction/'

        for path in [save_data_img_path, save_figure_path, save_model_path, save_prediction_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    # Step 7. Get only CSV files with gate labels
    training_csv_files = [f for f in os.listdir(csv_conversion_dir) if f.endswith('_with_gate_label.csv')]
    print(f"Found {len(training_csv_files)} files with gate labels")
    print("If 0 files found - likely already moved to gated csv files to correct dir")

    # Step 8. Move the gated .csv files to the UNITO_csv_conversion/train folder (Disk or RAM Disk)
    # for training_csv_file in training_csv_files:
    #     source_path = os.path.join(csv_conversion_dir, training_csv_file)
    #     destination_path = os.path.join(csv_train_dir, training_csv_file)
    #     if os.path.exists(source_path):  # Check if file exists before moving
    #         shutil.move(source_path, destination_path)

    # OR (Optional) Step 8. Downsample train .csv files and move to RAM disk
    for f in training_csv_files:
        csv = os.path.join(csv_conversion_dir, f)
        downsample_csv(csv, max_rows=200_000, out_dir= downsample_path)
    csv_train_dir = downsample_path

    # Step 8a. Add Gate Labels to the test .csv files
    #add_gate_labels_to_test_files(test_dir = csv_conversion_dir, train_dir = csv_train_dir)

    # Step 8b. Set csv_train_dir for the UNITO unpacking
    path2_lastgate_pred_list = [csv_conversion_dir]

    for idx in range(1, len(gate_list)):
            parent_gate = gate_pre_list[idx]
            path2_lastgate_pred_list.append(f'./prediction/{parent_gate}/')

    # Step 9. UNITO
    hyperparameter_df = pd.DataFrame(columns = ['gate','learning_rate','batch_size'])
    all_predictions = {} 
    
    with cd(dest):
        for i, (gate_pre, gate, x_axis, y_axis, path_raw) in enumerate(zip(gate_pre_list, gate_list, x_axis_list, y_axis_list, path2_lastgate_pred_list)):

            print(f"start UNITO for {gate}")

            # 9a. preprocess training data
            process_table(x_axis, y_axis, gate_pre, gate, csv_train_dir, convex = True, seq = (gate_pre!=None), dest = dest)
            train_test_val_split(gate, csv_train_dir, dest, "train")

            # 9b. train
            best_lr, best_bs = tune(gate, hyperparameter_set, device, epoches, n_worker, dest)
            hyperparameter_df.loc[len(hyperparameter_df)] = [gate, best_lr, best_bs]
            train(gate, best_lr, device, best_bs, epoches, n_worker, dest)

            # 9c. preprocess prediction data
            print(f"Start prediction for {gate}")

            if i == 0:  # Capture the files being processed for the FIRST gate only
                processed_files_list = [f for f in os.listdir(path_raw) 
                                            if f.endswith('.csv') 
                                            and not f.endswith('_with_gate_label.csv')]
            print(f"Captured file processing order: {len(processed_files_list)} files")

            process_table(x_axis, y_axis, gate_pre, gate, path_raw, convex = True, seq = (gate_pre!=None), dest = dest)
            train_test_val_split(gate, path_raw, dest, 'pred')

            # 9d. predict
            model_path = f'{dest}/model/{gate}_model.pt'
            gate_prediction_path = f'{save_prediction_path}/{gate}'  # Gate-specific path
            os.makedirs(gate_prediction_path, exist_ok=True)  # Ensure directory exists
            data_df_pred, predictions_dict = UNITO_gating(model_path, x_axis, y_axis, gate, path_raw, n_worker, device, gate_prediction_path, dest, seq = (gate_pre!=None), gate_pre=gate_pre)
            
            # Collect all predictions for this gate, across all files, to the all_predictions dict - gate_predictions is a nested dict of {key = {gate}_pred: value = [binary classifiers]}
            for filename, gate_predictions in predictions_dict.items():
                if filename not in all_predictions:
                    all_predictions[filename] = {}
                all_predictions[filename].update(gate_predictions)

            # 9e. Evaluation
            accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)
            print(f"{gate}: accuracy:{accuracy}, recall:{recall}, precision:{precision}, f1 score:{f1}")

            # 9f. Plot gating results <- skipping this for the moment NEED TO CHANGE UNITO code to just os.path.splitext() instead of whatever weirdness it is doing.
            # plot_all(gate_pre, gate, x_axis, y_axis, path_raw, save_figure_path)
            # print("All UNITO prediction visualization saved")

    # Step 10. Apply predicitons from dict to csv files
    apply_predictions_to_csv(all_predictions, csv_conversion_dir)

    # Step 11. Save hyperparameters for future
    hyperparameter_df.to_csv('./hyperparameter_tuning.csv')

    # Flush RAM DISK
    flush_ramdisk_to_disk(disk_dest)

if __name__ == '__main__':
    mount_ramdisk(True)
    try:
        main(ram_disk=True)
    finally:
        cleanup_ramdisk()