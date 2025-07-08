import os
import warnings
import ssl
import urllib3
import pandas as pd

# Removal of unnessecary error msgs caused by shit UNITO code.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
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
import atexit

# UNITO Imports
from UNITO_Train_Predict.hyperparameter_tunning import tune
from UNITO_Train_Predict.Train import train
from UNITO_Train_Predict.Validation_Recon_Plot_Single import plot_all
from UNITO_Train_Predict.Data_Preprocessing import process_table, train_test_val_split
from UNITO_Train_Predict.Predict import UNITO_gating, evaluation

# Own Imports
from generate_gating_strategy import parse_fcs_add_gate_label, extract_gating_strategy, clean_gating_strategy, add_gate_labels_to_test_files
from apply_unito_to_fcs import create_hierarchical_gates_from_unito

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
     Empties RAM disk on completion for next gate, leaves the .npy data alone. """
    ram_dest = os.getenv("UNITO_DEST")
    subdirs = ["figures", "model", "prediction", "Data"]

    for sub in subdirs:
        src = os.path.join(ram_dest, sub)
        dst = os.path.join(disk_dest, sub)
        if os.path.exists(src):
            os.makedirs(dst, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

    # Clear RAM for next gate
    for sub in ["figures", "model", "prediction"]:
        ram_sub = os.path.join(ram_dest, sub)
        if os.path.exists(ram_sub):
            shutil.rmtree(ram_sub)
            os.makedirs(ram_sub, exist_ok=True)
    
    # Copy summary .csv if possible
    for fn in ["gating_strategy.csv", "hyperparameter_tunning.csv"]:
        if os.path.exists(fn):
            shutil.copy(fn, os.path.join(disk_dest, fn))

    print(f"Flushed RAMDisk contents from {ram_dest} to {disk_dest}")
# ────────────────────────────────────────────────────────────────────────────────

# Torch settings - for reproducibility
train = torch.compile(train)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Setting Directories
fcs_dir = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/altered_fcs_files/post_flowai/'
csv_conversion_dir = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/processing_outputs/autogating_reports_and_data/autogating_csv_conversions/'
train_path = os.path.join(csv_conversion_dir, 'train')
fcs_files = [f for f in os.listdir(fcs_dir) if f.endswith('.fcs')]
wsp_path = '/Users/user/Documents/UNITO_train_wsp/WSP_22052025.wsp'
wsp_files_path = '/Users/user/Documents/UNITO_train_wsp/'
disk_dest = '/Users/user/Documents/UNITO_train_data'


# UNITO requires .csv so convert files
def _convert_fcs_to_csv(fcs_file, csv_conversion_dir):
    """ Takes a .fcs file (in specificed csv_conversion_dir, generates .csv of that fcs_file required for UNITO processing """
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
    parse_fcs_add_gate_label(wsp_path, wsp_files_path, csv_conversion_dir)
    print("Gate parsing complete!")

def generate_gating_strategy():
    """ Generate and save gating strategy """
    print("Generating gating strategy...")
    gating_strategy = extract_gating_strategy(wsp_path, wsp_files_path)
    panel_meta = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/panel_metadata_all_batches.csv'
    final_gating_strategy = clean_gating_strategy(panel_meta, gating_strategy)
    out_file = os.path.join('./', "gating_strategy.csv")
    final_gating_strategy.to_csv(out_file, index=False)
    print("Gating strategy saved!")
    return final_gating_strategy

def main(ram_disk):
    """ Main pipeline execution """
    # Step 1: Convert FCS to CSV
    #convert_all_fcs()
    
    # Step 2: Parse gates and add to CSV files
    #parse_gates()

    # Step 4: Generate gating strategy
    final_gating_strategy = generate_gating_strategy()
    
    # Step 5: Continue with rest of pipeline
    gating = final_gating_strategy
    
    gate_pre_list = list(gating.Parent_Gate)
    gate_pre_list[0] = None # the first gate does not have parent gate
    gate_list = list(gating.Gate)
    x_axis_list = list(gating.X_axis)
    y_axis_list = list(gating.Y_axis)
    path2_lastgate_pred_list = ['./prediction/' for x in range(len(gate_list))]

    device = 'mps'
    n_worker = 24
    epoches = 100

    hyperparameter_set = [
                          [1e-3, 128],
                          [1e-4, 128]
                          ]

    # Step 6. Define paths and build dirs
    if ram_disk == True:
        dest                 = os.getenv("UNITO_DEST")
        save_data_img_path   = f"{dest}/Data"
        save_figure_path     = f"{dest}/figures"
        save_model_path      = f"{dest}/model"
        save_prediction_path = f"{dest}/prediction"

        for path in [
            save_data_img_path,
            save_figure_path,
            save_model_path,
            save_prediction_path
        ]:
            os.makedirs(path, exist_ok=True)
    else:
        dest                  = '/Users/user/Documents/UNITO_train_data'
        save_data_img_path    = f'{dest}/Data'
        save_figure_path      = f'{dest}/figures'
        save_model_path       = f'{dest}/model'
        save_prediction_path  = f'{dest}/prediction'

        for path in [save_data_img_path, save_figure_path, save_model_path, save_prediction_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    # Step 7. 'Train' dir Generation and Management
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    # Step 7a. Get only CSV files with gate labels
    training_csv_files = [f for f in os.listdir(csv_conversion_dir) if f.endswith('_with_gate_label.csv')]
    print(f"Found {len(training_csv_files)} files with gate labels")
    print("If 0 files found - likely already moved to gated csv files to correct dir")

    # Step 8. Move the .csv files with gates to the train folder
    for training_csv_file in training_csv_files:
        source_path = os.path.join(csv_conversion_dir, training_csv_file)
        destination_path = os.path.join(train_path, training_csv_file)
        if os.path.exists(source_path):  # Check if file exists before moving
            shutil.move(source_path, destination_path)

    # Step 8a. Add Gate Labels to the .csv files in the train folder
    #add_gate_labels_to_test_files(test_dir = csv_conversion_dir, train_dir = train_path)

    # Step 9. UNITO
    path2_lastgate_pred_list[0] = csv_conversion_dir
    
    hyperparameter_df = pd.DataFrame(columns = ['gate','learning_rate','batch_size'])

    for i, (gate_pre, gate, x_axis, y_axis, _ignored) in enumerate(zip(gate_pre_list, gate_list, x_axis_list, y_axis_list, path2_lastgate_pred_list)):
        print(f"start UNITO for {gate}")

        path_raw = csv_conversion_dir # path_raw logic was messed up as it was expecting the test .csv in ./predicitons folder. Dummy variable in _ignored to get around this, with path_raw set each iteration.
        
        # 9a. preprocess training data
        process_table(x_axis, y_axis, gate_pre, gate, train_path, convex = True, seq = (gate_pre!=None), dest = dest)
        train_test_val_split(gate, train_path, dest, "train")

        # 9b. train
        best_lr, best_bs = tune(gate, hyperparameter_set, device, epoches, n_worker, dest)
        hyperparameter_df.loc[len(hyperparameter_df)] = [gate, best_lr, best_bs]
        train(gate, best_lr, device, best_bs, epoches, n_worker, dest)

        # 9c. preprocess prediction data
        print(f"Start prediction for {gate}")
        process_table(x_axis, y_axis, gate_pre, gate, csv_conversion_dir, convex = True, seq = (gate_pre!=None), dest = dest)
        train_test_val_split(gate, csv_conversion_dir, dest, 'pred')

        # 9d. predict
        model_path = f'{dest}/model/{gate}_model.pt'
        data_df_pred = UNITO_gating(model_path, x_axis, y_axis, gate, csv_conversion_dir, n_worker, device, save_prediction_path, dest, seq = (gate_pre!=None), gate_pre=gate_pre)

        # 9e. Evaluation
        accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)
        print(f"{gate}: accuracy:{accuracy}, recall:{recall}, precision:{precision}, f1 score:{f1}")

        # 9f. Plot gating results
        plot_all(gate_pre, gate, x_axis, y_axis, path_raw, save_figure_path)
        print("All UNITO prediction visualization saved")

        # Flush RAM DISK for next gate.
        flush_ramdisk_to_disk(disk_dest)

# Step 10. Create hierarchical gates from all predictions
    print("Creating hierarchical gates...")
    save_fcs_with_gates_path = f'{disk_dest}/fcs_with_hierarchical_unito_gates'
    if not os.path.exists(save_fcs_with_gates_path):
        os.makedirs(save_fcs_with_gates_path)
    disk_prediction_path = f'{disk_dest}/prediction'

    create_hierarchical_gates_from_unito(final_gating_strategy, disk_prediction_path, save_fcs_with_gates_path, fcs_dir)
    
    print("Sequential autogating with hierarchical gates finished")
    hyperparameter_df.to_csv('./hyperparameter_tunning.csv')

if __name__ == '__main__':
    mount_ramdisk(True)
    try:
        main(ram_disk=True)
    finally:
        cleanup_ramdisk()