
import os
import ssl
import urllib3
import warnings
import pandas as pd

# Removal of unnessecary error msgs caused by shit UNITO code.
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Add this at the very top
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 

from pathlib import Path
import concurrent.futures
import random
import shutil
import sys
import fcsparser

from UNITO_Train_Predict.hyperparameter_tunning import tune
from UNITO_Train_Predict.Train import train
from UNITO_Train_Predict.Validation_Recon_Plot_Single import plot_all
from UNITO_Train_Predict.Data_Preprocessing import process_table, train_test_val_split
from UNITO_Train_Predict.Predict import UNITO_gating, evaluation

from generate_gating_strategy import parse_fcs_add_gate_label, extract_gating_strategy, clean_gating_strategy
from apply_unito_to_fcs import create_hierarchical_gates_from_unito


# For reproducibility
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

# Setting Directories
fcs_dir = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/altered_fcs_files/post_flowai/'
csv_conversion_dir = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/processing_outputs/autogating_reports_and_data/autogating_csv_conversions/'
fcs_files = [f for f in os.listdir(fcs_dir) if f.endswith('.fcs')]
wsp_path = '/Users/user/Documents/UNITO_train_wsp/WSP_22052025.wsp' # <- Set the path to the WSP
wsp_files_path = '/Users/user/Documents/UNITO_train_wsp/'

# UNITO requires .csv so convert files
def _convert_fcs_to_csv(fcs_file, csv_conversion_dir):
    ''' Takes a .fcs file (in specificed csv_conversion_dir, generates .csv of that fcs_file required for UNITO processing'''
    fcs_filename = os.path.basename(fcs_file)
    meta, data = fcsparser.parse(fcs_file, reformat_meta=True)
    df = pd.DataFrame(data)
    csv_filename = fcs_filename.replace('.fcs', '.csv')  # Convert extension
    df_output = Path(csv_conversion_dir, csv_filename)
    df.to_csv(df_output, index=False)
    print(f'{fcs_filename} converted to csv')

def convert_all_fcs():
    """Convert all FCS files to CSV"""
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
    """Parse gates from WSP file and add to CSV files"""
    print("Parsing gates from WSP file...")
    parse_fcs_add_gate_label(wsp_path, wsp_files_path, csv_conversion_dir)
    print("Gate parsing complete!")

def generate_gating_strategy():
    """Generate and save gating strategy"""
    print("Generating gating strategy...")
    gating_strategy = extract_gating_strategy(wsp_path, wsp_files_path)
    panel_meta = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/panel_metadata_all_batches.csv'
    final_gating_strategy = clean_gating_strategy(panel_meta, gating_strategy)
    out_file = os.path.join('./', "gating_strategy.csv")
    final_gating_strategy.to_csv(out_file, index=False)
    print("Gating strategy saved!")
    return final_gating_strategy

def main():
    """Main pipeline execution"""
    # Step 1: Convert FCS to CSV
    #convert_all_fcs()
    
    # Step 2: Parse gates and add to CSV files
    #parse_gates()
    
    # Step 3: Generate gating strategy
    final_gating_strategy = generate_gating_strategy()
    
    # Step 4: Continue with rest of pipeline
    gating = final_gating_strategy
    
    gate_pre_list = list(gating.Parent_Gate)
    gate_pre_list[0] = None # the first gate does not have parent gate
    gate_list = list(gating.Gate)
    x_axis_list = list(gating.X_axis)
    y_axis_list = list(gating.Y_axis)
    path2_lastgate_pred_list = ['./prediction/' for x in range(len(gate_list))]

    device = 'mps'
    n_worker = 16
    epoches = 1000

    hyperparameter_set = [
                          [1e-3, 8],
                          [1e-4, 8],
                          [1e-3, 16],
                          [1e-4, 16]
                          ]

    #4. Define paths and build dirs
    dest = '/Users/user/Documents/UNITO_train_data'
    save_data_img_path = f'{dest}/Data'
    save_figure_path = f'{dest}/figures'
    save_model_path = f'{dest}/model'
    save_prediction_path = f'{dest}/prediction'

    for path in [save_data_img_path, save_figure_path, save_model_path, save_prediction_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # #5. 'Train' dir Generation and Management
    train_path = Path(csv_conversion_dir, 'train')
    if not os.path.exists(train_path):
        os.mkdir(train_path)

    #5a. Get only CSV files with gate labels
    training_csv_files = [f for f in os.listdir(csv_conversion_dir) if f.endswith('_with_gate_label.csv')]
    print(f"Found {len(training_csv_files)} files with gate labels")
    print("If 0 files found - likely already moved to gated csv files to correct dir")

    #5b. Move the .csv files with gates to the train folder
    for training_csv_file in training_csv_files:
        source_path = os.path.join(csv_conversion_dir, training_csv_file)
        destination_path = os.path.join(train_path, training_csv_file)
        if os.path.exists(source_path):  # Check if file exists before moving
            shutil.move(source_path, destination_path)

    pred_path = csv_conversion_dir
    path2_lastgate_pred_list[0] = pred_path

    hyperparameter_df = pd.DataFrame(columns = ['gate','learning_rate','batch_size'])


    for i, (gate_pre, gate, x_axis, y_axis, path_raw) in enumerate(zip(gate_pre_list, gate_list, x_axis_list, y_axis_list, path2_lastgate_pred_list)):
        print(f"start UNITO for {gate}")

        # 7a. preprocess training data
        process_table(x_axis, y_axis, gate_pre, gate, str(train_path), convex = None, seq = (gate_pre!=None), dest = dest)
        train_test_val_split(gate, str(train_path), dest, "train")

        # 7b. train
        best_lr, best_bs = tune(gate, hyperparameter_set, device, epoches, n_worker, dest)
        hyperparameter_df.loc[len(hyperparameter_df)] = [gate, best_lr, best_bs]
        train(gate, best_lr, device, best_bs, epoches, n_worker, dest)

        # 7c. preprocess prediction data
        print(f"Start prediction for {gate}")
        process_table(x_axis, y_axis, gate_pre, gate, pred_path, seq = (gate_pre!=None), dest = dest)
        train_test_val_split(gate, pred_path, dest, 'pred')

        # 7d. predict
        model_path = f'{dest}/model/{gate}_model.pt'
        data_df_pred = UNITO_gating(model_path, x_axis, y_axis, gate, path_raw, n_worker, device, save_prediction_path, dest, seq = (gate_pre!=None), gate_pre=gate_pre)

        # 7e. Evaluation
        accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)
        print(f"{gate}: accuracy:{accuracy}, recall:{recall}, precision:{precision}, f1 score:{f1}")

        # 7f. Plot gating results
        plot_all(gate_pre, gate, x_axis, y_axis, path_raw, save_figure_path)
        print("All UNITO prediction visualization saved")

# 8. Create hierarchical gates from all predictions
    print("Creating hierarchical gates...")
    
    save_fcs_with_gates_path = f'{dest}/fcs_with_hierarchical_unito_gates'
    if not os.path.exists(save_fcs_with_gates_path):
        os.makedirs(save_fcs_with_gates_path)
         
    create_hierarchical_gates_from_unito(final_gating_strategy, save_prediction_path, save_fcs_with_gates_path, fcs_dir)
    
    print("Sequential autogating with hierarchical gates finished")
    hyperparameter_df.to_csv('./hyperparameter_tunning.csv')

if __name__ == '__main__':
    main()