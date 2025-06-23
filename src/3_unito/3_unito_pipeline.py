import os
import pandas as pd
import warnings
from pathlib import Path
import concurrent.futures
import random
import shutil
import sys

from UNITO.UNITO_Train_Predict.hyperparameter_tunning import tune
from UNITO.UNITO_Train_Predict.Train import train
from UNITO.UNITO_Train_Predict.Validation_Recon_Plot_Single import plot_all
from UNITO.UNITO_Train_Predict.Data_Preprocessing import process_table, train_test_val_split
from UNITO.UNITO_Train_Predict.Predict import UNITO_gating, evaluation
from generate_gating_strategy import parse_fcs_add_gate_label, extract_gating_strategy, clean_gating_strategy
warnings.filterwarnings("ignore")

# For reproducibility
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

#1. Setting Directories
fcs_dir = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/altered_fcs_files/post_flowai/'
output_dir = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/processing_outputs/autogating_reports_and_data/autogating_csv_conversions/'

##2. UNITO requires .csv so convert files
#def _convert_fcs_to_csv(fcs_file, output_dir):
#    ''' Takes a .fcs file (in specificed output_dir, generates .csv of that fcs_file required for UNITO processing'''
#    fcs_filename = os.path.basename(fcs_file)
#    meta, data = fcsparser.parse(fcs_file, reformat_meta=True)
#    df = pd.DataFrame(data)
#    csv_filename = fcs_filename.replace('.fcs', '.csv')  # Convert extension
#    df_output = Path(output_dir, csv_filename)
#    df.to_csv(df_output, index=False)
#    print(f'{fcs_filename} converted to csv')

# if __name__ == '__main__':

#    fcs_files = [f for f in os.listdir(fcs_dir) if f.endswith('.fcs')]

#    with concurrent.futures.ProcessPoolExecutor() as executor:
#        results = [executor.submit(_convert_fcs_to_csv, os.path.join(fcs_dir, fcs_file), output_dir) for fcs_file in fcs_files]
#        for future in concurrent.futures.as_completed(results):
#            try:
#                future.result()  # Raise any exceptions that occurred
#            except Exception as e:
#                print(f"Error processing file: {e}")


# 2b. Parse gates .fcs files - generate gating metadata for training. 
        # Requires gated .fcs files, gating data is appended to the corresponding .csv file generated above
        # Requires .wsp contianing the .fcs files.

wsp_path = '/Users/user/Documents/test_wsp/WSP_22052025.wsp' # <- Set the path to the WSP
wsp_files_path = '/Users/user/Documents/test_wsp/'

#if __name__ == '__main__':
#        parse_fcs_add_gate_label(wsp_path, wsp_files_path, './')

# 2c. Generate gating strategy from .wsp

if __name__ == '__main__':
        gating_strategy = extract_gating_strategy(wsp_path, wsp_files_path) #Add output_path arguement if you want the gating strat to go somewhere else. Otherwise easier in ./dir for Setting gates below.

        panel_meta = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/panel_metadata_all_batches.csv'
        final_gating_strategy = clean_gating_strategy(panel_meta, gating_strategy)

        out_file = os.path.join('./', "gating_strategy.csv")
        final_gating_strategy.to_csv(out_file, index=False)

# #3. Setting gates
gating = final_gating_strategy

gate_pre_list = list(gating.Parent_Gate)
gate_pre_list[0] = None # the first gate does not have parent gate
gate_list = list(gating.Gate)
x_axis_list = list(gating.X_axis)
y_axis_list = list(gating.Y_axis)
path2_lastgate_pred_list = ['./prediction/' for x in range(len(gate_list))]

device = 'mps' if torch.backends.mps.is_available() else 'cpu' # Use Apple GPU
n_worker = 0
epoches = 1000

hyperparameter_set = [
                      [1e-3, 8],
                      [1e-4, 8],
                      [1e-3, 16],
                      [1e-4, 16]
                      ]

 #4. Define paths and build dirs
dest = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/processing_outputs/autogating_reports_and_data/' # change depending on your needs
save_data_img_path = f'{dest}/Data'
save_figure_path = f'{dest}/figures'
save_model_path = f'{dest}/model'
save_prediction_path = f'{dest}/prediction'

if not os.path.exists(save_data_img_path):
    os.mkdir(save_data_img_path)
if not os.path.exists(save_figure_path):
    os.mkdir(save_figure_path)
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
if not os.path.exists(save_prediction_path):
    os.mkdir(save_prediction_path)

# #5. 'Train' dir Generation and Management - Only use when there is not a reference set of data - This grabs a random 30 and trains the gates on them.
#train_path = Path(output_dir / 'train')
#os.mkdir(train_path) 
#for fcs_file in random.sample(fcs_files, 30):
#    shutil.move(os.path.join(output_dir, fcs_file), dest)

train_path = output_dir

# #6. 'Pred' dir - should be the original dir from which the random 30 were taken for training.
pred_path = output_dir
path2_lastgate_pred_list[0] = pred_path # the first gate should take data from raw folder

# #7. Send it off to train on the .csv files with the set gates. This is where OMIQ API integration needs to be applied.
hyperparameter_df = pd.DataFrame(columns = ['gate','learning_rate','batch_size'])

for i, (gate_pre, gate, x_axis, y_axis, path_raw) in enumerate(zip(gate_pre_list, gate_list, x_axis_list, y_axis_list, path2_lastgate_pred_list)):

    print(f"start UNITO for {gate}")

    # 7a. preprocess training data
    process_table(x_axis, y_axis, gate_pre, gate, train_path, seq = (gate_pre!=None), dest = dest)
    train_test_val_split(gate, train_path, dest, "train")

    # 7b. train
    best_lr, best_bs = tune(gate, hyperparameter_set, device, epoches, n_worker, dest)
    hyperparameter_df.loc[len(hyperparameter_df)] = [gate, best_lr, best_bs]
    train(gate, best_lr, device, best_bs, epoches, n_worker, dest)

    # 7c. preprocess training data
    print(f"Start prediction for {gate}")
    process_table(x_axis, y_axis, gate_pre, gate, pred_path, seq = (gate_pre!=None), dest = dest)
    train_test_val_split(gate, pred_path, dest, 'pred')

    # 7d. predict
    model_path = f'{dest}/model/{gate}_model.pt'
    data_df_pred = UNITO_gating(model_path, x_axis, y_axis, gate, path_raw, n_worker, device, save_prediction_path, dest, seq = (gate_pre!=None), gate_pre=gate_pre)

    # 7e. Evaluation
    accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)
    print(f"{gate}: accuracy:{accuracy}, recall:{recall}, precition:{precision}, f1 score:{f1}")

    # 7f. Plot gating results
    plot_all(gate_pre, gate, x_axis, y_axis, path_raw, save_figure_path)
    print("All UNITO prediction visualization saved")

print("Seqential autogating prediction finished")


hyperparameter_df.to_csv('./hyperparameter_tunning.csv')