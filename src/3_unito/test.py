import os
import pandas as pd
import warnings
import fcsparser
from pathlib import Path
import concurrent.futures
import random
import shutil
from generate_gating_strategy import _parse_fcs_add_gate_label, _extract_gating_strategy
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

#if __name__ == '__main__':
#
#    fcs_files = [f for f in os.listdir(fcs_dir) if f.endswith('.fcs')]
#
#    with concurrent.futures.ProcessPoolExecutor() as executor:
#        results = [executor.submit(_convert_fcs_to_csv, os.path.join(fcs_dir, fcs_file), output_dir) for fcs_file in fcs_files]
#        for future in concurrent.futures.as_completed(results):
#            try:
#                future.result()  # Raise any exceptions that occurred
#            except Exception as e:
#                print(f"Error processing file: {e}")


#2b. Parse gates .fcs files - generate gating metadata for training. 

#Requires gated .fcs files, gating data is appended to the corresponding .csv file generated above
#Requires .wsp contianing the .fcs files.

wsp_path = '/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube16_250425-XV1/Unmixed/30sample_gatingWSP_22052025.wsp' #<- Set the path to the WSP

if __name__ == '__main__':

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(_parse_fcs_add_gate_label(wsp_path = wsp_path, fcs_dir = fcs_dir, csv_dir = output_dir))]
        for future in concurrent.futures.as_completed(results):
            try:
                future.result()  # Raise any exceptions that occurred
            except Exception as e:
                print(f"Error processing file: {e}")

#2c. Generate gating strategy from .wsp
_extract_gating_strategy(wsp_path) #Add output_path arguement if you want the gating strat to go somewhere else. Otherwise easier in ./dir for Setting gates below.
