import flowkit as fk
import os
import pandas as pd
import numpy as np
from metric_functions import calc_gmfi, calc_percent_gate, calc_percent_marker_positive

# Setting Directories
unito_processed_fcs_dir = '' # Usually disk_dest from -> disk_dest = '/Users/user/Documents/UNITO_train_data'
fcs_files = [f for f in os.listdir(unito_processed_fcs_dir) if f.endswith('.fcs')]


# Rather than gate shapes, UNITO results are saved as paramteres. So we basically do population statistics on the UNITO parameters, rather than rebuilding the gates
# All UNITO adjusted parameters have the 'UNITO_' prefix to distinguish them from past parameters.

metrics_list = []

for fcs_file in fcs_files:
    sample_path = os.path.join(unito_processed_fcs_dir, fcs_file)
    sample = fk.Sample(sample_path)
    sample_events_df = sample.as_dataframe()

    # Assign the gates you wish to use for metrics to variables ||  This is setting metrics -> change these to metrics you want to output.
    cd45_positive = sample_events_df['UNITO_CD45'] == 1
    lymph_positive = sample_events_df['UNITO_Lymphocytes'] == 1

    # Perform your metric calculations
    cd45_in_lymph_percent = ((cd45_positive & lymph_positive).sum() / lymph_positive.sum()) * 100

    # Send results to a dictionary
