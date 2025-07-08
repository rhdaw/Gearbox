import flowkit as fk
import os
import pandas as pd
import numpy as np
from metric_functions import calc_gmfi, calc_percent_gate

# Setting Directories
unito_processed_fcs_dir = '' # Usually disk_dest from -> disk_dest = '/Users/user/Documents/UNITO_train_data'
fcs_files = [f for f in os.listdir(unito_processed_fcs_dir) if f.endswith('.fcs')]

# Rather than gate shapes, UNITO results are saved as paramteres. So we basically do population statistics on the UNITO parameters, rather than rebuilding the gates
# All UNITO adjusted parameters have the 'UNITO_' prefix to distinguish them from past parameters.

stats_list = []

for fcs_file in fcs_files:
    sample_path = os.path.join(unito_processed_fcs_dir, fcs_file)
    sample = fk.Sample(sample_path)
    sample_events_df = sample.as_dataframe()

    # Perform your metric calculations
    cd45_gmfi = calc_gmfi(sample_events_df, 'CD45', 'UNITO_Lymphocytes')
    cd45_percent_lymph = calc_percent_gate(sample_events_df, 'CD45', 'UNITO_Lymphocytes')

    # Send results to a dictionary
    stats_dict = {'Sample': str(fcs_file), 'CD45_gmfi': cd45_gmfi, 'CD45_percent_lymph': cd45_percent_lymph}

    # Append results to master list
    stats_list.append(stats_dict)

# Build and save dataframe
output_stats = pd.DataFrame(stats_list, orient='columns')
output_stats.to_csv('UNITO_flowcytometry_stats.csv')