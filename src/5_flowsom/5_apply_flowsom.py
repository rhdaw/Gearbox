import flowsom as fs
import os
import flowkit as fk

# Directories
disk_dest = '/Users/user/Documents/UNITO_train_data'
save_fcs_with_gates_path = f'{disk_dest}/fcs_with_hierarchical_unito_gates'
fcs_files = [f for f in os.listdir(save_fcs_with_gates_path) if f.endswith('.fcs')]

# Filter by cell type if you want here (i.e. everything that isn't a Neutrophil)
def filter_out_cell(fcs_dir, cell_gate):
    fcs_files = [f for f in os.listdir(fcs_dir) if f.endswith('.fcs')]
    for f in fcs_files:
        sample_path = os.path.join(fcs_dir, f)
        sample = fk.Sample(sample_path)
        sample_events_df = sample.as_dataframe()
        filtered_df = sample_events_df[sample_events_df[cell_gate] != 1]
        
        # Create new filename with dropped cell type
        base_name = os.path.splitext(f)[0]
        cell_type = cell_gate.replace('UNITO_', '')
        new_filename = f"{base_name}_dropped_{cell_type}.fcs"
        new_filepath = os.path.join(fcs_dir, new_filename)
        
        # Create new sample with filtered data and save
        filtered_sample = fk.Sample(filtered_df, sample_id=base_name, sample_data_set=sample.sample_data_set)
        filtered_sample.export(new_filepath, source='dataframe')
        
        print(f"Removed {cell_type} from .fcs file. Saved filtered file: {new_filename}")
        
    return

filter_out_cell(save_fcs_with_gates_path, 'UNITO_Neutrophils')
# Load the FCS file
ff = fs.pp.aggregate_flowframes(fcs_files)

# Run the FlowSOM algorithm -> cols to use must be UNITO ones. Function to find col numbers of 'UNITO_' cols -> list - > give list as arg
fsom = fs.FlowSOM(
    ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], xdim=10, ydim=10, n_clusters=10, seed=42
)

# Plot the FlowSOM results
p = fs.pl.plot_stars(fsom, background_values=fsom.get_cluster_data().obs.metaclustering)
p.show()