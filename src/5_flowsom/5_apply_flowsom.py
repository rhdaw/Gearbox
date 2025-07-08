import flowsom as fs
import os
import flowkit as fk

# Directories
disk_dest = '/Users/user/Documents/UNITO_train_data'
save_fcs_with_gates_path = f'{disk_dest}/fcs_with_hierarchical_unito_gates'
fcs_files = [f for f in os.listdir(save_fcs_with_gates_path) if f.endswith('.fcs')]

def filter_out_cell(fcs_dir, cell_gate):
    fcs_files = [f for f in os.listdir(fcs_dir) if f.endswith('.fcs')]
    new_filepath_list = []

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
        new_filepath_list.append(new_filename)

        # Create new sample with filtered data and save
        filtered_sample = fk.Sample(filtered_df, sample_id=base_name)
        filtered_sample.export(new_filepath, source='dataframe')
        print(f"Removed {cell_type} from .fcs file. Saved filtered file: {new_filename}") 

    return new_filepath_list

def get_unito_col_idx(fcs_dir):
    fcs_files = [f for f in os.listdir(fcs_dir) if f.endswith('.fcs')]
    sample_path = os.path.join(fcs_dir, fcs_files[0])
    sample = fk.Sample(sample_path)
    sample_events_df = sample.as_dataframe()
    col_names = sample_events_df.columns[sample_events_df.columns.str.contains('UNITO_')].tolist()
    col_indices = [sample_events_df.columns.get_loc(col) for col in col_names]

    return col_indices

# Get UNITO column indicies
col_indices = get_unito_col_idx(save_fcs_with_gates_path)

# Filter by cell type if you want here (i.e. everything that isn't a Neutrophil)
filtered_fcs_path_list = filter_out_cell(save_fcs_with_gates_path, 'UNITO_Neutrophils')

# Load the FCS file
ff = fs.pp.aggregate_flowframes(filtered_fcs_path_list)

# Run the FlowSOM algorithm
fsom = fs.FlowSOM(
    ff, cols_to_use = col_indices, xdim=10, ydim=10, n_clusters=10, seed=42
)

# Plot the FlowSOM results
p = fs.pl.plot_stars(fsom, background_values=fsom.get_cluster_data().obs.metaclustering)
p.show()