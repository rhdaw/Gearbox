import fcswrite
import fcsparser
from scipy.spatial import ConvexHull
import os
import pandas as pd
from pathlib import Path
import concurrent.futures
import fcsparser
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def run_unito_inference(model_dir, disk_prediction_dir, gating_strategy, csv_conversion_dir):
    """Run inference using trained UNITO models to generate gate predictions
    
    Args:
        model_dir (str): Directory containing trained UNITO model files (.pt)
        disk_prediction_dir (str): Output directory where prediction CSV files will be saved
        gating_strategy (pd.DataFrame): DataFrame with columns: Gate, X_axis, Y_axis, Parent_Gate, Step -> Built during generate_gating_strategy() 
        csv_conversion_dir (str): Directory containing CSV files converted from FCS files -> use csv_conversion_dir set before main
    
    Returns:
        None: Saves prediction CSV files to disk_prediction_dir
    """
    
    for csv_file in os.listdir(csv_conversion_dir):
        if not csv_file.endswith('.csv'):
            continue

        print(f"Running inference on {csv_file}...")
        csv_path = os.path.join(csv_conversion_dir, csv_file)
        data = pd.read_csv(csv_path)

        # Run inference for each gate, on each cell
        for _, row in gating_strategy.iterrows():
            gate_name = row['Gate']
            x_axis = row['X_axis']
            y_axis = row['Y_axis']
            
            # Look for trained model (.pt file)
            model_path = os.path.join(model_dir, f"{gate_name}_model.pt")
            if not os.path.exists(model_path):
                print(f"No trained model found for {gate_name}")
                continue
            
            # Load model
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            
            # Prepare input data
            input_data = data[[x_axis, y_axis]].values.astype(np.float32)
            dataset = TensorDataset(torch.from_numpy(input_data))
            dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)
            
            # Run inference
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    batch_input = batch[0]
                    batch_pred = model(batch_input)
                    batch_pred = torch.sigmoid(batch_pred) > 0.5
                    predictions.extend(batch_pred.cpu().numpy())
            
            # Merge predictions onto original csv
            data[f"UNITO_{gate_name}"] = [int(p) for p in predictions]

        # Save the complete CSV with all gate predictions after processing all gates
        sample_name = os.path.splitext(csv_file)[0]
        output_path = os.path.join(disk_prediction_dir, f"{sample_name}_with_predictions.csv")
        data.to_csv(output_path, index=False)
        print(f"Saved complete predictions for {sample_name}")   
        

# def create_hierarchical_gates_from_unito(gating_strategy, disk_prediction_dir, fcs_dir):
#     """Create hierarchical gates that respect parent-child relationships"""
    
#     print("Creating hierarchical gates...")
#     save_fcs_with_gates_path = os.join(disk_prediction_dir, 'fcs_with_hierarchical_unito_gates')
#     if not os.path.exists(save_fcs_with_gates_path):
#         os.makedirs(save_fcs_with_gates_path)

#     # Sort gates by hierarchy (parents before children)
#     gating_strategy_sorted = gating_strategy.sort_values('Step')
    
#     for fcs_file in os.listdir(fcs_dir):
#         if not fcs_file.endswith('.fcs'):
#             continue
            
#         print(f"Processing {fcs_file} for hierarchical gating...")
        
#         # Read original FCS
#         fcs_path = os.path.join(fcs_dir, fcs_file)
#         meta, data = fcsparser.parse(fcs_path, reformat_meta=True)
        
#         # Initialize all gate columns
#         for _, row in gating_strategy_sorted.iterrows():
#             gate_name = row['Gate']
#             data[f"UNITO_{gate_name}"] = 0  # Start with all cells as negative
        
#         # Apply gates in hierarchical order
#         for _, row in gating_strategy_sorted.iterrows():
#             gate_name = row['Gate']
#             parent_gate = row['Parent_Gate']
#             x_axis = row['X_axis']
#             y_axis = row['Y_axis']
            
#             # Load predictions for this gate
#             sample_name = fcs_file.replace('.fcs', '')
            
#             # Try different possible prediction file naming patterns
#             possible_files = [
#                 f"{gate_name}_predictions.csv",
#                 f"{sample_name}_{gate_name}.csv",
#                 f"predictions_{gate_name}.csv"
#             ]
            
#             predictions_file = None
#             for possible_file in possible_files:
#                 test_path = os.path.join(disk_prediction_dir, possible_file)
#                 if os.path.exists(test_path):
#                     predictions_file = test_path
#                     break
            
#             if not predictions_file:
#                 print(f"No predictions found for {gate_name}")
#                 continue
            
#             predictions_df = pd.read_csv(predictions_file)
            
#             # Find cells predicted positive for this gate
#             gate_pred_col = f"{gate_name}_pred" if f"{gate_name}_pred" in predictions_df.columns else gate_name
#             positive_predictions = predictions_df[predictions_df[gate_pred_col] == 1]
            
#             if len(positive_predictions) < 3:
#                 print(f"Not enough positive predictions for {gate_name}")
#                 continue
            
#             # Create gate boundary using ConvexHull
#             points = positive_predictions[[x_axis, y_axis]].to_numpy()
#             try:
#                 hull = ConvexHull(points)
#                 hull_vertices = points[hull.vertices]
                
#                 # Apply gate hierarchically
#                 gate_path = Path(hull_vertices)
                
#                 if parent_gate == 'None' or parent_gate is None:
#                     # Root gate - apply to all cells
#                     cell_coords = data[[x_axis, y_axis]].values
#                     inside_gate = gate_path.contains_points(cell_coords)
#                 else:
#                     # Child gate - only apply to cells that passed parent gate
#                     parent_col = f"UNITO_{parent_gate}"
#                     if parent_col not in data.columns:
#                         print(f"Parent gate {parent_gate} not found for {gate_name}")
#                         continue
                    
#                     # Only consider cells that passed parent gate
#                     parent_positive_mask = data[parent_col] == 1
                    
#                     if parent_positive_mask.sum() == 0:
#                         print(f"No cells passed parent gate {parent_gate} for {gate_name}")
#                         continue
                    
#                     # Apply gate only to parent-positive cells
#                     parent_positive_coords = data.loc[parent_positive_mask, [x_axis, y_axis]].values
#                     inside_gate_subset = gate_path.contains_points(parent_positive_coords)
                    
#                     # Initialize all as negative, then set positive where both parent and current gate are positive
#                     inside_gate = np.zeros(len(data), dtype=bool)
#                     inside_gate[parent_positive_mask] = inside_gate_subset
                
#                 # Set gate values
#                 data[f"UNITO_{gate_name}"] = inside_gate.astype(int)
                
#                 print(f"Applied hierarchical gate {gate_name}: {inside_gate.sum()} cells positive")
                
#             except Exception as e:
#                 print(f"Error creating gate for {gate_name}: {e}")
        
#         # Update FCS metadata for all new parameters
#         original_param_count = int(meta.get('$PAR', 0))
#         new_param_count = original_param_count
        
#         for _, row in gating_strategy_sorted.iterrows():
#             gate_name = row['Gate']
#             param_name = f"UNITO_{gate_name}"
            
#             if param_name in data.columns:
#                 new_param_count += 1
#                 meta[f'$P{new_param_count}N'] = param_name
#                 meta[f'$P{new_param_count}S'] = param_name
#                 meta[f'$P{new_param_count}R'] = '2'
#                 meta[f'$P{new_param_count}B'] = '32'
#                 meta[f'$P{new_param_count}E'] = '0,0'
        
#         meta['$PAR'] = str(len(data.columns))
        
#         # Save FCS with hierarchical gates
#         sample_name = fcs_file.replace('.fcs', '')
#         output_path = os.path.join(save_fcs_with_gates_path, f"{sample_name}_with_hierarchical_UNITO_gates.fcs")
        
#         fcswrite.write_fcs(output_path, data.columns.tolist(), data.values, text_kw_pr=meta,
#                           endianness='little', compat_chn_names=False)
        
#         print(f"Saved hierarchical gates for {sample_name}")
    
#     print("All FCS files processed with hierarchical UNITO gates!")