import os
import pandas as pd
import concurrent.futures
import os

def apply_predictions_to_csv(all_predictions, csv_conversion_dir):
    """Apply UNITO gate predictions stored in the all_predictions dictionary to test CSV files

    Args:
        all_predictions (dict): Nested dict {filename: {gate_name: [predictions]}}
        csv_conversion_dir (str): Directory containing original test CSV files
    """

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(_csv_load_addgates_save, gate_predictions, csv_conversion_dir, filename) for filename, gate_predictions in all_predictions.items()]
        
        for future in concurrent.futures.as_completed(results):
            try:
                future.result() 
            except Exception as e:
                print(f"Error processing file: {e}")


def _csv_load_addgates_save(gate_predictions, csv_conversion_dir, filename):
    """Load CSV, add gate predictions, and save
    
    Args:
        perfile_gate_predictions (dict): {gate_name: [predictions]} for this specific file
        csv_conversion_dir (str): Directory containing CSV files
        filename (str): Name of the CSV file to process
    """
    csv_file = f"{filename}.csv"
    csv_path = os.path.join(csv_conversion_dir, csv_file)

    if not os.path.exists(csv_path):
        print(f"ERROR: File {csv_file} not found in {csv_conversion_dir}")
        return None

    original_data = pd.read_csv(csv_path)
    
    # Add predictions to csv as new columns
    gates_added = 0
    for gate_name, predictions in gate_predictions.items():
        
        if len(predictions) != len(original_data):
            print(f"WARNING: Prediction mismatch for {gate_name}: {len(predictions)} vs {len(original_data)} - SKIPPING")
            continue
        
        original_data[f'UNITO_{gate_name}'] = predictions
        gates_added += 1
    
    if gates_added == 0:
        print(f"WARNING: No gates successfully added to {filename}")
        return None
    
    output_path = os.path.join(csv_conversion_dir, f"{filename}_with_UNITO_predictions.csv")
    original_data.to_csv(output_path, index=False)
    
    print(f"Saved {filename} with UNITO gate predictions to {output_path}")
    return filename