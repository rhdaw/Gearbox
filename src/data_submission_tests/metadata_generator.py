import os
import polars as pl
import numpy as np
import re
from pathlib import Path
path = '/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/original_fcs_files'

dir_list = os.listdir(path)
dir_arr = np.array(dir_list, dtype = str)

def extract_metadata(filename):
    """ Regex of the filename to populate metadata """
    site = re.search(r'\s(\w)', filename)
    patient = re.search(r'\s([MS]-\d+-(?:6M|InP)|[MS]-C-\d+)', filename)
    condition = re.search(r'-(\d{3})_Mainstain', filename)

    condition_mapping = {
        '001': 'Excipient',
        '020': 'IFNa2', 
        '040': 'PolyIC',
        '300': 'LPS'
    }

    condition_name = None
    if condition:
        condition_name = condition_mapping.get(condition.group(1))

    return {
        "Filename": filename,
        "batch": None,
        "condition": condition_name,
        "Patient_id": patient.group(1) if patient else None,
        "Site": site.group(1) if site else None
    }


def batch_loc(filename):
    """ Define the batch number based on file location """
    dir = [
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube1_271024_V2/Unmixed/Mainstain', 1), 
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube2_311024_V2/Unmixed/Mainstain', 2), 
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube3_211124_V4/Unmixed/Mainstain', 3), 
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube4_22012025/Unmixed/Mainstain', 4), 
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube5_050225x/Unmixed/Mainstain', 5), 
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube6_120225x/Unmixed/Mainstain', 6),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube7_220225/Unmixed/Mainstain', 7),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube8_260225/Unmixed/Mainstain', 8),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube9_270225/Unmixed/Mainstain', 9),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube10_050325-V2/Unmixed/Mainstain', 10),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube11_110325-V2/Unmixed/Mainstain', 11),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube12_190325-V2/Unmixed/Mainstain', 12),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube13_250325-X2/Unmixed/Mainstain', 13),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube14_250325-X2/Unmixed/Mainstain', 14),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube15_100425-X2/Unmixed/Mainstain', 15),
        ('/Volumes/grainger/Common/stroke_impact_smart_tube/base_flow_files/unzipped_flow_files/StrokeIMPaCT_SmartTube16_250425-XV1/Unmixed/Mainstain', 16),
    ]
    for loc, batch_num in dir:
        for root, dirs, files in os.walk(loc):
            if filename in files:  # Direct check - stops immediately when found
                return batch_num
    return None    


rows = []
for file in dir_arr:
    if file.endswith('.fcs'):
        row_data = extract_metadata(file)
        row_data["batch"] = batch_loc(file) 
        print(f'Found {file} in {row_data["batch"]}')
        rows.append(row_data)


# Create DataFrame from all rows at once (more efficient)
metadata_df = pl.DataFrame(rows)

metadata_dir = Path('/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files')
output_path = metadata_dir / "stroke_impact_metadata_all_batches.csv"
metadata_df.write_csv(output_path, separator=",")


