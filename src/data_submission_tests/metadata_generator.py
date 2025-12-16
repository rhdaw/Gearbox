import os
import numpy as np
import re
import pandas as pd
from pathlib import Path
import shutil

path = "/Users/user/Documents/final_stroke_data/complete_stroke_data"
move_dir = "/Users/user/Documents/final_stroke_data/original_fcs_files"
metadata_dir = Path("/Users/user/Documents/final_stroke_data/metadata_files")
output_path = metadata_dir / "stroke_impact_metadata_all_batches.csv"


dir_list = os.listdir(move_dir)
dir_arr = np.array(dir_list, dtype=str)
mainstain_dir = [
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube1_271024_V2_ST_UM2510/Unmixed/Mainstain",
        1,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube10_050325-UM2508/Unmixed/Mainstain",
        10,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube11_110325-UM2508/Unmixed/Mainstain",
        11,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube12_190325-UM2508/Unmixed/Mainstain",
        12,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube13_250325-UM1508/Unmixed/Mainstain",
        13,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube14_250325_UM1208/Unmixed/Mainstain",
        14,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube15_100425-UN1508/Unmixed/Mainstain",
        15,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube16_250425-UM2508/Unmixed/Mainstain",
        16,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube17_26082025-UM2808/Unmixed/Mainstain",
        17,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube18_24072025_UM0608/Unmixed/Mainstain",
        18,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube2_311024_V2_ST_UM2510/Unmixed/Mainstain",
        2,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube3_211124_V4_ST_UM2510/Unmixed/Mainstain",
        3,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube4_22012025_ST_UM2510/Unmixed/Mainstain",
        4,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube5_050225x 2_ST_UM2510/Unmixed/Mainstain",
        5,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube6_120225x_ST_UM2510_new/Unmixed/Mainstain",
        6,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube7_220225_ST_UM2510/Unmixed/Mainstain",
        7,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube8_260225_ST_UM2510/Unmixed/Mainstain",
        8,
    ),
    (
        "/Users/user/Documents/final_stroke_data/complete_stroke_data/StrokeIMPaCT_SmartTube9_270225_UM2510_ST_2/Unmixed/Mainstain",
        9,
    ),
]


def move_fcs_files(mainstain_dir, move_to_dir):
    fcs_map = {}
    for md in mainstain_dir:
        base_dir = md[0]
        matches = []
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.lower().endswith(".fcs"):
                    matches.append(os.path.join(root, fname))
        fcs_map[base_dir] = matches

    for paths in fcs_map.values():
        for p in paths:
            dest = os.path.join(move_to_dir, os.path.basename(p))
            shutil.copy(p, dest)


def extract_metadata(filename):
    """Regex of the filename to populate metadata"""
    site = re.search(r"\s(\w)", filename)
    patient = re.search(r"\s([MS]-\d+-(?:6M|InP)|[MS]-C-\d+)", filename)
    condition = re.search(r"-(\d{3})_Mainstain", filename)

    condition_mapping = {
        "001": "Excipient",
        "020": "IFNa2",
        "040": "PolyIC",
        "300": "LPS",
    }

    condition_name = None
    if condition:
        condition_name = condition_mapping.get(condition.group(1))

    return {
        "Filename": filename,
        "batch": None,
        "condition": condition_name,
        "Patient_id": patient.group(1) if patient else None,
        "Site": site.group(1) if site else None,
    }


def batch_loc(filename, dir):
    """Define the batch number based on file location"""
    for loc, batch_num in dir:
        for root, dirs, files in os.walk(loc):
            if filename in files:
                return batch_num
    return None


def apply_qc_criteria(
    flowai_qc_file_path,
    fcs_file_dir,
    event_criteria: int = 29999,
    percent_anomal: int = 40,
):
    qc_file = pd.read_excel(flowai_qc_file_path)
    df_to_remove = pd.merge(
        qc_file.query("`n. of events` <= @event_criteria"),
        qc_file.query("`% anomalies` >= @percent_anomal"),
    )
    remove_list = df_to_remove["Name file"].tolist()

    for rl in remove_list:
        rl += ".fcs"
        if rl in os.listdir(fcs_file_dir):
            os.remove(os.path.join(fcs_file_dir, rl))
            print(f"Removed {rl} - does not meet qc criteria")


move_fcs_files(mainstain_dir, move_dir)
apply_qc_criteria(
    "/Users/user/Documents/final_stroke_data/flowai_fcs_files_results_qc/Flow_AI_QC.xlsx",
    move_dir,
)

rows = []
for file in dir_arr:
    if file.endswith(".fcs"):
        row_data = extract_metadata(file)
        row_data["batch"] = batch_loc(file, mainstain_dir)
        print(f"Found {file} in {row_data['batch']}")
        rows.append(row_data)


additional_remove = [
    "A12 Unstained_Mainstain.fcs",
    "C12 Unstained_Mainstain.fcs",
    "D12 Unstained_Mainstain.fcs",
    "A11 Unstained_Mainstain.fcs",
    "B11 Unstained_Mainstain.fcs",
    "G10 Unstained_Mainstain.fcs",
    "G10 Batchcontrol_PolyIC_040_20112024_Mainstain.fcs",
    "G9 Batchcontrol_LPS_300_20112024_Mainstain.fcs",
    "H1 Unstained_Mainstain.fcs",
    "H2 Unstained_Mainstain.fcs",
    "A5 Batchcontrol3_001_Mainstain.fcs",
    "A6 Batchcontrol3_IFNa2_Mainstain.fcs",
    "A8 Batchcontrol_PolyIC_Mainstain.fcs",
    "H1 BATCHCONTROL_Batch8_001_Mainstain.fcs",
    "H2 BATCHCONTROL_Batch8_020_Mainstain.fcs",
    "H3 BATCHCONTROL_Batch8_300_Mainstain.fcs",
    "H10 Unstained_Mainstain.fcs",
    "H11 Unstained_Mainstain.fcs",
    "H12 Unstained_Mainstain.fcs",
    "H5 Unstained_Mainstain.fcs",
    "H6 Unstained_Mainstain.fcs",
    "H7 Unstained_Mainstain.fcs",
    "H8 Unstained_Mainstain.fcs",
    "H9 Unstained_Mainstain.fcs",
    "H1 Batchcontrol_190325_hc3_001_Mainstain.fcs",
    "H2 Batchcontrol_190325_hc3_020_Mainstain.fcs",
    "H3 Batchcontrol_190325_hc3_300_Mainstain.fcs",
    "H4 Batchcontrol13_250325_hc3_001_Mainstain.fcs",
    "H5 Batchcontrol13_250325_hc3_020_Mainstain.fcs",
    "H6 Batchcontrol13_250325_hc3_300_Mainstain.fcs",
    "G5 H5 Batchcontrol25042025_IFN3_Mainstain.fcs",
    "E4 S-127-InP-020_Mainstain.fcs",
    "F2 S-117-InP-300_Mainstain.fcs",
    "E3 S-108-InP-020_Mainstain.fcs",
    "E1 S-180-6M-020_Mainstain.fcs",
    "E2 S-180-6M-300_Mainstain.fcs",
    "D7 S-178-InP-001_Mainstain.fcs",
    "C5 M-163-InP-001_Mainstain.fcs",
    "G6 S-244-InP-020_Mainstain.fcs",
    "E4 S-108-InP-300_Mainstain.fcs",
    "E6 S-110-InP-020_Mainstain.fcs",
    "F8 S-171-6M-020_Mainstain.fcs",
    "E6 S-182-6M-001_Mainstain.fcs",
    "E9 S-124-InP-020_Mainstain.fcs",
    "E10 S-124-InP-300_Mainstain.fcs",
    "E8 S-182-6M-300_Mainstain.fcs",
    "E7 S-182-6M-020_Mainstain.fcs",
    "H1 S-194-6M-040_Mainstain.fcs",
    "E7 S-110-InP-300_Mainstain.fcs",
    "F7 S-193-InP-001_Mainstain.fcs",
]

for ar in additional_remove:
    if ar in os.listdir(move_dir):
        os.remove(os.path.join(move_dir, ar))
    if ar in rows:
        rows.remove(ar)
    print(f"additional removal of {ar}")


# Create DataFrame from all rows at once (more efficient)
metadata_df = pd.DataFrame(rows)
metadata_df.to_csv(output_path)
