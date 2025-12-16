from pathlib import Path
import os

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

post_ai_path = Path(
    "/Users/user/Documents/final_stroke_data/flowai_fcs_files_results_qc"
)
directory_files = os.listdir(post_ai_path)

for file in directory_files:
    if "_QC" in file:
        new_filename = file.replace("_QC", "")
        old_full_path = os.path.join(post_ai_path, file)
        new_full_path = os.path.join(post_ai_path, new_filename)

        os.rename(old_full_path, new_full_path)
        print(f"Renamed: {file} -> {new_filename}")

    for ar in additional_remove:
        if ar in file:
            os.remove(os.path.join(post_ai_path, file))
            print(f"removed {file} as matches {ar}")
