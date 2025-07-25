from pathlib import Path
import os

post_ai_path = Path('/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/altered_fcs_files/post_flowai')
directory_files = os.listdir(post_ai_path)

for file in directory_files:
    if '_QC' in file:
        new_filename = file.replace('_QC', '')
        old_full_path = os.path.join(post_ai_path, file)
        new_full_path = os.path.join(post_ai_path, new_filename)
        
        os.rename(old_full_path, new_full_path)
        print(f'Renamed: {file} -> {new_filename}')