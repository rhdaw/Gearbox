from pathlib import Path
from unito_objects import PipelineConfig, UNITOPipeline

def main():
    # Configuration
    config = PipelineConfig(
        fcs_dir=Path('/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/altered_fcs_files/post_flowai/'),
        wsp_path=Path('/Users/user/Documents/UNITO_train_wsp/WSP_22052025.wsp'),
        wsp_files_dir=Path('/Users/user/Documents/UNITO_train_wsp/'),
        panel_meta_path=Path('/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/panel_metadata_all_batches.csv'),
        csv_conversion_dir=Path('/Users/user/Documents/UNITO_csv_conversion/'),
        disk_dest=Path('/Users/user/Documents/UNITO_train_data'),
        default_hyperparameters=[[1e-3, 128], [1e-4, 256], [5e-4, 512]],
        problematic_gate_hyperparameters=[[1e-4, 32], [1e-5, 128], [2e-4, 64]]
    )

    # Run pipeline
    pipeline = UNITOPipeline(config)
    pipeline.run(use_ram_disk=True)

if __name__ == '__main__':
    main()