from unito_objects import PipelineConfig, UNITOPipeline


def main():
    config = PipelineConfig(
        fcs_dir="/Users/user/Documents/final_stroke_data/flowai_fcs_files_results_qc/",  # Dir of your .fcs files
        wsp_path="/Users/user/Documents/flow_gating/50samples_CD15pos_gating_same_for_every_sample/StrokeIMPaCT_GatesForTraining_CD15pos_50.wsp",  # Location of you .wsp file
        wsp_files_dir="/Users/user/Documents/flow_gating/50samples_CD15pos_gating_same_for_every_sample",  # Dir holding the .wsp
        panel_meta_path="/Users/user/Documents/final_stroke_data/metadata_files/panel_metadata_all_batches.csv",  # Path to panel metadatafile needed for cycombine
        ram_disk=False,  # Want to use a RAMDisk (MacOS only)?
        csv_conversion_dir="/Users/user/Documents/UNITO_csv_conversion/",
        csv_conversion_dir_metadir="/Users/user/Documents/UNITO_csv_conversion/metadata/",
        csv_conversion_dir_predict="/Users/user/Documents/UNITO_csv_conversion_predict/",
        disk_dest="/Users/user/Documents/UNITO_train_data/",  # If using a RAMDisk - this is the physical save location for outputs
        default_hyperparameters=[
            [1e-3, 8],
            [1e-4, 8],
            [1e-3, 16],
            [1e-4, 16],
        ],
        problematic_gate_hyperparameters=[
            [5e-4, 8],
            [2e-4, 8],
            [1e-4, 8],
            [5e-5, 8],
            [5e-4, 16],
            [2e-4, 16],
            [1e-4, 16],
            [5e-5, 16],
        ],
        problematic_gate_list=[
            "Single Cells",
            "Neutrophils",
        ],  # Must match gate labels
        epochs=1000,
        problematic_epochs=2000,  # More epochs for problematic gates
        downsample_max_rows=600_000,  # Downsample number of events per .fcs file to train on
        n_worker=0,
        device="mps",  # use CPU if on Windows
    )

    # Run pipeline
    pipeline = UNITOPipeline(config)
    # pipeline.run(downsample=False)

    pipeline.predict_only(
        predict_dir="/Users/user/Documents/UNITO_csv_conversion/train"
    )


if __name__ == "__main__":
    main()
