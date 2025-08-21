from unito_objects import PipelineConfig, UNITOPipeline

def main():

    config = PipelineConfig(
        fcs_dir='/Users/user/Documents/UNITO_csv_conversion/fcs_dir_testing/', # Dir of your .fcs files
        wsp_path='/Users/user/Documents/UNITO_train_wsp/WSP_22052025.wsp', # Location of you .wsp file
        wsp_files_dir='/Users/user/Documents/UNITO_train_wsp/', # Dir holding the .wsp
        panel_meta_path='/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/panel_metadata_all_batches.csv', # Path to panel metadatafile needed for cycombine
        ram_disk= True, # Want to use a RAMDisk (MacOS only)?
        csv_conversion_dir='/Users/user/Documents/UNITO_csv_conversion/', # Dir for the converted files to go to
        csv_conversion_dir_metadir='/Users/user/Documents/UNITO_csv_conversion/metadata/', # Dir for the converstedd file metadata to go to (must be different from above)
        disk_dest='/Users/user/Documents/UNITO_train_data/', # If using a RAMDisk - this is the physical save location for outputs
        default_hyperparameters=[
            [1e-3, 128],
            [1e-4, 256],
            [5e-4, 512]
            ],
        problematic_gate_hyperparameters=[
            [1e-4, 16],
            [5e-6, 32],
            [1e-6, 64],
            [1e-5, 8]
            ],
        problematic_gate_list=['Neutrophils', 'Non-neutrophil leukocytes'], # Must match gate labels
        downsample_max_rows=100_000, # Downsample number of events per .fcs file to train on
        n_worker=30,
        device='mps' # use CPU if on Windows
    )

    # Run pipeline
    pipeline = UNITOPipeline(config)
    pipeline.run(downsample=True)

if __name__ == '__main__':
    main()
