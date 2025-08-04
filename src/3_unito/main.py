from unito_objects import PipelineConfig, UNITOPipeline

def main():

    config = PipelineConfig(
        fcs_dir='/Users/user/Documents/UNITO_csv_conversion/fcs_dir_testing/',
        wsp_path='/Users/user/Documents/UNITO_train_wsp/WSP_22052025.wsp',
        wsp_files_dir='/Users/user/Documents/UNITO_train_wsp/',
        panel_meta_path='/Volumes/grainger/Common/stroke_impact_smart_tube/computational_outputs/fcs_files/metadata_files/panel_metadata_all_batches.csv',
        ram_disk= True,
        csv_conversion_dir='/Users/user/Documents/UNITO_csv_conversion/',
        csv_conversion_dir_metadir='/Users/user/Documents/UNITO_csv_conversion/metadata',
        disk_dest='/Users/user/Documents/UNITO_train_data',
        default_hyperparameters=[
            [1e-3, 64],
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
        problematic_gate_list=['Neutrophils', 'Non-neutrophil leukocytes'],
        downsample_max_rows=200_000,
        n_worker=30,
        device='mps'
    )

    # Run pipeline
    pipeline = UNITOPipeline(config)
    pipeline.run(downsample=True)

if __name__ == '__main__':
    main()
