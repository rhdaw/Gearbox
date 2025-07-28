from unito_objects import PipelineConfig, UNITOPipeline

def main():

    config = PipelineConfig(
        fcs_dir='/path/to/fcs_files/',
        wsp_path='/path/to/WSP_file.wsp',
        wsp_files_dir='/path/to/wsp_files_dir/',
        panel_meta_path='/path/to/panel_metadata.csv',
        ram_disk= True,
        csv_conversion_dir='/path/to/csv_conversion_dir/',
        disk_dest='/path/to/disk_dest/',
        default_hyperparameters=[[1e-3, 128], [1e-4, 256], [5e-4, 512]],
        problematic_gate_hyperparameters=[[1e-4, 32], [1e-5, 128], [2e-4, 64]],
        problematic_gate_list=['Cell_gate', 'Cell_gate_2'],
        downsample_max_rows=200_000,
        n_worker=30,
        device='mps'
    )

    # Run pipeline
    pipeline = UNITOPipeline(config)
    pipeline.run(downsample=True)

if __name__ == '__main__':
    main()
