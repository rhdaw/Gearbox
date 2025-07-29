# Gearbox
 
A pipeline for .fcs file QC, Batch Correction, ML-based gating, and data output.
Designed to be used on a local HPC, to perform analysis of 100s-1000s of .fcs files in one go.
Works well with both standard and spectral flow cytometry data. The pipeline was originally built using spectral flow cytometry data.

## Relies on the following major packages:
1) flowAI (R)
2) CyCombine (Py)
3) UNITO (Py)

For required py packages, please see the pyproject.toml file.

## For installation of the correct UNITO version, install to your .venv via:
Using pip:
>pip install git+https://github.com/happyhypocrite/UNITO
Using uv:
>uv add git+https://github.com/happyhypocrite/UNITO

Multiple functions and modules are designed to help transition files between pipeline steps. Mostly, these helper functions are built around UNITO to get it working correctly with little input. Details on these helper functions can be found in the documentation below (See 'UNITO Pipeline Documentation' below). Functions specific to flowAI, CyCombine, and UNITO can be found in their respective repos.

## Additional Requirements:
- Metadata and Paneldata .csv files according to Cycombine's documented requirements (https://github.com/biosurf/cyCombine).
- Gated .fcs files held in a FlowJo WSP, with WSP and .fcs files together in a specific directory.
- Note: Gearbox does not require a manual gating strategy .csv as per UNITO's limited documentation (https://github.com/KyleeCJ/UNITO) - Gearbox contains a helper function to generate the required gating strategy (in the correct format) directly from the FlowJo WSP file (which must contain gated-samples).


# UNITO Pipeline Documentation

## Overview
The UNITO Pipeline built here automates flow cytometry data processing, gating strategy extraction, machine learning model training, and evaluation. It supports FCS-to-CSV conversion, hierarchical gating, and robust classification of cell populations (e.g., neutrophils, lymphocytes) using configurable hyperparameters and optional RAM disk acceleration.

## Features
- FCS to CSV conversion
- Automated gating strategy extraction from WSP files
- Configurable machine learning training and evaluation
- Gate-specific hyperparameter tuning
- RAM disk support for high-speed I/O
- Parallel processing for scalability

## Requirements
- Python 3.9+ (Built in Python 3.9.12)
- Required packages: see pyproject.toml
- macOS (RAM disk support tested)

## Installation
```bash
git clone https://github.com/your-org/Gearbox.git
cd Gearbox
python -m venv .venv
source .venv/bin/activate

Then:
# Using pip:
pip install -r requirements.txt
# Or using uv:
uv pip install -r pyproject.toml
```

## Configuration
All pipeline settings are managed via the PipelineConfig object in main.py.
### Example:
```Python
config = PipelineConfig(
    fcs_dir='/path/to/fcs_files/',
    wsp_path='/path/to/wsp_file.wsp',
    wsp_files_dir='/path/to/wsp_files/',
    panel_meta_path='/path/to/panel_metadata.csv',
    ram_disk=True,
    csv_conversion_dir='/path/to/csv_conversion/',
    disk_dest='/path/to/output_data',
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
    problematic_gate_list=['cell_gate_1', 'cell_gate_2'],
    downsample_max_rows=200_000,
    n_worker=30,
    device='mps'
)
```
## Usage
Run the pipeline from the command line:
``` bash 
python src/3_unito/main.py
``` 

## Output
- Processed CSV files with hierarchical gates
- Trained model metrics (accuracy, precision, recall, F1)
- Hyperparameter tuning results (hyperparameter_tuning.csv)
- Print statements for logs and error reports

## Troubleshooting
- RAM disk errors: Ensure macOS used and sufficient memory. Check environment variable UNITO_DEST.
- Path errors: Verify all input/output paths exist and are accessible.
- Performance: Adjust n_worker and batch sizes for your hardware.
- Poor ML performance: adjust downsample size, epochs and hyperparameters in PipelineConfig.
- Slow runtime: adjust to cuda if applicable, enable ram_disk.

## Contributing
- Fork the repo, create a feature branch, and submit a pull request.
- Please add tests and update documentation for new features.

## License
MIT license

## Example Workflow
1) Place your FCS files in the specified directory.
2) Update main.py with your configuration.
3) Run the pipeline.
4) Review output metrics and processed files.
