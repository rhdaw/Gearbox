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

Multiple functions and modules are designed to help transition files between pipeline steps. Mostly, these helper functions are built around UNITO to get it working correctly with little input. Details on these helper functions can be found in the documentation below. Functions specific to flowAI, CyCombine, and UNITO can be found in their respective repos.

## Additional Requirements:
- Metadata and Paneldata .csv files according to Cycombine's documented requirements (https://github.com/biosurf/cyCombine).
- Gated .fcs files held in a FlowJo WSP, with WSP and .fcs files together in a specific directory.
- Note: Gearbox does not require a manual gating strategy .csv as per UNITO's limited documentation (https://github.com/KyleeCJ/UNITO) - Gearbox contains a helper function to generate the required gating strategy (in the correct format) directly from the FlowJo WSP file.


## Pipeline
