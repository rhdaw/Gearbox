import numpy as np

def calc_gmfi(sample_df, marker, gate):
    """
    Calculate geometric Mean Fluorescence Intensity (gMFI) of a marker within a gated population.
    
    Parameters
    ----------
    sample_df : pandas.DataFrame
        DataFrame containing flow cytometry data, typically obtained from 
        flowkit Sample.as_dataframe(). Should contain both marker intensity 
        columns and UNITO gate columns (0/1 values).
    marker : str
        Name of the marker column for which to calculate gMFI (e.g., 'CD45', 'Ki67').
        This should be the fluorescence intensity parameter, not the gate parameter.
    gate : str
        Name of the gate column to filter cells (e.g., 'UNITO_Lymphocytes', 'UNITO_CD4_Tcells').
        Expected to contain binary values (0/1) indicating gate membership.
    
    Returns
    -------
    float
        Geometric mean fluorescence intensity of the specified marker within 
        the gated population. Returns NaN if no cells are positive for the gate.
    
    Examples
    --------
    >>> sample = fk.Sample('sample.fcs')
    >>> df = sample.as_dataframe()
    >>> cd45_gmfi = calc_gmfi(df, 'CD45', 'UNITO_Lymphocytes')
    >>> ki67_gmfi = calc_gmfi(df, 'Ki67', 'UNITO_CD4_Tcells')
    """
    gate_positive = sample_df[gate] == 1
    marker_intensities = sample_df.loc[gate_positive, marker]
    result = np.exp(np.log(marker_intensities + 1).mean()) - 1
    return result

def calc_percent_gate(sample_df, marker, gate):
    """
    Calculate % expression of a marker relative to a gate.
    
    Parameters
    ----------
    sample_df : pandas.DataFrame
        DataFrame containing flow cytometry data, typically obtained from 
        flowkit Sample.as_dataframe(). Should contain both marker intensity 
        columns and UNITO gate columns (0/1 values).
    marker : str
        Name of the marker column for which to calculate percent expression (e.g., 'CD45', 'Ki67').
        This should be the fluorescence intensity parameter, not the gate parameter.
    gate : str
        Name of the gate column to filter cells (e.g., 'UNITO_Lymphocytes', 'UNITO_CD4_Tcells').
        Expected to contain binary values (0/1) indicating gate membership.
    
    Returns
    -------
    float
        Percentage marker expression of the specified marker within 
        the relative population. Returns NaN if no cells are positive for the gate.
    
    Examples
    --------
    >>> sample = fk.Sample('sample.fcs')
    >>> df = sample.as_dataframe()
    >>> cd45_percent_lymph = calc_percent_gate(df, 'CD45', 'UNITO_Lymphocytes')
    >>> ki67_percent_tcell = calc_percent_gate(df, 'Ki67', 'UNITO_CD4_Tcells')
    """
    gate_positive = sample_df[gate] == 1


def calc_percent_marker_positive(sample_df, marker, gate, threshold):
    """Calculate % of cells positive for a marker within a gate"""
    gate_positive = sample_df[gate] == 1
    marker_positive_in_gate = (sample_df.loc[gate_positive, marker] > threshold).sum()
    return marker_positive_in_gate / gate_positive.sum() * 100