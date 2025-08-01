import flowsom as fs
import os
import flowkit as fk
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import concurrent.futures

@dataclass
class PipelineConfig:
    """ Configuration of flowsom pipeline """
    # Input paths
    unitogated_csv_dir: str
    filtered_fcs_path: str
    # FlowSOM settings
    cluster_num: int
    seed: int
    # Cell filter
    filter_out: List = field(default_factory=list)
    # Marker list for flowSOM
    marker_list: List = field(default_factory=list)

class DataFilterConverter:
    """ Filters and Converts .csv flow cytometry files to .fcs files."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.csv_files = [f for f in os.listdir(config.unitogated_csv_dir) if
                          f.endswith('.csv')]

    def _filter_out_cell(self, f):
        """ Filters out specified cell types from config.filter_out,
            saves .csv as .fcs for flowSOM """
        sample_path = os.path.join(self.config.unitogated_csv_dir, f)
        sample_events_df = pd.read_csv(sample_path)
        filtered_df = sample_events_df.copy()
        for c in self.config.filter_out:
            filtered_df = filtered_df[filtered_df[c] != 1]

        # Create new filename with dropped cell type(s)
        base_name = os.path.splitext(f)[0]
        new_filename = f"{base_name}_dropped.fcs"
        new_filepath = os.path.join(self.config.filtered_fcs_path, new_filename)

        # Create new sample with filtered data and save
        filtered_sample = fk.Sample(filtered_df,
                                    sample_id=base_name,
                                    parameters=parameters)
        filtered_sample.export(new_filepath, source='orig')
        print(f"Removed {self.config.filter_out} from .csv file.")
        print(f"Saved filtered file now .fcs: {new_filename}")

        return new_filepath

    def multi_filter_cell(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self._filter_out_cell, f)
                        for f in self.csv_files]
            self.new_filepath_list = []
            for future in concurrent.futures.as_completed(results):
                try:
                    result = future.result()
                    self.new_filepath_list.append(result)
                except Exception as e:
                    print(f"Error processing file: {e}")

class FlowSOMProcessor:
    """ """
    def __init__(self, config: PipelineConfig, datafilter: DataFilterConverter):
        self.config = config
        self.datafilter = datafilter

    def get_col_idx(self):
        """ Gets col idx from csv for use in flowSOM """
        sample_path = os.path.join(self.config.unitogated_csv_dir,
                                    self.datafilter.csv_files[0])
        sample_events_df = pd.read_csv(sample_path)
        self.marker_col_indices = [sample_events_df.columns.get_loc(col)
                              for col in self.config.marker_list]

    def run_flowSOM(self):
        """ """
        ff = fs.pp.aggregate_flowframes(self.datafilter.new_filepath_list, c_total=100000000)
        fsom = fs.FlowSOM(
            ff, cols_to_use = self.marker_col_indices,
            xdim=10,
            ydim=10,
            n_clusters=self.config.cluster_num,
            seed=self.config.seed
        )

        return fsom

class FlowSOMPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.datafilter = DataFilterConverter(config)
        self.processor = FlowSOMProcessor(config, self.datafilter)

    def run(self):
        self.processor.get_col_idx()
        self.datafilter.multi_filter_cell()
        fsom = self.processor.run_flowSOM()
        return fsom

    def plot_flowSOM(self, fsom):
        """ """
        p = fs.pl.plot_stars(fsom,
                              background_values=fsom.get_cluster_data().obs.metaclustering)
        p.show()

        return p
