import flowsom as fs
import flowio
import os
from dataclasses import dataclass, field
from typing import List
import pandas as pd
import concurrent.futures
import ast
import io
import re
import numpy as np

@dataclass
class PipelineConfig:
    """ Configuration of flowsom pipeline """
    # Input paths
    unitogated_csv_dir: str
    csv_dir_metadir: str
    filtered_fcs_path: str
    # FlowSOM settings
    cluster_num: int
    seed: int
    # Cell filter
    filter_out: List = field(default_factory=list)
    # Marker list for flowSOM
    marker_list: List = field(default_factory=list)

    def __dir_assign__(self):
        for path in [
                self.csv_dir_metadir,
                self.filtered_fcs_path,
            ]:
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

class DataFilter:
    """ Filters .csv flow cytometry files"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.csv_files = [f for f in os.listdir(config.unitogated_csv_dir) if
                          f.endswith('.csv')]

    def _filter_out_cell(self, f):
        """ Filters out specified cell types from config.filter_out

        Parameters:
        - f: file to be filtered

            """
        sample_path = os.path.join(self.config.unitogated_csv_dir, f)
        sample_events_df = pd.read_csv(sample_path)
        filtered_df = sample_events_df.copy()
        for c in self.config.filter_out:
            filtered_df = filtered_df[filtered_df[c] != 1]

        # Create new filename with dropped cell type(s)
        base_name = os.path.splitext(f)[0]
        new_filename = f"{base_name}_dropped.csv"
        new_filepath = os.path.join(self.config.filtered_fcs_path, new_filename)

        # Create new sample with filtered data and save
        filtered_df.to_csv(new_filepath, index = False)

        return new_filepath

    def multi_filter_cell(self):
        """ Process pool executor for _filter_out_cell """
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
        print(f"Removed {self.config.filter_out} from .csv files.")

class FCSFileBuilder:
    """ Builds .fcs files from .csv files and their metadata counterparts """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.csv_files = [f for f in os.listdir(config.filtered_fcs_path) if
                          f.endswith('.csv')
                          and not f.endswith('_metadata.csv')]
        self.meta_csv_files =  [f for f in os.listdir(config.csv_dir_metadir) if
                          f.endswith('_metadata.csv')]

    def _parse_metadata_dataframe(self, metadata_df):
        """Parse metadata DataFrame (key-value format) for flowio"""
        self.meta_dict = dict(zip(metadata_df['key'], metadata_df['value']))
        # Get channel names
        channel_names_str = self.meta_dict.get('_channel_names_')
        self.channel_names = ast.literal_eval(channel_names_str)
        # Get channels string
        channels_str = self.meta_dict.get('_channels_')
        if channels_str.startswith('"') and channels_str.endswith('"'):
            channels_str = channels_str[1:-1]
        channels_str_clean = re.sub(r' {2,}', '\t', channels_str)
        channels_str_clean = channels_str_clean.replace('[', '').replace(']', '')
        self.channels_df = pd.read_csv(io.StringIO(channels_str_clean),
                                        sep='\t',
                                        index_col=0)

    def _create_fcs_from_csvs(self, data_csv_path, metadata_csv_path, output_fcs_path):
        """
        Create FCS file from separate data and metadata CSV files using flowio

        Parameters:
        - data_csv_path: path to CSV with flow cytometry event data
        - metadata_csv_path: path to CSV with channel metadata
        - output_fcs_path: path for output FCS file

        """
        metadata_df = pd.read_csv(metadata_csv_path)
        self._parse_metadata_dataframe(metadata_df)

        data_df = pd.read_csv(data_csv_path)
        cols_to_keep = [col for col in self.channel_names if col in data_df.columns]
        data_df = data_df[cols_to_keep]
        data_array = data_df.to_numpy()
        flattened_data = data_array.flatten()

        metadata_dict = {
            '$TOT': self.meta_dict.get('$TOT', str(data_array.shape[0])),
            '$PAR': self.meta_dict.get('$PAR', str(data_array.shape[1])),
            '$MODE': self.meta_dict.get('$MODE', 'L'),
            '$DATATYPE': self.meta_dict.get('$DATATYPE', 'F'),
            '$BYTEORD': self.meta_dict.get('$BYTEORD', '1,2,3,4'),
            '$DATE': self.meta_dict.get('$DATE', '25-Apr-25'),
            '$BTIM': self.meta_dict.get('$BTIM', '12:00:00'),
            '$ETIM': self.meta_dict.get('$ETIM', '12:01:00'),
            '$INST': self.meta_dict.get('$INST', 'Unknown'),
            '$SYS': 'flowio Python CSV Import',
        }

        channels_df_no_header = self.channels_df.drop('Channel Number')
        for i, (idx, row) in enumerate(channels_df_no_header.iterrows(), 1):
            metadata_dict[f'$P{i}N'] = self.channel_names[i-1]
            metadata_dict[f'$P{i}S'] = self.channel_names[i-1]
            if '$PnB' in self.channels_df.columns:
                metadata_dict[f'$P{i}B'] = str(row['$PnB'])
            if '$PnR' in self.channels_df.columns:
                metadata_dict[f'$P{i}R'] = str(row['$PnR'])
            if '$PnE' in self.channels_df.columns:
                metadata_dict[f'$P{i}E'] = str(row['$PnE'])

        for key, value in self.meta_dict.items():
            if key.startswith('$P') and key.endswith('V'):
                metadata_dict[key] = str(value)

        print(f"\nCreating FCS file: {output_fcs_path}")
        try:
            with open(output_fcs_path, 'wb') as fh:
                flowio.create_fcs(
                    file_handle=fh,
                    event_data=flattened_data,
                    channel_names=self.channel_names,
                    metadata_dict=metadata_dict
                )
        except Exception as e:
            print(f"Error creating FCS file: {e}")
            raise

        return output_fcs_path

    def _get_base_name(self, filepath):
        """Extract meaningful part of filenamne for csv and meta matching

        Parameters:
        - filepath: filepath of file to extract filename from.

        """
        return os.path.splitext(os.path.basename(filepath)
                                )[0].replace('_dropped', '').replace('_metadata', '')

    def multi_fcs_create(self):
        """ Process pool executor for _create_fcs_from_csvs """
        file_dict = {self._get_base_name(m): m for m in self.meta_csv_files}
        if file_dict is None:
            raise Exception('meta_csv_files found empty - ensure correct directory chosen')
        matching_pairs = [
            (f, file_dict[self._get_base_name(f)])
            for f in self.csv_files
            if self._get_base_name(f) in file_dict
        ]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self._create_fcs_from_csvs,
                                       os.path.join(self.config.filtered_fcs_path, f),
                                       os.path.join(self.config.csv_dir_metadir, m),
                                       os.path.join(self.config.filtered_fcs_path,
                                                    f.replace('_dropped.csv', '.fcs')))
                        for f, m in matching_pairs]
            self.new_filepath_list = []
            for future in concurrent.futures.as_completed(results):
                try:
                    result = future.result()
                    self.new_filepath_list.append(result)
                except Exception as e:
                    print(f"Error processing file: {e}")
            list_filled = len(self.new_filepath_list) > 0
            print(".fcs files created")
            print(f"self.new_filepath_list:{self.new_filepath_list}")

            return list_filled

class FlowSOMProcessor:
    """ """
    def __init__(self, config: PipelineConfig,
                 datafilter: DataFilter,
                 builder: FCSFileBuilder):
        self.config = config
        self.datafilter = datafilter
        self.builder = builder

    def get_col_idx(self):
        """ Gets col idx from csv for use in flowSOM """
        sample_path = os.path.join(self.config.unitogated_csv_dir,
                                    self.datafilter.csv_files[0])
        sample_events_df = pd.read_csv(sample_path)
        self.marker_col_indices = [sample_events_df.columns.get_loc(col)
                              for col in self.config.marker_list]

    def run_flowsom(self):
        """ Runs FlowSOM module automatically using self.config
        and self.datafilter settings

        Returns:
        - FlowSOM object

        """
        fcs_files_array = np.array(self.builder.new_filepath_list)
        ff = fs.pp.aggregate_flowframes(fcs_files_array,
                                         c_total=100000000)
        fsom = fs.FlowSOM(
            ff,
            cols_to_use = self.marker_col_indices,
            xdim=10,
            ydim=10,
            n_clusters=self.config.cluster_num,
            seed=self.config.seed
        )

        return fsom

class FlowSOMPipeline:
    """ User facing object to run FlowSOM on csv files """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.datafilter = DataFilter(config)
        self.builder = FCSFileBuilder(config)
        self.processor = FlowSOMProcessor(config, self.datafilter, self.builder)

    def run(self):
        self.config.__dir_assign__()
        self.processor.get_col_idx()
        print('.fcs files will be generated from your .csv files for FlowSOM analysis')
        self.datafilter.multi_filter_cell()
        list_filled = self.builder.multi_fcs_create()
        if list_filled:
            fsom = self.processor.run_flowsom()
            return fsom

    def plot_flowSOM(self, fsom):
        """ """
        p = fs.pl.plot_stars(fsom,
                              background_values=fsom.get_cluster_data().obs.metaclustering)
        p.show()

        return p
