import os
import warnings
import ssl
import urllib3
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
# Env settings
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning) 

# Standard Imports
from pathlib import Path
import concurrent.futures
import random
import shutil
import fcsparser
import torch
import subprocess
import numpy as np
import psutil
from contextlib import contextmanager

# UNITO Imports
from UNITO_Train_Predict.hyperparameter_tunning import tune
from UNITO_Train_Predict.Train import train
from UNITO_Train_Predict.Validation_Recon_Plot_Single import plot_all
from UNITO_Train_Predict.Data_Preprocessing import process_table, train_test_val_split
from UNITO_Train_Predict.Predict import UNITO_gating, evaluation

# Own Imports
from generate_gating_strategy import parse_fcs_add_gate_label, extract_gating_strategy, clean_gating_strategy, add_gate_labels_to_test_files
from apply_unito_to_fcs import apply_predictions_to_csv

@dataclass
class PipelineConfig:
    """Configuration for the UNITO pipeline"""
    # Input paths
    fcs_dir: str
    wsp_path: str
    wsp_files_dir: str
    panel_meta_path: str
    ram_disk: bool
    # Output paths
    csv_conversion_dir: str
    disk_dest: str
    # Processing settings
    max_rows: int = 200_000
    device: str = 'mps'
    n_worker: int = 30
    epochs: int = 7
    problematic_epochs: int = 15
    # Hyperparameters
    default_hyperparameters: List = []
    problematic_gate_hyperparameters: List = []

    def __dir_assign__(self, ram_disk: bool):
        if self.ram_disk:
            dest = os.getenv("UNITO_DEST")
            save_data_img_path   = f"{dest}/Data/"
            save_figure_path     = f"{dest}/figures/"
            save_model_path      = f"{dest}/model/"
            save_prediction_path = f"{dest}/prediction/"
            downsample_path      = f"{dest}/downsample/"
        else:
            dest = self.disk_dest
            self.save_data_img_path = f"{dest}/Data/"
            self.save_figure_path = f"{dest}/figures/"
            self.save_model_path = f"{dest}/model/"
            self.save_prediction_path = f"{dest}/prediction/"
            self.downsample_path = f"{dest}/downsample/"
        for path in [
            save_data_img_path,
            save_figure_path,
            save_model_path,
            save_prediction_path,
            downsample_path
        ]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

class FileConverter:
    """Handles FCS to CSV conversion"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.fcs_files = [f for f in os.listdir(config.fcs_dir) if f.endswith('.fcs')]

    def convert_all_fcs(self) -> None:
        """ Convert all FCS files to CSV """
        print("Converting FCS files to CSV...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self._convert_fcs_to_csv,
                                       os.path.join(self.config.fcs_dir, fcs_file),
                                       self.config.csv_conversion_dir)
                       for fcs_file in self.fcs_files]
            for future in concurrent.futures.as_completed(results):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")
        print("FCS to CSV conversion complete!")

    def _convert_fcs_to_csv(self, fcs_file: str, output_dir: str) -> None:
        """ Generate .csv of fcs_file required for UNITO processing """
        fcs_filename = os.path.basename(fcs_file)
        meta, data = fcsparser.parse(fcs_file, reformat_meta=True)
        df = pd.DataFrame(data)
        csv_filename = fcs_filename.replace('.fcs', '.csv')
        df_output = os.path.join(self.config.csv_conversion_dir, csv_filename)
        df.to_csv(df_output, index=False)
        print(f'{fcs_filename} converted to csv')

    def downsample_csv(self, csv_file: str, max_rows: int, out_dir: str) -> str:
        """Downsample a CSV file to max_rows and save to out_dir"""
        df = pd.read_csv(csv_file)
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=0)
        out_path = os.path.join(out_dir, os.path.basename(csv_file))
        df.to_csv(out_path, index=False)
        return out_path

class GateProcessor:
    """Handles gate parsing and strategy generation"""
    def __init__(self, config: PipelineConfig):
        self.config = config

    def parse_gates(self) -> None:
        """ Parse gates from WSP file and add to CSV files """
        print("Parsing gates from WSP file...")
        parse_fcs_add_gate_label(self.config.wsp_path,
                                 self.config.wsp_files_dir,
                                 self.config.csv_conversion_dir)
        print("Gate parsing complete!")

    def generate_gating_strategy(self) -> pd.DataFrame:
        """ Generate and save gating strategy """
        print("Generating gating strategy...")
        gating_strategy = extract_gating_strategy(self.config.wsp_path,
                                                  self.config.wsp_files_dir)
        final_gating_strategy = clean_gating_strategy(self.config.panel_meta_path,
                                                      gating_strategy)
        out_file = os.path.join('./', "gating_strategy.csv")
        final_gating_strategy.to_csv(out_file, index=False)
        print("Gating strategy saved!")
        return final_gating_strategy

class UNITOTrainer:
    """Handles UNITO training and prediction"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.hyperparameter_df = pd.DataFrame(columns=['gate',
                                                       'learning_rate',
                                                       'batch_size'])
        self.all_predictions = {}

    def train_gate(self, gate_config: Dict) -> Dict:
        """Train a single gate"""
        gate = gate_config['gate']
        hyperparameters = self._get_hyperparameters(gate)
        # Your existing 9a-9e logic for a single gate
        # Return metrics
    def _get_hyperparameters(self, gate: str) -> List:
        """Choose hyperparameters based on gate type"""
        if 'neutrophil' in gate.lower():
            return self.config.problematic_gate_hyperparameters
        return self.config.default_hyperparameters

class RAMDiskManager:
    """Handles RAM disk operations"""
    def __init__(self, config: PipelineConfig):
        self.config = config

    @staticmethod
    def mount_ramdisk(ram_disk: bool) -> None:
        """Your existing mount_ramdisk logic"""
        # ... existing code ...
    @staticmethod
    def cleanup_ramdisk() -> None:
        """Your existing cleanup_ramdisk logic"""
        # ... existing code ...
    @staticmethod
    def flush_ramdisk_to_disk(disk_dest: str) -> None:
        """Your existing flush logic"""
        # ... existing code ...

class UNITOPipeline:
    """Main pipeline orchestrator"""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.converter = FileConverter(config)
        self.gate_processor = GateProcessor(config)
        self.trainer = UNITOTrainer(config)

    def run(self,
            ram_disk: bool = True,
            downsample: bool = True,
            max_rows: Optional[int] = None) -> None:
        """Run the complete pipeline"""
        self.config.__dir_assign__(ram_disk)
        if ram_disk:
            RAMDiskManager.mount_ramdisk(True)
        try:
            # Pytorch settings
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)
            # Step 1: Convert FCS files
            # self.converter.convert_all_fcs()
            # Step 2: Parse gates
            # self.gate_processor.parse_gates()
            # Step 3: Generate strategy
            gating_strategy = self.gate_processor.generate_gating_strategy()
            gate_pre_list = list(gating_strategy.Parent_Gate)
            gate_pre_list[0] = None # the first gate does not have parent gate
            gate_list = list(gating_strategy.Gate)
            x_axis_list = list(gating_strategy.X_axis)
            y_axis_list = list(gating_strategy.Y_axis)

            # Step 4: Downsampling
            if downsample:
                max_rows = self.config.max_rows
                csv_files = [f for f in os.listdir(self.config.csv_conversion_dir)
                             if f.endswith('.csv')]
                for csv_file in csv_files:
                    csv_path = os.path.join(self.config.csv_conversion_dir, csv_file)
                    self.converter.downsample_csv(csv_path,
                                                  max_rows,
                                                  self.config.csv_conversion_dir)
            # Step 4: Train all gates
            self._train_all_gates(gating_strategy)
            # Step 5: Apply predictions and save results
            self._finalize_results()
        finally:
            if ram_disk:
                RAMDiskManager.flush_ramdisk_to_disk(str(self.config.disk_dest))
                RAMDiskManager.cleanup_ramdisk()

    def _train_all_gates(self, gating_strategy: pd.DataFrame) -> None:
        """Train all gates in sequence"""
        # Your existing Step 9 logic, but calling self.trainer.train_gate()
        pass

    def _finalize_results(self) -> None:
        """Apply predictions and save final results"""
        # Your existing Step 10-11 logic
        pass
