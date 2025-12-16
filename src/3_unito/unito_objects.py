import os
import warnings
import ssl
import urllib3
import pandas as pd
from dataclasses import dataclass, field
from typing import List

# Env settings
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DEBUG_DIR"] = "/tmp/torch_compile_debug"
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Standard Imports
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
from UNITO_Train_Predict.Data_Preprocessing import process_table, train_test_val_split
from UNITO_Train_Predict.Predict import UNITO_gating, evaluation

# Own Imports
from generate_gating_strategy import (
    parse_fcs_add_gate_label,
    extract_gating_strategy,
    clean_gating_strategy,
    add_gate_labels_to_test_files,
)
from apply_unito_to_fcs import apply_predictions_to_csv


@contextmanager
def cd(newdir):
    prev = os.getcwd()
    os.chdir(newdir)
    try:
        yield
    finally:
        os.chdir(prev)


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
    csv_conversion_dir_metadir: str
    disk_dest: str
    # Hyperparameters
    default_hyperparameters: List[List[float]]
    problematic_gate_hyperparameters: List[List[float]]
    # Processing settings
    downsample_max_rows: int = 200_000
    device: str = "mps"
    n_worker: int = 30
    epochs: int = 7
    problematic_epochs: int = 18
    n_test_files: int = 6
    # Problematic gates
    problematic_gate_list: List = field(default_factory=list)

    def __dir_assign__(self):
        if self.ram_disk:
            self.dest = os.getenv("UNITO_DEST")
            self.save_data_img_path = f"{self.dest}/Data/"
            self.save_figure_path = f"{self.dest}/figures/"
            self.save_model_path = f"{self.dest}/model/"
            self.save_prediction_path = f"{self.dest}/prediction/"
            self.downsample_path = f"{self.dest}/downsample/"
        else:
            self.dest = self.disk_dest
            self.save_data_img_path = f"{self.dest}/Data/"
            self.save_figure_path = f"{self.dest}/figures/"
            self.save_model_path = f"{self.dest}/model/"
            self.save_prediction_path = f"{self.dest}/prediction/"
            self.downsample_path = f"{self.dest}/downsample/"

        for path in [
            self.save_data_img_path,
            self.save_figure_path,
            self.save_model_path,
            self.save_prediction_path,
            self.downsample_path,
            self.csv_conversion_dir_metadir,
        ]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)


class FileConverter:
    """Handles FCS to CSV conversion"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.fcs_files = [f for f in os.listdir(config.fcs_dir) if f.endswith(".fcs")]

    def convert_all_fcs(self) -> None:
        """Convert all FCS files to CSV"""
        print("Converting FCS files to CSV...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [
                executor.submit(
                    self._convert_fcs_to_csv,
                    os.path.join(self.config.fcs_dir, fcs_file),
                    self.config.csv_conversion_dir,
                )
                for fcs_file in self.fcs_files
            ]
            for future in concurrent.futures.as_completed(results):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")
        print("FCS to CSV conversion complete!")

    def _convert_fcs_to_csv(self, fcs_file: str, output_dir: str) -> None:
        """Generate .csv of fcs_file required for UNITO processing"""
        try:
            fcs_filename = os.path.basename(fcs_file)
            m, data = fcsparser.parse(fcs_file, reformat_meta=True)
            # Save Data
            df = pd.DataFrame(data)
            csv_filename = fcs_filename.replace(".fcs", ".csv")
            df_output = os.path.join(self.config.csv_conversion_dir, csv_filename)
            df.to_csv(df_output, index=False)
            print(f"{fcs_filename} converted to csv")
            # Save Meta
            meta_df = pd.DataFrame(list(m.items()), columns=["key", "value"])
            meta_filename = fcs_filename.replace(".fcs", "_metadata.csv")
            meta_output = os.path.join(
                self.config.csv_conversion_dir_metadir, meta_filename
            )
            meta_df.to_csv(meta_output, index=False)
            print(f"Metadata saved for {meta_filename}")
        except Exception as e:
            print(f"Error saving data, metadata for {fcs_filename}: {e}")

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
        """Parse gates from WSP file and add to CSV files"""
        print("Parsing gates from WSP file...")
        parse_fcs_add_gate_label(
            self.config.wsp_path,
            self.config.wsp_files_dir,
            self.config.csv_conversion_dir,
        )
        print("Gate parsing complete!")

    def generate_gating_strategy(self) -> pd.DataFrame:
        """Generate and save gating strategy"""
        print("Generating gating strategy...")
        gating_strategy = extract_gating_strategy(
            self.config.wsp_path, self.config.wsp_files_dir
        )
        final_gating_strategy = clean_gating_strategy(
            self.config.panel_meta_path, gating_strategy
        )
        out_file = os.path.join("./", "gating_strategy.csv")
        final_gating_strategy.to_csv(out_file, index=False)
        print("Gating strategy saved!")
        return final_gating_strategy


class UNITOTrainer:
    """Handles UNITO training and prediction"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.hyperparameter_set = self.config.default_hyperparameters
        self.epochs = self.config.epochs
        self.all_predictions = {}
        self.csv_train_dir = os.path.join(self.config.csv_conversion_dir, "train")
        self.test_files = []

        if not os.path.exists(self.csv_train_dir):
            os.mkdir(self.csv_train_dir)

    def train_all_gates(
        self,
        gate_pre_list,
        gate_list,
        x_axis_list,
        y_axis_list,
        path2_lastgate_pred_list,
        csv_train_dir,
        dest,
        save_prediction_path,
        save_figure_path,
        hyperparameter_df,
        problematic_gate_hyperparameters,
        all_predictions,
        n_worker,
        device,
    ):
        for i, (gate_pre, gate, x_axis, y_axis, path_raw) in enumerate(
            zip(
                gate_pre_list,
                gate_list,
                x_axis_list,
                y_axis_list,
                path2_lastgate_pred_list,
            )
        ):
            # Granular hyperparameter settings for problematic gates
            if self.config.problematic_gate_list and any(
                g in gate for g in self.config.problematic_gate_list
            ):
                self.hyperparameter_set = problematic_gate_hyperparameters
                self.epochs = self.config.problematic_epochs
                print(f"Using specialized hyperparameters and epochs for {gate}")
            else:
                print(f"Using default hyperparameters and epochs for {gate}")

            print(f"start UNITO for {gate}")
            # 9a. preprocess training data
            process_table(
                x_axis,
                y_axis,
                gate_pre,
                gate,
                csv_train_dir,
                convex=True,
                seq=(gate_pre is not None),
                dest=dest,
            )
            train_test_val_split(gate, csv_train_dir, dest, "train")

            # 9b. train
            best_lr, best_bs = tune(
                gate, self.hyperparameter_set, device, self.epochs, n_worker, dest
            )
            hyperparameter_df.loc[len(hyperparameter_df)] = [gate, best_lr, best_bs]
            train(gate, best_lr, device, best_bs, self.epochs, n_worker, dest)

            # 9c. preprocess prediction data
            print(f"Start prediction for {gate}")
            # if i == 0:
            #     processed_files_list = [
            #         f
            #         for f in os.listdir(path_raw)
            #         if f.endswith(".csv") and not f.endswith("_with_gate_label.csv")
            #     ]
            process_table(
                x_axis,
                y_axis,
                gate_pre,
                gate,
                path_raw,
                convex=True,
                seq=(gate_pre is not None),
                dest=dest,
            )
            train_test_val_split(gate, path_raw, dest, "pred")

            # 9d. predict
            model_path = f"{dest}/model/{gate}_model.pt"
            gate_prediction_path = f"{save_prediction_path}/{gate}"
            os.makedirs(gate_prediction_path, exist_ok=True)
            data_df_pred, predictions_dict = UNITO_gating(
                model_path,
                x_axis,
                y_axis,
                gate,
                path_raw,
                n_worker,
                device,
                gate_prediction_path,
                dest,
                seq=(gate_pre is not None),
                gate_pre=gate_pre,
            )

            # Collect all predictions for this gate, across all files,
            # to the all_predictions dict
            # gate_predictions is a nested dict of {key = {gate}_pred: value = [binary classifiers]}

            for filename, gate_predictions in predictions_dict.items():
                if filename not in all_predictions:
                    all_predictions[filename] = {}
                all_predictions[filename].update(gate_predictions)

            # 9e. Evaluation - filter to only test files with ground truth
            if self.test_files:
                # Strip _with_gate_label.csv to match all_predictions keys
                test_file_basenames = [
                    f.replace("_with_gate_label.csv", "") for f in self.test_files
                ]
                print(f"Evaluating gate '{gate}' on test files: {test_file_basenames}")

                # Filter all_predictions to only test files
                test_predictions = {
                    filename: preds
                    for filename, preds in all_predictions.items()
                    if filename in test_file_basenames
                }

                if test_predictions:
                    # Read the ground truth CSV files for test files and merge with predictions
                    test_dfs = []
                    for filename in test_predictions.keys():
                        # Check if this file has predictions for the CURRENT gate
                        if gate not in test_predictions[filename]:
                            print(
                                f"Skipping {filename}: No prediction for gate '{gate}' yet"
                            )
                            continue
                        #  ALWAYS read ground truth from original CSV (has all gate columns)
                        gt_csv = os.path.join(
                            self.config.csv_conversion_dir,
                            f"{filename}_with_gate_label.csv",
                        )

                        if not os.path.exists(gt_csv):
                            print(f"Warning: Ground truth file not found: {gt_csv}")
                            continue

                        df = pd.read_csv(gt_csv)

                        # For subsequent gates, filter to cells that passed parent gate
                        if i > 0:
                            # Read parent predictions to match row indices
                            parent_gate = gate_pre
                            parent_pred_csv = os.path.join(
                                self.config.save_prediction_path,
                                parent_gate,
                                f"{filename}.csv",
                            )

                            if os.path.exists(parent_pred_csv):
                                df_parent = pd.read_csv(parent_pred_csv)
                                # Match by Time column (event number) if available
                                if "Time" in df.columns and "Time" in df_parent.columns:
                                    df = df[
                                        df["Time"].isin(df_parent["Time"])
                                    ].reset_index(drop=True)
                                else:
                                    # Fallback: assume same order, take first N rows
                                    print(
                                        f"WARNING: No Time column, using first {len(df_parent)} rows"
                                    )
                                    df = df.iloc[: len(df_parent)].reset_index(
                                        drop=True
                                    )
                            else:
                                print(
                                    f"ERROR: Parent prediction file not found: {parent_pred_csv}"
                                )
                                continue

                        print(f"DEBUG: Reading from {gt_csv}")
                        print(f"DEBUG: CSV has {len(df)} rows")
                        print(f"DEBUG: CSV columns: {df.columns.tolist()}")
                        print(
                            f"DEBUG: Prediction array length: {len(test_predictions[filename][gate])}"
                        )

                        # Verify ground truth column exists
                        if gate not in df.columns:
                            print(f"ERROR: Ground truth column '{gate}' not in CSV!")
                            print(f"Available columns: {df.columns.tolist()}")
                            continue

                        # Check length match
                        if len(test_predictions[filename][gate]) != len(df):
                            print(
                                f"ERROR: Length mismatch - pred:{len(test_predictions[filename][gate])}, gt:{len(df)}"
                            )
                            continue

                        # Add prediction column
                        df[f"{gate}_pred"] = test_predictions[filename][gate]
                        test_dfs.append(df)

                    if test_dfs:
                        data_df_pred_test = pd.concat(test_dfs, ignore_index=True)
                        print(
                            f"Evaluating on {len(data_df_pred_test)} rows from {len(test_dfs)} test files"
                        )
                        # DEBUG: Check what's actually in the predictions
                        print(f"\nDEBUG: Ground truth '{gate}' distribution:")
                        print(data_df_pred_test[gate].value_counts())
                        print(f"\nDEBUG: Predictions '{gate}_pred' distribution:")
                        print(data_df_pred_test[f"{gate}_pred"].value_counts())
                        print("\nDEBUG: First 20 rows:")
                        print(data_df_pred_test[[gate, f"{gate}_pred"]].head(20))
                        accuracy, recall, precision, f1 = evaluation(
                            data_df_pred_test, gate
                        )

                    else:
                        print(
                            f"WARNING: No test files with ground truth found for gate '{gate}', skipping evaluation"
                        )
                        accuracy, recall, precision, f1 = 0, 0, 0, 0
            else:
                print("WARNING: No test files set, using all data for evaluation")
                accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)

            print(
                f"{gate}: accuracy:{accuracy}, recall:{recall}, precision:{precision}, f1 score:{f1}"
            )


class RAMDiskManager:
    """Handles RAM disk operations"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def cleanup_ramdisk(self) -> None:
        """Unmount any current RAMDisks so diskimages-helper exits and frees the RAM."""
        try:
            info = subprocess.check_output(["hdiutil", "info"], text=True)
            print("Checked existing mounts")
        except Exception as e:
            print(f"Error checking existing mounts: {e}")
            return

        for line in info.splitlines():
            if "/Volumes/RAMDisk" in line or line.strip().startswith("/dev/ram"):
                dev = line.split()[0]
                try:
                    subprocess.check_call(
                        ["hdiutil", "detach", dev, "-force"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print(f"Detached existing RAM disk: {dev}")
                except subprocess.CalledProcessError:
                    pass

    def mount_ramdisk(self, ram_disk: bool) -> None:
        if not ram_disk:
            return
        print("Clearing current RAM disk...")
        try:
            self.cleanup_ramdisk()
        except Exception as e:
            print(f"Warning: Initial cleanup failed, continuing: {e}")

        print("Starting RAM disk creation...")
        try:
            vm = psutil.virtual_memory()
            usable_bytes = vm.available * 0.9
            blocks = int(usable_bytes // 512)
            print(
                f"Creating RAM disk: {blocks} blocks ({usable_bytes / 1024 / 1024 / 1024:.1f} GB)"
            )

            dev = subprocess.check_output(
                ["hdiutil", "attach", "-nomount", f"ram://{blocks}"], text=True
            ).strip()
            print(f"RAM device created: {dev}")

            subprocess.check_call(
                ["diskutil", "eraseVolume", "HFS+", "RAMDisk", dev],
                stdout=subprocess.DEVNULL,
            )
            print("RAM disk formatted")

            os.environ["UNITO_DEST"] = "/Volumes/RAMDisk/UNITO_train_data"
            print(f"UNITO_DEST set to: {os.environ['UNITO_DEST']}")

            if os.path.exists("/Volumes/RAMDisk"):
                print("✅ RAM disk successfully mounted at /Volumes/RAMDisk")
            else:
                print("❌ RAM disk mount failed")

        except Exception as e:
            print(f"Error creating RAM disk: {e}")

    def flush_ramdisk_to_disk(self, disk_dest: str) -> None:
        """Copy the four UNITO output subfolders plus strategy &
        hyperparam CSVs from the RAM disk ($UNITO_DEST) into disk_dest.
        Empties RAM disk on completion."""
        ram_dest = str(os.getenv("UNITO_DEST"))
        subdirs = ["figures", "model", "prediction", "Data"]

        for sub in subdirs:
            src = os.path.join(ram_dest, sub)
            dst = os.path.join(disk_dest, sub)
            if os.path.exists(src):
                os.makedirs(dst, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)

        for sub in ["figures", "model", "prediction", "Data"]:
            ram_sub = os.path.join(ram_dest, sub)
            if os.path.exists(ram_sub):
                shutil.rmtree(ram_sub)
                os.makedirs(ram_sub, exist_ok=True)

        for fn in ["gating_strategy.csv", "hyperparameter_tunning.csv"]:
            if os.path.exists(fn):
                shutil.copy(fn, os.path.join(disk_dest, fn))

        print(f"Flushed RAMDisk contents from {ram_dest} to {disk_dest}")


class UNITOPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.converter = FileConverter(config)
        self.gate_processor = GateProcessor(config)
        self.trainer = UNITOTrainer(config)
        self.hyperparameter_df = pd.DataFrame(
            columns=["gate", "learning_rate", "batch_size"]
        )
        self.path2_lastgate_pred_list = [self.config.csv_conversion_dir]
        self.gate_processor = GateProcessor(config)
        self.gating_strategy = self.gate_processor.generate_gating_strategy()
        self.gate_pre_list = list(self.gating_strategy.Parent_Gate)
        self.gate_pre_list[0] = None
        self.gate_list = list(self.gating_strategy.Gate)
        self.x_axis_list = list(self.gating_strategy.X_axis)
        self.y_axis_list = list(self.gating_strategy.Y_axis)

        for idx in range(1, len(self.gate_list)):
            parent_gate = self.gate_pre_list[idx]
            self.path2_lastgate_pred_list.append(f"./prediction/{parent_gate}/")

    def run(self, downsample: bool = True):
        """Run the complete pipeline"""

        if self.config.ram_disk:
            ramdisk_manager = RAMDiskManager(self.config)
            ramdisk_manager.mount_ramdisk(True)
        self.config.__dir_assign__()
        try:
            # Pytorch settings
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)

            # # Step 1: Convert FCS files
            self.converter.convert_all_fcs()

            # # Step 2: Parse gates
            self.gate_processor.parse_gates()

            # # Step 3: Find train files, move to train directory
            self._find_train_csv_files()
            self._move_gated_csv_files_to_train()

            # Step 4: Downsampling (Optional)
            if downsample:
                max_rows = self.config.downsample_max_rows

                # Downsample training files
                csv_files = [
                    f
                    for f in os.listdir(self.trainer.csv_train_dir)
                    if f.endswith(".csv")
                ]
                print(
                    f"Downsampling {len(csv_files)} training files to {max_rows} rows..."
                )
                for csv_file in csv_files:
                    csv_path = os.path.join(self.trainer.csv_train_dir, csv_file)
                    self.converter.downsample_csv(
                        csv_path, max_rows, self.trainer.csv_train_dir
                    )

                # Also downsample test files (those left in csv_conversion_dir)
                test_csv_files = [
                    f
                    for f in os.listdir(self.config.csv_conversion_dir)
                    if f.endswith("_with_gate_label.csv")
                ]
                print(
                    f"Downsampling {len(test_csv_files)} test files to {max_rows} rows..."
                )
                for csv_file in test_csv_files:
                    csv_path = os.path.join(self.config.csv_conversion_dir, csv_file)
                    self.converter.downsample_csv(
                        csv_path, max_rows, self.config.csv_conversion_dir
                    )

            # Step 5: Add gate labels columns to test csv
            self._gate_col_added_test_files()

            # Verify test files are set correctly
            print(f"Test files for evaluation: {self.trainer.test_files}")

            with cd(self.config.dest):
                # Step 6: Train all gates
                self.trainer.train_all_gates(
                    self.gate_pre_list,
                    self.gate_list,
                    self.x_axis_list,
                    self.y_axis_list,
                    self.path2_lastgate_pred_list,
                    self.trainer.csv_train_dir,
                    self.config.dest,
                    self.config.save_prediction_path,
                    self.config.save_figure_path,
                    self.hyperparameter_df,
                    self.config.problematic_gate_hyperparameters,
                    self.trainer.all_predictions,
                    self.config.n_worker,
                    self.config.device,
                )

                # Step 7: Apply predictions and save results
                self._finalize_results(
                    self.hyperparameter_df,
                    self.trainer.all_predictions,
                    self.config.csv_conversion_dir,
                )

            if self.config.ram_disk:
                ramdisk_manager.flush_ramdisk_to_disk(str(self.config.disk_dest))
                ramdisk_manager.cleanup_ramdisk()

        except Exception as e:
            if self.config.ram_disk:
                print(f"\nPipeline failed with error: {type(e).__name__}: {e}")
                print("RAMDisk left mounted for debugging.")
                print("To manually cleanup: hdiutil detach /Volumes/RAMDisk -force")
            raise

    def _finalize_results(
        self, hyperparameter_df, all_predictions, csv_conversion_dir
    ) -> None:
        """Applies predicitons to the test .csv files and saves the hyperparams"""
        apply_predictions_to_csv(all_predictions, csv_conversion_dir)
        hyperparameter_df.to_csv("./hyperparameter_tuning.csv")
        pass

    def _find_train_csv_files(self):
        """Get list of csv files with gate labels (i.e. gated by user)"""
        self.training_csv_files = [
            f
            for f in os.listdir(self.config.csv_conversion_dir)
            if f.endswith("_with_gate_label.csv")
        ]
        print(f"Found {len(self.training_csv_files)} files with gate labels")
        if len(self.training_csv_files) == 0:
            print("Check if already moved to gated csv files to correct dir")

    def _move_gated_csv_files_to_train(
        self,
    ):
        """Move gated .csv files to UNITO_csv_conversion/train dir (Disk or RAM Disk)
        Keeps n_test_files separate for testing/validation metrics"""

        # Shuffle and split
        # random.shuffle(self.training_csv_files)
        # train_files = self.training_csv_files[self.config.n_test_files :]
        # test_files = self.training_csv_files[: self.config.n_test_files]

        self.training_csv_files.sort()
        train_files = self.training_csv_files[: -self.config.n_test_files]
        test_files = self.training_csv_files[-self.config.n_test_files :]

        # Store test files in trainer for evaluation filtering
        self.trainer.test_files = test_files

        print(
            f"Splitting: {len(train_files)} files for training, {len(test_files)} files held out for testing"
        )
        print(f"Test files: {test_files}")

        for f in train_files:
            source_path = os.path.join(self.config.csv_conversion_dir, f)
            destination_path = os.path.join(self.trainer.csv_train_dir, f)
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)

    def _gate_col_added_test_files(self):
        """Add gate labels to the test .csv files as column labels"""
        add_gate_labels_to_test_files(
            test_dir=self.config.csv_conversion_dir,
            train_dir=self.trainer.csv_train_dir,
        )
