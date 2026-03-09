# main.py
"""
Main execution script for the Building Structure Optimization process.

This script orchestrates the workflow involving:
- Reading initial structural configurations.
- Generating variations of configurations.
- Analyzing configurations using TQS or a faster geometric estimation.
- Collecti
ng training data (features and pycorresponding material quantities).
- Training a Neural Network model to predict material quantities.
- Evaluating the trained model.
- Saving results and configurations.

Author: Adriana
Date: March 2025
"""
# Standard library imports
import copy
from random import random
import random as py_random
import traceback
import time # Added for potential delays and timing
from typing import List, Tuple, Optional, Dict
import numpy as np
from shapely.geometry import Polygon
from sklearn.metrics import r2_score, mean_absolute_error # Added for evaluation metrics
# import numpy as np # Uncomment if needed for advanced splitting or direct normalization here

# Third-party importspy
from TQS import TQSUtil
import torch # TQS Utility functions
import threading
import shutil
import urllib.request
import smtplib
import ssl
import os

# Project-specific imports
# Configuration - Make sure these files exist and are configured
from config.settings import BuildingConfig, RunConfig, NeuralNetConfig # General settings, NN config, analysis mode flag
from config.paths import FINAL_VECTORS_CSV_PATH # Path for saving final vectors CSV

# Algorithm components - Core logic for processing and ML
from geometry.binary_input_processor import BinaryProcessor
from geometry.length_input_processor import LengthProcessor # Handles vector input format
from models.dnnmodel import SimpleNN # NN Model definition (used indirectly via manager)
# Note: train_model, test_model are likely used within nn_manager
from data.genBinary import generate_new_binary_vector # For binary input variations

# TQS API interactions - Encapsulates TQS usage
# Ensure 'tqsapi' is the correct package name in your structure
from tqs_interface.tqs_manager import TQSModelManager # Manages TQS model creation/saving
from tqs_interface.tqs_exec import RunModel # Executes TQS analysis jobs

# Neural Network interaction - Encapsulates NN training/prediction
# Ensure 'tqsapi' or 'models' is the correct path for nn_manager
from models.nn_manager import NeuralNetworkManager # Using the path from your previous code

# Results and Visualization - Handling outputs and plotting
from results.resultsext import extract_material_summary # Extracts data from TQS output file
from tqs_interface.tqs_errors import TQSErrorReader
from visualization.segment_plotter import SegmentPlotter # Plots individual configurations
from visualization.results_plotter import ResultsPlotter # Plots NN performance graphs
from visualization.nn_diagnostics import run_full_diagnostics

# Utilities - Helper functions
# Ensure 'utils' directory exists and contains these files
from utils.geometric_calculator import get_geometric_concrete_volume # Fast geometric volume estimate
from utils.file_handler import save_final_vectors_to_csv # Saves generated vectors to CSV
from utils.feature_engineer import FeatureEngineer
from utils.feature_pipeline import FeaturePipeline

from utils.experiment_manager import ExperimentManager
from config import paths # Importa o mÃ³dulo de caminhos
import logging
import hashlib
import json
from pathlib import Path

# =============================================================================
# Main Optimizer Class
# =============================================================================

class BuildingOptimizer:
    """
    Orchestrates the building structure optimization process using structural
    analysis (TQS or geometric estimation) and a Neural Network model.

    Handles data generation, model training, prediction, and result evaluation.

    Attributes:
        use_vector_input (bool): Flag indicating input format (vector vs. binary).
        use_geometric_estimate (bool): Flag for using fast geometric calc instead of TQS.
        analysis_mode (str): String representation of the analysis mode ('TQS Analysis' or 'Geometric Estimate').
        num_target_samples (int): Desired number of valid samples to collect.
        train_split_ratio (float): Ratio of data to use for training.
        tqs_manager (TQSModelManager): Handles TQS model interactions.
        nn_manager (NeuralNetworkManager): Manages the Neural Network lifecycle.
        bynary_input_processor (BinaryProcessor): Processes segments from binary input.
        length_input_processor (LengthProcessor): Processes segments from vector input.
        segment_plotter (SegmentPlotter): Visualizes segment configurations.
        results_plotter (ResultsPlotter): Plots training/testing results.
        current_iteration (int): Counter for configuration generation attempts.
        generated_valid_configurations (List[List[Dict]]): Stores segment dicts of *valid* configurations analyzed.
        normalization_params (Optional[Dict]): Stores {'X_mean':..., 'X_std':..., 'y_mean':..., 'y_std':...} after training.
    """

    def __init__(self, exp_manager: ExperimentManager):
        """Initializes all components required for the optimization."""
        print("--- Initializing Building Optimizer ---")
        self.exp_manager = exp_manager

        # Os componentes agora recebem os caminhos do ExperimentManager
        self.results_plotter = ResultsPlotter(output_dir=self.exp_manager.plots_dir)
        self.segment_plotter = SegmentPlotter(output_dir=self.exp_manager.images_dir)

        # --- Configuration Flags ---
        # Read from BuildingConfig, providing defaults if attributes are missing
        self.use_vector_input = RunConfig.USE_VECTOR_INPUT
        self.use_geometric_estimate = RunConfig.USE_GEOMETRIC_ESTIMATE
        self.analysis_mode = "Geometric Estimate" if self.use_geometric_estimate else "TQS Analysis"
        self.num_target_samples = RunConfig.NUM_SAMPLES
        
        # Print configuration summary
        print(f"Input Format:         {'Vector Lengths' if self.use_vector_input else 'Binary Grid'}")
        print(f"Analysis Mode:        {self.analysis_mode}")
        print(f"Target Valid Samples: {self.num_target_samples}")

        # --- Component Initialization ---
        self.tqs_manager = TQSModelManager(BuildingConfig.NAME)
        self.nn_manager = NeuralNetworkManager()
        # Seeds para reprodutibilidade
        try:
            seed = getattr(RunConfig, 'SEED', 42)
            py_random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        self.length_processor = LengthProcessor()
        self.feature_pipeline = FeaturePipeline(self.length_processor)
        self.binary_processor = BinaryProcessor()

        # --- State Variables ---
        self.current_iteration = 0 # Counts attempts to generate configurations
        self.generated_valid_configurations = [] # Stores List[Dict] for each valid config
        self._clf_features: List[List[float]] = []
        self._clf_labels: List[int] = []  # 1 = válido, 0 = inválido
        self._error_reader = TQSErrorReader()
        self.metrics_dir = self.exp_manager.get_metrics_dir()
        self.nn_manager.metrics_dir = self.metrics_dir
        try:
            self._preflight_checks()
        except Exception:
            pass
        self._seen_segments_hash = set()
        self._monitor_stop = False
        self._monitor_thread = None
        self._heartbeat_ts = time.time()
        if getattr(RunConfig, 'MONITORING_ENABLED', True):
            try:
                self._monitor_thread = threading.Thread(target=self._monitor_system_health_loop, daemon=True)
                self._monitor_thread.start()
            except Exception:
                pass

        print("--- Optimizer Initialized Successfully ---")


    # -------------------------------------------------------------------------
    # Main Workflow Method
    # -------------------------------------------------------------------------

    def run_optimization(self):
        """Executes the main optimization workflow."""
        print(f"\n--- Starting Optimization Workflow ---")
        start_time_total = time.time()
        try:
            # --- FASE 1: GERAÃ‡ÃƒO DE DADOS ---
            print("\n--- PHASE 1: DATA COLLECTION ---")
            initial_segments = self._load_initial_segments()
            # Este mÃ©todo retorna os dados brutos (nÃ£o normalizados)
            feature_vectors, output_values = self._collect_training_data(initial_segments)
            self._validate_collected_data(feature_vectors, output_values)
            print(f"--- PHASE 1 COMPLETE: {len(feature_vectors)} samples generated. ---")

            # --- FASE 2: TREINAMENTO E AVALIAÃ‡ÃƒO ---
            print("\n--- PHASE 2: PIPELINE & MODEL TRAINING ---")
            # Este novo mÃ©todo orquestra o treinamento da pipeline e do modelo
            self._train_and_evaluate(feature_vectors, output_values)

            # --- FASE 3: SALVAR RESULTADOS ADICIONAIS ---
            self._save_results()

            end_time_total = time.time()
            print(f"\n--- Workflow Finished Successfully ({end_time_total - start_time_total:.2f}s) ---")
        except Exception as e:
            end_time_total = time.time()
            print(f"\n--- ERROR DURING OPTIMIZATION WORKFLOW ({end_time_total - start_time_total:.2f}s) ---")
            error_message = f"Optimization failed: {str(e)}"
            TQSUtil.writef(error_message) # Log to TQS console too
            print(error_message)
            # Optional: Log detailed traceback for debugging
            print("Traceback:")
            print(traceback.format_exc())
            # Consider re-raising or exiting based on desired behavior on error
            # raise # Uncomment to stop execution completely on error

    # -------------------------------------------------------------------------
    # Helper Methods for run_optimization Workflow Steps
    # -------------------------------------------------------------------------

    def _load_initial_segments(self) -> List[dict]:
        """Loads the initial set of structural segments based on configuration."""
        print("\n--- Loading Initial Segments ---")
        if self.use_vector_input:
            segments = self.length_processor.read_length_from_csv()
            print(f"Loaded {len(segments)} initial vector segments.")
        else:
            segments = self.binary_processor.read_binary_from_csv()
            print(f"Loaded {len(segments)} initial binary grid segments.")
        if not segments:
             raise ValueError("Failed to load initial segments. Check CSV paths and content.")
        return segments

    def _validate_collected_data(self, features: list, outputs: list):
        """Validates the collected features and outputs before processing."""
        print("\n--- Validating Collected Data ---")
        if not features:
            raise RuntimeError("Data collection resulted in zero valid feature vectors. Cannot proceed.")
        if not outputs:
             raise RuntimeError("Data collection resulted in zero valid output vectors. Cannot proceed.")
        if len(features) != len(outputs):
            raise RuntimeError(f"Data mismatch: {len(features)} features vs {len(outputs)} outputs collected.")

        num_samples = len(features)
        # Check feature/output dimensions consistency
        try:
             num_features = len(features[0])
             num_outputs = len(outputs[0])
        except IndexError:
             raise RuntimeError("Collected data lists seem to contain empty elements.")

        print(f"Validation passed: {num_samples} samples collected.")
        print(f"Feature vector size: {num_features}")
        print(f"Output vector size:  {num_outputs} ({'Steel only' if num_outputs==1 else 'Steel & Concrete'})")

    def _save_results(self):
        """Saves generated configurations or other desired results."""
        # Save Final Generated Configurations if applicable
        if self.use_vector_input and self.generated_valid_configurations:
            print(f"\n--- Saving Generated Valid Configurations to CSV ---")
            try:
                # Assumes save_final_vectors_to_csv uses CSV_FINAL_PATH from constants
                # and is imported from utils.file_handler
                save_final_vectors_to_csv(self.generated_valid_configurations)
                print(f"Successfully saved {len(self.generated_valid_configurations)} configurations to {FINAL_VECTORS_CSV_PATH}.")
            except Exception as e:
                print(f"Error saving configuration CSV: {e}")
                # Log to TQS console as well?
                # TQSUtil.writef(f"Error saving configuration CSV: {e}")
        else:
            print("\n--- Skipping Configuration CSV Saving ---")
            if not self.use_vector_input:
                print("(Reason: Input mode is not vector-based)")
            elif not self.generated_valid_configurations:
                 print("(Reason: No valid configurations were generated/stored)")

    # -------------------------------------------------------------------------
    # Data Collection and Analysis Helper Methods
    # -------------------------------------------------------------------------

    def _collect_training_data(self, initial_segments: List[dict]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Generates configurations and collects feature vectors and corresponding
        analysis outputs (TQS or Geometric).

        Args:
            initial_segments: The starting list of segment dictionaries.

        Returns:
            A tuple containing:
                - feature_vectors (List[List[float]]): List of input vectors for the NN.
                - output_values (List[List[float]]): List of corresponding output vectors
                                                     [steel, concrete] or [0.0, concrete].
        """
        print(f"\n--- Starting Data Collection ({self.analysis_mode} Mode) ---")
        feature_vectors = []
        output_values = []
        processed_valid_configs_count = 0
        if getattr(RunConfig, 'CHECKPOINTS_ENABLED', True) and getattr(RunConfig, 'RESUME_FROM_CHECKPOINT', False):
            ck = self._load_checkpoint()
            if ck:
                feature_vectors = ck.get('feature_vectors', [])
                output_values = ck.get('output_values', [])
                processed_valid_configs_count = int(ck.get('valid_count', 0))
                self.current_iteration = int(ck.get('current_iteration', 0))
                print("Resuming from checkpoint.")
        # Calculate max attempts to prevent infinite loops if analysis consistently fails
        max_iterations = self.num_target_samples * RunConfig.MAX_ITERATION_FACTOR

        # --- Analyze Initial Configuration ---
        print(f"\nAnalyzing Initial Configuration (Attempt 0)...")
        analysis_start_time = time.time()
        steel, concrete, column_polygons, beam_definitions, is_valid = self._get_analysis_results(initial_segments)
        analysis_end_time = time.time()
        print(f"Initial analysis took {analysis_end_time - analysis_start_time:.2f}s")

        if concrete is not None: # Check if analysis (TQS or geometric) was successful
            print(f"Initial Results -> Steel: {steel if steel is not None else 'N/A'} kgf, Concrete: {concrete:.4f} mÂ³")
            feature_vector = self._extract_feature_vector(column_polygons, beam_definitions)
            # Em _collect_training_data, logo apÃ³s extrair o feature_vector da amostra inicial
            
            print(f"[DEBUG MAIN] Vetor de Features da Semente: {np.array(feature_vector)}")

            # Rótulo de validade para classificador
            self._clf_features.append(feature_vector)
            self._clf_labels.append(1 if is_valid else 0)
            # Apenas amostras válidas entram no treino de aço
            if is_valid:
                feature_vectors.append(feature_vector)
                if steel is None:
                    raise RuntimeError("Steel is None. For steel-only training, use TQS mode (RunConfig.USE_GEOMETRIC_ESTIMATE=False).")
                output_values.append([steel])
                if self.use_vector_input:
                    self.generated_valid_configurations.append(copy.deepcopy(initial_segments))
                processed_valid_configs_count += 1
        else:
            # If the very first configuration fails, it's critical.
            print("CRITICAL: Initial configuration failed analysis. Check input data, TQS setup, or geometric calculation logic.")
            raise RuntimeError("Initial configuration analysis failed. Cannot proceed.")

        # --- Generate and Analyze Subsequent Configurations ---
        print(f"\n--- Starting Generation Loop (Target: {self.num_target_samples} valid samples) ---")
        # Use initial_segments as the base for variations for simplicity.
        base_segments_for_variation = initial_segments

        last_ck_ts = time.time()
        while processed_valid_configs_count < self.num_target_samples and self.current_iteration < max_iterations:
            self._heartbeat_ts = time.time()
            self.current_iteration += 1 # Increment attempt counter
            print(f"\n--- Iteration Attempt {self.current_iteration}/{max_iterations} (Valid Samples Collected: {processed_valid_configs_count}/{self.num_target_samples}) ---")

            # 1. Generate a new variation
            print("Generating segment variation...")
            try:
                 new_segments = self._generate_segment_variation(base_segments_for_variation, variation_strategy="random")
                 seg_hash = hashlib.sha256(json.dumps(new_segments, ensure_ascii=False).encode("utf-8")).hexdigest()
                 if seg_hash in getattr(self, '_seen_segments_hash', set()):
                     print("Duplicate geometry detected. Skipping re-analysis.")
                     continue
                 self._seen_segments_hash.add(seg_hash)
            except Exception as gen_e:
                 print(f"Error during segment variation generation: {gen_e}. Skipping iteration.")
                 continue # Skip to next iteration

            # Optional: Plot the generated configuration
            # print("Plotting current segment configuration...")
            try:
                  self.segment_plotter.plot_segments(new_segments, self.current_iteration)
            except Exception as plot_e:
                  print(f"Warning: Failed to plot segment configuration: {plot_e}")

            # 2. Get analysis results (TQS or Geometric)
            analysis_start_time = time.time()
            steel, concrete, column_polygons, beam_definitions, is_valid = self._get_analysis_results(new_segments)
            analysis_end_time = time.time()
            print(f"Analysis took {analysis_end_time - analysis_start_time:.2f}s")

            # 3. Process results
            if concrete is not None: # Check if analysis was successful
                processed_valid_configs_count += 1 # Increment valid sample count
                print(f"Config {self.current_iteration} Results (Valid Sample {processed_valid_configs_count}) -> Steel: {steel if steel is not None else 'N/A'} kgf, Concrete: {concrete:.4f} mÂ³")
                feature_vector = self._extract_feature_vector(column_polygons, beam_definitions)
                # Ensure feature vector extraction was successful
                if feature_vector:
                     # Rótulo para classificador (válida/inválida)
                     self._clf_features.append(feature_vector)
                     self._clf_labels.append(1 if is_valid else 0)
                     # Apenas válidas no treino de aço
                     if is_valid:
                         feature_vectors.append(feature_vector)
                         if steel is None:
                             raise RuntimeError("Steel is None. For steel-only training, use TQS mode (RunConfig.USE_GEOMETRIC_ESTIMATE=False).")
                         output_values.append([steel])
                         if self.use_vector_input:
                             self.generated_valid_configurations.append(copy.deepcopy(new_segments))
                         # Optional: Update base segments if varying from last success
                         # base_segments_for_variation = new_segments
                else:
                     print(f"Warning: Failed to extract feature vector for valid config {self.current_iteration}. Skipping sample.")
                     processed_valid_configs_count -= 1 # Decrement count as it's not fully usable
            else:
                print(f"Config {self.current_iteration} failed analysis. Skipping sample.")
            if getattr(RunConfig, 'CHECKPOINTS_ENABLED', True):
                if time.time() - last_ck_ts >= max(60, int(getattr(RunConfig, 'CHECKPOINT_INTERVAL_MIN', 60)) * 60):
                    last_ck_ts = time.time()
                    try:
                        self._save_checkpoint(feature_vectors, output_values, processed_valid_configs_count)
                    except Exception:
                        pass

            # Optional small delay
            # time.sleep(0.05)

        # --- Loop End Summary ---
        print(f"\n--- Data Collection Loop Finished ({self.current_iteration} iterations attempted) ---")
        if processed_valid_configs_count < self.num_target_samples:
             print(f"Warning: Only generated {processed_valid_configs_count} valid samples out of the target {self.num_target_samples}.")
             print("Consider increasing NUM_SAMPLES or MAX_ITERATION_FACTOR if analysis fails often, or check analysis logs.")
        else:
             print(f"Successfully generated {processed_valid_configs_count} valid samples.")

        return feature_vectors, output_values

    def _monitor_system_health_loop(self):
        interval = max(60, int(getattr(RunConfig, 'MONITOR_INTERVAL_MIN', 30)) * 60)
        stuck_thr = max(60, int(getattr(RunConfig, 'ALERT_STUCK_THRESHOLD_MIN', 90)) * 60)
        while not self._monitor_stop:
            ts = time.time()
            cpu = None
            mem = None
            try:
                import psutil
                cpu = float(psutil.cpu_percent(interval=1))
                mem = float(psutil.virtual_memory().percent)
            except Exception:
                pass
            try:
                du = shutil.disk_usage(str(self.exp_manager.run_dir))
                disk_used_ratio = (du.used / du.total) if du.total else None
            except Exception:
                disk_used_ratio = None
            net_ok = None
            try:
                with urllib.request.urlopen('http://example.com', timeout=5) as _resp:
                    net_ok = True
            except Exception:
                net_ok = False
            rec = {
                'timestamp': ts,
                'cpu_percent': cpu,
                'mem_percent': mem,
                'disk_used_ratio': disk_used_ratio,
                'network_ok': net_ok
            }
            try:
                with open(self.metrics_dir / 'system_health.ndjson', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                pass
            try:
                if disk_used_ratio is not None and disk_used_ratio >= 0.8:
                    self._send_alert('Disk space high', f'Usage={disk_used_ratio:.2f}')
            except Exception:
                pass
            try:
                if (ts - self._heartbeat_ts) >= stuck_thr:
                    self._send_alert('Process may be stuck', f'No heartbeat for {int(ts - self._heartbeat_ts)}s')
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            time.sleep(interval)

    def _send_alert(self, subject: str, body: str):
        to_addr = os.getenv('BUILDOPT_ALERT_EMAIL_TO')
        host = os.getenv('BUILDOPT_ALERT_SMTP_HOST')
        port = int(os.getenv('BUILDOPT_ALERT_SMTP_PORT', '0'))
        user = os.getenv('BUILDOPT_ALERT_SMTP_USER')
        pwd = os.getenv('BUILDOPT_ALERT_SMTP_PASS')
        if to_addr and host and port:
            try:
                msg = f"Subject: {subject}\n\n{body}"
                context = ssl.create_default_context()
                with smtplib.SMTP(host, port, timeout=10) as server:
                    try:
                        server.starttls(context=context)
                    except Exception:
                        pass
                    if user and pwd:
                        server.login(user, pwd)
                    server.sendmail(user or 'noreply@localhost', [to_addr], msg)
                return
            except Exception:
                pass
        try:
            with open(self.metrics_dir / 'alerts.ndjson', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'timestamp': time.time(), 'subject': subject, 'body': body}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _log_error(self, label: str, message: str):
        try:
            with open(self.metrics_dir / 'errors.ndjson', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'timestamp': time.time(), 'label': label, 'message': message}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _save_checkpoint(self, feature_vectors: List[List[float]], output_values: List[List[float]], valid_count: int):
        obj = {
            'timestamp': time.time(),
            'current_iteration': self.current_iteration,
            'valid_count': valid_count,
            'feature_vectors': feature_vectors,
            'output_values': output_values
        }
        with open(self.exp_manager.run_dir / 'checkpoint.json', 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    def _load_checkpoint(self) -> Optional[Dict]:
        p = self.exp_manager.run_dir / 'checkpoint.json'
        if not p.exists():
            return None
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def _preflight_checks(self):
        try:
            du = shutil.disk_usage(str(self.exp_manager.run_dir))
            if du.total and (du.used / du.total) >= 0.9:
                self._send_alert('Disk space critical', f'Usage={(du.used/du.total):.2f}')
        except Exception:
            pass
        try:
            p = self.metrics_dir / 'preflight.tmp'
            with open(p, 'w', encoding='utf-8') as f:
                f.write('ok')
            p.unlink(missing_ok=True)
        except Exception as e:
            self._send_alert('Write permission error', str(e))

    def _train_and_evaluate(self, feature_vectors: list, output_values: list):
        """
        Orquestra o treinamento da pipeline, do modelo e a avaliaÃ§Ã£o final.
        Este mÃ©todo substitui a lÃ³gica que estava espalhada em _train_model e _predict_on_test_set.
        """
        # 1. SPLIT ANTES DO SCALING E TRANSFORMAR OS DADOS
        from sklearn.model_selection import train_test_split
        print("\n[Step 1/5] Splitting data and fitting pipeline...")
        t_split_start = time.time()
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            feature_vectors, output_values, test_size=getattr(NeuralNetConfig, "TEST_SPLIT_RATIO", 0.15), random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=getattr(NeuralNetConfig, "VALIDATION_SPLIT_RATIO", 0.2), random_state=42
        )
        t_split_end = time.time()
        self.feature_pipeline.fit(X_train, y_train)
        t_scale_start = time.time()
        X_train_scaled = self.feature_pipeline.transform_features(X_train)
        y_train_scaled = self.feature_pipeline.transform_outputs(y_train)
        X_val_scaled = self.feature_pipeline.transform_features(X_val)
        y_val_scaled = self.feature_pipeline.transform_outputs(y_val)
        X_test_scaled = self.feature_pipeline.transform_features(X_test)
        y_test_scaled = self.feature_pipeline.transform_outputs(y_test)
        t_scale_end = time.time()
        try:
            h = hashlib.sha256()
            h.update(np.array(feature_vectors, dtype=np.float32).tobytes())
            h.update(np.array(output_values, dtype=np.float32).tobytes())
            dataset_hash = h.hexdigest()
        except Exception:
            dataset_hash = None
        try:
            print(f"[DEBUG MAIN] Exemplo de feature NORMALIZADA (train): {X_train_scaled[0]}")
        except Exception:
            pass
        # Salva a pipeline TREINADA usando o caminho do ExperimentManager
        self.feature_pipeline.save(self.exp_manager.get_pipeline_path())
        try:
            with open(self.exp_manager.get_metrics_dir() / "feature_names.json", "w", encoding="utf-8") as f:
                json.dump({"feature_names": FeatureEngineer.feature_names()}, f)
        except Exception:
            pass

        # 2. TREINAR O MODELO NEURAL
        # O nn_manager recebe os dados JÃ normalizados e retorna os conjuntos de teste (tambÃ©m normalizados).
        print("\n[Step 2/5] Training the Neural Network...")
        t_train_nn_start = time.time()
        self.nn_manager.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        t_train_nn_end = time.time()
        
        # Salva o modelo treinado usando o caminho do ExperimentManager
        self.nn_manager.save_model(self.exp_manager.get_model_path())

        # 2b. Treinar classificador de validade e salvar métricas
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve
            import joblib, json
            if len(self._clf_features) > 0 and len(self._clf_labels) > 0:
                print("\n[Step 2b/5] Training validity classifier...")
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight='balanced')
                )
                from sklearn.model_selection import train_test_split
                Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                    self._clf_features, self._clf_labels, test_size=0.2, stratify=self._clf_labels, random_state=42
                )
                t_train_clf_start = time.time()
                clf.fit(Xc_train, yc_train)
                t_train_clf_end = time.time()
                joblib.dump(clf, self.exp_manager.run_dir / "validity_classifier.pkl")
                y_train_pred = clf.predict(Xc_train)
                train_metrics = {"accuracy": float(accuracy_score(yc_train, y_train_pred))}
                pr, rc, f1, _ = precision_recall_fscore_support(yc_train, y_train_pred, labels=[0,1], zero_division=0)
                train_metrics.update({
                    "precision_by_class": {"0": float(pr[0]), "1": float(pr[1])},
                    "recall_by_class": {"0": float(rc[0]), "1": float(rc[1])},
                    "f1_by_class": {"0": float(f1[0]), "1": float(f1[1])},
                    "confusion_matrix": confusion_matrix(yc_train, y_train_pred, labels=[0,1]).tolist()
                })
                with open(self.exp_manager.get_metrics_dir() / "classifier.json", "w", encoding="utf-8") as f:
                    json.dump(train_metrics, f)
                y_test_pred = clf.predict(Xc_test)
                test_metrics = {"accuracy": float(accuracy_score(yc_test, y_test_pred))}
                pr_t, rc_t, f1_t, _ = precision_recall_fscore_support(yc_test, y_test_pred, labels=[0,1], zero_division=0)
                test_metrics.update({
                    "precision_by_class": {"0": float(pr_t[0]), "1": float(pr_t[1])},
                    "recall_by_class": {"0": float(rc_t[0]), "1": float(rc_t[1])},
                    "f1_by_class": {"0": float(f1_t[0]), "1": float(f1_t[1])},
                    "confusion_matrix": confusion_matrix(yc_test, y_test_pred, labels=[0,1]).tolist()
                })
                with open(self.exp_manager.get_metrics_dir() / "classifier_test.json", "w", encoding="utf-8") as f:
                    json.dump(test_metrics, f)
                try:
                    proba_test = clf.predict_proba(Xc_test)
                    classes = list(clf.named_steps['logisticregression'].classes_)
                    idx_invalid = int(classes.index(0))
                    y_inv_test = (np.array(yc_test) == 0).astype(int)
                    roc_auc = float(roc_auc_score(y_inv_test, proba_test[:, idx_invalid]))
                    fpr, tpr, thr = roc_curve(y_inv_test, proba_test[:, idx_invalid])
                    roc_data = {"fpr": list(map(float, fpr)), "tpr": list(map(float, tpr)), "thresholds": list(map(float, thr)), "auc": roc_auc}
                    with open(self.exp_manager.get_metrics_dir() / "roc_curve.json", "w", encoding="utf-8") as f:
                        json.dump(roc_data, f)
                    try:
                        j = tpr - fpr
                        best_idx = int(np.argmax(j))
                        best_thr = float(thr[best_idx])
                        with open(self.exp_manager.get_metrics_dir() / "validity_threshold.json", "w", encoding="utf-8") as f:
                            json.dump({"threshold": best_thr, "method": "youden", "class_index": idx_invalid}, f)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    lr = clf.named_steps['logisticregression']
                    coeffs = {
                        "classes": list(map(int, lr.classes_.tolist())),
                        "coef": lr.coef_.tolist(),
                        "intercept": lr.intercept_.tolist()
                    }
                    with open(self.exp_manager.get_metrics_dir() / "classifier_coeffs.json", "w", encoding="utf-8") as f:
                        json.dump(coeffs, f)
                except Exception:
                    pass
                print("Validity classifier trained and metrics saved (train/test/ROC).")
            else:
                print("[Step 2b/5] Skipping validity classifier training (no labels).")
        except Exception as e:
            print(f"Warning: Failed to train validity classifier: {e}")

        # 3. FAZER PREDIÃ‡Ã•ES NO CONJUNTO DE TESTE
        if X_test_scaled.size > 0:
            print("\n[Step 3/5] Predicting on the test set...")
            # O modelo recebe dados normalizados e retorna prediÃ§Ãµes normalizadas
            predictions_scaled = self.nn_manager.predict(X_test_scaled)

            # 4. DESNORMALIZAR OS RESULTADOS PARA AVALIAÃ‡ÃƒO
            # Usamos a pipeline para converter prediÃ§Ãµes e valores reais de volta para a escala original (kgf, mÂ³).
            print("\n[Step 4/5] Inverse-transforming predictions for evaluation...")
            predictions_final = self.feature_pipeline.inverse_transform_outputs(predictions_scaled)
            actuals_final = self.feature_pipeline.inverse_transform_outputs(y_test_scaled)

            # 5. AVALIAR E PLOTAR OS RESULTADOS FINAIS
            print("\n[Step 5/5] Evaluating and plotting final results...")
            
            # Converte para arrays numpy para facilitar a indexaÃ§Ã£o por coluna
            actuals_np = np.array(actuals_final)
            predictions_np = np.array(predictions_final)
            
            # DicionÃ¡rio para armazenar as mÃ©tricas calculadas
            final_metrics = {}

            # Calcula mÃ©tricas para Concreto (coluna de Ã­ndice 1)
            # Calcula métricas para Concreto somente se houver 2 saídas
            if predictions_np.shape[1] >= 2 and actuals_np.shape[1] >= 2:
                r2_concrete = r2_score(actuals_np[:, 1], predictions_np[:, 1])
                mae_concrete = mean_absolute_error(actuals_np[:, 1], predictions_np[:, 1])
                final_metrics['concrete'] = {
                    'r2_score': r2_concrete,
                    'mean_absolute_error_m3': mae_concrete
                }
            else:
                print("Concreto: métricas ignoradas (modelo com 1 saída).")
            # Calcula mÃ©tricas para AÃ§o (coluna de Ã­ndice 0), se nÃ£o estiver em modo geomÃ©trico
            if not self.use_geometric_estimate:
                try:
                    r2_steel = r2_score(actuals_np[:, 0], predictions_np[:, 0])
                    mae_steel = mean_absolute_error(actuals_np[:, 0], predictions_np[:, 0])
                    rmse_steel = float(np.sqrt(np.mean((actuals_np[:, 0] - predictions_np[:, 0])**2)))
                    abs_err = np.abs(actuals_np[:, 0] - predictions_np[:, 0])
                    resid_stats = {
                        'mean_abs_error_kgf': float(np.mean(abs_err)),
                        'std_abs_error_kgf': float(np.std(abs_err)),
                        'max_abs_error_kgf': float(np.max(abs_err))
                    }
                    final_metrics['steel'] = {
                        'r2_score': r2_steel,
                        'mean_absolute_error_kgf': mae_steel,
                        'rmse_kgf': rmse_steel,
                        'residual_stats': resid_stats
                    }
                except IndexError:
                    print("Aviso: NÃ£o foi possÃ­vel calcular mÃ©tricas para o aÃ§o.")


            self._evaluate_and_report(predictions_final.tolist(), actuals_final.tolist())
            self._plot_results(predictions_final.tolist(), actuals_final.tolist(), output_values)
                    # Loga os metadados com as mÃ©tricas calculadas

            # Loga os metadados com as mÃ©tricas REAIS que acabamos de calcular
            summary = {
                "experiment_id": self.exp_manager.run_dir.name,
                "timestamp_start": None,
                "timestamp_end": None,
                "device": str(self.nn_manager.device),
                "analysis_mode": self.analysis_mode,
                "dataset_hash": dataset_hash,
                "num_samples_trained": len(X_train),
                "num_test_samples": len(actuals_final),
                "splits": {
                    "test_ratio": getattr(NeuralNetConfig, "TEST_SPLIT_RATIO", None),
                    "val_ratio": getattr(NeuralNetConfig, "VALIDATION_SPLIT_RATIO", None)
                },
                "nn_architecture": {
                    "hidden_layers": getattr(NeuralNetConfig, "HIDDEN_LAYERS", None),
                    "dropout_rate": getattr(NeuralNetConfig, "DROPOUT_RATE", None),
                    "output_size": getattr(NeuralNetConfig, "OUTPUT_SIZE", None)
                },
                "hyperparams": {
                    "learning_rate": getattr(NeuralNetConfig, "LEARNING_RATE", None),
                    "weight_decay": getattr(NeuralNetConfig, "WEIGHT_DECAY", None),
                    "loss_type": getattr(NeuralNetConfig, "LOSS_TYPE", None),
                    "lr_scheduler": getattr(NeuralNetConfig, "LR_SCHEDULER", None),
                    "patience": getattr(NeuralNetConfig, "EARLY_STOPPING_PATIENCE", None)
                },
                "final_metrics": final_metrics,
                "tqs_phase_times_sec": {
                    "modeling": float(self._last_tqs_model_time) if self._last_tqs_model_time is not None else None,
                    "execution": float(self._last_tqs_exec_time) if self._last_tqs_exec_time is not None else None
                },
                "timings_detailed": {
                    "split_sec": float(t_split_end - t_split_start),
                    "scaling_sec": float(t_scale_end - t_scale_start),
                    "train_nn_sec": float(t_train_nn_end - t_train_nn_start),
                    "train_classifier_sec": float(t_train_clf_end - t_train_clf_start)
                }
            }
            try:
                feature_names = FeatureEngineer.feature_names()
                baseline_val_pred_scaled = self.nn_manager.predict(X_val_scaled)
                baseline_val_pred = self.feature_pipeline.inverse_transform_outputs(baseline_val_pred_scaled)
                baseline_val_mae = float(mean_absolute_error(
                    np.array(self.feature_pipeline.inverse_transform_outputs(y_val_scaled))[:, 0],
                    np.array(baseline_val_pred)[:, 0]
                ))
                deltas = []
                Xv = np.array(X_val_scaled, copy=True)
                for j in range(Xv.shape[1]):
                    Xp = Xv.copy()
                    Xp[:, j] = np.random.permutation(Xp[:, j])
                    pv_scaled = self.nn_manager.predict(Xp)
                    pv = self.feature_pipeline.inverse_transform_outputs(pv_scaled)
                    mae_j = float(mean_absolute_error(
                        np.array(self.feature_pipeline.inverse_transform_outputs(y_val_scaled))[:, 0],
                        np.array(pv)[:, 0]
                    ))
                    deltas.append({"feature": feature_names[j] if j < len(feature_names) else f"f{j}", "delta_mae": float(mae_j - baseline_val_mae)})
                with open(self.exp_manager.get_metrics_dir() / "feature_importance.json", "w", encoding="utf-8") as f:
                    json.dump({"baseline_val_mae": baseline_val_mae, "deltas": deltas}, f)
            except Exception:
                pass
            try:
                import psutil
                proc = psutil.Process()
                rss_mb = float(proc.memory_info().rss) / (1024*1024)
                cpu_snapshot = float(psutil.cpu_percent(interval=0))
                summary["resources_snapshot"] = {
                    "cpu_percent": cpu_snapshot,
                    "memory_rss_mb": rss_mb
                }
            except Exception:
                pass
            try:
                libs = {}
                import torch, sklearn, numpy, pandas, shapely, bs4, lxml
                libs = {
                    "torch": getattr(torch, "__version__", None),
                    "sklearn": getattr(sklearn, "__version__", None),
                    "numpy": getattr(numpy, "__version__", None),
                    "pandas": getattr(pandas, "__version__", None),
                    "shapely": getattr(shapely, "__version__", None),
                    "bs4": getattr(bs4, "__version__", None),
                    "lxml": getattr(lxml, "__version__", None)
                }
                summary["libraries"] = libs
            except Exception:
                pass
            summary["summary_text"] = "Resumo executivo do experimento e métricas finais."
            try:
                steel_metrics = final_metrics.get('steel') or {}
                mae_ok = None
                r2_ok = None
                rmse_ok = None
                if steel_metrics:
                    mae = steel_metrics.get('mean_absolute_error_kgf')
                    r2s = steel_metrics.get('r2_score')
                    rmse = steel_metrics.get('rmse_kgf')
                    med_steel = float(np.median(actuals_np[:, 0])) if actuals_np.shape[1] > 0 else None
                    if med_steel and mae is not None:
                        mae_ok = bool(mae <= 0.10 * med_steel)
                    if r2s is not None:
                        r2_ok = bool(r2s >= 0.80)
                    if rmse is not None and med_steel:
                        rmse_ok = bool(rmse <= 0.12 * med_steel)
                clf_test_path = self.exp_manager.get_metrics_dir() / "classifier_test.json"
                clf_ok = None
                try:
                    if clf_test_path.exists():
                        with open(clf_test_path, 'r', encoding='utf-8') as f:
                            obj = json.load(f)
                        pr = obj.get('precision_by_class', {})
                        rc = obj.get('recall_by_class', {})
                        auc = obj.get('roc_auc')
                        if auc is not None:
                            clf_ok = bool(auc >= 0.80)
                        if rc.get('0') is not None and pr.get('0') is not None:
                            clf_ok = bool((rc['0'] >= 0.80) and (pr['0'] >= 0.60)) if clf_ok is None else (clf_ok and (rc['0'] >= 0.80) and (pr['0'] >= 0.60))
                except Exception:
                    pass
                tqs_ok = None
                if self._last_tqs_exec_time is not None:
                    tqs_ok = True
                summary["criteria_status"] = {
                    "steel_mae_ok": mae_ok,
                    "steel_r2_ok": r2_ok,
                    "steel_rmse_ok": rmse_ok,
                    "classifier_ok": clf_ok,
                    "tqs_ok": tqs_ok
                }
            except Exception:
                pass
            with open(self.exp_manager.get_metrics_dir() / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f)
            self.exp_manager.log_metadata({
                "num_samples_trained": len(feature_vectors),
                "num_test_samples": len(actuals_final),
                "final_metrics": final_metrics
            })
            try:
                _lc = locals()
                _clf_diag = _lc.get('clf')
                _yc_test  = _lc.get('yc_test')
                _Xc_test  = _lc.get('Xc_test')
                run_full_diagnostics(
                    experiment_dir=self.exp_manager.run_dir,
                    feature_names=feature_names,
                    nn_manager=self.nn_manager,
                    X_test=X_test_scaled,
                    y_test_steel=actuals_np[:, 0],
                    y_pred_steel=predictions_np[:, 0],
                    feature_pipeline=self.feature_pipeline,
                    classifier=_clf_diag,
                    y_test_valid=_yc_test,
                    X_test_clf=_Xc_test,
                )
            except Exception as _diag_exc:
                print(f"[NNDiagnostics] Skipped due to error: {_diag_exc}")
            try:
                print("Sugestão: execute tuning offline com 'python tuning/tune_model.py' para otimizar hiperparâmetros.")
            except Exception:
                pass
        else:
            print("Skipping prediction and evaluation (no test data available).")
            self.exp_manager.log_metadata({
                "num_samples_trained": len(feature_vectors),
                "num_test_samples": 0,
                "final_metrics": "No test data to evaluate."
            })
        



    def _generate_segment_variation(self, base_segments: List[dict], variation_strategy: str = "random") -> List[dict]:
        """Generates a new variation of segments based on the input mode."""
        if self.use_vector_input:
            # length_processor.generate_variation should handle variation logic
            # It typically takes the segments to vary as input
            return self.length_processor.generate_variation(base_segments, variation_strategy)
        else:
            # Ensure generate_new_binary_vector works as expected
            return generate_new_binary_vector(base_segments)


    def _extract_feature_vector(self, column_polygons: List[Polygon], beam_definitions: List[Dict]) -> Optional[List[float]]:
        """
        Extracts the feature vector (input for NN) from the structure's geometry.
        Args:
            column_polygons: List of Shapely Polygons for columns.
            beam_definitions: List of dictionaries defining beams.
        Returns:
            A list of floats representing the feature vector, or None if extraction fails.
        """
        if not column_polygons:
            print("Warning: Cannot extract features from empty column polygon list.")
            return None
        try:
            feature_engineer = FeatureEngineer(column_polygons, beam_definitions)
            features = feature_engineer.extract_features()
            if not features:
                print("Warning: Extracted feature vector is empty.")
                return None
            return features
        except Exception as e:
            print(f"Error extracting feature vector: {e}. Traceback: {traceback.format_exc()}")
            return None


    def _get_analysis_results(self, segments: List[dict]) -> Tuple[Optional[float], Optional[float], Optional[List[Polygon]], Optional[List[Dict]], bool]:
        """
        Performs structural analysis based on the configured mode (Geometric or TQS).

        Args:
            segments: The list of segment dictionaries for the configuration.

        Returns:
            A tuple (steel_kgf, concrete_m3, column_polygons, beam_definitions).
            Steel is None in geometric mode. Returns (None, None, None, None) if analysis fails.
        """
        print(f"Performing analysis using: {self.analysis_mode}")
        if self.use_geometric_estimate:
            print("  [Geometric] Starting geometric volume calculation...")
            # --- Geometric Estimation Mode ---
            try:
                print("  [Geometric] Step 1: Processing segments into geometric shapes...")
                # Process segments to get beam definitions needed for volume calc
                # Assume processors return tuple: (column_geometry, beam_definitions)
                if self.use_vector_input:
                    column_polygons, beam_definitions = self.length_processor.process_segments(segments)
                else:
                    column_polygons, beam_definitions = self.binary_processor.process_segments(segments)

                if not column_polygons:
                    print("  [Geometric] Warning: No column polygons were generated. Concrete volume might be underestimated.")
                    column_polygons = [] # Ensure it's a list

                # Handle case where beam definitions might not be generated
                if beam_definitions is None:
                     print("   Warning: Could not determine beam definitions. Calculating volume based on pillars only.")
                     beam_definitions = [] # Use empty list for calculator

                print("  [Geometric] Step 2: Calculating total concrete volume...")
                concrete_volume = get_geometric_concrete_volume(column_polygons, beam_definitions)
                print(f"   Geometric Concrete Volume Estimated: {concrete_volume:.4f} mÂ³")
                return None, concrete_volume, column_polygons, beam_definitions, True # Steel is None; geometric mode assumed válido
                

            except Exception as e:
                print(f"   Error during geometric calculation: {e}")
                TQSUtil.writef(f"Error during geometric calculation: {e}")
                return None, None, None, None # Indicate failure
        else:
            # --- TQS Analysis Mode ---
            print("  [TQS] Starting full TQS analysis pipeline...")

            model_was_created = self._create_tqs_structural_model(segments)

            if model_was_created:
                # This second function runs the analysis and extracts results.
                steel_kgf, concrete_m3, is_valid = self._execute_tqs_analysis_and_get_results(segments)
                # We need to get the geometry that was used for the analysis
                if self.use_vector_input:
                    column_polygons, beam_definitions = self.length_processor.process_segments(segments)
                else:
                    column_polygons, beam_definitions = self.binary_processor.process_segments(segments)

                # Return the results, which could be (None, None) if execution failed.
                return steel_kgf, concrete_m3, column_polygons, beam_definitions, is_valid

            else:
                # If model creation failed, abort the process for this sample.
                print("  [TQS] Error: Aborting analysis because model creation failed.")
                return None, None, None, None, False


    def _create_tqs_structural_model(self, segments: List[dict]) -> bool:
        """
        Creates and saves a TQS model file based on a given structural geometry.

        This function handles the entire modeling pipeline:
        1. Processes the abstract segment geometry into TQS-compatible polygons and definitions.
        2. Invokes the TQSModelManager to create and save the building model with all
        structural elements (columns, beams, slabs).
        
        Args:
            segments: A list of segment dictionaries defining the structure's geometry.

        Returns:
            bool: True if the TQS model file was created and saved successfully, False otherwise.
        """
        print("  [Modeling]  Starting TQS model creation pipeline...")
        try:
            t0 = time.time()
            # 1. Process segments into TQS-compatible geometry
            print("      Step 1: Processing segments for TQS geometry...")
            if self.use_vector_input:
                column_polygons, beam_definitions = self.length_processor.process_segments(segments)
            else:
                column_polygons, beam_definitions = self.binary_processor.process_segments(segments)

            # Validate processing results
            if not column_polygons:
                 print("      TQS Error: Segment processing yielded no column polygons.")
                 return False
            if beam_definitions is None:
                 print("      TQS Warning: Segment processing yielded no beam definitions. Proceeding with columns only.")
                 beam_definitions = [] # Ensure list format

            print(f"      Processed into {len(column_polygons)} column groups and {len(beam_definitions)} beam definitions.")

            # 2. Create TQS building model
            print("      Step 2: Creating TQS building model...")
            # Ensure TQS Manager logs details on failure
            model_created = self.tqs_manager.create_building_model_and_elements(column_polygons, beam_definitions)
            if not model_created:
                print("      TQS Error: Failed to create building model via TQS Manager.")
                return False
            print("      TQS model created successfully.")
            self._last_tqs_model_time = time.time() - t0
            return True

        except Exception as e:
            print(f"  [Modeling]  Error: An unexpected exception occurred during model creation: {e}")
            print(traceback.format_exc())
            return False # Indicate failure 
        
    def _execute_tqs_analysis_and_get_results(self, segments: List[dict]) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Runs the TQS analysis on a pre-existing model and extracts the material results.    

        Args:
            segments: List of segment dictionaries defining the structure.

        Returns:
            Tuple (steel_kgf, concrete_m3) or (None, None) if analysis fails at any step.
        """
        print("  [Execution] Starting TQS analysis and result extraction...")
        try:
            start_time_exec  = time.time()
           
            print("      Step 3: Executing TQS global processing...")
            max_attempts = int(getattr(RunConfig, 'TQS_MAX_ATTEMPTS', 3))
            attempts = 0
            while attempts < max_attempts:
                RunModel(BuildingConfig.NAME)
                print("      TQS global processing command issued.")
                tqs_output_file = BuildingConfig.TQS_RESULTS_FILE
                print(f"      Step 4: Extracting results from {tqs_output_file}...")
                timeout = int(getattr(RunConfig, 'TQS_TIMEOUT_SEC', 120))
                start_wait_time  = time.time()
                while not tqs_output_file.exists():
                    if time.time() - start_wait_time  > timeout:
                        attempts += 1
                        print(f"  [Execution] Timeout waiting for results. Retrying ({attempts}/{max_attempts})...")
                        if attempts >= max_attempts:
                            self._send_alert('TQS timeout', 'RESDES.HTM not produced')
                            return None, None, False
                        break
                    time.sleep(0.5)
                if tqs_output_file.exists():
                    break

            print("  [Execution] Results file found. Extracting summary...")
            steel_value_str, concrete_value_str = extract_material_summary(tqs_output_file)

            if steel_value_str is None or concrete_value_str is None:
                print(f"      TQS Error: Could not extract 'Totais' row or values from '{tqs_output_file}'. Check file content and format.")
                return None, None, False

            # 5. Convert results to float
            print("      Step 5: Parsing results...")
            try:
                # Replace comma decimal separator if used in TQS output
                steel_kgf = float(steel_value_str.replace(",", "."))
                concrete_m3 = float(concrete_value_str.replace(",", "."))
            except ValueError as ve:
                 print(f"      TQS Error: Could not convert extracted results ('{steel_value_str}', '{concrete_value_str}') to numbers: {ve}")
                 return None, None, False

            det_ok = False
            try:
                det_ok = bool(self._error_reader._dlls_available())
            except Exception:
                det_ok = False
            if not det_ok:
                print("   TQS Critical Errors detection unavailable (DLLs).")
            else:
                try:
                    print(f"   TQS Critical Errors detection active. DLL dir: {getattr(self._error_reader, '_dll_dir', None)}")
                except Exception:
                    pass
            critical_errors = self._error_reader.get_critical_errors()
            is_valid = len(critical_errors) == 0
            end_time_tqs = time.time()
            self._last_tqs_exec_time = end_time_tqs - start_time_exec
            print(f"   TQS Analysis successful ({end_time_tqs - start_time_exec:.2f}s). Steel: {steel_kgf:.2f} kgf, Concrete: {concrete_m3:.3f} mÂ³ | Valid: {is_valid}")
            if not is_valid and critical_errors:
                print("   TQS Critical Errors detected:")
                for err in critical_errors:
                    try:
                        print(f"     - Element {err.elm_number}: {err.error_header}")
                    except Exception:
                        pass
                try:
                    if self.metrics_dir:
                        rec = {
                            "iteration": self.current_iteration,
                            "errors": [{"elm_number": int(e.elm_number), "error_header": str(e.error_header)} for e in critical_errors]
                        }
                        with open(self.metrics_dir / "tqs_errors.ndjson", "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            return steel_kgf, concrete_m3, is_valid
        
        except Exception as e:
            # Catch any unexpected errors during the TQS pipeline
            error_time = time.time()
            print(f"   TQS Error: An unexpected exception occurred during TQS pipeline at {error_time:.0f}: {e}")
            TQSUtil.writef(f"Error during TQS model run/extraction: {str(e)}")
            print(traceback.format_exc())
            try:
                self._log_error('TQS pipeline', str(e))
            except Exception:
                pass
            return None, None, False # Indicate failure 

    # -------------------------------------------------------------------------
    # Evaluation and Plotting Helper Methods (Placeholder implementations)
    # -------------------------------------------------------------------------

    def _evaluate_and_report(self, predictions: List[List[float]], actual_values: List[List[float]]):
        """
        Calculates and prints evaluation metrics (R2, MAE, percentage error) for the test set predictions.
        """
        print("\n--- Test Set Evaluation ---")

        if len(predictions) == 0 or len(actual_values) == 0:
            print("Evaluation skipped: No predictions or actual values available.")
            return
    
        if len(predictions) != len(actual_values):
             print(f"Evaluation Warning: Mismatch in number of predictions ({len(predictions)}) and actual values ({len(actual_values)}). Evaluating based on shorter list.")
             min_len = min(len(predictions), len(actual_values))
             predictions = predictions[:min_len]
             actual_values = actual_values[:min_len]

        num_test_samples = len(actual_values)
        predicts_steel = self.nn_manager.is_trained and not self.use_geometric_estimate

        # Extract material-specific lists for easier calculation
        has_concrete = all(len(p) >= 2 for p in predictions) and all(len(a) >= 2 for a in actual_values)
        has_steel = all(len(p) >= 1 for p in predictions) and all(len(a) >= 1 for a in actual_values)
        concrete_predictions = [p[1] for p in predictions] if has_concrete else []
        concrete_actuals = [a[1] for a in actual_values] if has_concrete else []
        # Concrete Metrics
        if len(concrete_actuals) > 0:
            r2_concrete = r2_score(concrete_actuals, concrete_predictions)
            mae_concrete = mean_absolute_error(concrete_actuals, concrete_predictions)
            print(f"Concrete R2: {r2_concrete:.4f}")
            print(f"Concrete MAE: {mae_concrete:.4f} m3")
        else:
            print("Concrete metrics not calculated (single-output model or no data).")

        # Steel Metrics (if applicable)
        if predicts_steel and has_steel:
            steel_predictions = [p[0] for p in predictions]
            steel_actuals = [a[0] for a in actual_values]
            if len(steel_actuals) > 0:
                r2_steel = r2_score(steel_actuals, steel_predictions)
                mae_steel = mean_absolute_error(steel_actuals, steel_predictions)
                print(f"Steel R2: {r2_steel:.4f}")
                print(f"Steel MAE: {mae_steel:.4f} kgf")
            else:
                print("Steel metrics not calculated (no data).")
        else:
            print("Steel metrics not applicable (Geometric mode or model not trained for steel). ")

        print("\n--- Detailed Sample Comparison (Test Set) ---")
        total_steel_error_perc = 0.0
        total_concrete_error_perc = 0.0
        valid_steel_samples = 0
        valid_concrete_samples = 0

        print("Comparing predictions against actual values for test samples:")
        print("-" * 60)
        for i in range(num_test_samples):
            pred = predictions[i]
            actual = actual_values[i]

            has_concrete_sample = len(pred) >= 2 and len(actual) >= 2
            has_steel_sample = predicts_steel and len(pred) >= 1 and len(actual) >= 1

            if not has_concrete_sample and not has_steel_sample:
                print(f"Skipping sample {i+1}: No comparable outputs (pred len={len(pred)}, actual len={len(actual)}).")
                continue

            print(f"Test Sample {i+1}/{num_test_samples}:")

            # Concrete Evaluation
            if has_concrete_sample:
                concrete_pred = pred[1]
                concrete_actual = actual[1]
                print(f"  Concrete -> Predicted: {concrete_pred:>8.2f} m3 | Actual: {concrete_actual:>8.2f} m3")
                if abs(concrete_actual) > 1e-6:
                    concrete_err = abs(concrete_pred - concrete_actual) / concrete_actual * 100
                    print(f"                 Error: {concrete_err:>8.2f}%")
                    total_concrete_error_perc += concrete_err
                    valid_concrete_samples += 1
                else:
                    absolute_diff = abs(concrete_pred - concrete_actual)
                    print(f"                 Actual is ~0. Absolute Difference: {absolute_diff:.4f} m3")
            else:
                print("  Concrete -> N/A (single-output model)")

            # Steel Evaluation (if applicable)
            if has_steel_sample:
                steel_pred = pred[0]
                steel_actual = actual[0]
                print(f"  Steel    -> Predicted: {steel_pred:>8.2f} kgf | Actual: {steel_actual:>8.2f} kgf")
                if abs(steel_actual) > 1e-6:
                    steel_err = abs(steel_pred - steel_actual) / steel_actual * 100
                    print(f"                 Error: {steel_err:>8.2f}%")
                    total_steel_error_perc += steel_err
                    valid_steel_samples += 1
                else:
                    absolute_diff = abs(steel_pred - steel_actual)
                    print(f"                 Actual is ~0. Absolute Difference: {absolute_diff:.2f} kgf")
            elif predicts_steel:
                print("  Steel    -> N/A for this sample (missing data)")
            print("-" * 60)

        # Average Errors
        print("\n--- Average Prediction Errors (Test Set) ---")
        if valid_concrete_samples > 0:
             avg_concrete_error = total_concrete_error_perc / valid_concrete_samples
             print(f"Concrete Avg. Error: {avg_concrete_error:.2f}% ({valid_concrete_samples} valid samples)")
        else:
             print("Concrete Avg. Error: Not calculated (no valid samples)")

        if predicts_steel:
            if valid_steel_samples > 0:
                avg_steel_error = total_steel_error_perc / valid_steel_samples
                print(f"Steel Avg. Error:    {avg_steel_error:.2f}% ({valid_steel_samples} valid samples)")
            else:
                 print("Steel Avg. Error:    Not calculated (no valid samples)")
        print("-" * 50)

    def _plot_results(self, predictions: List[List[float]], actual_values: List[List[float]], all_output_values: List[List[float]]):
        """
        Generates comparison (predicted vs actual) and distribution plots.
        """
        print("\n--- Generating Result Plots ---")
        try:
            # Plot comparison (Predicted vs Actual for Test Set)
            # Only plot steel if the model is trained and not in geometric estimate mode
            if self.nn_manager.is_trained and not self.use_geometric_estimate:
                 print("   Plotting Steel comparison...")
                 self.results_plotter.plot_comparison(predictions, actual_values, 'steel')
            else:
                 print("   Skipping Steel comparison plot (Geometric mode or model not trained for steel). ")

            # Plot concrete only if model has 2 outputs
            has_conc = any(len(v) >= 2 for v in actual_values) and any(len(v) >= 2 for v in predictions)
            if has_conc:
                print("   Plotting Concrete comparison...")
                self.results_plotter.plot_comparison(predictions, actual_values, 'concrete')
            else:
                print("   Skipping Concrete comparison plot (single-output model).")

            # Plot distribution of all collected output values
            if all_output_values:
                 print("   Plotting overall material distribution...")
                 self.results_plotter.plot_distribution(all_output_values)
            else:
                 print("   Skipping distribution plot (no overall output values available).")

            print("Plots generated successfully (check results/plots directory).")
            # Plot residuals
            # Residuals: plot concrete only if model has 2 outputs
            if any(len(v) >= 2 for v in actual_values) and any(len(v) >= 2 for v in predictions):
                self.results_plotter.plot_residuals(predictions, actual_values, 'concrete')
            if self.nn_manager.is_trained and not self.use_geometric_estimate:
                self.results_plotter.plot_residuals(predictions, actual_values, 'steel')
        except Exception as e:
            print(f"Warning: Plot generation failed: {str(e)}")

# =============================================================================
# Script Execution Entry Point
# =============================================================================

def main():
    """Main function to run the building optimization process."""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    # 1. Inicializa o gerenciador de experimentos.
    #    Ele usarÃ¡ o diretÃ³rio definido em config/paths.py.
    #    VocÃª pode dar um nome descritivo para a execuÃ§Ã£o.
    exp_manager = ExperimentManager(
        base_dir=paths.EXPERIMENTS_DIR,
        run_name=f"Treino_com_{getattr(RunConfig, 'NUM_SAMPLES', getattr(RunConfig, 'NUMSAMPLES', 0))}_amostras"
    )

    # (Opcional, mas recomendado) Configurar o logging para salvar no diretÃ³rio do experimento
    # setup_logging(log_dir=exp_manager.run_dir)

    try:
        # 2. Instancia o otimizador, passando o gerenciador de experimento
        optimizer = BuildingOptimizer(exp_manager)
        
        # 3. Executa o fluxo de otimizaÃ§Ã£o
        optimizer.run_optimization()
        
        logging.info(f"ExecuÃ§Ã£o {exp_manager.run_dir.name} finalizada com sucesso.")

    except Exception as e:
        logging.error(f"ExecuÃ§Ã£o {exp_manager.run_dir.name} falhou.", exc_info=True)

    except Exception as main_error:
         print("\n--- A CRITICAL ERROR OCCURRED IN MAIN EXECUTION ---")
         print(f"Error Type: {type(main_error).__name__}")
         print(f"Error Details: {main_error}")
         # Log detailed traceback
         print("\nTraceback:")
         print(traceback.format_exc())
         print("--- Script execution aborted ---")
    finally:
        # This block executes whether an error occurred or not
        print("\n===========================================")
        print("   Building Optimization Script Finished   ")
        print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("===========================================")


if __name__ == '__main__':
    # This ensures the main function runs only when the script is executed directly
    main()
