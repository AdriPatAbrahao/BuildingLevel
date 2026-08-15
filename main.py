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
import argparse
import random as py_random
import traceback
import time # Added for potential delays and timing
from typing import List, Tuple, Optional, Dict
import numpy as np
from shapely.geometry import Polygon
from sklearn.metrics import r2_score, mean_absolute_error # Added for evaluation metrics

# Third-party imports
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
from config.settings import BuildingConfig, RunConfig, NeuralNetConfig, ParallelConfig, ObjectiveConfig # General settings, NN config, analysis mode flag
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
from tqs_interface.tqs_worker_pool import TQSWorkerPool, _run_model_with_timeout  # Parallel TQS execution pool

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


def _pid_exists(pid: int) -> bool:
    """Return whether a process exists, using psutil when available."""
    try:
        import psutil
        return bool(psutil.pid_exists(pid))
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def _acquire_collection_lock(lock_path: Path, run_dir: Path) -> None:
    """Atomically prevent two collectors from sharing the same TQS slot."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "run_dir": str(Path(run_dir).resolve()),
        "created_at": time.time(),
    }
    for _ in range(2):
        try:
            with open(lock_path, "x", encoding="utf-8") as stream:
                json.dump(payload, stream)
            return
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid", -1))
            except Exception:
                existing_pid = -1
            if existing_pid > 0 and _pid_exists(existing_pid):
                raise RuntimeError(
                    "Another collection process is active for this TQS slot: "
                    f"PID={existing_pid}, lock='{lock_path}'."
                )
            lock_path.unlink(missing_ok=True)
    raise RuntimeError(f"Could not acquire collection lock '{lock_path}'.")


def _release_collection_lock(lock_path: Optional[Path]) -> None:
    """Remove only a lock owned by the current process."""
    if lock_path is None or not lock_path.exists():
        return
    try:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if int(existing.get("pid", -1)) == os.getpid():
            lock_path.unlink(missing_ok=True)
    except Exception:
        pass

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

    def run_optimization(
        self,
        *,
        collect_only: bool = False,
        train_from_checkpoint: bool = False,
    ):
        """Execute collection and/or training according to the selected mode."""
        print(f"\n--- Starting Optimization Workflow ---")
        start_time_total = time.time()
        try:
            if train_from_checkpoint:
                checkpoint = self._load_checkpoint()
                if not checkpoint:
                    raise RuntimeError(
                        "Training requested but checkpoint.json was not found."
                    )
                (
                    feature_vectors,
                    output_values,
                    _,
                    self.current_iteration,
                    _,
                ) = self._restore_collection_state(checkpoint)
                self._validate_collected_data(feature_vectors, output_values)
                print("\n--- TRAINING FROM SAVED COLLECTION ---")
                self._train_and_evaluate(feature_vectors, output_values)
                self._save_results()
                print("--- TRAINING FROM CHECKPOINT COMPLETE ---")
                return

            # --- FASE 1: GERAÃ‡ÃƒO DE DADOS ---
            print("\n--- PHASE 1: DATA COLLECTION ---")
            initial_segments = self._load_initial_segments()
            # Este mÃ©todo retorna os dados brutos (nÃ£o normalizados)
            feature_vectors, output_values = self._collect_training_data(initial_segments)
            self._validate_collected_data(feature_vectors, output_values)
            print(f"--- PHASE 1 COMPLETE: {len(feature_vectors)} samples generated. ---")

            if collect_only:
                print(
                    "--- COLLECTION-ONLY MODE COMPLETE: "
                    "training was not started. ---"
                )
                return

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
            raise

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

    def _collect_training_data_parallel(
        self, initial_segments: List[dict]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Parallel variant of ``_collect_training_data`` using
        :class:`~tqs_interface.tqs_worker_pool.TQSWorkerPool`.

        Keeps all worker slots continuously busy with a sliding-window
        dispatch strategy:
          1. Prime the pipeline by submitting one job per worker slot.
          2. As each result arrives, immediately submit the next job.
          3. Feature extraction (CPU-only) runs in the main process while
             workers process the next batch.

        The initial configuration is still evaluated sequentially in the
        main process so that the first sample is always the seed geometry.
        """
        import copy as _copy

        num_workers   = int(getattr(ParallelConfig, "NUM_WORKERS", 2))
        base_name     = str(getattr(ParallelConfig, "BASE_NAME", "OptimBuilding"))
        timeout_sec   = int(getattr(ParallelConfig, "TIMEOUT_SEC", 180))
        max_iterations = self.num_target_samples * RunConfig.MAX_ITERATION_FACTOR

        feature_vectors: List[List[float]] = []
        output_values:   List[List[float]] = []
        processed_valid = 0
        current_iter    = 0
        seed_processed  = False
        last_ck_ts      = time.time()

        if (
            getattr(RunConfig, 'CHECKPOINTS_ENABLED', True)
            and getattr(RunConfig, 'RESUME_FROM_CHECKPOINT', False)
        ):
            checkpoint = self._load_checkpoint()
            if checkpoint:
                if int(checkpoint.get('checkpoint_version', 1)) < 2:
                    raise RuntimeError(
                        "Legacy checkpoint does not contain classifier state; "
                        "parallel resume aborted."
                    )
                (
                    feature_vectors,
                    output_values,
                    processed_valid,
                    current_iter,
                    seed_processed,
                ) = self._restore_collection_state(checkpoint)
                print(
                    "Resuming parallel collection from checkpoint: "
                    f"iteration={current_iter}, valid={processed_valid}."
                )

        # ── Process seed configuration sequentially in the main process ──────
        print(
            f"\n--- Parallel Data Collection (TQS Mode) "
            f"| {num_workers} worker(s) | "
            f"target={self.num_target_samples} samples ---"
        )
        if not seed_processed:
            print("\nAnalyzing seed configuration (sequential, main process)...")
            steel, concrete, col_polys_s, beam_defs_s, is_valid_s = (
                self._get_analysis_results(initial_segments)
            )
            if concrete is None:
                raise RuntimeError(
                    "Seed configuration analysis failed. Cannot start parallel collection."
                )
            fv_seed = self._extract_feature_vector(col_polys_s, beam_defs_s)
            if not fv_seed:
                raise RuntimeError("Seed feature extraction failed.")
            self._clf_features.append(fv_seed)
            self._clf_labels.append(1 if is_valid_s else 0)
            if is_valid_s:
                if steel is None:
                    raise RuntimeError(
                        "Steel is None for seed. "
                        "Use TQS mode (RunConfig.USE_GEOMETRIC_ESTIMATE=False)."
                    )
                feature_vectors.append(fv_seed)
                output_values.append([steel])
                if self.use_vector_input:
                    self.generated_valid_configurations.append(
                        _copy.deepcopy(initial_segments)
                    )
                processed_valid += 1
            seed_processed = True
            print(
                f"Seed -> steel={steel:.1f} kgf  concrete={concrete:.4f} m3  "
                f"valid={is_valid_s}"
            )
            if getattr(RunConfig, "CHECKPOINTS_ENABLED", True):
                self._save_checkpoint(
                    feature_vectors,
                    output_values,
                    processed_valid,
                    current_iteration=current_iter,
                    seed_processed=True,
                )

        # ── Maps job_id → (col_polys, beam_defs, new_segments) ───────────────
        # Kept in the main process so we can do feature extraction after results
        in_flight: Dict[int, Tuple] = {}

        def _generate_and_submit(pool: TQSWorkerPool) -> Optional[int]:
            """Generate one variation, process segments, submit to pool."""
            nonlocal current_iter
            if current_iter >= max_iterations:
                return None
            current_iter += 1
            try:
                valid_labels = sum(1 for label in self._clf_labels if label == 1)
                invalid_labels = len(self._clf_labels) - valid_labels
                invalid_fraction = (
                    invalid_labels / len(self._clf_labels)
                    if self._clf_labels
                    else 0.0
                )
                variation_strategy = (
                    "upper_biased" if invalid_fraction >= 0.35 else "random"
                )
                new_segs = self._generate_segment_variation(
                    initial_segments,
                    variation_strategy,
                )
                seg_hash = hashlib.sha256(
                    json.dumps(new_segs, ensure_ascii=False).encode()
                ).hexdigest()
                if seg_hash in getattr(self, "_seen_segments_hash", set()):
                    return None
                self._seen_segments_hash.add(seg_hash)

                # Segment processing is CPU-only — stays in the main process.
                if self.use_vector_input:
                    col_polys, beam_defs = (
                        self.length_processor.process_segments(new_segs)
                    )
                else:
                    col_polys, beam_defs = (
                        self.binary_processor.process_segments(new_segs)
                    )
                if not col_polys:
                    return None

                job_id = pool.submit(col_polys, beam_defs)
                in_flight[job_id] = (col_polys, beam_defs, new_segs)
                return job_id

            except Exception as gen_err:
                print(f"[Parallel] Generation error (iter {current_iter}): {gen_err}")
                return None

        # ── Start the pool and fill the sliding window ────────────────────────
        validity_check_dll = bool(getattr(ParallelConfig, "VALIDITY_CHECK_DLL", False))
        with TQSWorkerPool(
            num_workers=num_workers,
            base_name=base_name,
            timeout_sec=timeout_sec,
            validity_check_dll=validity_check_dll,
        ) as pool:
            # Prime: one job per worker slot
            primed = 0
            while primed < num_workers and current_iter < max_iterations:
                if _generate_and_submit(pool) is not None:
                    primed += 1

            # Sliding-window loop: collect → replenish → repeat
            max_consec_timeouts = int(getattr(ParallelConfig, "MAX_CONSECUTIVE_TIMEOUTS", 3))
            consec_timeouts = 0
            while in_flight and processed_valid < self.num_target_samples:
                self._heartbeat_ts = time.time()

                try:
                    res = pool.get_result(timeout=float(timeout_sec) + 30)
                    consec_timeouts = 0  # reset on any successful result
                except Exception as wait_err:
                    consec_timeouts += 1
                    print(
                        f"[Parallel] Timeout waiting for result ({consec_timeouts}/"
                        f"{max_consec_timeouts}): {wait_err}."
                    )
                    if consec_timeouts >= max_consec_timeouts:
                        print("[Parallel] Too many consecutive timeouts. Stopping collection.")
                        break
                    continue

                entry = in_flight.pop(res.job_id, None)
                if entry is None:
                    continue  # stale / unexpected job_id
                col_polys, beam_defs, new_segs = entry

                print(
                    f"\n[Parallel] Job #{res.job_id} from [{res.slot_name}] "
                    f"in {res.elapsed:.1f}s - success={res.success}"
                )

                if res.success:
                    fv = self._extract_feature_vector(col_polys, beam_defs)
                    if fv:
                        # Determine validity: DLL check (when enabled) OR output bounds.
                        _steel_min = getattr(ObjectiveConfig, "STEEL_MIN_KGF", None)
                        _steel_max = getattr(ObjectiveConfig, "STEEL_MAX_KGF", None)
                        _conc_min  = getattr(ObjectiveConfig, "CONCRETE_MIN_M3", None)
                        _out_of_bounds = (
                            (_steel_min is not None and res.steel < _steel_min) or
                            (_steel_max is not None and res.steel > _steel_max) or
                            (_conc_min  is not None and res.concrete < _conc_min)
                        )
                        is_valid_sample = res.is_valid and not _out_of_bounds
                        self._clf_features.append(fv)
                        self._clf_labels.append(1 if is_valid_sample else 0)
                        if is_valid_sample:
                            feature_vectors.append(fv)
                            output_values.append([res.steel])
                            if self.use_vector_input:
                                self.generated_valid_configurations.append(
                                    _copy.deepcopy(new_segs)
                                )
                            processed_valid += 1
                            if processed_valid % 100 == 0:
                                try:
                                    self.segment_plotter.plot_segments(
                                        new_segs, current_iter,
                                        steel=res.steel, concrete=res.concrete,
                                    )
                                except Exception as _pe:
                                    print(f"[Parallel] Segment plot skipped: {_pe}")
                            print(
                                f"  [OK] steel={res.steel:.1f} kgf  "
                                f"concrete={res.concrete:.4f} m3  "
                                f"valid_count={processed_valid}/"
                                f"{self.num_target_samples}"
                            )
                        else:
                            reason = "DLL check" if not res.is_valid else "output bounds"
                            print(
                                f"  [INVALID] job #{res.job_id} rejected by {reason} "
                                f"(steel={res.steel:.1f} kgf  concrete={res.concrete:.4f} m3)."
                            )
                    else:
                        print(
                            f"  [WARN] Feature extraction failed for job #{res.job_id}."
                        )
                else:
                    print(f"  [FAILED] Job #{res.job_id}: {res.error}")

                # Checkpoint
                if getattr(RunConfig, "CHECKPOINTS_ENABLED", True):
                    ck_interval = max(
                        60,
                        int(getattr(RunConfig, "CHECKPOINT_INTERVAL_MIN", 60)) * 60
                    )
                    if time.time() - last_ck_ts >= ck_interval:
                        last_ck_ts = time.time()
                        self._save_checkpoint(
                            feature_vectors,
                            output_values,
                            processed_valid,
                            current_iteration=current_iter,
                            seed_processed=seed_processed,
                        )

                # Replenish: keep pipeline full while target not reached
                if (
                    processed_valid < self.num_target_samples
                    and current_iter < max_iterations
                ):
                    _generate_and_submit(pool)

        # ── Summary ───────────────────────────────────────────────────────────
        print(
            f"\n--- Parallel Collection Finished "
            f"({current_iter} iterations, {processed_valid} valid samples) ---"
        )
        if processed_valid < self.num_target_samples:
            print(
                f"Warning: only {processed_valid}/{self.num_target_samples} "
                "valid samples collected."
            )
        if getattr(RunConfig, "CHECKPOINTS_ENABLED", True):
            self._save_checkpoint(
                feature_vectors,
                output_values,
                processed_valid,
                current_iteration=current_iter,
                seed_processed=seed_processed,
                collection_complete=processed_valid >= self.num_target_samples,
            )
        return feature_vectors, output_values

    def _collect_training_data(self, initial_segments: List[dict]) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Generates configurations and collects feature vectors and corresponding
        analysis outputs (TQS or Geometric).

        Delegates to the parallel implementation when
        ``ParallelConfig.ENABLED`` is ``True`` and the analysis mode is TQS
        (geometric mode is already CPU-only and does not benefit from the pool).

        Args:
            initial_segments: The starting list of segment dictionaries.

        Returns:
            A tuple containing:
                - feature_vectors (List[List[float]]): List of input vectors for the NN.
                - output_values (List[List[float]]): List of corresponding output vectors
                                                     [steel, concrete] or [0.0, concrete].
        """
        use_parallel = (
            getattr(ParallelConfig, "ENABLED", False)
            and not self.use_geometric_estimate  # TQS mode only
        )
        if use_parallel:
            return self._collect_training_data_parallel(initial_segments)

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

    def _save_checkpoint(
        self,
        feature_vectors: List[List[float]],
        output_values: List[List[float]],
        valid_count: int,
        *,
        current_iteration: Optional[int] = None,
        seed_processed: bool = True,
        collection_complete: bool = False,
    ):
        seed_path = Path(self.length_processor.csv_path).resolve()
        obj = {
            'checkpoint_version': 3,
            'feature_schema_version': int(NeuralNetConfig.FEATURE_SCHEMA_VERSION),
            'timestamp': time.time(),
            'current_iteration': (
                self.current_iteration
                if current_iteration is None
                else int(current_iteration)
            ),
            'valid_count': valid_count,
            'feature_vectors': feature_vectors,
            'output_values': output_values,
            'classifier_features': self._clf_features,
            'classifier_labels': self._clf_labels,
            'generated_valid_configurations': self.generated_valid_configurations,
            'seen_segments_hashes': sorted(self._seen_segments_hash),
            'python_random_state': py_random.getstate(),
            'seed_processed': bool(seed_processed),
            'collection_complete': bool(collection_complete),
            'target_valid_samples': int(self.num_target_samples),
            'parallel_base_name': str(ParallelConfig.BASE_NAME),
            'worker_count': int(ParallelConfig.NUM_WORKERS),
            'seed_csv': str(seed_path),
            'seed_sha256': hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        }
        checkpoint_path = self.exp_manager.run_dir / 'checkpoint.json'
        temporary_path = checkpoint_path.with_suffix('.json.tmp')
        with open(temporary_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)
        temporary_path.replace(checkpoint_path)

    def _load_checkpoint(self) -> Optional[Dict]:
        p = self.exp_manager.run_dir / 'checkpoint.json'
        if not p.exists():
            return None
        try:
            with open(p, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Could not read checkpoint '{p}': {exc}") from exc

        stored_seed_hash = checkpoint.get('seed_sha256')
        if stored_seed_hash:
            seed_path = Path(self.length_processor.csv_path).resolve()
            current_seed_hash = hashlib.sha256(seed_path.read_bytes()).hexdigest()
            if stored_seed_hash != current_seed_hash:
                raise RuntimeError(
                    "Checkpoint seed differs from the current seed CSV; "
                    "resume aborted."
                )
        stored_feature_schema = checkpoint.get('feature_schema_version')
        current_feature_schema = int(NeuralNetConfig.FEATURE_SCHEMA_VERSION)
        if stored_feature_schema != current_feature_schema:
            raise RuntimeError(
                "Checkpoint feature schema differs from the current extractor "
                f"(checkpoint={stored_feature_schema!r}, current={current_feature_schema}); "
                "resume aborted."
            )
        return checkpoint

    def _restore_collection_state(self, checkpoint: Dict):
        """Restore all regression, classifier and deduplication state."""
        feature_vectors = checkpoint.get('feature_vectors', [])
        output_values = checkpoint.get('output_values', [])
        valid_count = int(checkpoint.get('valid_count', len(feature_vectors)))
        current_iteration = int(checkpoint.get('current_iteration', 0))
        self._clf_features = checkpoint.get('classifier_features', [])
        self._clf_labels = checkpoint.get('classifier_labels', [])
        self.generated_valid_configurations = checkpoint.get(
            'generated_valid_configurations', []
        )
        self._seen_segments_hash = set(
            checkpoint.get('seen_segments_hashes', [])
        )
        random_state = checkpoint.get('python_random_state')
        if random_state is not None:
            def _lists_to_tuples(value):
                if isinstance(value, list):
                    return tuple(_lists_to_tuples(item) for item in value)
                return value

            py_random.setstate(_lists_to_tuples(random_state))
        seed_processed = bool(checkpoint.get('seed_processed', False))
        return (
            feature_vectors,
            output_values,
            valid_count,
            current_iteration,
            seed_processed,
        )

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
        _seed = getattr(RunConfig, 'SEED', 42)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            feature_vectors, output_values, test_size=getattr(NeuralNetConfig, "TEST_SPLIT_RATIO", 0.15), random_state=_seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=getattr(NeuralNetConfig, "VALIDATION_SPLIT_RATIO", 0.2), random_state=_seed
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

        # ── Save raw + scaled arrays for post-training analysis ───────────────
        # Enables running nn_diagnostics.py standalone later without retraining.
        try:
            np.savez_compressed(
                self.exp_manager.run_dir / "arrays.npz",
                X_train=np.array(X_train, dtype=np.float32),
                X_val=np.array(X_val, dtype=np.float32),
                X_test=np.array(X_test, dtype=np.float32),
                y_train=np.array(y_train, dtype=np.float32),
                y_val=np.array(y_val, dtype=np.float32),
                y_test=np.array(y_test, dtype=np.float32),
                X_train_scaled=X_train_scaled,
                X_val_scaled=X_val_scaled,
                X_test_scaled=X_test_scaled,
                y_train_scaled=y_train_scaled,
                y_val_scaled=y_val_scaled,
                y_test_scaled=y_test_scaled,
            )
            print(f"[Arrays] Saved arrays.npz ({len(X_train)} train / {len(X_val)} val / {len(X_test)} test)")
        except Exception as _e_arr:
            print(f"[Arrays] Failed to save arrays.npz: {_e_arr}")

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
            feature_names = FeatureEngineer.feature_names()
        except Exception:
            feature_names = []
        try:
            with open(self.exp_manager.get_metrics_dir() / "feature_names.json", "w", encoding="utf-8") as f:
                json.dump({"feature_names": feature_names}, f)
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
        #
        # Split em 3 vias (train/val/test) em vez de train/test simples:
        #   - train (60%): ajusta os pesos da regressão logística.
        #   - val   (20%): calibra o limiar de decisão (Youden) — a ÚNICA finalidade
        #                  desse subconjunto é escolher o ponto de corte da curva ROC.
        #   - test  (20%): nunca é usado em nenhuma decisão de calibração; reporta
        #                  o desempenho esperado da regra já calibrada
        #                  (prob_invalid >= threshold), tal como ela é usada de
        #                  fato na função objetivo. Isso evita reportar métricas
        #                  "de teste" otimistas por terem sido calculadas no mesmo
        #                  conjunto usado para escolher o limiar.
        t_train_clf_start = t_train_clf_end = None
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix, roc_curve
            from sklearn.model_selection import train_test_split
            import joblib, json
            if len(self._clf_features) > 0 and len(self._clf_labels) > 0:
                n_classes = len(set(self._clf_labels))
                if n_classes < 2:
                    print(
                        f"\n[Step 2b/5] Skipping validity classifier: only {n_classes} class "
                        f"present in labels (need at least 2). "
                        f"Set ParallelConfig.VALIDITY_CHECK_DLL=True or define "
                        f"ObjectiveConfig.STEEL_MIN_KGF/STEEL_MAX_KGF to generate invalid samples."
                    )
                else:
                    print("\n[Step 2b/5] Training validity classifier (train/val/test split)...")
                    _clf_seed = getattr(RunConfig, 'SEED', 42)

                    # 60% train / 20% val (calibra limiar) / 20% test (reporta apenas)
                    Xc_train, Xc_temp, yc_train, yc_temp = train_test_split(
                        self._clf_features, self._clf_labels,
                        test_size=0.4, stratify=self._clf_labels, random_state=_clf_seed
                    )
                    Xc_val, Xc_test, yc_val, yc_test = train_test_split(
                        Xc_temp, yc_temp,
                        test_size=0.5, stratify=yc_temp, random_state=_clf_seed
                    )
                    n_inv_train = sum(1 for v in yc_train if v == 0)
                    n_inv_val = sum(1 for v in yc_val if v == 0)
                    n_inv_test = sum(1 for v in yc_test if v == 0)
                    print(
                        f"   - Split: {len(Xc_train)} train / {len(Xc_val)} val / {len(Xc_test)} test "
                        f"(inválidos: {n_inv_train}/{n_inv_val}/{n_inv_test})"
                    )

                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight='balanced')
                    )
                    t_train_clf_start = time.time()
                    clf.fit(Xc_train, yc_train)
                    t_train_clf_end = time.time()
                    joblib.dump(clf, self.exp_manager.run_dir / "validity_classifier.pkl")

                    classes = list(clf.named_steps['logisticregression'].classes_)
                    idx_invalid = int(classes.index(0)) if 0 in classes else 0

                    def _clf_split_metrics(y_true, y_pred) -> dict:
                        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
                        return {
                            "accuracy": float(accuracy_score(y_true, y_pred)),
                            "precision_by_class": {"0": float(pr[0]), "1": float(pr[1])},
                            "recall_by_class": {"0": float(rc[0]), "1": float(rc[1])},
                            "f1_by_class": {"0": float(f1[0]), "1": float(f1[1])},
                            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
                        }

                    # --- Diagnóstico no treino (limiar padrão 0.5 do clf.predict) ---
                    train_metrics = _clf_split_metrics(yc_train, clf.predict(Xc_train))
                    train_metrics.update({"split": "train", "n_samples": len(Xc_train)})
                    with open(self.exp_manager.get_metrics_dir() / "classifier.json", "w", encoding="utf-8") as f:
                        json.dump(train_metrics, f)

                    # --- Calibração do limiar (Youden) — SOMENTE na validação ---
                    best_thr = 0.5
                    try:
                        proba_val = clf.predict_proba(Xc_val)
                        y_inv_val = (np.array(yc_val) == 0).astype(int)
                        fpr, tpr, thr = roc_curve(y_inv_val, proba_val[:, idx_invalid])
                        roc_val_auc = float(roc_auc_score(y_inv_val, proba_val[:, idx_invalid]))
                        roc_data = {
                            "fpr": list(map(float, fpr)), "tpr": list(map(float, tpr)),
                            "thresholds": list(map(float, thr)), "auc": roc_val_auc,
                            "split": "validation", "n_samples": len(Xc_val),
                        }
                        with open(self.exp_manager.get_metrics_dir() / "roc_curve.json", "w", encoding="utf-8") as f:
                            json.dump(roc_data, f)

                        j = tpr - fpr
                        best_idx = int(np.argmax(j))
                        best_thr = float(thr[best_idx])
                        with open(self.exp_manager.get_metrics_dir() / "validity_threshold.json", "w", encoding="utf-8") as f:
                            json.dump({
                                "threshold": best_thr, "method": "youden", "class_index": idx_invalid,
                                "split": "validation", "n_samples": len(Xc_val),
                            }, f)
                    except Exception as exc:
                        print(f"Warning: threshold calibration on validation split failed, using default 0.5: {exc}")

                    # --- Métricas finais no teste, aplicando o limiar já calibrado ---
                    # O teste nunca participa da calibração: mede o desempenho real da
                    # regra de decisão (prob_invalid >= best_thr) tal como ela é usada
                    # na função objetivo (optimization/objective_function.py).
                    try:
                        proba_test = clf.predict_proba(Xc_test)
                        y_pred_test = np.where(proba_test[:, idx_invalid] >= best_thr, 0, 1)
                        y_inv_test = (np.array(yc_test) == 0).astype(int)
                        test_metrics = _clf_split_metrics(yc_test, y_pred_test)
                        test_metrics.update({
                            "split": "test", "n_samples": len(Xc_test),
                            "threshold_used": best_thr,
                            "auc": float(roc_auc_score(y_inv_test, proba_test[:, idx_invalid])),
                        })
                        with open(self.exp_manager.get_metrics_dir() / "classifier_test.json", "w", encoding="utf-8") as f:
                            json.dump(test_metrics, f)

                        # ROC do teste é só para relato — nunca usada para calibrar nada.
                        fpr_t, tpr_t, thr_t = roc_curve(y_inv_test, proba_test[:, idx_invalid])
                        roc_test_data = {
                            "fpr": list(map(float, fpr_t)), "tpr": list(map(float, tpr_t)),
                            "thresholds": list(map(float, thr_t)), "auc": test_metrics["auc"],
                            "split": "test", "n_samples": len(Xc_test),
                        }
                        with open(self.exp_manager.get_metrics_dir() / "roc_curve_test.json", "w", encoding="utf-8") as f:
                            json.dump(roc_test_data, f)
                    except Exception as exc:
                        print(f"Warning: failed to compute final test metrics: {exc}")

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
                    print("Validity classifier trained: threshold calibrated on validation, metrics reported on held-out test.")
            else:
                print("[Step 2b/5] Skipping validity classifier training (no labels).")
        except Exception as e:
            print(f"Warning: Failed to train validity classifier: {e}")

        # 2c. Feature Importance analysis (PFI sklearn + SHAP)
        try:
            _lc2   = locals()
            _clf_fi   = _lc2.get('clf')
            _yc_fi    = np.array(_lc2.get('yc_test', []))
            _Xc_fi    = np.array(_lc2.get('Xc_test', [])) if _lc2.get('Xc_test') is not None else None
            self.nn_manager.run_feature_importance_analysis(
                X_val=X_val_scaled,
                y_val_real=self.feature_pipeline.inverse_transform_outputs(y_val_scaled)[:, 0],
                feature_names=feature_names,
                plotter=self.results_plotter,
                feature_pipeline=self.feature_pipeline,
                classifier=_clf_fi,
                X_val_clf=_Xc_fi if (_Xc_fi is not None and len(_Xc_fi) > 0) else None,
                y_val_clf=_yc_fi if len(_yc_fi) > 0 else None,
            )
        except Exception as _fi_exc:
            print(f"[FeatureImportance] Skipped due to error: {_fi_exc}")

        # 3. FAZER PREDIÇÕES NO CONJUNTO DE TESTE
        if X_test_scaled.size > 0:
            print("\n[Step 3/5] Predicting on the test set...")
            # Benchmark de tempo de inferência do surrogate (100 passes → média por amostra)
            _n_bench = 100
            _t_bench_start = time.perf_counter()
            for _ in range(_n_bench):
                self.nn_manager.predict(X_test_scaled)
            _t_bench_end = time.perf_counter()
            _surrogate_ms_per_sample = ((_t_bench_end - _t_bench_start) / (_n_bench * len(X_test_scaled))) * 1000
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
                    y_true_s  = actuals_np[:, 0]
                    y_pred_s  = predictions_np[:, 0]
                    abs_err   = np.abs(y_true_s - y_pred_s)

                    r2_steel   = r2_score(y_true_s, y_pred_s)
                    mae_steel  = mean_absolute_error(y_true_s, y_pred_s)
                    rmse_steel = float(np.sqrt(np.mean((y_true_s - y_pred_s) ** 2)))

                    # MAPE — ignora amostras com aço real ≈ 0 para evitar divisão instável
                    _nonzero = y_true_s != 0
                    mape_steel = float(np.mean(abs_err[_nonzero] / np.abs(y_true_s[_nonzero])) * 100) if _nonzero.any() else None

                    # P90 do erro absoluto: 90% das predições erram menos que este valor
                    p90_steel = float(np.percentile(abs_err, 90))

                    # Bootstrap CI (95%) para R², MAE e RMSE — 1000 iterações
                    _rng = np.random.default_rng(getattr(RunConfig, 'SEED', 42))
                    _n   = len(y_true_s)
                    _boot_r2, _boot_mae, _boot_rmse = [], [], []
                    for _ in range(1000):
                        _idx = _rng.integers(0, _n, size=_n)
                        _yt, _yp = y_true_s[_idx], y_pred_s[_idx]
                        _boot_r2.append(r2_score(_yt, _yp))
                        _boot_mae.append(float(np.mean(np.abs(_yt - _yp))))
                        _boot_rmse.append(float(np.sqrt(np.mean((_yt - _yp) ** 2))))
                    ci_95 = {
                        'r2':   [float(np.percentile(_boot_r2,   2.5)), float(np.percentile(_boot_r2,   97.5))],
                        'mae':  [float(np.percentile(_boot_mae,  2.5)), float(np.percentile(_boot_mae,  97.5))],
                        'rmse': [float(np.percentile(_boot_rmse, 2.5)), float(np.percentile(_boot_rmse, 97.5))],
                    }

                    resid_stats = {
                        'mean_abs_error_kgf': float(np.mean(abs_err)),
                        'std_abs_error_kgf':  float(np.std(abs_err)),
                        'max_abs_error_kgf':  float(np.max(abs_err)),
                        'p90_abs_error_kgf':  p90_steel,
                    }
                    final_metrics['steel'] = {
                        'r2_score':                  r2_steel,
                        'mean_absolute_error_kgf':   mae_steel,
                        'rmse_kgf':                  rmse_steel,
                        'mape_pct':                  mape_steel,
                        'p90_abs_error_kgf':         p90_steel,
                        'bootstrap_ci_95':           ci_95,
                        'residual_stats':            resid_stats,
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
                "surrogate_inference_ms_per_sample": float(_surrogate_ms_per_sample),
                "timings_detailed": {
                    "split_sec": float(t_split_end - t_split_start),
                    "scaling_sec": float(t_scale_end - t_scale_start),
                    "train_nn_sec": float(t_train_nn_end - t_train_nn_start),
                    "train_classifier_sec": float(t_train_clf_end - t_train_clf_start) if (t_train_clf_end is not None and t_train_clf_start is not None) else None
                }
            }
            try:
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
            validity_required = bool(
                getattr(ParallelConfig, "VALIDITY_CHECK_DLL", False)
            )
            if validity_required and not det_ok:
                raise RuntimeError(
                    "TQS validity DLLs are required but unavailable."
                )
            critical_errors = self._error_reader.get_critical_errors(
                strict=validity_required
            )
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

def main(argv=None):
    """Run collection and training with explicit, resumable modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect TQS data and checkpoint it without starting training.",
    )
    mode.add_argument(
        "--train-from-checkpoint",
        type=Path,
        help="Train only, using checkpoint.json from an existing run.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help="Existing experiment directory to resume in collection-only mode.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Override the target number of valid samples for this run.",
    )
    parser.add_argument(
        "--checkpoint-minutes",
        type=int,
        help="Override the checkpoint interval for this collection run.",
    )
    args = parser.parse_args(argv)

    if args.resume_run and not args.collect_only:
        parser.error("--resume-run requires --collect-only.")
    if args.num_samples is not None:
        if args.num_samples <= 0:
            parser.error("--num-samples must be positive.")
        RunConfig.NUM_SAMPLES = args.num_samples
    if args.checkpoint_minutes is not None:
        if args.checkpoint_minutes <= 0:
            parser.error("--checkpoint-minutes must be positive.")
        RunConfig.CHECKPOINT_INTERVAL_MIN = args.checkpoint_minutes

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    # 1. Inicializa o gerenciador de experimentos.
    #    Ele usarÃ¡ o diretÃ³rio definido em config/paths.py.
    #    VocÃª pode dar um nome descritivo para a execuÃ§Ã£o.
    if args.train_from_checkpoint:
        checkpoint_path = args.train_from_checkpoint.resolve()
        if checkpoint_path.name != 'checkpoint.json':
            parser.error("--train-from-checkpoint must point to checkpoint.json.")
        exp_manager = ExperimentManager.from_existing(checkpoint_path.parent)
    elif args.resume_run:
        exp_manager = ExperimentManager.from_existing(args.resume_run)
    else:
        mode_name = "Coleta" if args.collect_only else "Treino"
        exp_manager = ExperimentManager(
            base_dir=paths.EXPERIMENTS_DIR,
            run_name=f"{mode_name}_com_{RunConfig.NUM_SAMPLES}_amostras",
        )

    # (Opcional, mas recomendado) Configurar o logging para salvar no diretÃ³rio do experimento
    # setup_logging(log_dir=exp_manager.run_dir)

    collection_lock = None
    if args.collect_only:
        collection_lock = (
            paths.OUTPUTS_DIR
            / f".{ParallelConfig.BASE_NAME}_collection.lock"
        )
        _acquire_collection_lock(collection_lock, exp_manager.run_dir)

    try:
        # 2. Instancia o otimizador, passando o gerenciador de experimento
        optimizer = BuildingOptimizer(exp_manager)
        
        # 3. Executa o fluxo de otimizaÃ§Ã£o
        optimizer.run_optimization(
            collect_only=args.collect_only,
            train_from_checkpoint=bool(args.train_from_checkpoint),
        )
        
        logging.info(f"ExecuÃ§Ã£o {exp_manager.run_dir.name} finalizada com sucesso.")

    except Exception as e:
        logging.error(f"ExecuÃ§Ã£o {exp_manager.run_dir.name} falhou.", exc_info=True)

        raise

    finally:
        _release_collection_lock(collection_lock)
        # This block executes whether an error occurred or not
        print("\n===========================================")
        print("   Building Optimization Script Finished   ")
        print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("===========================================")


if __name__ == '__main__':
    # This ensures the main function runs only when the script is executed directly
    main()
