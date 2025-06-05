# main.py
"""
Main execution script for the Building Structure Optimization process.

This script orchestrates the workflow involving:
- Reading initial structural configurations.
- Generating variations of configurations.
- Analyzing configurations using TQS or a faster geometric estimation.
- Collecting training data (features and corresponding material quantities).
- Training a Neural Network model to predict material quantities.
- Evaluating the trained model.
- Saving results and configurations.

Author: Adriana
Date: March 2025
"""
# Standard library imports
import copy
import time # Added for potential delays and timing
from typing import List, Tuple, Optional, Dict
# import numpy as np # Uncomment if needed for advanced splitting or direct normalization here

# Third-party imports
from TQS import TQSUtil # TQS Utility functions

# Project-specific imports
# Configuration - Make sure these files exist and are configured
from config.settings import BuildingConfig # General settings, NN config, analysis mode flag
from config.constants import CSV_FINAL_PATH # Path for saving final vectors CSV

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
from visualization.segment_plotter import SegmentPlotter # Plots individual configurations
from visualization.results_plotter import ResultsPlotter # Plots NN performance graphs

# Utilities - Helper functions
# Ensure 'utils' directory exists and contains these files
from utils.geometric_calculator import get_geometric_concrete_volume # Fast geometric volume estimate
from utils.file_handler import save_final_vectors_to_csv # Saves generated vectors to CSV

# --- Constants --- (Consider moving to config if used elsewhere)
DEFAULT_TRAIN_SPLIT_RATIO = 0.8
# Safety factor to prevent excessively long data collection loops if analysis fails often
MAX_ITERATION_FACTOR = 2 # Will attempt up to NUM_SAMPLES * 2 iterations

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

    def __init__(self):
        """Initializes all components required for the optimization."""
        print("--- Initializing Building Optimizer ---")

        # --- Configuration Flags ---
        # Read from BuildingConfig, providing defaults if attributes are missing
        self.use_vector_input = getattr(BuildingConfig, 'USE_VECTOR_INPUT', False)
        self.use_geometric_estimate = getattr(BuildingConfig, 'USE_GEOMETRIC_VOLUME_ESTIMATE', False)
        self.analysis_mode = "Geometric Estimate" if self.use_geometric_estimate else "TQS Analysis"
        self.num_target_samples = getattr(BuildingConfig, 'NUM_SAMPLES', 100)
        self.train_split_ratio = getattr(BuildingConfig, 'TRAIN_SPLIT_RATIO', DEFAULT_TRAIN_SPLIT_RATIO)

        # Print configuration summary
        print(f"Input Format:         {'Vector Lengths' if self.use_vector_input else 'Binary Grid'}")
        print(f"Analysis Mode:        {self.analysis_mode}")
        print(f"Target Valid Samples: {self.num_target_samples}")
        print(f"Train/Test Split:     {self.train_split_ratio*100:.0f}% / {(1-self.train_split_ratio)*100:.0f}%")

        # --- Component Initialization ---
        self.tqs_manager = TQSModelManager(BuildingConfig.NAME)
        # Consider passing NeuralNetConfig here if nn_manager needs it
        self.nn_manager = NeuralNetworkManager()
        self.binary_processor = BinaryProcessor() # Used if not vector_input
        self.length_processor = LengthProcessor() # Used if vector_input
        self.segment_plotter = SegmentPlotter() # Used for plotting configurations
        self.results_plotter = ResultsPlotter() # Used for plotting NN results

        # --- State Variables ---
        self.current_iteration = 0 # Counts attempts to generate configurations
        self.generated_valid_configurations = [] # Stores List[Dict] for each valid config
        self.normalization_params = None # To store NN normalization stats {'X_mean':..., 'X_std':..., 'y_mean':..., 'y_std':...}

        print("--- Optimizer Initialized Successfully ---")

    # -------------------------------------------------------------------------
    # Main Workflow Method
    # -------------------------------------------------------------------------

    def run_optimization(self):
        """Executes the main optimization workflow."""
        print(f"\n--- Starting Optimization Workflow ---")
        start_time_total = time.time()
        try:
            # 1. Load initial data
            initial_segments = self._load_initial_segments()

            # 2. Generate training/testing samples by analyzing variations
            feature_vectors, output_values = self._collect_training_data(initial_segments)

            # 3. Validate generated data before proceeding
            self._validate_collected_data(feature_vectors, output_values)
            num_valid_samples = len(feature_vectors) # Get actual number generated

            # 4. Split data into training and testing sets
            train_features, test_features, train_outputs, test_outputs = self.train_test_split(
            feature_vectors, output_values, train_size=self.train_split_ratio, shuffle=True, random_state=42
            )

            # 5. Train the Neural Network model
            self._train_model(train_features, train_outputs)

            # 6. Make predictions on the test set
            predictions = self._predict_on_test_set(test_features)

            # 7. Evaluate predictions and Report Results
            self._evaluate_and_report(predictions, test_outputs, test_features)

            # 8. Plot Results (Comparison and Distribution)
            self._plot_results(predictions, test_outputs, output_values)

            # 9. Save Final Generated Configurations (if applicable)
            self._save_results()

            end_time_total = time.time()
            print(f"\n--- Optimization Workflow Finished Successfully ({end_time_total - start_time_total:.2f}s) ---")

        except Exception as e:
            end_time_total = time.time()
            print(f"\n--- ERROR DURING OPTIMIZATION WORKFLOW ({end_time_total - start_time_total:.2f}s) ---")
            error_message = f"Optimization failed: {str(e)}"
            TQSUtil.writef(error_message) # Log to TQS console too
            print(error_message)
            # Optional: Log detailed traceback for debugging
            import traceback
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
        print(f"Output vector size:  {num_outputs} ({'Concrete Only' if self.use_geometric_estimate else 'Steel & Concrete'})")


    def _split_data(self, features: list, outputs: list) -> Tuple[list, list, list, list]:
        """Splits the collected data into training and testing sets."""
        print("\n--- Splitting Data into Train/Test Sets ---")
        num_samples = len(features)
        num_training_samples = int(self.train_split_ratio * num_samples)
        num_test_samples = num_samples - num_training_samples

        # Ensure at least one sample in each set if possible
        if num_training_samples == 0 or num_test_samples == 0:
            raise ValueError(
                f"Cannot split {num_samples} samples with ratio {self.train_split_ratio}. "
                "Need more samples or adjust split ratio for valid training and testing sets."
            )

        # Simple chronological split - for random split, use:
        # from sklearn.model_selection import train_test_split
        # train_features, test_features, train_outputs, test_outputs = train_test_split(
        #     features, outputs, train_size=self.train_split_ratio, shuffle=True
        # )
        train_features = features[:num_training_samples]
        train_outputs = outputs[:num_training_samples]
        test_features = features[num_training_samples:]
        test_outputs = outputs[num_training_samples:]

        print(f"Total samples:    {num_samples}")
        print(f"Training samples: {len(train_features)}")
        print(f"Test samples:     {len(test_features)}")
        return train_features, train_outputs, test_features, test_outputs


    def _train_model(self, train_features: list, train_outputs: list):
        """Trains the Neural Network model and stores normalization parameters."""
        print("\n--- Training Neural Network ---")
        # The nn_manager.train method should handle normalization internally
        # and return the parameters used.
        try:
            # Pass training hyperparams from config if needed, e.g.,
            # num_epochs = getattr(BuildingConfig, 'NUM_EPOCHS', 100) ...
            self.normalization_params = self.nn_manager.train(
                train_features, train_outputs #, num_epochs=..., batch_size=...
            )
            if self.normalization_params is None:
                # This case should ideally be handled by an exception in nn_manager.train if it fails
                print("Warning: NN training finished but did not return normalization parameters.")
            else:
                print("Normalization parameters stored.")
            print("Training phase complete.")
        except Exception as e:
             print(f"CRITICAL ERROR during NN training: {e}")
             # Decide how to handle training failure - maybe stop the whole process?
             raise RuntimeError(f"Neural network training failed: {e}") from e


    def _predict_on_test_set(self, test_features: list) -> list:
        """Makes predictions on the test set using the trained model."""
        print("\n--- Predicting on Test Set ---")
        # The nn_manager.predict method should handle normalization internally
        # using the stored self.normalization_params and return denormalized results.
        if not test_features:
             print("Warning: No test features to predict on.")
             return []
        try:
            predictions = self.nn_manager.predict(test_features)
            print(f"Generated {len(predictions)} predictions for the test set.")
            if len(predictions) != len(test_features):
                 print(f"Warning: Number of predictions ({len(predictions)}) does not match number of test samples ({len(test_features)}).")
            return predictions
        except Exception as e:
            print(f"CRITICAL ERROR during NN prediction: {e}")
            raise RuntimeError(f"Neural network prediction failed: {e}") from e


    def _save_results(self):
        """Saves generated configurations or other desired results."""
        # Save Final Generated Configurations if applicable
        if self.use_vector_input and self.generated_valid_configurations:
            print(f"\n--- Saving Generated Valid Configurations to CSV ---")
            try:
                # Assumes save_final_vectors_to_csv uses CSV_FINAL_PATH from constants
                # and is imported from utils.file_handler
                save_final_vectors_to_csv(self.generated_valid_configurations)
                print(f"Successfully saved {len(self.generated_valid_configurations)} configurations to {CSV_FINAL_PATH}.")
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
        # Calculate max attempts to prevent infinite loops if analysis consistently fails
        max_iterations = self.num_target_samples * MAX_ITERATION_FACTOR

        # --- Analyze Initial Configuration ---
        print(f"\nAnalyzing Initial Configuration (Attempt 0)...")
        analysis_start_time = time.time()
        steel, concrete = self._get_analysis_results(initial_segments)
        analysis_end_time = time.time()
        print(f"Initial analysis took {analysis_end_time - analysis_start_time:.2f}s")

        if concrete is not None: # Check if analysis (TQS or geometric) was successful
            print(f"Initial Results -> Steel: {steel if steel is not None else 'N/A'} kgf, Concrete: {concrete:.4f} m³")
            feature_vector = self._extract_feature_vector(initial_segments)
            feature_vectors.append(feature_vector)
            # Use 0.0 for steel if geometric mode, otherwise use TQS result
            output_values.append([steel if steel is not None else 0.0, concrete])
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

        while processed_valid_configs_count < self.num_target_samples and self.current_iteration < max_iterations:
            self.current_iteration += 1 # Increment attempt counter
            print(f"\n--- Iteration Attempt {self.current_iteration}/{max_iterations} (Valid Samples Collected: {processed_valid_configs_count}/{self.num_target_samples}) ---")

            # 1. Generate a new variation
            print("Generating segment variation...")
            try:
                 new_segments = self._generate_segment_variation(base_segments_for_variation)
            except Exception as gen_e:
                 print(f"Error during segment variation generation: {gen_e}. Skipping iteration.")
                 continue # Skip to next iteration

            # Optional: Plot the generated configuration
            # print("Plotting current segment configuration...")
            # try:
            #      self.segment_plotter.plot_segments(new_segments, self.current_iteration)
            # except Exception as plot_e:
            #      print(f"Warning: Failed to plot segment configuration: {plot_e}")

            # 2. Get analysis results (TQS or Geometric)
            analysis_start_time = time.time()
            steel, concrete = self._get_analysis_results(new_segments)
            analysis_end_time = time.time()
            print(f"Analysis took {analysis_end_time - analysis_start_time:.2f}s")

            # 3. Process results
            if concrete is not None: # Check if analysis was successful
                processed_valid_configs_count += 1 # Increment valid sample count
                print(f"Config {self.current_iteration} Results (Valid Sample {processed_valid_configs_count}) -> Steel: {steel if steel is not None else 'N/A'} kgf, Concrete: {concrete:.4f} m³")
                feature_vector = self._extract_feature_vector(new_segments)
                # Ensure feature vector extraction was successful
                if feature_vector:
                     feature_vectors.append(feature_vector)
                     output_values.append([steel if steel is not None else 0.0, concrete])
                     if self.use_vector_input:
                         self.generated_valid_configurations.append(copy.deepcopy(new_segments))
                     # Optional: Update base segments if varying from last success
                     # base_segments_for_variation = new_segments
                else:
                     print(f"Warning: Failed to extract feature vector for valid config {self.current_iteration}. Skipping sample.")
                     processed_valid_configs_count -= 1 # Decrement count as it's not fully usable
            else:
                print(f"Config {self.current_iteration} failed analysis. Skipping sample.")

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


    def _generate_segment_variation(self, base_segments: List[dict]) -> List[dict]:
        """Generates a new variation of segments based on the input mode."""
        if self.use_vector_input:
            # length_processor.generate_variation should handle variation logic
            # It typically takes the segments to vary as input
            return self.length_processor.generate_variation(base_segments)
        else:
            # Ensure generate_new_binary_vector works as expected
            return generate_new_binary_vector(base_segments)


    def _extract_feature_vector(self, segments: List[dict]) -> Optional[List[float]]:
        """
        Extracts the feature vector (input for NN) from a list of segments.
        Returns None if extraction fails or segments are invalid.
        """
        if not segments:
             print("Warning: Cannot extract features from empty segment list.")
             return None
        try:
            if self.use_vector_input:
                # Extract length, handle potential missing key with default 0.0
                features = [seg.get("length", 0.0) for seg in segments]
            else:
                # Extract binary flag, handle potential missing key with default 0
                features = [float(seg.get("binary", 0)) for seg in segments] # Ensure float type
            # Basic validation
            if not features:
                 print("Warning: Extracted feature vector is empty.")
                 return None
            return features
        except (TypeError, KeyError, AttributeError) as e:
             print(f"Error extracting feature vector: {e}. Segment data: {segments}")
             return None


    def _get_analysis_results(self, segments: List[dict]) -> Tuple[Optional[float], Optional[float]]:
        """
        Performs structural analysis based on the configured mode (Geometric or TQS).

        Args:
            segments: The list of segment dictionaries for the configuration.

        Returns:
            A tuple (steel_kgf, concrete_m3). Steel is None in geometric mode.
            Returns (None, None) if analysis fails.
        """
        print(f"Performing analysis using: {self.analysis_mode}")
        if self.use_geometric_estimate:
            # --- Geometric Estimation Mode ---
            try:
                print("   Processing segments for geometric calculation...")
                # Process segments to get beam definitions needed for volume calc
                # Assume processors return tuple: (column_geometry, beam_definitions)
                if self.use_vector_input:
                    _, beam_definitions = self.length_processor.process_segments(segments)
                else:
                    _, beam_definitions = self.binary_processor.process_segments(segments)

                # Handle case where beam definitions might not be generated
                if beam_definitions is None:
                     print("   Warning: Could not determine beam definitions. Calculating volume based on pillars only.")
                     beam_definitions = [] # Use empty list for calculator

                # Calculate geometric volume (steel is None)
                concrete_volume = get_geometric_concrete_volume(segments, beam_definitions)
                print(f"   Geometric Concrete Volume Estimated: {concrete_volume:.4f} m³")
                return None, concrete_volume # Steel is None

            except Exception as e:
                print(f"   Error during geometric calculation: {e}")
                TQSUtil.writef(f"Error during geometric calculation: {e}")
                return None, None # Indicate failure
        else:
            # --- TQS Analysis Mode ---
            return self._run_tqs_model(segments)


    def _run_tqs_model(self, segments: List[dict]) -> Tuple[Optional[float], Optional[float]]:
        """
        Executes the full TQS structural analysis for a given segment configuration.
        Handles segment processing, model creation, execution, and result extraction.

        Args:
            segments: List of segment dictionaries defining the structure.

        Returns:
            Tuple (steel_kgf, concrete_m3) or (None, None) if analysis fails at any step.
        """
        try:
            print("   Starting TQS Analysis Pipeline...")
            start_time_tqs = time.time()

            # 1. Process segments into TQS-compatible geometry
            print("      Step 1: Processing segments for TQS geometry...")
            if self.use_vector_input:
                column_polygons, beam_definitions = self.length_processor.process_segments(segments)
            else:
                column_polygons, beam_definitions = self.binary_processor.process_segments(segments)

            # Validate processing results
            if not column_polygons:
                 print("      TQS Error: Segment processing yielded no column polygons.")
                 return None, None
            if beam_definitions is None:
                 print("      TQS Warning: Segment processing yielded no beam definitions. Proceeding with columns only.")
                 beam_definitions = [] # Ensure list format

            print(f"      Processed into {len(column_polygons)} column groups and {len(beam_definitions)} beam definitions.")

            # 2. Create TQS building model
            print("      Step 2: Creating TQS building model...")
            # Ensure TQS Manager logs details on failure
            model_created = self.tqs_manager.create_building_model(column_polygons, beam_definitions)
            if not model_created:
                print("      TQS Error: Failed to create building model via TQS Manager.")
                return None, None
            print("      TQS model created successfully.")

            # 3. Run TQS analysis executables
            print("      Step 3: Executing TQS global processing...")
            RunModel(BuildingConfig.NAME) # Assumes RunModel handles TQS execution flow
            print("      TQS global processing command issued.")

            # 4. Extract results (Add delay for file writing if needed)
            tqs_output_file = BuildingConfig.RESULTS_PATH
            print(f"      Step 4: Extracting results from {tqs_output_file}...")
            # Optional delay: If RunModel is async, TQS might need time to write the file.
            timeout = 10  # seconds
            start = time.time()
            while not tqs_output_file.exists():
                if time.time() - start > timeout:
                    print("Timeout waiting for TQS output file.")
                    return None, None
                time.sleep(0.2)

            steel_value_str, concrete_value_str = extract_material_summary(tqs_output_file)

            if steel_value_str is None or concrete_value_str is None:
                print(f"      TQS Error: Could not extract 'Totais' row or values from '{tqs_output_file}'. Check file content and format.")
                return None, None

            # 5. Convert results to float
            print("      Step 5: Parsing results...")
            try:
                # Replace comma decimal separator if used in TQS output
                steel_kgf = float(steel_value_str.replace(",", "."))
                concrete_m3 = float(concrete_value_str.replace(",", "."))
            except ValueError as ve:
                 print(f"      TQS Error: Could not convert extracted results ('{steel_value_str}', '{concrete_value_str}') to numbers: {ve}")
                 return None, None

            end_time_tqs = time.time()
            print(f"   TQS Analysis successful ({end_time_tqs - start_time_tqs:.2f}s). Steel: {steel_kgf:.2f} kgf, Concrete: {concrete_m3:.3f} m³")
            return steel_kgf, concrete_m3

        except Exception as e:
            # Catch any unexpected errors during the TQS pipeline
            error_time = time.time()
            print(f"   TQS Error: An unexpected exception occurred during TQS pipeline at {error_time:.0f}: {e}")
            TQSUtil.writef(f"Error during TQS model run/extraction: {str(e)}")
            import traceback # Log detailed traceback for debugging
            print(traceback.format_exc())
            return None, None # Indicate failure

    # -------------------------------------------------------------------------
    # Evaluation and Plotting Helper Methods (Placeholder implementations)
    # -------------------------------------------------------------------------

    def _evaluate_and_report(self, predictions: List[List[float]], actual_values: List[List[float]], test_features: List[List[float]]):
        """
        Calculates and prints evaluation metrics for the test set predictions.
        (Implementation adapted from previous version, ensures correct handling)
        """
        print("\n--- Test Set Evaluation ---")
        # TODO: Ensure predictions and actual_values are denormalized here if normalization was used.
        #       The nn_manager.predict should ideally return denormalized values.

        if len(predictions) == 0 or len(actual_values) == 0:
            print("Evaluation skipped: No predictions or actual values available.")
            return
    
        if len(predictions) != len(actual_values):
             print(f"Evaluation Warning: Mismatch in number of predictions ({len(predictions)}) and actual values ({len(actual_values)}). Evaluating based on shorter list.")
             min_len = min(len(predictions), len(actual_values))
             predictions = predictions[:min_len]
             actual_values = actual_values[:min_len]
             # test_features = test_features[:min_len] # Also truncate features if needed

        num_test_samples = len(actual_values)
        predicts_steel = not self.use_geometric_estimate

        total_steel_error_perc = 0.0
        total_concrete_error_perc = 0.0
        valid_steel_samples = 0
        valid_concrete_samples = 0

        print("Comparing predictions against actual values for test samples:")
        print("-" * 60)
        for i in range(num_test_samples):
            pred = predictions[i]
            actual = actual_values[i]

            if len(pred) < 2 or len(actual) < 2:
                print(f"Skipping sample {i+1}: Invalid prediction/actual data structure.")
                continue

            concrete_pred = pred[1]
            concrete_actual = actual[1]

            print(f"Test Sample {i+1}/{num_test_samples}:")

            # Concrete Evaluation
            print(f"  Concrete -> Predicted: {concrete_pred:>8.2f} m³ | Actual: {concrete_actual:>8.2f} m³")
            if abs(concrete_actual) > 1e-6:
                concrete_err = abs(concrete_pred - concrete_actual) / concrete_actual * 100
                print(f"                 Error: {concrete_err:>8.2f}%")
                total_concrete_error_perc += concrete_err
                valid_concrete_samples += 1
            else:
                absolute_diff = abs(concrete_pred - concrete_actual)
                print(f"                 Actual is ~0. Absolute Difference: {absolute_diff:.4f} m³")

            # Steel Evaluation (if applicable)
            if predicts_steel:
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
        (Implementation adapted from previous version)
        """
        print("\n--- Generating Result Plots ---")
        # TODO: Ensure predictions and actual_values are denormalized before plotting.
        try:
            # Plot comparison (Predicted vs Actual for Test Set)
            if not self.use_geometric_estimate:
                 print("   Plotting Steel comparison...")
                 self.results_plotter.plot_comparison(predictions, actual_values, 'steel')
            else:
                 print("   Skipping Steel comparison plot (Geometric mode).")

            print("   Plotting Concrete comparison...")
            self.results_plotter.plot_comparison(predictions, actual_values, 'concrete')

            # Plot distribution of all collected output values
            if all_output_values:
                 print("   Plotting overall material distribution...")
                 self.results_plotter.plot_distribution(all_output_values)
            else:
                 print("   Skipping distribution plot (no overall output values available).")

            print("Plots generated successfully (check results/plots directory).")
        except Exception as e:
            print(f"Warning: Plot generation failed: {str(e)}")


# =============================================================================
# Script Execution Entry Point
# =============================================================================

def main():
    """Main function to run the building optimization process."""
    print("===========================================")
    print("   Starting Building Optimization Script   ")
    print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("===========================================")

    try:
        # --- Configuration should be set in config/settings.py ---
        # Example: Ensure these lines are in your config/settings.py:
        # class BuildingConfig:
        #     NAME = "OptimizedBuilding"
        #     USE_VECTOR_INPUT = True
        #     USE_GEOMETRIC_VOLUME_ESTIMATE = False # Set True for fast mode
        #     NUM_SAMPLES = 100 # Target number of *valid* samples
        #     TRAIN_SPLIT_RATIO = 0.8
        #     RESULTS_PATH = Path(r"C:\TQS\OptimizedBuilding\ESPACIAL\RESDES.HTM") # Adjust TQS path
        #     # Add other necessary paths and parameters from your original config
        #
        # class NeuralNetConfig: # If needed by nn_manager
        #     INPUT_SIZE = 24 # Or determine dynamically
        #     HIDDEN_SIZE = 128
        #     OUTPUT_SIZE = 2
        # ----------------------------------------------------------

        optimizer = BuildingOptimizer()
        optimizer.run_optimization()

    except Exception as main_error:
         print("\n--- A CRITICAL ERROR OCCURRED IN MAIN EXECUTION ---")
         print(f"Error Type: {type(main_error).__name__}")
         print(f"Error Details: {main_error}")
         # Log detailed traceback
         import traceback
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