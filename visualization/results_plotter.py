import matplotlib
from sklearn.metrics import r2_score
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

class ResultsPlotter:
    def __init__(self, output_dir: Path):
        """Initialize plotter with output directory"""
        self.output_dir = output_dir
        # Garante que o diretório exista (o ExperimentManager já faz isso, mas é uma boa prática)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"ResultsPlotter configurado para salvar gráficos em: {self.output_dir.resolve()}")
        
    def plot_comparison(self, predictions, actual_values, material_type='steel'):
        """Plot predicted vs actual values comparison."""
        try:
            plt.close('all')  # Close any existing figures
            
            # Create new figure
            plt.figure(figsize=(10, 6))
            
            # Extract values for the specific material
            idx = 0 if material_type == 'steel' else 1
            pred_values = [p[idx] for p in predictions]
            true_values = [a[idx] for a in actual_values]

            if not true_values or not pred_values:
                print(f"Warning: No data to plot for {material_type}.")
                return
            
            # Plot predicted vs actual
            plt.scatter(true_values, pred_values, c='blue', alpha=0.5, label='Test Points')
            print(f"Plotting {material_type}:")
            print(f"  Actual - Min: {min(true_values):.2f}, Max: {max(true_values):.2f}, Count: {len(true_values)}")
            print(f"  Predicted - Min: {min(pred_values):.2f}, Max: {max(pred_values):.2f}, Count: {len(pred_values)}")
            # Add perfect prediction line
            min_val = min(min(true_values), min(pred_values))
            max_val = max(max(true_values), max(pred_values))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            # Calculate R² score (Coefficient of Determination)
            r2 = r2_score(true_values, pred_values)
            
            # Add labels and title
            plt.xlabel(f'TQS Calculated {material_type.title()} {"(kgf)" if material_type=="steel" else "(m³)"}')
            plt.ylabel(f'DNN Predicted {material_type.title()} {"(kgf)" if material_type=="steel" else "(m³)"}')
            plt.title(f'{material_type.title()} Prediction Comparison\nR² = {r2:.3f}')
            
            plt.grid(True)
            plt.legend()
            
            # Save and close
            filepath = self.output_dir / f'{material_type}_comparison.png'
            plt.savefig(str(filepath))
            plt.close()
        except Exception as e:
            print(f"Error plotting comparison: {str(e)}")
        
    def plot_distribution(self, outputs: List[List[float]]):
        """Plot distribution of available outputs (steel-only or steel+concrete)."""
        if not outputs:
            print("Warning: No outputs provided for distribution plot.")
            return

        steel_values = [out[0] for out in outputs if len(out) >= 1]
        has_concrete = all(len(out) >= 2 for out in outputs)
        concrete_values = [out[1] for out in outputs] if has_concrete else []

        if has_concrete:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

        # Steel distribution
        ax1.hist(steel_values, bins=30)
        ax1.set_title('Steel Distribution')
        ax1.set_xlabel('Steel (kgf)')
        ax1.set_ylabel('Frequency')

        # Concrete distribution (if present)
        if has_concrete:
            ax2.hist(concrete_values, bins=30)
            ax2.set_title('Concrete Distribution')
            ax2.set_xlabel('Concrete (m³)')
            ax2.set_ylabel('Frequency')

        plt.tight_layout()
        filepath = self.output_dir / 'material_distribution.png'
        plt.savefig(filepath)
        plt.close()

        # Print statistics
        print("\nSample Statistics:")
        print(f"Steel (kgf):")
        if steel_values:
            print(f"  Min: {min(steel_values):.2f}")
            print(f"  Max: {max(steel_values):.2f}")
            print(f"  Range: {max(steel_values) - min(steel_values):.2f}")
            print(f"  Std Dev: {np.std(steel_values):.2f}")
        if has_concrete and concrete_values:
            print(f"\nConcrete (m³):")
            print(f"  Min: {min(concrete_values):.2f}")
            print(f"  Max: {max(concrete_values):.2f}")
            print(f"  Range: {max(concrete_values) - min(concrete_values):.2f}")
            print(f"  Std Dev: {np.std(concrete_values):.2f}")

    def plot_residuals(self, predictions, actual_values, material_type='steel'):
        """Plot residuals (prediction errors) to diagnose model bias."""
        try:
            plt.close('all')
            plt.figure(figsize=(10, 6))

            idx = 0 if material_type == 'steel' else 1
            pred_values = np.array([p[idx] for p in predictions])
            true_values = np.array([a[idx] for a in actual_values])
            
            residuals = true_values - pred_values

            plt.scatter(true_values, residuals, c='green', alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--') # Zero error line
            
            plt.xlabel(f'Actual {material_type.title()} {"(kgf)" if material_type=="steel" else "(m³)"}')
            plt.ylabel('Residuals (Actual - Predicted)')
            plt.title(f'Residual Plot for {material_type.title()}')
            plt.grid(True)

            filepath = self.output_dir / f'{material_type}_residuals.png'
            plt.savefig(str(filepath))
            plt.close()
        except Exception as e:
            print(f"Error plotting residuals: {str(e)}")
