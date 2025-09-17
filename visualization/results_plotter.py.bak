import matplotlib
from sklearn.metrics import r2_score
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

class ResultsPlotter:
    def __init__(self):
        """Initialize plotter with output directory"""
        # Get project root directory and create plots folder
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "results" / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Comparison plots will be saved to: {self.output_dir}")
        
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
        """Plot distribution of steel and concrete values"""
        steel_values = [out[0] for out in outputs]
        concrete_values = [out[1] for out in outputs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Steel distribution
        ax1.hist(steel_values, bins=30)
        ax1.set_title('Steel Distribution')
        ax1.set_xlabel('Steel (kgf)')
        ax1.set_ylabel('Frequency')
        
        # Concrete distribution
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
        print(f"  Min: {min(steel_values):.2f}")
        print(f"  Max: {max(steel_values):.2f}")
        print(f"  Range: {max(steel_values) - min(steel_values):.2f}")
        print(f"  Std Dev: {np.std(steel_values):.2f}")
        
        print(f"\nConcrete (m³):")
        print(f"  Min: {min(concrete_values):.2f}")
        print(f"  Max: {max(concrete_values):.2f}")
        print(f"  Range: {max(concrete_values) - min(concrete_values):.2f}")
        print(f"  Std Dev: {np.std(concrete_values):.2f}")