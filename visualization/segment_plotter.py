import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import os
from tqs_interface.geometry import Segment

class SegmentPlotter:
    def __init__(self):
        """Initialize plotter with output directory"""
        self.colors = {1: 'black', 0: 'grey'}
        self.line_width = 2
        
        # Get project root directory and create images folder
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Images will be saved to: {self.output_dir}")  # Debug print
        
    def plot_segments(self, segments: List[Dict], iteration: int, steel: float = None, concrete: float = None, show: bool = False) -> None:
        """Plot segments configuration with material quantities"""
        plt.figure(figsize=(10, 10))
        
        # Plot segments
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            binary = segment["binary"]
            plt.plot([start[0], end[0]], [start[1], end[1]], 
                    color=self.colors[binary],
                    linewidth=self.line_width)
        
        # Add title with material quantities if provided
        title = f'Configuration - Iteration {iteration}'
        if steel is not None and concrete is not None:
            title += f'\nSteel: {steel:.2f} kgf, Concrete: {concrete:.2f} m³'
        plt.title(title)
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        
        filename = self.output_dir / f'configuration_{iteration}.png'
        plt.savefig(str(filename))
        #if show:
        #    plt.show()
        plt.close()