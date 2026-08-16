import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from visualization.thesis_style import (
    SQUARE,
    SUBTITLE_SIZE,
    apply_thesis_style,
    save_thesis_figure,
)

class SegmentPlotter:
    def __init__(self, output_dir: Path):
        """Initialize plotter with output directory"""
        self.colors = {1: 'black', 0: 'grey'}
        self.line_width = 2
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        apply_thesis_style()
        
        print(f"SegmentPlotter configurado para salvar imagens em: {self.output_dir.resolve()}")
        
    def plot_segments(self, segments: List[Dict], iteration: int, steel: float = None, concrete: float = None, show: bool = False) -> None:
        """Plot segments configuration with material quantities"""
        fig, ax = plt.subplots(figsize=SQUARE)
        
        # Plot segments
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            binary = segment["binary"]
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=self.colors[binary],
                    linewidth=self.line_width)
        
        # Keep the main title on one line; quantities use the subtitle tier.
        title = f'Column Layout — Iteration {iteration}'
        subtitle = None
        if steel is not None and concrete is not None:
            subtitle = (
                f'Reinforcement steel weight: {steel:.2f} kgf · '
                f'Concrete volume: {concrete:.2f} m³'
            )
        fig.suptitle(title)
        if subtitle:
            fig.text(
                0.5, 0.91, subtitle, ha='center', va='top',
                fontsize=SUBTITLE_SIZE, fontweight='normal',
            )
            fig.subplots_adjust(top=0.82)
        else:
            fig.subplots_adjust(top=0.88)
        
        ax.set_xlabel('Global X coordinate')
        ax.set_ylabel('Global Y coordinate')
        ax.set_aspect('equal', adjustable='box')
        
        filename = self.output_dir / f'configuration_{iteration}.png'
        save_thesis_figure(fig, filename)
        #if show:
        #    plt.show()
        plt.close()
