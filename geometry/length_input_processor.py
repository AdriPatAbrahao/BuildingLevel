from typing import List, Tuple, Dict, Set
import pandas as pd
from shapely.geometry import LineString, Polygon
from config.constants import DEFAULT_BEAM_WIDTH_CM
from config.paths import SEED_VECTOR_CSV
from config.vector_config import VectorConfig
import random
import copy
import numpy as np
from .geometry_utils import GeometryProcessor
from pathlib import Path

class LengthProcessor:
    def __init__(self, csv_filepath: str = None):
        self.wall_segments = VectorConfig.WALL_SEGMENTS
        if csv_filepath:
            self.csv_path = Path(csv_filepath)
        else:
            self.csv_path = SEED_VECTOR_CSV
        print(f"LengthProcessor inicializado para ler o arquivo: '{self.csv_path}'")
        
    def read_length_from_csv(self) -> List[dict]:
        """Reads column vectors from CSV and converts to segments"""
        # Força o pandas a usar ponto como separador decimal
        #if not self.csv_path.exists():
        #    print(f"ERRO: Arquivo CSV não encontrado em '{self.csv_path}'")
        #    return [] # Retorna lista vazia se o arquivo não existe
        
        segments = []
        try:
            df = pd.read_csv(self.csv_path, delimiter=';', decimal=',')

            for _, row in df.iterrows():
                x = float(row["x"])
                y = float(row["y"])
                dx = float(row["dx"])
                dy = float(row["dy"])
                length = float(row["length"])
                maxlength = float(row["maxlength"]) if "maxlength" in row and not pd.isna(row["maxlength"]) else None

                segments.append({
                "start": (x, y),
                "end": (x + dx * length, y + dy * length),
                "length": length,
                "maxlength": maxlength,
                "binary": 1
            })
                
        except FileNotFoundError:
            print(f"Erro: Arquivo CSV não encontrado em '{self.csv_path}'")
            return [] # Retorna lista vazia em caso de erro de arquivo não encontrado
        except Exception as e:
            # Captura outros erros de parsing ou processamento
            print(f"Erro ao ler ou processar o CSV/buffer em LengthProcessor: {e}")
            return [] 
            
        return segments
    
    def process_segments(self, segments: List[dict] = None) -> Tuple[List[Polygon], List[dict]]:
        """Process segments into column groups and find beam locations"""
        # Use provided segments or read from file
        if segments is None:
            segments = self.read_vector_segments()
            
        # Group column segments
        column_groups = self._group_connected_segments(segments)
        
        # Create polygons for column groups
        column_polygons = []
        for group in column_groups:
            rectangles = GeometryProcessor.create_rectangles_from_segments(group, DEFAULT_BEAM_WIDTH_CM  / 2.0) # Passando half_thickness
            polygons = GeometryProcessor.convert_vertices_to_polygons(rectangles)
            united = GeometryProcessor.union_polygons(polygons)
            column_polygons.extend(united)
            
        # Find beam locations at walls without columns
        beam_groups = self._find_beam_locations(column_polygons)
        
        return column_polygons, beam_groups
        
    def _group_connected_segments(self, segments: List[dict]) -> List[List[dict]]:
        graph = self._build_column_connection_graph(segments)
        components_indices = GeometryProcessor.find_connected_components(graph) # Ou GeometryUtils.
        return [[segments[original_idx] for original_idx in component] for component in components_indices]
    
    def _build_column_connection_graph(self, all_segments: List[Dict]) -> Dict[int, Set[int]]:
        segments_to_process: List[Dict] = all_segments
        original_indices: List[int] = list(range(len(all_segments))) # Mapeamento direto de índice

        if not segments_to_process:
            return {}

        # Chama a função utilitária estática
        return GeometryProcessor.build_graph_from_polygon_intersections(
            segments_to_process,
            original_indices,
            DEFAULT_BEAM_WIDTH_CM    # Passa a espessura TOTAL do elemento
        )
    
    def _find_beam_locations(self, column_polygons: List[Polygon]) -> List[dict]:
        """
        Finds beam locations along wall segments. Creates one beam per wall segment,
        starting/ending at column faces instead of centerlines.
        """
        beam_groups = []
        
        for wall in self.wall_segments:
            wall_line = LineString([wall["start"], wall["end"]])
            column_intersections = []
            
            # Find all column intersections with this wall
            for column in column_polygons:
                if column.distance(wall_line) < DEFAULT_BEAM_WIDTH_CM:
                    # Get column boundaries instead of centroid
                    minx, miny, maxx, maxy = column.bounds
                    
                    # Project appropriate column face onto wall line
                    if abs(wall["end"][0] - wall["start"][0]) < 0.001:  # Vertical wall
                        column_intersections.append({
                            "y": column.centroid.coords[0][1],
                            "min_face": miny,
                            "max_face": maxy
                        })
                    else:  # Horizontal wall
                        column_intersections.append({
                            "x": column.centroid.coords[0][0],
                            "min_face": minx,
                            "max_face": maxx
                        })
            
            if column_intersections:
                # Sort intersections along wall direction
                if abs(wall["end"][0] - wall["start"][0]) < 0.001:  # Vertical wall
                    column_intersections.sort(key=lambda c: c["y"])
                    first_col = column_intersections[0]
                    last_col = column_intersections[-1]
                    
                    # Create beam from bottom face of first column to top face of last column
                    beam_groups.append({
                        "node_1": (wall["start"][0], first_col["min_face"]),
                        "node_2": (wall["start"][0], last_col["max_face"])
                    })
                else:  # Horizontal wall
                    column_intersections.sort(key=lambda c: c["x"])
                    first_col = column_intersections[0]
                    last_col = column_intersections[-1]
                    
                    # Create beam from left face of first column to right face of last column
                    beam_groups.append({
                        "node_1": (first_col["min_face"], wall["start"][1]),
                        "node_2": (last_col["max_face"], wall["start"][1])
                    })
        
        return beam_groups

    def generate_variation(self, segments: List[dict], variation_strategy: str = "random") -> List[dict]:
        new_segments = copy.deepcopy(segments)
        made_changes = False
        step = 5.0 # Variação em múltiplos de 5 cm

        if variation_strategy == "random":
            max_attempts = 10
            attempt = 0
            while not made_changes and attempt < max_attempts:
                for segment in new_segments:
                    original_length = segment["length"]
                    max_length = segment.get("maxlength")
                    
                    # Só varia se existe maxlength e é maior que o original
                    if max_length is not None and max_length > original_length:
                        if random.random() < 0.4:  # 40% chance de variar
                            variation = max_length - original_length
                            max_steps = int(variation / step)

                            if max_steps > 0:
                                num_steps = random.randint(1, max_steps)
                                new_length = original_length + (num_steps * step)
                            
                                self._update_segment_length(segment, new_length)
                                made_changes = True    
                attempt += 1
        
            if not made_changes:
                # Se nenhum segmento foi alterado, força pelo menos um
                valid_segments = [s for s in new_segments if s.get("maxlength") is not None and s["maxlength"] > s["length"]]
                if valid_segments:
                    segment = random.choice(valid_segments)
                    original_length = segment["length"]
                    max_length = segment["maxlength"]

                    max_variation = max_length - original_length
                    max_steps = int(max_variation / step)

                    if max_steps > 0:
                        num_steps = random.randint(1, max_steps)
                        new_length = original_length + (num_steps * step)
                    
                        self._update_segment_length(segment, new_length)
        elif variation_strategy == "guided_by_volume":
            # Implement a more intelligent variation strategy here
            # For now, it will just do a random variation
            print("Guided by volume strategy not yet implemented. Falling back to random.")
            return self.generate_variation(segments, variation_strategy="random")

        return new_segments

    def _update_segment_length(self, segment: dict, new_length: float):
        """Helper to update segment with new length while maintaining direction"""
        start_x, start_y = segment["start"]
        dx = segment["end"][0] - start_x
        dy = segment["end"][1] - start_y
        
        # Get direction vector
        length = np.sqrt(dx*dx + dy*dy)
        dx, dy = dx/length, dy/length
        
        # Update endpoint
        segment["length"] = new_length
        segment["end"] = (
            start_x + dx * new_length,
            start_y + dy * new_length
        )