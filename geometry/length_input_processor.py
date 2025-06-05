from typing import List, Tuple, Dict, Set
import pandas as pd
from shapely.geometry import LineString, Polygon
from config.constants import BEAM_THICKNESS, VECTOR_CSV_PATH
from config.vector_config import VectorConfig
import random
import copy
import numpy as np
from .geometry_utils import GeometryProcessor

class LengthProcessor:
    def __init__(self):
        self.wall_segments = VectorConfig.WALL_SEGMENTS
        
    def read_length_from_csv(self) -> List[dict]:
        """Reads column vectors from CSV and converts to segments"""
        df = pd.read_csv(VECTOR_CSV_PATH, delimiter=';')
        segments = []
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
            rectangles = GeometryProcessor.create_rectangles_from_segments(group, BEAM_THICKNESS / 2.0) # Passando half_thickness
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
            BEAM_THICKNESS    # Passa a espessura TOTAL do elemento
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
                if column.distance(wall_line) < BEAM_THICKNESS:
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

    def generate_variation(self, segments: List[dict]) -> List[dict]:
        new_segments = copy.deepcopy(segments)
        made_changes = False
        max_attempts = 10
        attempt = 0
    
        while not made_changes and attempt < max_attempts:
            for segment in new_segments:
                original_length = segment["length"]
                max_length = segment.get("maxlength")
                
                # Só varia se existe maxlength e é maior que o original
                if max_length is not None and max_length > original_length:
                    if random.random() < 0.4:  # 40% chance de variar
                        variation = random.uniform(0, max_length - original_length)
                        new_length = original_length + variation
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
                new_length = original_length + random.uniform(0, max_length - original_length)
                self._update_segment_length(segment, new_length)
    
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