from typing import List, Tuple, Dict, Set
import pandas as pd
from shapely.geometry import LineString, Polygon
from config.constants import DEFAULT_BEAM_WIDTH_CM, SPLIT_BEAM_COLUMN_THRESHOLD_CM
from config.paths import SEED_VECTOR_CSV
from config.vector_config import VectorConfig
import random
import copy
import numpy as np
from .geometry_utils import GeometryProcessor
from pathlib import Path

class LengthProcessor:
    """
    Read length vectors from CSV and derive structural geometry.

    Parameters
    ----------
    csv_filepath : str, optional
        Path to the CSV with columns `x,y,dx,dy,length,maxlength`.
        Defaults to `SEED_VECTOR_CSV` from configuration.
    """
    def __init__(self, csv_filepath: str = None):
        self.wall_segments = VectorConfig.WALL_SEGMENTS
        if csv_filepath:
            self.csv_path = Path(csv_filepath)
        else:
            self.csv_path = SEED_VECTOR_CSV
        print(f"LengthProcessor inicializado para ler o arquivo: '{self.csv_path}'")
        
    def read_length_from_csv(self) -> List[dict]:
        """
        Read segments from the configured CSV.

        Returns
        -------
        List[dict]
            Each segment dict contains `start`, `end`, `length`, `maxlength`, `binary`.

        Raises
        ------
        ValueError
            On parsing issues; returns empty list for missing files.
        """
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

                seg = {
                    "start": (x, y),
                    "end": (x + dx * length, y + dy * length),
                    "length": length,
                    "maxlength": maxlength,
                    "binary": 1
                }
                if "group_id" in df.columns:
                    gid = row["group_id"]
                    if pd.isna(gid):
                        seg["group_id"] = None
                    elif isinstance(gid, str) and gid.strip() == "":
                        seg["group_id"] = None
                    else:
                        seg["group_id"] = str(gid)
                segments.append(seg)
                
        except FileNotFoundError:
            print(f"Erro: Arquivo CSV não encontrado em '{self.csv_path}'")
            return [] # Retorna lista vazia em caso de erro de arquivo não encontrado
        except Exception as e:
            # Captura outros erros de parsing ou processamento
            print(f"Erro ao ler ou processar o CSV/buffer em LengthProcessor: {e}")
            return [] 
            
        return segments
    
    def process_segments(self, segments: List[dict] = None) -> Tuple[List[Polygon], List[dict]]:
        """
        Convert segments into column polygons and beam locations.

        Parameters
        ----------
        segments : List[dict], optional
            Segments to process; if `None`, reads from CSV.

        Returns
        -------
        Tuple[List[Polygon], List[dict]]
            Column polygons and beam definitions along walls.
        """
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
        Determine beam locations along wall segments using column faces.

        Parameters
        ----------
        column_polygons : List[Polygon]
            Polygons representing unioned column groups.

        Returns
        -------
        List[dict]
            Beam definitions with `node_1` and `node_2` endpoints.
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
                is_vertical_wall = abs(wall["end"][0] - wall["start"][0]) < 0.001
                if is_vertical_wall:
                    column_intersections.sort(key=lambda c: c["y"])
                else:
                    column_intersections.sort(key=lambda c: c["x"])

                # Require at least two columns to form a beam
                if len(column_intersections) < 2:
                    continue

                # Check intermediate columns exceeding threshold
                has_large_intermediate = False
                if len(column_intersections) >= 3:
                    for idx in range(1, len(column_intersections) - 1):
                        ci = column_intersections[idx]
                        dim = float(ci["max_face"] - ci["min_face"])  # width or height aligned to wall
                        if dim > float(SPLIT_BEAM_COLUMN_THRESHOLD_CM):
                            has_large_intermediate = True
                            break

                if has_large_intermediate:
                    # Create beams between adjacent columns
                    if is_vertical_wall:
                        for i in range(len(column_intersections) - 1):
                            curr_col = column_intersections[i]
                            next_col = column_intersections[i + 1]
                            beam_groups.append({
                                "node_1": (wall["start"][0], curr_col["max_face"]),
                                "node_2": (wall["start"][0], next_col["min_face"])
                            })
                    else:
                        for i in range(len(column_intersections) - 1):
                            curr_col = column_intersections[i]
                            next_col = column_intersections[i + 1]
                            beam_groups.append({
                                "node_1": (curr_col["max_face"], wall["start"][1]),
                                "node_2": (next_col["min_face"], wall["start"][1])
                            })
                else:
                    # Single continuous beam from first to last
                    first_col = column_intersections[0]
                    last_col = column_intersections[-1]
                    if is_vertical_wall:
                        beam_groups.append({
                            "node_1": (wall["start"][0], first_col["min_face"]),
                            "node_2": (wall["start"][0], last_col["max_face"])
                        })
                    else:
                        beam_groups.append({
                            "node_1": (first_col["min_face"], wall["start"][1]),
                            "node_2": (last_col["max_face"], wall["start"][1])
                        })
        
        return beam_groups

    def generate_variation(self, segments: List[dict], variation_strategy: str = "random") -> List[dict]:
        new_segments = copy.deepcopy(segments)
        made_changes = False
        step = 5.0 # Variação em múltiplos de 5 cm

        if variation_strategy in {"random", "upper_biased"}:
            # ------------------------------------------------------------------
            # Pre-compute rectangular constraint data from the ORIGINAL segments.
            # If seed rectangles from an x-group and a y-group physically overlap,
            # including offset starts at an external corner, only
            # the group with the larger deviation from its seed length is allowed
            # to change.  This mirrors the logic in DesignSpace._apply_rect_constraint
            # so that training samples match the geometry seen during optimisation.
            # ------------------------------------------------------------------
            # Build group index from original segments
            seed_groups: dict = {}
            for idx, seg in enumerate(segments):
                gid = seg.get("group_id")
                key = gid if gid is not None else f"__solo_{idx}"
                seed_groups.setdefault(key, []).append(idx)

            # Per-group seed length (max across members — same rule as lower_bounds)
            seed_len = {
                gid: max(segments[i]["length"] for i in idxs)
                for gid, idxs in seed_groups.items()
            }

            rect_pairs = GeometryProcessor.find_orthogonal_group_pairs(
                segments,
                DEFAULT_BEAM_WIDTH_CM,
            )

            # ------------------------------------------------------------------
            # Variation loop — probability raised from 0.4 → 0.7 for better
            # design-space coverage.
            # ------------------------------------------------------------------
            max_attempts = 10
            attempt = 0
            while not made_changes and attempt < max_attempts:
                groups = {}
                for idx, seg in enumerate(new_segments):
                    gid = seg.get("group_id")
                    groups.setdefault(gid if gid is not None else f"__solo_{idx}", []).append(idx)

                for gid, idxs in groups.items():
                    orig_lengths = [new_segments[i]["length"] for i in idxs]
                    max_lengths = [new_segments[i].get("maxlength") for i in idxs]
                    if any(m is None for m in max_lengths):
                        continue
                    group_min_allowed = max(orig_lengths)
                    group_max_allowed = min(max_lengths)
                    if group_max_allowed <= group_min_allowed:
                        continue
                    change_probability = (
                        1.0 if variation_strategy == "upper_biased" else 0.7
                    )
                    if random.random() < change_probability:
                        variation = group_max_allowed - group_min_allowed
                        max_steps = int(variation / step)
                        if max_steps > 0:
                            minimum_step = 1
                            if variation_strategy == "upper_biased":
                                # Generic validity-oriented stratum: explore
                                # only the upper half of each CSV-defined range.
                                minimum_step = max(1, int(np.ceil(max_steps * 0.5)))
                            num_steps = random.randint(minimum_step, max_steps)
                            new_length = group_min_allowed + (num_steps * step)
                            for i in idxs:
                                self._update_segment_length(new_segments[i], new_length)
                            made_changes = True
                attempt += 1

            if not made_changes:
                # Força pelo menos um grupo a mudar
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

            # ------------------------------------------------------------------
            # Post-process: enforce rectangular column constraint.
            # For each conflicting (x-group, y-group) pair, the group with the
            # smaller deviation from its seed length is reset to the original.
            # Ties go to x (same rule as DesignSpace._apply_rect_constraint).
            # ------------------------------------------------------------------
            if rect_pairs:
                curr_groups: dict = {}
                for idx, seg in enumerate(new_segments):
                    gid = seg.get("group_id")
                    key = gid if gid is not None else f"__solo_{idx}"
                    curr_groups.setdefault(key, []).append(idx)

                for xg, yg in rect_pairs:
                    x_idxs = curr_groups.get(xg, [])
                    y_idxs = curr_groups.get(yg, [])
                    if not x_idxs or not y_idxs:
                        continue
                    x_curr = max(new_segments[i]["length"] for i in x_idxs)
                    y_curr = max(new_segments[i]["length"] for i in y_idxs)
                    x_dev = x_curr - seed_len.get(xg, x_curr)
                    y_dev = y_curr - seed_len.get(yg, y_curr)
                    if x_dev >= y_dev:
                        # y-group deviates less (or tied) → reset it to seed lengths
                        for i in y_idxs:
                            self._update_segment_length(new_segments[i], segments[i]["length"])
                    else:
                        # x-group deviates less → reset it to seed lengths
                        for i in x_idxs:
                            self._update_segment_length(new_segments[i], segments[i]["length"])

        elif variation_strategy == "guided_by_volume":
            print("Guided by volume strategy not yet implemented. Falling back to random.")
            return self.generate_variation(segments, variation_strategy="random")

        return new_segments

    def _update_segment_length(self, segment: dict, new_length: float):
        """
        Update segment `end` preserving direction given a new length.

        Parameters
        ----------
        segment : dict
            Segment with `start` and `end` coordinates and `length`.
        new_length : float
            New length value (cm).
        """
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
