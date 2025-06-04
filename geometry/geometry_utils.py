# Em algorithm/geometry_utils.py (ou algorithm/polygon_processor.py)

from config.constants import BEAM_THICKNESS
from shapely.geometry import Polygon
import pyclipper 
from typing import List, Dict, Tuple, Set, Optional

class GeometryProcessor:
    """
    Provides static utility methods for geometric operations on segments and polygons,
    useful for structural analysis pre-processing.
    """

    @staticmethod
    def create_rectangles_from_segments(
        segments: List[Dict],
        half_thickness: float # Renomeado para clareza, representa metade da espessura total
    ) -> List[Optional[List[Tuple[float, float]]]]: # Retorna Optional para lidar com falhas
        """
        Converts a list of segment dictionaries into a list of rectangle vertices.
        Each segment dictionary must contain 'start': (x,y) and 'end': (x,y) coordinates.
        'half_thickness' is the distance from the segment's centerline to its edge.

        Args:
            segments: List of segment dictionaries.
            half_thickness: Half of the desired total thickness of the rectangles.

        Returns:
            A list where each element is either a list of (x,y) rectangle vertices
            or None if a rectangle could not be created for the corresponding segment.
        """
        rectangles_vertices_list: List[Optional[List[Tuple[float, float]]]] = []
        if not segments:
            return rectangles_vertices_list

        for segment in segments:
            start = segment.get("start")
            end = segment.get("end")

            if not start or not end:
                print(f"Warning (GP.create_rectangles): Segment missing 'start' or 'end' key: {segment}")
                rectangles_vertices_list.append(None)
                continue
            
            x1, y1 = start
            x2, y2 = end

            if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                print(f"Warning (GP.create_rectangles): Segment has non-numeric coordinates: {segment}")
                rectangles_vertices_list.append(None)
                continue

            if abs(y1 - y2) < 1e-6:  # Horizontal segment
                vertices = [
                    (x1, y1 - half_thickness), (x2, y1 - half_thickness),
                    (x2, y1 + half_thickness), (x1, y1 + half_thickness)
                ]
            elif abs(x1 - x2) < 1e-6:  # Vertical segment
                vertices = [
                    (x1 - half_thickness, y1), (x1 + half_thickness, y1),
                    (x1 + half_thickness, y2), (x1 - half_thickness, y2)
                ]
            else:
                print(f"Warning (GP.create_rectangles): Segment from {start} to {end} is diagonal. "
                      "Simple rectangle generation for H/V segments only. Skipping.")
                rectangles_vertices_list.append(None)
                continue
            
            rectangles_vertices_list.append(vertices)
        return rectangles_vertices_list

    @staticmethod
    def convert_vertices_to_polygons(
        vertices_list: List[Optional[List[Tuple[float, float]]]]
    ) -> List[Optional[Polygon]]:
        """
        Converts lists of rectangle vertices into Shapely Polygon objects.
        Skips entries in the input list that are None.

        Args:
            vertices_list: A list where each element is a list of (x,y) vertices or None.

        Returns:
            A list where each element is a Shapely Polygon or None.
        """
        polygons: List[Optional[Polygon]] = []
        for vertices in vertices_list:
            if vertices: # Only attempt to create a polygon if vertices are not None
                try:
                    polygons.append(Polygon(vertices))
                except Exception as e:
                    print(f"Warning (GP.convert_vertices_to_polygons): Could not create polygon from {vertices}: {e}")
                    polygons.append(None) # Append None if polygon creation fails
            else:
                polygons.append(None) # Maintain correspondence if input vertices were None
        return polygons

    @staticmethod
    def union_polygons(polygons: List[Polygon]) -> List[Polygon]:
        """
        Performs a union operation on a list of valid Shapely Polygon objects using PyClipper.
        Filters out any None values from the input list before processing.

        Args:
            polygons: List of Shapely Polygon objects (may contain None).

        Returns:
            List of resulting Shapely Polygon objects after union. Can be empty.
        """
        valid_polygons = [p for p in polygons if p is not None and p.is_valid]
        if not valid_polygons:
            return []

        pc = pyclipper.Pyclipper()
        for polygon in valid_polygons:
            coords = list(polygon.exterior.coords)
            # PyClipper generally prefers counter-clockwise for subjects,
            # but union operation is often robust. Explicit CCW can be added if issues arise.
            # if not polygon.exterior.is_ccw:
            #     coords = coords[::-1]
            pc.AddPath(coords, pyclipper.PT_SUBJECT, True)
        
        try:
            solution = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
            return [Polygon(path) for path in solution if path] # Ensure path is not empty
        except pyclipper.PyclipperException as e:
            print(f"Error (GP.union_polygons): PyClipper union failed: {e}")
            # Fallback: return original valid polygons without union, or an empty list
            return valid_polygons # Or [] depending on desired error handling

    @staticmethod
    def build_graph_from_polygon_intersections(
        segments_to_process: List[Dict],
        original_indices: List[int],
        segment_total_thickness: float # Espessura TOTAL do elemento
    ) -> Dict[int, Set[int]]:
        """
        Builds an intersection graph for a list of segments by creating polygons
        (rectangles) around them and checking for polygon intersections.

        Args:
            segments_to_process: List of segment dicts (must have 'start', 'end').
            original_indices: List of original indices for each segment in segments_to_process.
                              Ensures graph uses original indexing.
            segment_total_thickness: The full desired thickness of the elements.
                                     Rectangles will be created with half of this on each side.
        Returns:
            Adjacency list graph using original segment indices.
        """
        if not segments_to_process or len(segments_to_process) != len(original_indices):
            print("Warning (GP.build_graph): segments_to_process is empty or mismatch with original_indices.")
            return {}

        half_thickness = segment_total_thickness / 2.0
        
        rect_vertices_list = GeometryProcessor.create_rectangles_from_segments(
            segments_to_process, half_thickness
        )
        
        # Mapeia original_idx -> Polygon
        segment_polygons: Dict[int, Polygon] = {}
        
        for i, rect_verts in enumerate(rect_vertices_list):
            if rect_verts: # Se o retângulo foi criado com sucesso
                original_idx = original_indices[i] # Pega o índice original correspondente
                try:
                    poly = Polygon(rect_verts)
                    if poly.is_valid: # Adiciona apenas polígonos válidos
                        segment_polygons[original_idx] = poly
                    else:
                        print(f"Warning (GP.build_graph): Invalid polygon created for segment original_idx {original_idx}.")
                except Exception as e:
                    print(f"Warning (GP.build_graph): Failed to create polygon for segment original_idx {original_idx} from vertices {rect_verts}: {e}")

        graph: Dict[int, Set[int]] = {idx: set() for idx in segment_polygons.keys()}
        
        valid_polygon_original_indices = list(segment_polygons.keys())

        for i in range(len(valid_polygon_original_indices)):
            for j in range(i + 1, len(valid_polygon_original_indices)):
                idx1 = valid_polygon_original_indices[i]
                idx2 = valid_polygon_original_indices[j]
                
                poly1 = segment_polygons[idx1]
                poly2 = segment_polygons[idx2]
                
                if poly1.intersects(poly2):
                    graph[idx1].add(idx2)
                    graph[idx2].add(idx1)
        return graph

    @staticmethod
    def find_connected_components(graph: Dict[int, Set[int]]) -> List[List[int]]:
        """
        Finds connected components in a graph using Depth-First Search (DFS).
        """
        visited: Set[int] = set()
        components: List[List[int]] = []
        
        # Consider todos os nós que são chaves ou aparecem em conjuntos de adjacência
        all_nodes_in_graph = set(graph.keys())
        for neighbors in graph.values():
            all_nodes_in_graph.update(neighbors)

        def dfs(node: int, current_component: List[int]):
            visited.add(node)
            current_component.append(node)
            # graph.get(node, set()) para lidar com nós que podem não ter entrada de chave no grafo
            # mas são mencionados como vizinhos (embora um grafo bem formado deva ter chaves para todos os nós)
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, current_component)

        for node in sorted(list(all_nodes_in_graph)): # Ordenar para resultados consistentes (opcional)
            if node not in visited:
                component: List[int] = []
                dfs(node, component)
                if component: # Adiciona apenas se o componente não for vazio
                    components.append(component)
        return components