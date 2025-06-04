# Em algorithm/binary_input_processor.py
# (Anteriormente parte de datatreat.py)

import csv
from typing import List, Tuple, Dict, Set, Optional # Adicionado Optional
from shapely.geometry import Polygon
from .geometry_utils import GeometryProcessor # Importa sua classe de utilidades geométricas
from config.constants import CSV_PATH, BEAM_THICKNESS # CSV_PATH para o arquivo binário

class BinaryProcessor: # Renomeado de SegmentProcessor para BinaryProcessor
    """
    Processes segments from a binary grid input format (e.g., Building1.csv).
    It groups connected segments to form column polygons and beam definitions.
    Relies on GeometryProcessor for common geometric operations.
    """

    def __init__(self):
        # beam_thickness aqui é a espessura total para engrossar segmentos
        # para detecção de conexão e criação de polígonos.
        # A função create_rectangles_from_segments espera half_thickness.
        self.element_thickness = BEAM_THICKNESS
        # print(f"BinaryProcessor initialized with element_thickness: {self.element_thickness} cm")

    def read_binary_segments_from_csv(self, csv_file_path: str = CSV_PATH) -> List[Dict]:
        """
        Reads segments from a CSV file specific to the binary grid format.
        CSV should have columns: x1, y1, x2, y2, binary_flag (ou 'binario').

        Args:
            csv_file_path (str): Path to the CSV file. Defaults to CSV_PATH from constants.

        Returns:
            List[Dict]: A list of segment dictionaries, each with 'start', 'end',
                        and 'binary_flag' keys.
        """
        segments = []
        try:
            with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                for i, row in enumerate(reader):
                    try:
                        segments.append({
                            "id": i, # Adiciona um ID único baseado na ordem de leitura
                            "start": (float(row["x1"]), float(row["y1"])),
                            "end": (float(row["x2"]), float(row["y2"])),
                            # Padroniza a chave para 'binary_flag'
                            "binary_flag": int(row.get("binario", row.get("binary", row.get("binary_flag"))))
                        })
                    except KeyError as e:
                        print(f"Warning (read_binary_segments): Missing expected column {e} in row: {row}")
                    except ValueError as e:
                        print(f"Warning (read_binary_segments): Could not convert value in row {row}: {e}")
        except FileNotFoundError:
            print(f"Error (read_binary_segments): CSV file not found at {csv_file_path}")
            # Retorna lista vazia ou levanta exceção, dependendo do comportamento desejado
            return []
        except Exception as e:
            print(f"Error (read_binary_segments): Unexpected error reading CSV {csv_file_path}: {e}")
            return []
        return segments

    def _build_connection_graph(self, all_segments: List[Dict], binary_flag_value: int) -> Dict[int, Set[int]]:
        """
        Builds a connection graph for segments matching the binary_flag_value
        using the thickened polygon intersection method from GeometryProcessor.
        Uses original indices of segments from the 'all_segments' list.
        """
        segments_to_process: List[Dict] = []
        original_indices: List[int] = []

        for idx, seg_data in enumerate(all_segments):
            # A chave 'binary_flag' deve ser consistente com o que read_binary_segments_from_csv produz.
            if seg_data.get('binary_flag') == binary_flag_value:
                segments_to_process.append(seg_data)
                original_indices.append(idx) # Armazena o índice original do segmento na lista 'all_segments'
        
        if not segments_to_process:
            return {}

        # Chama a função utilitária estática de GeometryProcessor
        # Passa a espessura TOTAL do elemento (BEAM_THICKNESS)
        return GeometryProcessor.build_graph_from_polygon_intersections(
            segments_to_process,
            original_indices,
            self.element_thickness # Usa a espessura definida no __init__
        )

    def _group_segments_by_connectivity(self, all_segments: List[Dict], binary_flag_value: int) -> List[List[Dict]]:
        """
        Groups segments from 'all_segments' that match the 'binary_flag_value'
        based on their connectivity.
        """
        # Constrói o grafo usando os índices originais de 'all_segments'
        graph = self._build_connection_graph(all_segments, binary_flag_value)
        # Obtém componentes como listas de índices originais
        components_original_indices = GeometryProcessor.find_connected_components(graph)
        
        # Mapeia os índices de volta para os dicionários de segmentos originais
        grouped_segments = []
        for component_indices in components_original_indices:
            group = [all_segments[original_idx] for original_idx in component_indices]
            grouped_segments.append(group)
        return grouped_segments

    def _create_beam_definitions_from_binary_groups(self, beam_segment_groups: List[List[Dict]]) -> List[Dict]:
        """
        Processes groups of binary segments (binary_flag=0) into beam definitions
        (dictionaries with 'node_1' and 'node_2').
        This is similar to the old 'process_binary_0_groups'.
        """
        beam_definitions: List[Dict] = []
        for group in beam_segment_groups:
            if not group:
                continue
            
            # Encontra as coordenadas min/max do bounding box do grupo de segmentos de viga
            min_x = min(seg['start'][0] for seg in group + [{'start': (float('inf'),)}])
            min_x = min(min_x, min(seg['end'][0] for seg in group + [{'end': (float('inf'),)}]))
            min_y = min(seg['start'][1] for seg in group + [{'start': (0, float('inf'))}])
            min_y = min(min_y, min(seg['end'][1] for seg in group + [{'end': (0, float('inf'))}]))
            
            max_x = max(seg['start'][0] for seg in group + [{'start': (float('-inf'),)}])
            max_x = max(max_x, max(seg['end'][0] for seg in group + [{'end': (float('-inf'),)}]))
            max_y = max(seg['start'][1] for seg in group + [{'start': (0, float('-inf'))}])
            max_y = max(max_y, max(seg['end'][1] for seg in group + [{'end': (0, float('-inf'))}]))

            # Assume que vigas são principalmente horizontais ou verticais
            # e define node_1 e node_2 pelas extremidades do bounding box do grupo.
            # Esta lógica pode precisar de ajuste dependendo de como você quer definir as vigas
            # a partir dos segmentos binários.
            # Exemplo simples:
            if (max_x - min_x) > (max_y - min_y): # Provavelmente horizontal
                node_1 = (min_x, (min_y + max_y) / 2)
                node_2 = (max_x, (min_y + max_y) / 2)
            else: # Provavelmente vertical
                node_1 = ((min_x + max_x) / 2, min_y)
                node_2 = ((min_x + max_x) / 2, max_y)

            beam_definitions.append({"node_1": node_1, "node_2": node_2})
        return beam_definitions

    def process_segments(self, segments_from_csv: Optional[List[Dict]] = None) -> Tuple[List[Polygon], List[Dict]]:
        """
        Main processing method for binary input segments.
        Reads segments from CSV if not provided, then groups them to form
        column polygons and beam definitions.

        Args:
            segments_from_csv (Optional[List[Dict]]): List of segments read from CSV.
                                                       If None, reads from default CSV_PATH.

        Returns:
            Tuple[List[Polygon], List[Dict]]:
                - final_column_polygons: List of Shapely Polygons for column groups.
                - beam_definitions: List of dictionaries defining beams (with 'node_1', 'node_2').
        """
        if segments_from_csv is None:
            print("BinaryProcessor: No segments provided, reading from CSV_PATH...")
            segments_from_csv = self.read_binary_segments_from_csv()

        if not segments_from_csv:
            print("BinaryProcessor: No segments to process.")
            return [], []

        print(f"BinaryProcessor: Processing {len(segments_from_csv)} binary segments...")

        # Grupos de pilares (binary_flag = 1)
        pillar_segment_groups = self._group_segments_by_connectivity(segments_from_csv, binary_flag_value=1)
        
        final_column_polygons: List[Polygon] = []
        print(f"BinaryProcessor: Found {len(pillar_segment_groups)} potential pillar groups.")
        for i, group in enumerate(pillar_segment_groups):
            if not group:
                continue
            # Cria retângulos com METADE da espessura do elemento
            rect_vertices_list = GeometryProcessor.create_rectangles_from_segments(group, self.element_thickness / 2.0)
            shapely_polygons = GeometryProcessor.convert_vertices_to_polygons(rect_vertices_list)
            
            # Filtra Nones antes de unir
            valid_polygons_for_union = [p for p in shapely_polygons if p is not None]
            if valid_polygons_for_union:
                united_polygons = GeometryProcessor.union_polygons(valid_polygons_for_union)
                final_column_polygons.extend(united_polygons)
                # print(f"BinaryProcessor: Pillar group {i+1} processed into {len(united_polygons)} final polygon(s).")

        # Grupos de vigas (binary_flag = 0)
        beam_segment_groups = self._group_segments_by_connectivity(segments_from_csv, binary_flag_value=0)
        print(f"BinaryProcessor: Found {len(beam_segment_groups)} potential beam segment groups.")
        beam_definitions = self._create_beam_definitions_from_binary_groups(beam_segment_groups)
        print(f"BinaryProcessor: Created {len(beam_definitions)} beam definitions.")
        
        return final_column_polygons, beam_definitions