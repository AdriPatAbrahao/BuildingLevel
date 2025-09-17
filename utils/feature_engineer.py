
from typing import List, Dict
from shapely.geometry import Polygon, LineString, MultiLineString
import numpy as np
from config.constants import DEFAULT_BEAM_WIDTH_CM, DEFAULT_BEAM_HEIGHT_CM
from utils.geometric_calculator import (
    calculate_column_geometric_volume,
    calculate_beams_geometric_volume,
)
import shapely

class FeatureEngineer:
    """
    Extracts a rich feature set from the geometric properties of the structure.
    """
    def __init__(self, column_polygons: List[Polygon], beam_definitions: List[Dict]):
        self.column_polygons = column_polygons
        self.beam_definitions = beam_definitions

    def extract_features(self) -> List[float]:
        """
        Computes all engineered features and returns them as a single vector.
        """
        features = []
        
        # Column features
        column_areas = [p.area for p in self.column_polygons]
        total_column_area = sum(column_areas)
        num_columns = len(self.column_polygons)
        
        features.extend([
            total_column_area,
            num_columns,
            np.mean(column_areas) if column_areas else 0,
            np.std(column_areas) if column_areas else 0,
            min(column_areas) if column_areas else 0,
            max(column_areas) if column_areas else 0,
        ])
        
        # Beam features (effective after subtracting column intersections)
        beam_lines = [LineString([b['node_1'], b['node_2']]) for b in self.beam_definitions]

        def _intersect_len(line: LineString, poly: Polygon) -> float:
            try:
                inter = line.intersection(poly)
                if inter.is_empty:
                    return 0.0
                if isinstance(inter, LineString):
                    return float(inter.length)
                if isinstance(inter, MultiLineString):
                    return float(sum(seg.length for seg in inter.geoms))
                return 0.0
            except Exception:
                return 0.0

        effective_lengths = []
        for ln in beam_lines:
            subtract = 0.0
            for col_poly in self.column_polygons:
                subtract += _intersect_len(ln, col_poly)
            effective_lengths.append(max(ln.length - subtract, 0.0))

        total_eff_beam_length = float(sum(effective_lengths))
        num_beams = len(beam_lines)

        # Approximate beam volume (m^3): effective length (cm -> m) * width(m) * height(m)
        CM_TO_M = 0.01
        beam_width_m = DEFAULT_BEAM_WIDTH_CM * CM_TO_M
        beam_height_m = DEFAULT_BEAM_HEIGHT_CM * CM_TO_M
        total_beam_volume_m3 = (total_eff_beam_length * CM_TO_M) * beam_width_m * beam_height_m

        features.extend([
            total_eff_beam_length,
            num_beams,
            float(np.mean(effective_lengths)) if effective_lengths else 0.0,
            float(np.std(effective_lengths)) if effective_lengths else 0.0,
            float(np.max(effective_lengths)) if effective_lengths else 0.0,
        ])

        # Inertia features
        moments_of_inertia_xx = []
        moments_of_inertia_yy = []

        for p in self.column_polygons:
            # Substituímos a chamada hasattr pela nossa função
            try:
                # Chama nossa nova função
                Ixx, Iyy = calculate_centroidal_moment_of_inertia(p)
                moments_of_inertia_xx.append(Ixx)
                moments_of_inertia_yy.append(Iyy)
            except Exception as e:
                print(f"Warning: Error calculating moment of inertia for a polygon: {e}. Appending 0.")
                moments_of_inertia_xx.append(0.0)
                moments_of_inertia_yy.append(0.0)

        sum_Ix = float(np.sum(moments_of_inertia_xx)) if moments_of_inertia_xx else 0.0
        sum_Iy = float(np.sum(moments_of_inertia_yy)) if moments_of_inertia_yy else 0.0
        mean_Ix = float(np.mean(moments_of_inertia_xx)) if moments_of_inertia_xx else 0.0
        mean_Iy = float(np.mean(moments_of_inertia_yy)) if moments_of_inertia_yy else 0.0
        inertia_ratio = (sum_Iy / (sum_Ix + 1e-9)) if (sum_Ix > 0.0 or sum_Iy > 0.0) else 0.0

        features.extend([
            sum_Ix,
            sum_Iy,
            mean_Ix,
            mean_Iy,
            inertia_ratio,
        ])

        # Geometric concrete volumes in m^3 (separated): pillars and beams
        vol_columns_m3 = calculate_column_geometric_volume(self.column_polygons)
        # For beams prefer volume computed from effective lengths above to avoid double count
        vol_beams_m3 = total_beam_volume_m3
        features.extend([
            vol_columns_m3,
            vol_beams_m3,
        ])

        # Column perimeter/shape descriptors
        perims = [p.length for p in self.column_polygons]
        compact = []
        for p in self.column_polygons:
            A = p.area
            P = p.length
            if P > 0:
                compact.append(float(4.0 * np.pi * A / (P * P)))
        features.extend([
            float(np.sum(perims)) if perims else 0.0,
            float(np.mean(perims)) if perims else 0.0,
            float(np.std(perims)) if perims else 0.0,
            float(np.mean(compact)) if compact else 0.0,
        ])
        
        return features
    
def calculate_centroidal_moment_of_inertia(polygon: Polygon) -> tuple[float, float]:
    """
    Calcula os momentos de inércia (Ixx, Iyy) de um polígono em relação ao seu centroide.

    Args:
        polygon (Polygon): O polígono do Shapely.

    Returns:
        tuple[float, float]: Uma tupla contendo (Ixx, Iyy) em relação ao centroide.
                             Retorna (0.0, 0.0) se o polígono for inválido.
    """
    if not polygon.is_valid or polygon.is_empty:
        return (0.0, 0.0)

    # Pega as coordenadas do contorno externo
    coords = np.array(polygon.exterior.coords)
    x = coords[:, 0]
    y = coords[:, 1]

    # Usa a fórmula baseada no Teorema de Green para calcular a inércia em relação à ORIGEM
    # (x_i * y_{i+1} - x_{i+1} * y_i) é um termo comum
    a = x[:-1] * y[1:] - x[1:] * y[:-1]

    # Cálculo dos momentos de inércia em relação à ORIGEM (0,0)
    Ixx_origin = (1/12) * np.sum((y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2) * a)
    Iyy_origin = (1/12) * np.sum((x[:-1]**2 + x[:-1]*x[1:] + x[1:]**2) * a)
    
    # Pega a área e o centroide calculados pelo Shapely
    area = polygon.area
    centroid = polygon.centroid
    cx, cy = centroid.x, centroid.y

    # Usa o Teorema dos Eixos Paralelos para transladar a inércia para o centroide
    # I_centroid = I_origin - A * d^2
    Ixx_centroid = Ixx_origin - area * (cy**2)
    Iyy_centroid = Iyy_origin - area * (cx**2)

    return (Ixx_centroid, Iyy_centroid)
