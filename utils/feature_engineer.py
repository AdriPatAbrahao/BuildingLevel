
from typing import List, Dict
from shapely.geometry import Polygon, LineString, MultiLineString
import numpy as np
from config.constants import DEFAULT_BEAM_WIDTH_CM, DEFAULT_BEAM_HEIGHT_CM
from utils.geometric_calculator import (
    calculate_column_geometric_volume,
    calculate_beams_geometric_volume,
)

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

        Feature layout (43 total):
          [0-5]   Column area stats (6)
          [6-10]  Beam effective-length stats (5)
          [11-15] Inertia (sum_Ix, sum_Iy, mean_Ix, mean_Iy, ratio) (5)
          [16-17] Geometric volumes — columns, beams (2)
          [18-21] Column perimeter / compactness (4)
          [22-28] Spatial kept: excentricity, quadrant-area-ratio, slenderness×2, span×3 (7)
          [29-43] NEW structural features (14):
                  CS_x, CS_y, ecc_x, ecc_y, ecc_total (5)
                  mean_rx, mean_ry, mean_r_min, min_r_global (4)
                  mean_aspect, std_aspect, max_aspect (3)
                  max_quadrant_inertia_ratio (1)
                  J_polar_layout (1)
        """
        features = []

        # --- Block 1: Column area features (6) ---
        col_areas = np.array([p.area for p in self.column_polygons], dtype=float)
        num_columns = len(self.column_polygons)
        if col_areas.size > 0:
            total_column_area = float(col_areas.sum())
            mean_col_area     = float(col_areas.mean())
            std_col_area      = float(col_areas.std())
            min_col_area      = float(col_areas.min())
            max_col_area      = float(col_areas.max())
        else:
            total_column_area = mean_col_area = std_col_area = min_col_area = max_col_area = 0.0

        features.extend([
            total_column_area, num_columns, mean_col_area,
            std_col_area, min_col_area, max_col_area,
        ])

        # --- Block 2: Beam effective-length features (5) ---
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
            subtract = sum(_intersect_len(ln, cp) for cp in self.column_polygons)
            effective_lengths.append(max(ln.length - subtract, 0.0))

        total_eff_beam_length = float(sum(effective_lengths))
        num_beams = len(beam_lines)

        CM_TO_M = 0.01
        beam_width_m  = DEFAULT_BEAM_WIDTH_CM  * CM_TO_M
        beam_height_m = DEFAULT_BEAM_HEIGHT_CM * CM_TO_M
        total_beam_volume_m3 = (total_eff_beam_length * CM_TO_M) * beam_width_m * beam_height_m

        features.extend([
            total_eff_beam_length,
            num_beams,
            float(np.mean(effective_lengths)) if effective_lengths else 0.0,
            float(np.std(effective_lengths))  if effective_lengths else 0.0,
            float(np.max(effective_lengths))  if effective_lengths else 0.0,
        ])

        # --- Block 3: Inertia features (5) ---
        moments_of_inertia_xx = []
        moments_of_inertia_yy = []
        for p in self.column_polygons:
            try:
                Ixx, Iyy = calculate_centroidal_moment_of_inertia(p)
                moments_of_inertia_xx.append(Ixx)
                moments_of_inertia_yy.append(Iyy)
            except Exception as e:
                print(f"Warning: Error calculating moment of inertia: {e}. Appending 0.")
                moments_of_inertia_xx.append(0.0)
                moments_of_inertia_yy.append(0.0)

        sum_Ix = float(np.sum(moments_of_inertia_xx)) if moments_of_inertia_xx else 0.0
        sum_Iy = float(np.sum(moments_of_inertia_yy)) if moments_of_inertia_yy else 0.0
        mean_Ix = float(np.mean(moments_of_inertia_xx)) if moments_of_inertia_xx else 0.0
        mean_Iy = float(np.mean(moments_of_inertia_yy)) if moments_of_inertia_yy else 0.0
        inertia_ratio = (sum_Iy / (sum_Ix + 1e-9)) if (sum_Ix > 0.0 or sum_Iy > 0.0) else 0.0

        features.extend([sum_Ix, sum_Iy, mean_Ix, mean_Iy, inertia_ratio])

        # --- Block 4: Geometric volumes (2) ---
        vol_columns_m3 = calculate_column_geometric_volume(self.column_polygons)
        features.extend([vol_columns_m3, total_beam_volume_m3])

        # --- Block 5: Column perimeter / compactness (4) ---
        perims = [p.length for p in self.column_polygons]
        compact = [
            float(4.0 * np.pi * p.area / (p.length * p.length))
            for p in self.column_polygons if p.length > 0
        ]
        features.extend([
            float(np.sum(perims))   if perims   else 0.0,
            float(np.mean(perims))  if perims   else 0.0,
            float(np.std(perims))   if perims   else 0.0,
            float(np.mean(compact)) if compact  else 0.0,
        ])

        # --- Block 6: Spatial + structural features (21) ---
        if self.column_polygons:
            _cg         = [p.centroid for p in self.column_polygons]
            centroids_np = np.array([(float(c.x), float(c.y)) for c in _cg], dtype=float)
            areas_np     = col_areas
            perims_np    = np.array([p.length for p in self.column_polygons], dtype=float)
            ixx_arr      = np.array(moments_of_inertia_xx, dtype=float)
            iyy_arr      = np.array(moments_of_inertia_yy, dtype=float)

            # Area-weighted centroid (center of mass) — structurally correct reference
            # for eccentricity and J_polar.  Simple mean would give wrong results when
            # columns have different cross-section sizes.
            _total_area = float(areas_np.sum())
            if _total_area > 0:
                cx = float(np.sum(areas_np * centroids_np[:, 0]) / _total_area)
                cy = float(np.sum(areas_np * centroids_np[:, 1]) / _total_area)
            else:
                cx = float(centroids_np[:, 0].mean())
                cy = float(centroids_np[:, 1].mean())
            dists = np.sqrt((centroids_np[:, 0] - cx)**2 + (centroids_np[:, 1] - cy)**2)

            # Kept: area-weighted eccentricity
            excentricity_global = float(np.sum(areas_np * dists))

            # Kept: quadrant area asymmetry
            xc    = centroids_np[:, 0]
            yc    = centroids_np[:, 1]
            q_idx = np.where(xc >= cx,
                             np.where(yc >= cy, 0, 3),
                             np.where(yc >= cy, 1, 2))
            q_areas         = [float(np.sum(areas_np[q_idx == q])) for q in range(4)]
            total_area      = float(areas_np.sum())
            max_q_area_ratio = float(max(q_areas)) / total_area if total_area > 0 else 0.0

            # Kept: geometric slenderness
            valid_mask = areas_np > 0
            if valid_mask.any():
                slend            = perims_np[valid_mask] / np.sqrt(areas_np[valid_mask])
                mean_slenderness = float(slend.mean())
                p95_slenderness  = float(np.percentile(slend, 95))
            else:
                mean_slenderness = p95_slenderness = 0.0

            # Kept: beam span distribution
            span_max = float(np.max(effective_lengths)) if effective_lengths else 0.0
            span_p95 = float(np.percentile(effective_lengths, 95)) if effective_lengths else 0.0
            if effective_lengths:
                bins = max(10, min(50, int(np.sqrt(len(effective_lengths)))))
                hist, _ = np.histogram(np.array(effective_lengths), bins=bins, density=True)
                p = hist / (np.sum(hist) + 1e-9)
                span_entropy = float(-np.sum(p * np.log(p + 1e-9)))
            else:
                span_entropy = 0.0

            # NEW: Center of stiffness and structural eccentricity
            # CS_x resists lateral displacement in X → weighted by Iyy (bending about Y-axis)
            # CS_y resists lateral displacement in Y → weighted by Ixx (bending about X-axis)
            sum_Iyy = float(iyy_arr.sum())
            sum_Ixx = float(ixx_arr.sum())
            CS_x = float(np.sum(iyy_arr * centroids_np[:, 0])) / sum_Iyy if sum_Iyy > 0 else cx
            CS_y = float(np.sum(ixx_arr * centroids_np[:, 1])) / sum_Ixx if sum_Ixx > 0 else cy
            ecc_x     = CS_x - cx
            ecc_y     = CS_y - cy
            ecc_total = float(np.sqrt(ecc_x**2 + ecc_y**2))

            # NEW: Radius of gyration r = sqrt(I/A) — determines structural slenderness
            rx_list, ry_list, r_min_list = [], [], []
            for Ixx_i, Iyy_i, A_i in zip(moments_of_inertia_xx, moments_of_inertia_yy, areas_np):
                if A_i > 0:
                    rx_i = float(np.sqrt(abs(Ixx_i) / A_i))
                    ry_i = float(np.sqrt(abs(Iyy_i) / A_i))
                    rx_list.append(rx_i)
                    ry_list.append(ry_i)
                    r_min_list.append(min(rx_i, ry_i))
            mean_rx      = float(np.mean(rx_list))    if rx_list    else 0.0
            mean_ry      = float(np.mean(ry_list))    if ry_list    else 0.0
            mean_r_min   = float(np.mean(r_min_list)) if r_min_list else 0.0
            min_r_global = float(np.min(r_min_list))  if r_min_list else 0.0

            # NEW: Column aspect ratio h/b ≈ sqrt(Iyy/Ixx) — captures section directionality
            aspect_list = [
                float(np.sqrt(abs(Iyy_i) / (abs(Ixx_i) + 1e-9)))
                for Ixx_i, Iyy_i in zip(moments_of_inertia_xx, moments_of_inertia_yy)
            ]
            mean_aspect = float(np.mean(aspect_list)) if aspect_list else 1.0
            std_aspect  = float(np.std(aspect_list))  if aspect_list else 0.0
            max_aspect  = float(np.max(aspect_list))  if aspect_list else 1.0

            # NEW: Quadrant inertia asymmetry (stiffness-based, not area-based)
            total_I_per_col           = ixx_arr + iyy_arr
            total_I                   = float(total_I_per_col.sum())
            q_inertia                 = [float(np.sum(total_I_per_col[q_idx == q])) for q in range(4)]
            max_quadrant_inertia_ratio = float(max(q_inertia)) / total_I if total_I > 0 else 0.25

            # NEW: Polar moment of inertia of layout — torsional stiffness proxy
            # J = Σ(Ixx_i + Iyy_i + A_i·d_i²)  includes parallel-axis theorem term
            J_polar = float(np.sum(ixx_arr + iyy_arr + areas_np * dists**2))

            features.extend([
                # --- kept spatial (7) ---
                excentricity_global,
                max_q_area_ratio,
                mean_slenderness,
                p95_slenderness,
                span_max,
                span_p95,
                span_entropy,
                # --- new: center of stiffness + eccentricity (5) ---
                CS_x, CS_y,
                ecc_x, ecc_y, ecc_total,
                # --- new: radius of gyration (4) ---
                mean_rx, mean_ry, mean_r_min, min_r_global,
                # --- new: aspect ratio (3) ---
                mean_aspect, std_aspect, max_aspect,
                # --- new: stiffness asymmetry (1) ---
                max_quadrant_inertia_ratio,
                # --- new: torsional stiffness proxy (1) ---
                J_polar,
            ])

        assert len(features) == 43, (
            f"Feature count mismatch: expected 43, got {len(features)}. "
            "Update NeuralNetConfig.INPUT_SIZE and feature_names() if features were added/removed."
        )
        return features

    @staticmethod
    def feature_names() -> List[str]:
        base = [
            # Block 1 — column area stats (6)
            "columns_total_area_cm2",
            "columns_count",
            "columns_mean_area_cm2",
            "columns_std_area_cm2",
            "columns_min_area_cm2",
            "columns_max_area_cm2",
            # Block 2 — beam effective-length stats (5)
            "beams_total_effective_length_cm",
            "beams_count",
            "beams_mean_effective_length_cm",
            "beams_std_effective_length_cm",
            "beams_max_effective_length_cm",
            # Block 3 — inertia (5)
            "inertia_sum_Ix",
            "inertia_sum_Iy",
            "inertia_mean_Ix",
            "inertia_mean_Iy",
            "inertia_ratio_Iy_over_Ix",
            # Block 4 — geometric volumes (2)
            "vol_columns_m3",
            "vol_beams_m3",
            # Block 5 — perimeter / compactness (4)
            "columns_total_perimeter_cm",
            "columns_mean_perimeter_cm",
            "columns_std_perimeter_cm",
            "columns_mean_compactness",
        ]
        spatial = [
            # Block 6a — kept spatial (7)
            "pillars_excentricity_global",
            "pillars_max_quadrant_area_ratio",
            "pillars_mean_slenderness",
            "pillars_p95_slenderness",
            "beams_span_max_cm",
            "beams_span_p95_cm",
            "beams_span_entropy",
            # Block 6b — center of stiffness + eccentricity (5)
            "cs_x",
            "cs_y",
            "stiffness_ecc_x",
            "stiffness_ecc_y",
            "stiffness_ecc_total",
            # Block 6c — radius of gyration (4)
            "mean_radius_gyration_x",
            "mean_radius_gyration_y",
            "mean_radius_gyration_min",
            "min_radius_gyration_global",
            # Block 6d — column aspect ratio (3)
            "mean_col_aspect_ratio",
            "std_col_aspect_ratio",
            "max_col_aspect_ratio",
            # Block 6e — stiffness asymmetry (1)
            "max_quadrant_inertia_ratio",
            # Block 6f — torsional stiffness proxy (1)
            "J_polar_layout",
        ]
        return base + spatial
    
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
