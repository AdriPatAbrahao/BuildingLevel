
from typing import List, Dict
from shapely.geometry import Polygon, LineString, MultiLineString, GeometryCollection, box
from shapely.ops import unary_union
import numpy as np
from config.constants import DEFAULT_BEAM_WIDTH_CM, DEFAULT_BEAM_HEIGHT_CM
from config.settings import BuildingConfig
from utils.geometric_calculator import (
    calculate_column_geometric_volume,
)

class FeatureEngineer:
    """
    Extracts a rich feature set from the geometric properties of the structure.
    """
    def __init__(self, column_polygons: List[Polygon], beam_definitions: List[Dict]):
        self.column_polygons = column_polygons
        self.beam_definitions = beam_definitions
        self._spatial_diagnostics: Dict[str, float] = {}
        self._non_model_diagnostics: Dict[str, float] = {}

    def extract_features(self) -> List[float]:
        """
        Computes all engineered features and returns them as a single vector.

        Feature layout (27 total, schema v6):
          [0-3]   Non-redundant column area stats (4)
          [4-11]  Physical clear-span stats separated into X and Y (8)
          [12-16] Inertia (sum_Ix, sum_Iy, mean_Ix, mean_Iy, ratio) (5)
          [17]    Mean column compactness (1)
          [18-19] Fixed-reference area spread in X and Y (2)
          [20-21] Section-shape descriptors (2)
          [22-23] Directional radius of gyration (2)
          [24-26] Section directional aspect ratio (3)
        """
        features = []

        # --- Block 1: Non-redundant column area features (4) ---
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
            total_column_area, std_col_area, min_col_area, max_col_area,
        ])

        # --- Block 2: Physical clear-span features by direction (8) ---
        beam_lines = [LineString([b['node_1'], b['node_2']]) for b in self.beam_definitions]

        column_union = unary_union(self.column_polygons)
        clear_spans = {"x": [], "y": []}
        for ln in beam_lines:
            remaining = ln.difference(column_union)
            if isinstance(remaining, LineString):
                parts = [remaining]
            elif isinstance(remaining, MultiLineString):
                parts = list(remaining.geoms)
            elif isinstance(remaining, GeometryCollection):
                parts = [g for g in remaining.geoms if isinstance(g, LineString)]
            else:
                parts = []
            x0, y0 = ln.coords[0]
            x1, y1 = ln.coords[-1]
            axis = "x" if abs(x1 - x0) >= abs(y1 - y0) else "y"
            clear_spans[axis].extend(
                float(part.length) for part in parts if part.length > 1e-9
            )

        spans_x = clear_spans["x"]
        spans_y = clear_spans["y"]
        total_x = float(sum(spans_x))
        total_y = float(sum(spans_y))
        total_eff_beam_length = total_x + total_y

        def _span_entropy(lengths: List[float]) -> float:
            if not lengths:
                return 0.0
            hist, _ = np.histogram(np.asarray(lengths, dtype=float), bins=10, density=True)
            probabilities = hist / (np.sum(hist) + 1e-9)
            entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-9)))
            return max(entropy, 0.0)

        def _mean(lengths: List[float]) -> float:
            return float(np.mean(lengths)) if lengths else 0.0

        def _std(lengths: List[float]) -> float:
            return float(np.std(lengths)) if lengths else 0.0

        def _max(lengths: List[float]) -> float:
            return float(np.max(lengths)) if lengths else 0.0

        def _p95(lengths: List[float]) -> float:
            return float(np.percentile(lengths, 95)) if lengths else 0.0

        CM_TO_M = 0.01
        beam_width_m  = DEFAULT_BEAM_WIDTH_CM  * CM_TO_M
        beam_height_m = DEFAULT_BEAM_HEIGHT_CM * CM_TO_M
        total_beam_volume_m3 = (total_eff_beam_length * CM_TO_M) * beam_width_m * beam_height_m

        features.extend([
            total_x,
            total_y,
            _std(spans_x),
            _std(spans_y),
            _max(spans_x),
            _max(spans_y),
            _span_entropy(spans_x),
            _span_entropy(spans_y),
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

        # --- Derived geometric volumes (diagnostic only) ---
        vol_columns_m3 = calculate_column_geometric_volume(self.column_polygons)

        # --- Block 5: Column compactness (perimeter stats are diagnostic) ---
        perims = [p.length for p in self.column_polygons]
        compact = [
            float(4.0 * np.pi * p.area / (p.length * p.length))
            for p in self.column_polygons if p.length > 0
        ]
        total_perimeter = float(np.sum(perims)) if perims else 0.0
        mean_perimeter = float(np.mean(perims)) if perims else 0.0
        std_perimeter = float(np.std(perims)) if perims else 0.0
        features.append(float(np.mean(compact)) if compact else 0.0)

        # --- Block 6: Spatial + structural features (14) ---
        if self.column_polygons:
            _cg         = [p.centroid for p in self.column_polygons]
            centroids_np = np.array([(float(c.x), float(c.y)) for c in _cg], dtype=float)
            areas_np     = col_areas
            perims_np    = np.array([p.length for p in self.column_polygons], dtype=float)
            ixx_arr      = np.array(moments_of_inertia_xx, dtype=float)
            iyy_arr      = np.array(moments_of_inertia_yy, dtype=float)

            _total_area = float(areas_np.sum())
            load_center_x, load_center_y = map(float, BuildingConfig.LOAD_CENTER_CM)
            plan_width = float(BuildingConfig.PLAN_WIDTH_CM)
            plan_length = float(BuildingConfig.PLAN_LENGTH_CM)
            if plan_width <= 0 or plan_length <= 0:
                raise ValueError("Building plan dimensions must be positive.")

            # Dimensionless coordinates relative to the fixed center of loads.
            # LOAD_CENTER_CM and plan dimensions are explicit building inputs;
            # slab insertion points are deliberately not used as centroids.
            dx = (centroids_np[:, 0] - load_center_x) / plan_width
            dy = (centroids_np[:, 1] - load_center_y) / plan_length

            if _total_area > 0:
                area_offset_x = float(np.sum(areas_np * dx) / _total_area)
                area_offset_y = float(np.sum(areas_np * dy) / _total_area)
                area_spread_x = float(np.sum(areas_np * dx**2) / _total_area)
                area_spread_y = float(np.sum(areas_np * dy**2) / _total_area)
                area_coupling_xy = float(np.sum(areas_np * dx * dy) / _total_area)
            else:
                area_offset_x = area_offset_y = 0.0
                area_spread_x = area_spread_y = area_coupling_xy = 0.0

            # Fixed quadrants: intersect the actual polygons instead of assigning
            # each whole pillar by its centroid. A pillar crossing an axis is thus
            # split between adjacent quadrants without a positive-side bias.
            min_x = min(p.bounds[0] for p in self.column_polygons) - 1.0
            min_y = min(p.bounds[1] for p in self.column_polygons) - 1.0
            max_x = max(p.bounds[2] for p in self.column_polygons) + 1.0
            max_y = max(p.bounds[3] for p in self.column_polygons) + 1.0
            quadrant_polygons = (
                box(load_center_x, load_center_y, max_x, max_y),
                box(min_x, load_center_y, load_center_x, max_y),
                box(min_x, min_y, load_center_x, load_center_y),
                box(load_center_x, min_y, max_x, load_center_y),
            )
            q_areas = [
                float(sum(p.intersection(quadrant).area for p in self.column_polygons))
                for quadrant in quadrant_polygons
            ]
            max_q_area_ratio_fixed = (
                float(max(q_areas)) / _total_area if _total_area > 0 else 0.0
            )

            # Kept: geometric slenderness
            valid_mask = areas_np > 0
            if valid_mask.any():
                slend            = perims_np[valid_mask] / np.sqrt(areas_np[valid_mask])
                mean_slenderness = float(slend.mean())
                p95_slenderness  = float(np.percentile(slend, 95))
            else:
                mean_slenderness = p95_slenderness = 0.0

            # Signed stiffness eccentricities relative to the fixed load center.
            # X resistance is weighted by Iyy; Y resistance is weighted by Ixx.
            sum_Iyy = float(iyy_arr.sum())
            sum_Ixx = float(ixx_arr.sum())
            stiffness_ecc_x = (
                float(np.sum(iyy_arr * dx)) / sum_Iyy if sum_Iyy > 0 else 0.0
            )
            stiffness_ecc_y = (
                float(np.sum(ixx_arr * dy)) / sum_Ixx if sum_Ixx > 0 else 0.0
            )

            # These six quantities are structural diagnostics for the current
            # symmetry constraints. They are intentionally excluded from the
            # learning vector because they are constant over this design space.
            self._spatial_diagnostics = {
                "column_area_offset_x_norm": area_offset_x,
                "column_area_offset_y_norm": area_offset_y,
                "column_area_coupling_xy_norm": area_coupling_xy,
                "max_quadrant_area_ratio_fixed": max_q_area_ratio_fixed,
                "stiffness_ecc_x_norm": stiffness_ecc_x,
                "stiffness_ecc_y_norm": stiffness_ecc_y,
            }

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
            self._non_model_diagnostics = {
                "columns_count": float(num_columns),
                "columns_mean_area_cm2": mean_col_area,
                "vol_columns_m3": float(vol_columns_m3),
                "columns_total_perimeter_cm": total_perimeter,
                "columns_mean_perimeter_cm": mean_perimeter,
                "columns_std_perimeter_cm": std_perimeter,
                "beam_definition_count": float(len(beam_lines)),
                "clear_span_count_x": float(len(spans_x)),
                "clear_span_count_y": float(len(spans_y)),
                "beams_mean_clear_span_x_cm": _mean(spans_x),
                "beams_mean_clear_span_y_cm": _mean(spans_y),
                "beams_p95_clear_span_x_cm": _p95(spans_x),
                "beams_p95_clear_span_y_cm": _p95(spans_y),
                "vol_beams_m3": float(total_beam_volume_m3),
                "mean_radius_gyration_min": mean_r_min,
                "min_radius_gyration_global": min_r_global,
            }

            # NEW: Column aspect ratio h/b ≈ sqrt(Iyy/Ixx) — captures section directionality
            aspect_list = [
                float(np.sqrt(abs(Iyy_i) / (abs(Ixx_i) + 1e-9)))
                for Ixx_i, Iyy_i in zip(moments_of_inertia_xx, moments_of_inertia_yy)
            ]
            mean_aspect = float(np.mean(aspect_list)) if aspect_list else 1.0
            std_aspect  = float(np.std(aspect_list))  if aspect_list else 0.0
            max_aspect  = float(np.max(aspect_list))  if aspect_list else 1.0

            features.extend([
                # --- variable fixed-reference spatial distribution (2) ---
                area_spread_x,
                area_spread_y,
                # --- section shape (2) ---
                mean_slenderness,
                p95_slenderness,
                # --- directional radius of gyration (2) ---
                mean_rx, mean_ry,
                # --- directional aspect ratio (3) ---
                mean_aspect, std_aspect, max_aspect,
            ])

        assert len(features) == 27, (
            f"Feature count mismatch: expected 27, got {len(features)}. "
            "Update NeuralNetConfig.INPUT_SIZE and feature_names() if features were added/removed."
        )
        return features

    def get_spatial_diagnostics(self) -> Dict[str, float]:
        """Return fixed-reference symmetry metrics excluded from model input."""
        if not self._spatial_diagnostics:
            self.extract_features()
        return dict(self._spatial_diagnostics)

    def get_diagnostics(self) -> Dict[str, float]:
        """Return all metrics excluded from model input by design."""
        if not self._spatial_diagnostics or not self._non_model_diagnostics:
            self.extract_features()
        return {**self._spatial_diagnostics, **self._non_model_diagnostics}

    @staticmethod
    def feature_names() -> List[str]:
        base = [
            # Block 1 — non-redundant column area stats (4)
            "columns_total_area_cm2",
            "columns_std_area_cm2",
            "columns_min_area_cm2",
            "columns_max_area_cm2",
            # Block 2 — physical clear-span stats by direction (8)
            "beams_total_clear_length_x_cm",
            "beams_total_clear_length_y_cm",
            "beams_std_clear_span_x_cm",
            "beams_std_clear_span_y_cm",
            "beams_max_clear_span_x_cm",
            "beams_max_clear_span_y_cm",
            "beams_span_entropy_x",
            "beams_span_entropy_y",
            # Block 3 — inertia (5)
            "inertia_sum_Ix",
            "inertia_sum_Iy",
            "inertia_mean_Ix",
            "inertia_mean_Iy",
            "inertia_ratio_Iy_over_Ix",
            # Block 4 — compactness (1)
            "columns_mean_compactness",
        ]
        spatial = [
            # Block 6a — variable fixed-reference spatial distribution (2)
            "column_area_spread_x_norm",
            "column_area_spread_y_norm",
            # Block 6b — section shape (2)
            "pillars_mean_slenderness",
            "pillars_p95_slenderness",
            # Block 6c — directional radius of gyration (2)
            "mean_radius_gyration_x",
            "mean_radius_gyration_y",
            # Block 6d — directional column aspect ratio (3)
            "mean_col_aspect_ratio",
            "std_col_aspect_ratio",
            "max_col_aspect_ratio",
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
