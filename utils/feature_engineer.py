
from typing import List, Dict
from shapely.geometry import Polygon, LineString, MultiLineString, GeometryCollection, box
from shapely.geometry.polygon import orient
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

        Feature layout (23 total, schema v11):
          [0-3]   Non-redundant column area stats (4)
          [4-9]   Non-redundant clear-span stats separated into X and Y (6)
          [10-12] Inertia (sum_Ix, sum_Iy, ratio) (3)
          [13-16] Fixed-reference area and stiffness spreads (4)
          [17-18] Section-shape factors (2)
          [19]    Mean directional radius-of-gyration balance (1)
          [20-22] Logarithmic directional aspect descriptors (3)
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

        # --- Block 2: Non-redundant physical clear-span features (6) ---
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
            _std(spans_x),
            _std(spans_y),
            _max(spans_x),
            _max(spans_y),
            _span_entropy(spans_x),
            _span_entropy(spans_y),
        ])

        # --- Block 3: Inertia features (3) ---
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

        features.extend([sum_Ix, sum_Iy, inertia_ratio])

        # --- Derived geometric volumes (diagnostic only) ---
        vol_columns_m3 = calculate_column_geometric_volume(self.column_polygons)

        # --- Column compactness and perimeter stats (diagnostic only) ---
        perims = [p.length for p in self.column_polygons]
        compact = [
            float(4.0 * np.pi * p.area / (p.length * p.length))
            for p in self.column_polygons if p.length > 0
        ]
        total_perimeter = float(np.sum(perims)) if perims else 0.0
        mean_perimeter = float(np.mean(perims)) if perims else 0.0
        std_perimeter = float(np.std(perims)) if perims else 0.0
        mean_compactness = float(np.mean(compact)) if compact else 0.0

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

            # Dimensionless section-shape factor P/sqrt(A). This is not the
            # member slenderness L_e/r used in structural verification.
            valid_mask = areas_np > 0
            if valid_mask.any():
                shape_factors = perims_np[valid_mask] / np.sqrt(areas_np[valid_mask])
                mean_shape_factor = float(shape_factors.mean())
                p95_shape_factor = float(np.percentile(shape_factors, 95))
            else:
                mean_shape_factor = p95_shape_factor = 0.0

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

            # Stiffness distribution relevant to torsional response. Translation
            # in X uses Iyy with its perpendicular lever arm dy; translation in
            # Y uses Ixx with its perpendicular lever arm dx.
            stiffness_spread_x_response = (
                float(np.sum(iyy_arr * dy**2)) / sum_Iyy if sum_Iyy > 0 else 0.0
            )
            stiffness_spread_y_response = (
                float(np.sum(ixx_arr * dx**2)) / sum_Ixx if sum_Ixx > 0 else 0.0
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
            radius_sum = mean_rx + mean_ry
            directional_radius_balance = (
                (mean_ry - mean_rx) / radius_sum if radius_sum > 0.0 else 0.0
            )
            self._non_model_diagnostics = {
                "columns_count": float(num_columns),
                "columns_mean_area_cm2": mean_col_area,
                "vol_columns_m3": float(vol_columns_m3),
                "columns_total_perimeter_cm": total_perimeter,
                "columns_mean_perimeter_cm": mean_perimeter,
                "columns_std_perimeter_cm": std_perimeter,
                "columns_mean_compactness": mean_compactness,
                "beam_definition_count": float(len(beam_lines)),
                "clear_span_count_x": float(len(spans_x)),
                "clear_span_count_y": float(len(spans_y)),
                "beams_total_clear_length_x_cm": total_x,
                "beams_total_clear_length_y_cm": total_y,
                "beams_mean_clear_span_x_cm": _mean(spans_x),
                "beams_mean_clear_span_y_cm": _mean(spans_y),
                "beams_p95_clear_span_x_cm": _p95(spans_x),
                "beams_p95_clear_span_y_cm": _p95(spans_y),
                "vol_beams_m3": float(total_beam_volume_m3),
                "inertia_mean_Ix": mean_Ix,
                "inertia_mean_Iy": mean_Iy,
                "mean_radius_gyration_x": mean_rx,
                "mean_radius_gyration_y": mean_ry,
                "mean_radius_gyration_min": mean_r_min,
                "min_radius_gyration_global": min_r_global,
            }

            # Logarithmic aspect log(b/h) = 0.5*log(Iyy/Ixx). A physical 90°
            # rotation changes only its sign, so horizontal and vertical
            # elongation are treated symmetrically.
            log_aspect_list = []
            for Ixx_i, Iyy_i in zip(moments_of_inertia_xx, moments_of_inertia_yy):
                abs_ixx = abs(Ixx_i)
                abs_iyy = abs(Iyy_i)
                log_aspect_list.append(
                    float(0.5 * np.log(abs_iyy / abs_ixx))
                    if abs_ixx > 0.0 and abs_iyy > 0.0
                    else 0.0
                )
            mean_log_aspect = (
                float(np.mean(log_aspect_list)) if log_aspect_list else 0.0
            )
            std_log_aspect = (
                float(np.std(log_aspect_list)) if log_aspect_list else 0.0
            )
            max_abs_log_aspect = (
                float(np.max(np.abs(log_aspect_list))) if log_aspect_list else 0.0
            )

            features.extend([
                # --- variable fixed-reference spatial distribution (4) ---
                area_spread_x,
                area_spread_y,
                stiffness_spread_x_response,
                stiffness_spread_y_response,
                # --- dimensionless section-shape factors (2) ---
                mean_shape_factor,
                p95_shape_factor,
                # --- normalized mean directional radius balance (1) ---
                directional_radius_balance,
                # --- rotation-symmetric logarithmic aspect descriptors (3) ---
                mean_log_aspect, std_log_aspect, max_abs_log_aspect,
            ])

        assert len(features) == 23, (
            f"Feature count mismatch: expected 23, got {len(features)}. "
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
            # Block 2 — non-redundant physical clear-span stats (6)
            "beams_std_clear_span_x_cm",
            "beams_std_clear_span_y_cm",
            "beams_max_clear_span_x_cm",
            "beams_max_clear_span_y_cm",
            "beams_span_entropy_x",
            "beams_span_entropy_y",
            # Block 3 — inertia (3)
            "inertia_sum_Ix",
            "inertia_sum_Iy",
            "inertia_ratio_Iy_over_Ix",
        ]
        spatial = [
            # Block 4 — variable fixed-reference spatial distribution (4)
            "column_area_spread_x_norm",
            "column_area_spread_y_norm",
            "columns_stiffness_spread_x_response_norm",
            "columns_stiffness_spread_y_response_norm",
            # Block 5 — dimensionless section-shape factors P/sqrt(A) (2)
            "columns_mean_shape_factor",
            "columns_p95_shape_factor",
            # Block 6 — normalized mean directional radius balance (1)
            "columns_mean_radius_gyration_directional_balance",
            # Block 7 — logarithmic directional column aspect (3)
            "columns_mean_log_aspect_ratio",
            "columns_std_log_aspect_ratio",
            "columns_max_abs_log_aspect_ratio",
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

    # Normalize ring orientation: exterior counterclockwise (positive signed
    # integrals), holes clockwise (negative signed integrals). Physical inertia
    # must not depend on the input vertex ordering.
    normalized = orient(polygon, sign=1.0)

    def _ring_integrals(coords) -> tuple[float, float, float]:
        points = np.asarray(coords, dtype=float)
        x = points[:, 0]
        y = points[:, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        signed_area = 0.5 * float(np.sum(cross))
        ixx_origin = (1.0 / 12.0) * float(
            np.sum((y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2) * cross)
        )
        iyy_origin = (1.0 / 12.0) * float(
            np.sum((x[:-1] ** 2 + x[:-1] * x[1:] + x[1:] ** 2) * cross)
        )
        return signed_area, ixx_origin, iyy_origin

    signed_area, ixx_origin, iyy_origin = _ring_integrals(
        normalized.exterior.coords
    )
    for interior in normalized.interiors:
        hole_area, hole_ixx, hole_iyy = _ring_integrals(interior.coords)
        signed_area += hole_area
        ixx_origin += hole_ixx
        iyy_origin += hole_iyy

    centroid = normalized.centroid
    ixx_centroid = ixx_origin - signed_area * centroid.y**2
    iyy_centroid = iyy_origin - signed_area * centroid.x**2
    return float(ixx_centroid), float(iyy_centroid)
