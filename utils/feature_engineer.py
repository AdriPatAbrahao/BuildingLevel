
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
        
        # Column features — single numpy pass (avoids 5 separate iterations).
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
            total_column_area,
            num_columns,
            mean_col_area,
            std_col_area,
            min_col_area,
            max_col_area,
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
        
        try:
            from sklearn.neighbors import KDTree
            from sklearn.cluster import DBSCAN
        except Exception:
            KDTree = None
            DBSCAN = None

        # Build centroid/area/perimeter arrays once; reused throughout.
        if self.column_polygons:
            _cg = [p.centroid for p in self.column_polygons]
            centroids_np = np.array([(float(c.x), float(c.y)) for c in _cg], dtype=float)
            areas_np     = col_areas          # already computed above
            perims_np    = np.array([p.length for p in self.column_polygons], dtype=float)
        else:
            centroids_np = np.empty((0, 2), dtype=float)
            areas_np     = col_areas
            perims_np    = np.empty(0, dtype=float)

        # Keep list-of-tuples alias so downstream zip() calls are unchanged.
        centroids = list(map(tuple, centroids_np)) if centroids_np.size else []

        if centroids:
            cx = float(centroids_np[:, 0].mean())
            cy = float(centroids_np[:, 1].mean())
            dists = np.sqrt((centroids_np[:, 0] - cx)**2 + (centroids_np[:, 1] - cy)**2)
            mean_dist = float(np.mean(dists))
            median_dist = float(np.median(dists))
            max_dist = float(np.max(dists))
            excentricity_global = float(np.sum(areas_np * dists))

            # Quadrant analysis — vectorised
            xc = centroids_np[:, 0]
            yc = centroids_np[:, 1]
            q_idx = np.where(xc >= cx,
                             np.where(yc >= cy, 0, 3),
                             np.where(yc >= cy, 1, 2))
            q_counts = [int(np.sum(q_idx == q)) for q in range(4)]
            q_areas  = [float(np.sum(areas_np[q_idx == q])) for q in range(4)]
            total_count = float(len(centroids))
            total_area  = float(areas_np.sum())
            max_q_count_ratio = float(max(q_counts)) / total_count if total_count > 0 else 0.0
            max_q_area_ratio  = float(max(q_areas))  / total_area  if total_area  > 0 else 0.0

            # Slenderness — vectorised
            valid_mask = areas_np > 0
            if valid_mask.any():
                slend = perims_np[valid_mask] / np.sqrt(areas_np[valid_mask])
                mean_slenderness = float(slend.mean())
                p95_slenderness  = float(np.percentile(slend, 95))
            else:
                mean_slenderness = p95_slenderness = 0.0

            # KDTree — built once and reused for both neighbour distances and DBSCAN eps.
            k_neighbors = 4
            kd_mean = kd_std = kd_ratio_min_max = 0.0
            _kd = None  # shared KDTree instance
            if KDTree is not None and len(centroids) > k_neighbors:
                _kd = KDTree(centroids_np)
                d, _ = _kd.query(centroids_np, k=k_neighbors + 1)
                dn = d[:, 1:]
                avg_neighbor = np.mean(dn, axis=1)
                kd_mean = float(avg_neighbor.mean())
                kd_std  = float(avg_neighbor.std())
                if avg_neighbor.size > 0:
                    kd_ratio_min_max = float(avg_neighbor.min() / (avg_neighbor.max() + 1e-9))

            n_clusters = 0
            largest_cluster_size = 0
            proportion_in_clusters = 0.0
            if DBSCAN is not None and len(centroids) >= 5:
                # Reuse existing KDTree for eps estimation (no second build).
                if _kd is not None:
                    dnn, _ = _kd.query(centroids_np, k=2)
                    median_nn = float(np.median(dnn[:, 1]))
                elif KDTree is not None:
                    _kd2 = KDTree(centroids_np)
                    dnn, _ = _kd2.query(centroids_np, k=2)
                    median_nn = float(np.median(dnn[:, 1]))
                else:
                    median_nn = float(np.median(dists))
                eps = max(median_nn * 0.5, 1e-6)
                labels = DBSCAN(eps=eps, min_samples=3).fit(centroids_np).labels_
                valid = labels >= 0
                unique_labels = [l for l in set(labels) if l >= 0]
                n_clusters = int(len(unique_labels))
                counts = [int(np.sum(labels == l)) for l in unique_labels]
                largest_cluster_size = int(max(counts)) if counts else 0
                proportion_in_clusters = float(np.sum(valid)) / float(len(labels)) if len(labels) > 0 else 0.0

            span_max = float(np.max(effective_lengths)) if effective_lengths else 0.0
            span_p95 = float(np.percentile(effective_lengths, 95)) if effective_lengths else 0.0
            if effective_lengths:
                bins = max(10, min(50, int(np.sqrt(len(effective_lengths)))))
                hist, _ = np.histogram(np.array(effective_lengths), bins=bins, density=True)
                p = hist / (np.sum(hist) + 1e-9)
                span_entropy = float(-np.sum(p * np.log(p + 1e-9)))
            else:
                span_entropy = 0.0

            features.extend([
                mean_dist,
                median_dist,
                max_dist,
                excentricity_global,
                max_q_count_ratio,
                max_q_area_ratio,
                mean_slenderness,
                p95_slenderness,
                kd_mean,
                kd_std,
                kd_ratio_min_max,
                n_clusters,
                largest_cluster_size,
                proportion_in_clusters,
                span_max,
                span_p95,
                span_entropy,
            ])

        return features
    @staticmethod
    def feature_names() -> List[str]:
        base = [
            "columns_total_area_cm2",
            "columns_count",
            "columns_mean_area_cm2",
            "columns_std_area_cm2",
            "columns_min_area_cm2",
            "columns_max_area_cm2",
            "beams_total_effective_length_cm",
            "beams_count",
            "beams_mean_effective_length_cm",
            "beams_std_effective_length_cm",
            "beams_max_effective_length_cm",
            "inertia_sum_Ix",
            "inertia_sum_Iy",
            "inertia_mean_Ix",
            "inertia_mean_Iy",
            "inertia_ratio_Iy_over_Ix",
            "vol_columns_m3",
            "vol_beams_m3",
            "columns_total_perimeter_cm",
            "columns_mean_perimeter_cm",
            "columns_std_perimeter_cm",
            "columns_mean_compactness",
        ]
        spatial = [
            "pillars_mean_dist_to_center",
            "pillars_median_dist_to_center",
            "pillars_max_dist_to_center",
            "pillars_excentricity_global",
            "pillars_max_quadrant_count_ratio",
            "pillars_max_quadrant_area_ratio",
            "pillars_mean_slenderness",
            "pillars_p95_slenderness",
            "pillars_kd_mean_spacing",
            "pillars_kd_std_spacing",
            "pillars_kd_ratio_min_over_max",
            "pillars_dbscan_num_clusters",
            "pillars_dbscan_largest_cluster_size",
            "pillars_dbscan_proportion_in_clusters",
            "beams_span_max_cm",
            "beams_span_p95_cm",
            "beams_span_entropy",
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
