
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
        
        try:
            from sklearn.neighbors import KDTree
            from sklearn.cluster import DBSCAN
        except Exception:
            KDTree = None
            DBSCAN = None

        centroids = []
        areas = []
        perimeters = []
        for p in self.column_polygons:
            c = p.centroid
            centroids.append((float(c.x), float(c.y)))
            areas.append(float(p.area))
            perimeters.append(float(p.length))

        if centroids:
            cx = float(np.mean([c[0] for c in centroids]))
            cy = float(np.mean([c[1] for c in centroids]))
            dists = np.sqrt((np.array([c[0] for c in centroids]) - cx)**2 + (np.array([c[1] for c in centroids]) - cy)**2)
            mean_dist = float(np.mean(dists))
            median_dist = float(np.median(dists))
            max_dist = float(np.max(dists))
            excentricity_global = float(np.sum(np.array(areas) * dists)) if areas else 0.0

            q_counts = [0, 0, 0, 0]
            q_areas = [0.0, 0.0, 0.0, 0.0]
            for (x, y), a in zip(centroids, areas or [0.0]*len(centroids)):
                qi = 0
                if x >= cx and y >= cy:
                    qi = 0
                elif x < cx and y >= cy:
                    qi = 1
                elif x < cx and y < cy:
                    qi = 2
                else:
                    qi = 3
                q_counts[qi] += 1
                q_areas[qi] += a
            total_count = float(len(centroids))
            total_area = float(np.sum(areas)) if areas else 0.0
            max_q_count_ratio = float(max(q_counts)) / total_count if total_count > 0 else 0.0
            max_q_area_ratio = float(max(q_areas)) / total_area if total_area > 0 else 0.0

            slenderness_vals = []
            for A, P in zip(areas or [], perimeters or []):
                if A > 0:
                    slenderness_vals.append(float(P / np.sqrt(A)))
            mean_slenderness = float(np.mean(slenderness_vals)) if slenderness_vals else 0.0
            p95_slenderness = float(np.percentile(slenderness_vals, 95)) if slenderness_vals else 0.0

            k_neighbors = 4
            kd_mean = 0.0
            kd_std = 0.0
            kd_ratio_min_max = 0.0
            if KDTree is not None and len(centroids) > k_neighbors:
                arr = np.array(centroids)
                kd = KDTree(arr)
                d, _ = kd.query(arr, k=k_neighbors+1)
                dn = d[:, 1:]
                avg_neighbor = np.mean(dn, axis=1)
                kd_mean = float(np.mean(avg_neighbor))
                kd_std = float(np.std(avg_neighbor))
                if avg_neighbor.size > 0:
                    kd_ratio_min_max = float(np.min(avg_neighbor) / (np.max(avg_neighbor) + 1e-9))

            n_clusters = 0
            largest_cluster_size = 0
            proportion_in_clusters = 0.0
            if DBSCAN is not None and len(centroids) >= 5:
                arr = np.array(centroids)
                if KDTree is not None:
                    kd = KDTree(arr)
                    dnn, _ = kd.query(arr, k=2)
                    median_nn = float(np.median(dnn[:, 1]))
                else:
                    median_nn = float(np.median(dists))
                eps = max(median_nn * 0.5, 1e-6)
                labels = DBSCAN(eps=eps, min_samples=3).fit(arr).labels_
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
