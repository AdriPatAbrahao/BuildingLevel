# utils/geometric_calculator.py

import numpy as np
from typing import List, Dict
from shapely.geometry import Polygon, LineString, MultiLineString

# --- Constantes ---
# Fator de conversão
CM_TO_M = 0.01 # 1 cm = 0.01 m

# --- Dimensões Fixas (em CM) ---
# !!! AJUSTE ESTES VALORES CONFORME SEU PROJETO !!!
COLUMN_STORY_HEIGHT_CM = 300.0  # Ex: Pé direito de 3 metros
COLUMN_FIXED_WIDTH_CM = 20.0   # Ex: Largura fixa da seção do pilar de 20cm

BEAM_WIDTH_CM = 20.0          # Ex: Largura da viga de 20cm
BEAM_HEIGHT_CM = 40.0         # Ex: Altura da viga de 40cm

# Converter dimensões fixas para metros para cálculos de volume
COLUMN_STORY_HEIGHT_M = COLUMN_STORY_HEIGHT_CM * CM_TO_M
COLUMN_FIXED_WIDTH_M = COLUMN_FIXED_WIDTH_CM * CM_TO_M
BEAM_WIDTH_M = BEAM_WIDTH_CM * CM_TO_M
BEAM_HEIGHT_M = BEAM_HEIGHT_CM * CM_TO_M
# --- Fim das Constantes ---


def calculate_column_geometric_volume(column_polygons: List[Polygon]) -> float:
    """
    Calculates the total geometric volume for a list of pillar polygons.

    This function operates on the final polygon shapes of the columns, which
    are typically the result of union operations on overlapping segments. It
    calculates volume as (Polygon Base Area * Story Height).

    Args:
        column_polygons (List[Polygon]): A list of Shapely Polygon objects
                                         representing the base of each pillar.
                                         The coordinates of the polygons are
                                         assumed to be in centimeters (cm).

    Returns:
        float: The total volume of all pillars in cubic meters (m³).
    """
    total_column_volume_m3 = 0.0

    if not column_polygons:
        print("  [Geometric] Warning: No column polygons provided for volume calculation.")
        return 0.0

    for column in column_polygons:
        # Ensure the provided object is a valid, non-empty polygon
        if column and isinstance(column, Polygon) and not column.is_empty:
            
            # 1. Get the base area from the Shapely polygon.
            base_area_cm2 = column.area
            base_area_m2 = base_area_cm2 / 10000.0  # Convert cm² to m²

            # 3. Calculate the volume for the individual pillar in cubic meters (m³).
            column_volume_m3 = base_area_m2 * COLUMN_STORY_HEIGHT_M
            
            total_column_volume_m3 += column_volume_m3
        else:
            print(f"  [Geometric] Warning: An invalid or empty pillar polygon was skipped: {column}")

    return total_column_volume_m3

def calculate_beams_geometric_volume(beam_definitions: List[Dict]) -> float:
    """
    Calcula o volume geométrico total para uma lista de definições de vigas.

    Args:
        beam_definitions (List[Dict]): Lista de dicionários representando as vigas.
                                       Cada dicionário deve ter 'node_1': (x1, y1) e 'node_2': (x2, y2) em cm.

    Returns:
        float: Volume total das vigas em metros cúbicos (m³).
    """
    total_beam_volume_m3 = 0.0
    for beam in beam_definitions:
        node1 = beam.get("node_1")
        node2 = beam.get("node_2")

        if node1 and node2:
            # Calcula o comprimento da viga em cm usando a distância euclidiana
            length_cm = np.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2) - 2*BEAM_WIDTH_CM

            if length_cm > 0:
                length_m = length_cm * CM_TO_M
                # Calcula o volume da viga individual em m³
                volume_m3 = length_m * BEAM_WIDTH_M * BEAM_HEIGHT_M
                total_beam_volume_m3 += volume_m3
            else:
                print(f"Aviso: Viga com comprimento zero encontrada: {beam}")
        else:
            print(f"Aviso: Definição de viga inválida encontrada: {beam}")


    return total_beam_volume_m3


def calculate_beams_geometric_volume_with_subtractions(beam_definitions: List[Dict], column_polygons: List[Polygon]) -> float:
    """
    Calculates beam volume subtracting the portions that run inside columns.

    - Computes beam centerline length in cm
    - Subtracts the total length of intersections between the centerline and each column polygon
    - Converts effective length to meters and multiplies by beam cross-section (BxH)
    """
    total_beam_volume_m3 = 0.0
    if not beam_definitions:
        return 0.0

    for beam in beam_definitions:
        node1 = beam.get("node_1")
        node2 = beam.get("node_2")
        if not (node1 and node2):
            continue

        # Base length (cm)
        length_cm = float(np.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2))

        subtract_cm = 0.0
        try:
            beam_line = LineString([node1, node2])
            for col in (column_polygons or []):
                inter = beam_line.intersection(col)
                if inter.is_empty:
                    continue
                if isinstance(inter, LineString):
                    subtract_cm += float(inter.length)
                elif isinstance(inter, MultiLineString):
                    subtract_cm += float(sum(seg.length for seg in inter.geoms))
        except Exception:
            pass

        effective_cm = max(length_cm - subtract_cm, 0.0)
        if effective_cm <= 0.0:
            continue

        length_m = effective_cm * CM_TO_M
        volume_m3 = length_m * BEAM_WIDTH_M * BEAM_HEIGHT_M
        total_beam_volume_m3 += volume_m3

    return total_beam_volume_m3

def get_geometric_concrete_volume(column_polygons: List[Dict], beam_definitions: List[Dict]) -> float:
    """
    Calcula o volume geométrico total estimado de concreto para pilares e vigas.

    IMPORTANTE: Este cálculo é uma aproximação e ignora a sobreposição de
    volumes nas interseções entre pilares e vigas. O resultado pode ser
    maior que o volume calculado por softwares como o TQS.

    Args:
        column_polygons (List[Dict]): Lista de dicionários dos polígonos de pilares.
        beam_definitions (List[Dict]): Lista de dicionários das definições de vigas.

    Returns:
        float: Volume total estimado de concreto em metros cúbicos (m³).
    """
    # Validação de entrada para garantir que os argumentos são listas
    if not isinstance(column_polygons, list) or not isinstance(beam_definitions, list):
        raise ValueError("Entradas para 'column_polygons' e 'beam_definitions' devem ser listas.")

    volume_pilares = calculate_column_geometric_volume(column_polygons)
    volume_vigas = calculate_beams_geometric_volume_with_subtractions(beam_definitions, column_polygons)

    print(f"[Geométrico] Vol. Pilares: {volume_pilares:.4f} m³ | Vol. Vigas: {volume_vigas:.4f} m³")

    return volume_pilares + volume_vigas
