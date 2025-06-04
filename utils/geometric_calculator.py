# utils/geometric_calculator.py

import numpy as np
from typing import List, Dict

# --- Constantes ---
# Fator de conversão
CM_TO_M = 0.01 # 1 cm = 0.01 m

# --- Dimensões Fixas (em CM) ---
# !!! AJUSTE ESTES VALORES CONFORME SEU PROJETO !!!
PILLAR_STORY_HEIGHT_CM = 300.0  # Ex: Pé direito de 3 metros
PILLAR_FIXED_WIDTH_CM = 20.0   # Ex: Largura fixa da seção do pilar de 20cm

BEAM_WIDTH_CM = 20.0          # Ex: Largura da viga de 20cm
BEAM_HEIGHT_CM = 40.0         # Ex: Altura da viga de 40cm

# Converter dimensões fixas para metros para cálculos de volume
PILLAR_STORY_HEIGHT_M = PILLAR_STORY_HEIGHT_CM * CM_TO_M
PILLAR_FIXED_WIDTH_M = PILLAR_FIXED_WIDTH_CM * CM_TO_M
BEAM_WIDTH_M = BEAM_WIDTH_CM * CM_TO_M
BEAM_HEIGHT_M = BEAM_HEIGHT_CM * CM_TO_M
# --- Fim das Constantes ---


def calculate_pillars_geometric_volume(pillar_segments: List[Dict]) -> float:
    """
    Calcula o volume geométrico total para uma lista de segmentos de pilares.

    Args:
        pillar_segments (List[Dict]): Lista de dicionários representando os pilares.
                                       Cada dicionário deve ter a chave 'length' (em cm).

    Returns:
        float: Volume total dos pilares em metros cúbicos (m³).
    """
    total_pillar_volume_m3 = 0.0
    for pillar in pillar_segments:
        # Pega o comprimento variável do pilar (em cm)
        variable_length_cm = pillar.get("length")

        if variable_length_cm is not None and variable_length_cm > 0:
            variable_length_m = variable_length_cm * CM_TO_M
            # Calcula o volume do pilar individual em m³
            volume_m3 = PILLAR_STORY_HEIGHT_M * PILLAR_FIXED_WIDTH_M * variable_length_m
            total_pillar_volume_m3 += volume_m3
        else:
             print(f"Aviso: Segmento de pilar inválido encontrado: {pillar}")

    return total_pillar_volume_m3


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
            length_cm = np.sqrt((node2[0] - node1[0])**2 + (node2[1] - node1[1])**2)

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


def get_geometric_concrete_volume(pillar_segments: List[Dict], beam_definitions: List[Dict]) -> float:
    """
    Calcula o volume geométrico total estimado de concreto para pilares e vigas.

    IMPORTANTE: Este cálculo é uma aproximação e ignora a sobreposição de
    volumes nas interseções entre pilares e vigas. O resultado pode ser
    maior que o volume calculado por softwares como o TQS.

    Args:
        pillar_segments (List[Dict]): Lista de dicionários dos segmentos de pilares.
        beam_definitions (List[Dict]): Lista de dicionários das definições de vigas.

    Returns:
        float: Volume total estimado de concreto em metros cúbicos (m³).
    """
    volume_pilares = calculate_pillars_geometric_volume(pillar_segments)
    volume_vigas = calculate_beams_geometric_volume(beam_definitions)

    print(f"[Geométrico] Vol. Pilares: {volume_pilares:.4f} m³ | Vol. Vigas: {volume_vigas:.4f} m³")

    return volume_pilares + volume_vigas