
# tests/utils/test_geometric_calculator.py

import pytest
from shapely.geometry import Polygon
import numpy as np
from config.constants import DEFAULT_SLAB_VOLUME

# Módulo que estamos testando
from utils.geometric_calculator import (
    get_geometric_concrete_volume,
    calculate_column_geometric_volume,
    calculate_column_formwork_area,
    calculate_beams_geometric_volume,
    calculate_beams_geometric_volume_with_subtractions,
    COLUMN_STORY_HEIGHT_M,
    BEAM_WIDTH_M,
    BEAM_HEIGHT_M,
)

# --- Fixtures: Dados de teste reutilizáveis ---

@pytest.fixture
def simple_column_polygons():
    """
    Fornece uma lista com dois polígonos de pilares simples para os testes.
    - Um quadrado de 20x50 cm.
    - Um retângulo de 30x40 cm.
    """
    # Coordenadas em cm
    return [
        Polygon([(0, 0), (20, 0), (20, 50), (0, 50)]),  # Área: 20 * 50 = 1000 cm^2
        Polygon([(100, 100), (130, 100), (130, 140), (100, 140)]) # Área: 30 * 40 = 1200 cm^2
    ]

@pytest.fixture
def simple_beam_definitions():
    """
    Fornece uma lista com duas definições de vigas simples para os testes.
    - Uma viga horizontal de 300 cm.
    - Uma viga diagonal (3-4-5) de 500 cm.
    """
    # Coordenadas em cm
    return [
        {"node_1": (0, 0), "node_2": (300, 0)},  # Comprimento: 300 cm
        {"node_1": (0, 0), "node_2": (300, 400)}  # Comprimento: 500 cm (Teorema de Pitágoras)
    ]

# --- Testes para get_geometric_concrete_volume ---

def test_full_calculation_with_columns_beam_and_slabs():
    """
    Valida o contrato completo: pilares + vigas face a face + lajes.

    A viga vai do centro de dois pilares 20 x 20 cm. Dos 300 cm de eixo,
    10 cm em cada extremidade pertencem aos pilares, restando 280 cm.
    """
    columns = [
        Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        Polygon([(290, -10), (310, -10), (310, 10), (290, 10)]),
    ]
    beams = [{"node_1": (0, 0), "node_2": (300, 0)}]

    expected_column_volume = 2 * (400 / 10000) * COLUMN_STORY_HEIGHT_M
    expected_beam_volume = 2.8 * BEAM_WIDTH_M * BEAM_HEIGHT_M
    expected_total_volume = (
        expected_column_volume + expected_beam_volume + DEFAULT_SLAB_VOLUME
    )

    total_volume = get_geometric_concrete_volume(columns, beams)

    assert np.isclose(total_volume, expected_total_volume)


def test_beam_component_distinguishes_raw_and_face_to_face_volume():
    columns = [
        Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)]),
        Polygon([(290, -10), (310, -10), (310, 10), (290, 10)]),
    ]
    beams = [{"node_1": (0, 0), "node_2": (300, 0)}]

    raw_volume = calculate_beams_geometric_volume(beams)
    effective_volume = calculate_beams_geometric_volume_with_subtractions(
        beams, columns
    )

    assert np.isclose(raw_volume, 0.24)
    assert np.isclose(effective_volume, 0.224)


def test_fixed_slab_volume_matches_four_clear_panels():
    # Quatro painéis: (3,50 - 0,20) x (4,00 - 0,20) x 0,12 m.
    expected_slab_volume = 4 * 3.30 * 3.80 * 0.12
    assert np.isclose(DEFAULT_SLAB_VOLUME, expected_slab_volume)

def test_calculation_with_only_columns(simple_column_polygons):
    """
    Sem vigas, o total contém pilares e as lajes fixas.
    """
    expected_column_volume = (1000 / 10000 * COLUMN_STORY_HEIGHT_M) + (1200 / 10000 * COLUMN_STORY_HEIGHT_M)
    
    total_volume = get_geometric_concrete_volume(
        column_polygons=simple_column_polygons,
        beam_definitions=[]  # Lista de vigas vazia
    )
    
    assert np.isclose(total_volume, expected_column_volume + DEFAULT_SLAB_VOLUME)

def test_calculation_with_only_beams(simple_beam_definitions):
    """
    Sem pilares, não há descontos; o total contém vigas brutas e lajes.
    """
    expected_beam_volume = (3.0 * BEAM_WIDTH_M * BEAM_HEIGHT_M) + (5.0 * BEAM_WIDTH_M * BEAM_HEIGHT_M)

    total_volume = get_geometric_concrete_volume(
        column_polygons=[], # Lista de pilares vazia
        beam_definitions=simple_beam_definitions
    )

    assert np.isclose(total_volume, expected_beam_volume + DEFAULT_SLAB_VOLUME)

def test_calculation_without_columns_or_beams_returns_slab_volume():
    """
    O contrato total mantém os quatro painéis de laje fixos.
    """
    total_volume = get_geometric_concrete_volume(
        column_polygons=[],
        beam_definitions=[]
    )
    
    assert np.isclose(total_volume, DEFAULT_SLAB_VOLUME)

def test_calculation_with_invalid_inputs_raises_error():
    """
    Testa se a função levanta um ValueError quando as entradas não são listas.
    """
    with pytest.raises(ValueError, match="Entradas para .* devem ser listas."):
        get_geometric_concrete_volume(column_polygons=None, beam_definitions=[])

    with pytest.raises(ValueError, match="Entradas para .* devem ser listas."):
        get_geometric_concrete_volume(column_polygons=[], beam_definitions=None)


# --- Testes para calculate_column_formwork_area ---

def test_formwork_area_with_columns(simple_column_polygons):
    """
    Testa o cálculo da área de forma (perímetro x pé-direito) dos pilares.
    - Pilar 1: retângulo 20x50 cm -> perímetro = 2*(20+50) = 140 cm
    - Pilar 2: retângulo 30x40 cm -> perímetro = 2*(30+40) = 140 cm
    """
    expected_area = (140 / 100 * COLUMN_STORY_HEIGHT_M) + (140 / 100 * COLUMN_STORY_HEIGHT_M)

    total_area = calculate_column_formwork_area(simple_column_polygons)

    assert np.isclose(total_area, expected_area)


def test_formwork_area_with_no_columns():
    """
    Testa o caso de borda onde não há pilares. O resultado deve ser 0.
    """
    assert calculate_column_formwork_area([]) == 0.0

