
# tests/utils/test_geometric_calculator.py

import pytest
from shapely.geometry import Polygon
import numpy as np

# Módulo que estamos testando
from utils.geometric_calculator import get_geometric_concrete_volume, COLUMN_STORY_HEIGHT_M, BEAM_WIDTH_M, BEAM_HEIGHT_M

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

def test_full_calculation_with_columns_and_beams(simple_column_polygons, simple_beam_definitions):
    """
    Testa o cálculo completo com pilares e vigas.
    Verifica se o volume total é a soma correta dos volumes individuais.
    """
    # --- Valores Esperados ---
    # Volume dos Pilares (em m^3)
    # Pilar 1: (1000 cm^2 / 10000) * 3.0m = 0.1 m^2 * 3.0m = 0.3 m^3
    # Pilar 2: (1200 cm^2 / 10000) * 3.0m = 0.12 m^2 * 3.0m = 0.36 m^3
    # Total Pilares: 0.3 + 0.36 = 0.66 m^3
    expected_column_volume = (1000 / 10000 * COLUMN_STORY_HEIGHT_M) + (1200 / 10000 * COLUMN_STORY_HEIGHT_M)

    # Volume das Vigas (em m^3)
    # Viga 1: (300cm * 0.01) * BEAM_WIDTH_M * BEAM_HEIGHT_M = 3.0m * 0.2m * 0.4m = 0.24 m^3
    # Viga 2: (500cm * 0.01) * BEAM_WIDTH_M * BEAM_HEIGHT_M = 5.0m * 0.2m * 0.4m = 0.40 m^3
    # Total Vigas: 0.24 + 0.40 = 0.64 m^3
    expected_beam_volume = (3.0 * BEAM_WIDTH_M * BEAM_HEIGHT_M) + (5.0 * BEAM_WIDTH_M * BEAM_HEIGHT_M)

    expected_total_volume = expected_column_volume + expected_beam_volume

    # --- Execução ---
    total_volume = get_geometric_concrete_volume(
        column_polygons=simple_column_polygons, 
        beam_definitions=simple_beam_definitions
    )

    # --- Verificação ---
    # Usamos np.isclose para comparar floats com segurança
    assert np.isclose(total_volume, expected_total_volume)

def test_calculation_with_only_columns(simple_column_polygons):
    """
    Testa o cálculo quando apenas pilares são fornecidos.
    O volume das vigas deve ser zero.
    """
    expected_column_volume = (1000 / 10000 * COLUMN_STORY_HEIGHT_M) + (1200 / 10000 * COLUMN_STORY_HEIGHT_M)
    
    total_volume = get_geometric_concrete_volume(
        column_polygons=simple_column_polygons,
        beam_definitions=[]  # Lista de vigas vazia
    )
    
    assert np.isclose(total_volume, expected_column_volume)

def test_calculation_with_only_beams(simple_beam_definitions):
    """
    Testa o cálculo quando apenas vigas são fornecidas.
    O volume dos pilares deve ser zero.
    """
    expected_beam_volume = (3.0 * BEAM_WIDTH_M * BEAM_HEIGHT_M) + (5.0 * BEAM_WIDTH_M * BEAM_HEIGHT_M)

    total_volume = get_geometric_concrete_volume(
        column_polygons=[], # Lista de pilares vazia
        beam_definitions=simple_beam_definitions
    )

    assert np.isclose(total_volume, expected_beam_volume)

def test_calculation_with_no_geometry():
    """
    Testa o caso de borda onde não há pilares nem vigas.
    O resultado deve ser 0.
    """
    total_volume = get_geometric_concrete_volume(
        column_polygons=[],
        beam_definitions=[]
    )
    
    assert total_volume == 0.0

def test_calculation_with_invalid_inputs_raises_error():
    """
    Testa se a função levanta um ValueError quando as entradas não são listas.
    """
    with pytest.raises(ValueError, match="Entradas para .* devem ser listas."):
        get_geometric_concrete_volume(column_polygons=None, beam_definitions=[])
    
    with pytest.raises(ValueError, match="Entradas para .* devem ser listas."):
        get_geometric_concrete_volume(column_polygons=[], beam_definitions=None)

