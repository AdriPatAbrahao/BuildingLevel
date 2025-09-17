
# tests/geometry/test_length_input_processor.py

import pytest
from unittest.mock import patch
from shapely.geometry import Polygon
import random
import numpy as np

# Módulo que estamos testando
from geometry.length_input_processor import LengthProcessor

# --- Fixtures: Dados de teste reutilizáveis ---

@pytest.fixture
def simple_length_segments():
    """
    Fornece uma lista de segmentos de entrada simples.
    Dois segmentos verticais que devem se conectar para formar um pilar.
    Um segmento horizontal isolado que formará outro pilar.
    """
    return [
        # Pilar 1 (dois segmentos conectados)
        {"start": (10, 10), "end": (10, 60), "length": 50, "maxlength": 100, "binary": 1},
        {"start": (10, 60), "end": (10, 110), "length": 50, "maxlength": 100, "binary": 1},
        # Pilar 2 (isolado)
        {"start": (100, 50), "end": (150, 50), "length": 50, "maxlength": 100, "binary": 1}
    ]



# --- Testes para process_segments ---

# Usamos o 'patch' para substituir a configuração externa por nosso mock
@patch('geometry.length_input_processor.VectorConfig.WALL_SEGMENTS', new=[
    {"start": (10, 0), "end": (10, 200)},   # Parede vertical alinhada com o Pilar 1
    {"start": (80, 50), "end": (180, 50)}  # Parede horizontal alinhada com o Pilar 2
])
def test_process_segments_groups_columns_and_creates_beams(simple_length_segments):
    """
    Testa o fluxo principal de process_segments.
    Verifica se os pilares são agrupados corretamente e se as vigas são geradas.
    """
    processor = LengthProcessor()
    
    # Execução
    column_polygons, beam_groups = processor.process_segments(simple_length_segments)
    
    # --- Verificação dos Pilares ---
    assert len(column_polygons) == 2, "Deveria encontrar 2 grupos de pilares"
    
    # Calcula a área esperada (aproximada)
    # Pilar 1: (10,10) -> (10,110). Comprimento 100. Espessura (BEAM_THICKNESS) é 20.
    # Área esperada ~ 100 * 20 = 2000
    # Pilar 2: (100,50) -> (150,50). Comprimento 50. Espessura 20.
    # Área esperada ~ 50 * 20 = 1000
    # A união pode alterar um pouco a área, então usamos uma tolerância.
    total_area = sum(p.area for p in column_polygons)
    assert np.isclose(total_area, 3000, rtol=0.05), "A área total dos polígonos dos pilares está incorreta"

    # --- Verificação das Vigas ---
    assert len(beam_groups) == 2, "Deveria criar 2 vigas, uma para cada parede"
    
    # Verifica se as vigas foram criadas com os nós corretos (aproximado)
    # A lógica exata depende de _find_beam_locations, mas podemos checar a orientação
    beam1 = beam_groups[0]
    beam2 = beam_groups[1]
    assert beam1['node_1'][0] == beam1['node_2'][0], "A viga 1 deveria ser vertical"
    assert beam2['node_1'][1] == beam2['node_2'][1], "A viga 2 deveria ser horizontal"

def test_process_segments_with_empty_input():
    """
    Testa o caso de borda com uma lista de segmentos vazia.
    """
    processor = LengthProcessor()
    column_polygons, beam_groups = processor.process_segments([])
    
    assert column_polygons == []
    assert beam_groups == []

# --- Testes para generate_variation ---

def test_generate_variation_changes_length():
    """
    Testa se a função de variação realmente altera o comprimento de um segmento.
    """
    processor = LengthProcessor()
    segments = [
        {"start": (0, 0), "end": (0, 50), "length": 50, "maxlength": 100, "binary": 1}
    ]
    
    # Usamos random.seed para garantir que o resultado seja sempre o mesmo para o teste
    np.random.seed(42)
    random.seed(42)
    
    new_segments = processor.generate_variation(segments)
    
    assert new_segments[0]["length"] != segments[0]["length"], "O comprimento deveria ter sido alterado"
    assert new_segments[0]["length"] > 50
    assert new_segments[0]["length"] <= 100, "O novo comprimento não deve exceder o maxlength"

def test_generate_variation_updates_endpoint():
    """
    Testa se a atualização do comprimento também atualiza o ponto final (end).
    """
    processor = LengthProcessor()
    segments = [
        {"start": (10, 20), "end": (10, 70), "length": 50, "maxlength": 100, "binary": 1}
    ]
    
    np.random.seed(42)
    random.seed(42)
    
    new_segments = processor.generate_variation(segments)
    
    new_length = new_segments[0]["length"]
    new_end = new_segments[0]["end"]
    
    # O ponto final esperado para um segmento vertical que começa em (10, 20)
    expected_end = (10, 20 + new_length)
    
    assert np.isclose(new_end[0], expected_end[0])
    assert np.isclose(new_end[1], expected_end[1])

def test_generate_variation_with_no_possible_change():
    """
    Testa o que acontece se nenhum segmento puder ser variado (length == maxlength).
    A função deve retornar os segmentos originais sem alteração.
    """
    processor = LengthProcessor()
    segments = [
        {"start": (0, 0), "end": (0, 100), "length": 100, "maxlength": 100, "binary": 1}
    ]
    
    new_segments = processor.generate_variation(segments)
    
    assert new_segments[0]["length"] == segments[0]["length"]
    assert new_segments[0]["end"] == segments[0]["end"]

