from . import paths  # Importa o módulo de caminhos
from . import constants # Importa o módulo de constantes

class BuildingConfig:
    NAME = "OptimizedBuilding"
    BUILDING_COORDINATES = constants.DEFAULT_BUILDING_COORDINATES
    SLAB_COORDINATES = constants.DEFAULT_SLAB_COORDINATES

    TQS_RESULTS_FILE = paths.TQS_OUTPUT_DIR / NAME / "ESPACIAL" / "RESDES.HTM"
    TQS_ERROR_REPORT_FILE = paths.TQS_OUTPUT_DIR / NAME / "ESPACIAL" / "PGLOERR.HTM"
    TQS_FATAL_ERROR_MARKER = "<H4> Existem mensagens de erros graves:</H4>"

class RunConfig:
    """ Configurações que controlam COMO o script executa. """
    # --- Controle de Fluxo ---
    USE_VECTOR_INPUT = True
    USE_GEOMETRIC_ESTIMATE = False
    
    # --- Geração de Dados ---
    NUM_SAMPLES = 5
    MAX_ITERATION_FACTOR = 2 # Multiplicador para tentativas de geração


class NeuralNetConfig:
    """ Hiperparâmetros e arquitetura da Rede Neural. """
    # --- Arquitetura ---
    INPUT_SIZE = 24  # Idealmente, deveria ser determinado dinamicamente
    HIDDEN_LAYERS = [128, 128, 64]
    DROPOUT_RATE = 0.2
    OUTPUT_SIZE = 2

    # --- Treinamento ---
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 500
    BATCH_SIZE = 32

    # --- Divisão de Dados e Validação ---
    TEST_SPLIT_RATIO = 0.15
    VALIDATION_SPLIT_RATIO = 0.2
    EARLY_STOPPING_PATIENCE = 50

    
    


