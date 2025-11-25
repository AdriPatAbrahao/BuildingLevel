from . import paths  # Importa o módulo de caminhos
from . import constants # Importa o módulo de constantes

class BuildingConfig:
    NAME = "OptimizedBuilding"
    BUILDING_COORDINATES = constants.DEFAULT_BUILDING_COORDINATES
    SLAB_COORDINATES = constants.DEFAULT_SLAB_COORDINATES

    TQS_RESULTS_FILE = paths.TQS_OUTPUT_DIR / NAME / "ESPACIAL" / "RESDES.HTM"

class RunConfig:
    """ Configurações que controlam COMO o script executa. """
    # --- Controle de Fluxo ---
    USE_VECTOR_INPUT = True
    USE_GEOMETRIC_ESTIMATE = False
    
    # --- Geração de Dados ---
    NUM_SAMPLES = 600
    NUMSAMPLES = NUM_SAMPLES
    MAX_ITERATION_FACTOR = 3 # Multiplicador para tentativas de geração

    # --- Métricas e Logging ---
    METRICS_LOG_FORMAT = "json"
    LOG_EPOCH_RESOURCES = True
    LOG_EPOCH_GRADIENTS = "last_batch"
    LOG_CLASSIFIER_METRICS = True
    SEED = 42
    MONITORING_ENABLED = True
    MONITOR_INTERVAL_MIN = 30
    CHECKPOINTS_ENABLED = True
    CHECKPOINT_INTERVAL_MIN = 60
    RESUME_FROM_CHECKPOINT = True
    ALERT_STUCK_THRESHOLD_MIN = 90


class NeuralNetConfig:
    """ Hiperparâmetros e arquitetura da Rede Neural. """
    # --- Arquitetura ---
    INPUT_SIZE = 22  # Ajustado ao conjunto atual de features
    HIDDEN_LAYERS = [128, 128, 64]
    DROPOUT_RATE = 0.2
    OUTPUT_SIZE = 1

    # --- Treinamento ---
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 500
    BATCH_SIZE = 32

    # --- Divisão de Dados e Validação ---
    TEST_SPLIT_RATIO = 0.15
    VALIDATION_SPLIT_RATIO = 0.2
    EARLY_STOPPING_PATIENCE = 50

    # --- Training Enhancements ---
    LOSS_TYPE = "mse"  # options: "mse", "huber"
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER = True
    LR_SCHEDULER_PATIENCE = 10
    LR_SCHEDULER_FACTOR = 0.5

class ObjectiveConfig:
    """Parameters used by the optimization objective function."""
    CONCRETE_PRICE_M3 = 10.0
    STEEL_PRICE_KG = 100.0
    LENGTH_STEP_CM = 20.0
    INVALID_PROB_THRESHOLD = 0.5
    INVALID_COST_PENALTY = 1_000_000
    STOP_MIN_STEEL_KG = 0.0
    STOP_MAX_INVALID_PROB = 0.1
    MAX_TIME_SEC = 600

    
    


