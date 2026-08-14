from . import paths  # Importa o módulo de caminhos
from . import constants # Importa o módulo de constantes

class BuildingConfig:
    NAME = "OptimizedBuilding"
    BUILDING_COORDINATES = constants.DEFAULT_BUILDING_COORDINATES
    SLAB_COORDINATES = constants.DEFAULT_SLAB_COORDINATES

    TQS_RESULTS_FILE = paths.TQS_OUTPUT_DIR / NAME / "ESPACIAL" / "RESDES.HTM"
    TQS_TIMEOUT_SEC = 120  # segundos esperando pelo RESDES.HTM no modo single-thread

class RunConfig:
    """ Configurações que controlam COMO o script executa. """
    # --- Controle de Fluxo ---
    USE_VECTOR_INPUT = True
    USE_GEOMETRIC_ESTIMATE = False
    
    # --- Geração de Dados ---
    NUM_SAMPLES = 2500
    MAX_ITERATION_FACTOR = 5  # M5ultiplicador para tentativas de geração
    TQS_TIMEOUT_SEC = 120    # segundos esperando pelo RESDES.HTM no modo single-thread

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
    INPUT_SIZE = 43  # Número de features extraídas pelo FeatureEngineer atual
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

class ParallelConfig:
    """
    Controls the parallel TQS data-generation pool.

    Set ``ENABLED = True`` to run multiple TQS instances in parallel during
    training-data collection.  Each worker gets its own isolated building
    directory (``{BASE_NAME}_01``, ``_02``, …) so no file-level locking is
    ever needed.

    Tuning guidelines
    -----------------
    * ``NUM_WORKERS``: start at 2 and raise by 1 until CPU/RAM saturate.
      Each worker uses ~1 GB RAM and one TQS licence seat.
    * ``BASE_NAME``: slots are ``{BASE_NAME}_01``, ``_02``, … — the ``_NN``
      suffix keeps them distinct from ``BuildingConfig.NAME`` (no suffix).
    * ``TIMEOUT_SEC``: increase for very large buildings or slow machines.
    """
    ENABLED           = False          # False = caminho sequencial (OptimizedBuilding); True = pool paralelo
    NUM_WORKERS       = 1              # relevante apenas quando ENABLED=True
    BASE_NAME         = "OptimizedBuilding"  # slot prefix → OptimizedBuilding_01, _02 …
    TIMEOUT_SEC       = 180            # per-job RESDES.HTM wait timeout (seconds)
    MAX_CONSECUTIVE_TIMEOUTS = 3       # stop collection only after this many timeouts in a row
    VALIDITY_CHECK_DLL = False  # TQS DLL check: desativado — investigar erros classe==2 após treino


class ObjectiveConfig:
    """Parameters used by the optimization objective function."""
    CONCRETE_PRICE_M3 = 10.0
    STEEL_PRICE_KG = 100.0
    FORM_PRICE_M2 = 10.0  # R$/m² — custo de forma (fôrma) dos pilares; AJUSTAR para o valor real de mercado
    LENGTH_STEP_CM = 20.0
    INVALID_PROB_THRESHOLD = 0.5
    INVALID_COST_PENALTY = 1_000_000
    STOP_MIN_STEEL_KG = 0.0
    STOP_MAX_INVALID_PROB = 0.1
    MAX_TIME_SEC = 600

    STEEL_MIN_KGF   = None   # bounds desativados — validade determinada exclusivamente pelo TQS DLL
    STEEL_MAX_KGF   = None
    CONCRETE_MIN_M3 = None

    
    


