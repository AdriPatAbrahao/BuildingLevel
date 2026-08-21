from . import paths  # Importa o módulo de caminhos
from . import constants # Importa o módulo de constantes

class BuildingConfig:
    NAME = "OptimizedBuilding"
    BUILDING_COORDINATES = constants.DEFAULT_BUILDING_COORDINATES
    SLAB_COORDINATES = constants.DEFAULT_SLAB_COORDINATES

    # Building-specific geometric reference. Review this block when changing
    # the building; slab insertion points are not slab centroids.
    LOAD_CENTER_CM = (360.0, 410.0)
    PLAN_WIDTH_CM = 720.0
    PLAN_LENGTH_CM = 820.0

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
    CHECKPOINT_INTERVAL_MIN = 10
    RESUME_FROM_CHECKPOINT = True
    ALERT_STUCK_THRESHOLD_MIN = 90


class NeuralNetConfig:
    """ Hiperparâmetros e arquitetura da Rede Neural. """
    # --- Arquitetura ---
    INPUT_SIZE = 23  # Número de features extraídas pelo FeatureEngineer atual
    FEATURE_SCHEMA_VERSION = 11
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


class DataSplitConfig:
    """Regras do protocolo experimental para o conjunto de regressão."""

    # As primeiras 230 amostras já participaram da análise exploratória de
    # features e hiperparâmetros. Elas podem ser usadas no desenvolvimento,
    # mas nunca na estimativa final de generalização.
    PREUSED_DEVELOPMENT_PREFIX_SAMPLES = 230
    # The 230 valid pilot configurations correspond to 253 classifier rows
    # (valid + invalid TQS outcomes). These rows may support development, but
    # must never enter the untouched final classifier test set.
    PREUSED_CLASSIFIER_PREFIX_SAMPLES = 253
    REGRESSION_STRATIFICATION_BINS = 10

class ParallelConfig:
    """
    Controls the parallel TQS data-generation pool.

    Set ``ENABLED = True`` to run multiple TQS instances in parallel during
    training-data collection.  Each worker gets its own isolated building
    directory (``{BASE_NAME}_01``, ``_02``, …) so no file-level locking is
    ever needed.

    Tuning guidelines
    -----------------
    * ``NUM_WORKERS``: the 2026-08-16 concurrency pilot validated 2 workers on
      6 comparison cases (1.67x speedup, zero result divergence —
      ``outputs/validation/tqs_concurrency_pilot/summary.json``), but two
      real production canary runs on 2026-08-17 (``--num-samples 260`` off
      the 230-sample checkpoint) showed real instability under actual load:
      successful jobs took 143-151s vs. the pilot's steady-state 62-67s, and
      3 consecutive jobs then hung to the full 180s timeout each. This
      happened even after re-provisioning ``TrainBuild815_02`` cleanly via
      TQS's own ``SaveAs`` (ruling out a bad manual duplication) and after a
      TQS modal dialog (``TCZOFEXZ.EXE ... executável não existe``) suggested
      the two simultaneous engine instances may race over the *shared*
      ``T:\\TQSW\\EXEC\\X64`` executable directory — unlike the per-slot
      building directories, that path is NOT isolated per worker. Reverted
      to 1 worker for the real 2500-sample collection; do not raise this
      again without resolving that shared-executable-directory risk first.
    * ``ALLOW_SIMULTANEOUS_TQS``: required whenever ``NUM_WORKERS > 1`` —
      ``TQSWorkerPool`` raises otherwise. Disables each worker's own
      pre-run global ``NTQSHTM.EXE`` termination so workers don't kill each
      other's in-flight process. A *timeout* still kills ``NTQSHTM.EXE`` by
      image name globally (there is no PID-isolated kill yet), so every
      in-flight job can be lost when one worker hangs — this is why
      ``MAX_CONSECUTIVE_TIMEOUTS`` also counts worker-level job failures,
      not just queue-wait timeouts. Keep ``False`` at ``NUM_WORKERS = 1`` so
      the single worker keeps its normal pre-run cleanup of stray
      ``NTQSHTM.EXE`` processes.
    * Each slot directory (``{BASE_NAME}_01``, ``_02``, …) must already exist
      as a TQS building duplicated from the same validated seed model before
      raising ``NUM_WORKERS`` — TQSWorkerPool does not create slots itself.
    * ``BASE_NAME``: slots are ``{BASE_NAME}_01``, ``_02``, … — the ``_NN``
      suffix keeps them distinct from ``BuildingConfig.NAME`` (no suffix).
    * ``TIMEOUT_SEC``: increase for very large buildings or slow machines.
    """
    ENABLED           = False          # 2026-08-17: temporariamente sequencial p/ testar OptimizedBuilding como controle
    NUM_WORKERS       = 1              # 2 workers revertido em 2026-08-17: instável em produção real, ver acima
    ALLOW_SIMULTANEOUS_TQS = False     # só True ao validar NUM_WORKERS>1 de novo
    BASE_NAME         = "TrainBuild815"  # slots: TrainBuild815_01, TrainBuild815_02, ...
    TIMEOUT_SEC       = 180            # per-job RESDES.HTM wait timeout (seconds)
    MAX_CONSECUTIVE_TIMEOUTS = 3       # stop collection after this many consecutive timeouts/failures
    # Required for collection: unavailable/failed DLL checks reject the sample.
    VALIDITY_CHECK_DLL = True


class ObjectiveConfig:
    """Parameters used by the optimization objective function."""
    CONCRETE_PRICE_M3 = 450.0
    STEEL_PRICE_KG = 12.0
    FORM_PRICE_M2 = 70.0  # R$/m² — custo de forma (fôrma) dos pilares
    # Constructive discretization shared by data generation and optimization.
    LENGTH_STEP_CM = 5.0
    INVALID_PROB_THRESHOLD = 0.5
    INVALID_COST_PENALTY = 1_000_000
    STOP_MIN_STEEL_KG = 0.0
    STOP_MAX_INVALID_PROB = 0.1
    MAX_TIME_SEC = 600

    STEEL_MIN_KGF   = None   # bounds desativados — validade determinada exclusivamente pelo TQS DLL
    STEEL_MAX_KGF   = None
    CONCRETE_MIN_M3 = None

    
    


