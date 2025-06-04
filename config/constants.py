from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "Building1.csv"
VECTOR_CSV_PATH = PROJECT_ROOT / "data" / "Building1b.csv"
CSV_FINAL_PATH = PROJECT_ROOT / "results" / "final_tqs_input_vectors.csv"


# Building parameters
BEAM_THICKNESS = 20 

# TQS parameters
BUILDING_NAME = "OptimizedBuilding"
COORDINATES = (
    (10., 10.), (360., 10.),
    (10., 410.), (360., 410.), (710., 410.),
    (360., 810.), (710., 810.)
)


DEFAULT_TRAIN_SPLIT_RATIO = 0.8
# Safety factor to prevent excessively long data collection loops
MAX_ITERATION_FACTOR = 2

# Beam Defaults
DEFAULT_BEAM_WIDTH_CM = 20.0
DEFAULT_BEAM_HEIGHT_CM = 40.0
DEFAULT_BEAM_DEAD_LOAD_TF_M = 0.2
DEFAULT_BEAM_LIVE_LOAD_TF_M = 0.3

# Slab Defaults
DEFAULT_SLAB_THICKNESS_CM: float = 12.0
DEFAULT_SLAB_DEAD_LOAD_TF_M2: float = 0.15  # Carga permanente (morta)
DEFAULT_SLAB_LIVE_LOAD_TF_M2: float = 0.10  # Carga acidental (viva)
DEFAULT_SLAB_ANGLE_DEGREES: float = 0.0
DEFAULT_SLAB_LOAD_CASE: int = 1