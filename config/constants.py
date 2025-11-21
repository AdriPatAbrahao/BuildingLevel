DEFAULT_BUILDING_COORDINATES  = (
    (10., 10.), (360., 10.),
    (10., 410.), (360., 410.), (710., 410.),
    (360., 810.), (710., 810.)
)

DEFAULT_SLAB_COORDINATES = (
        (180., 105.), (540., 105.),
        (180., 615.), (540., 615.),
    )

# --- Parâmetros Físicos e de Engenharia (em cm ou unidades base) ---
DEFAULT_BEAM_WIDTH_CM = 20.0
DEFAULT_BEAM_HEIGHT_CM = 40.0
DEFAULT_SLAB_THICKNESS_CM: float = 12.0

# Regra de divisão de viga por pilar intermediário
SPLIT_BEAM_COLUMN_THRESHOLD_CM: float = 70.0

# --- Cargas Padrão (unidades consistentes, ex: tf/m ou tf/m²) ---
DEFAULT_BEAM_DEAD_LOAD_TF_M = 2.0
DEFAULT_BEAM_LIVE_LOAD_TF_M = 1.0
DEFAULT_SLAB_DEAD_LOAD_TF_M2: float = 1.00  
DEFAULT_SLAB_LIVE_LOAD_TF_M2: float = 1.00 

# Slab Defaults
DEFAULT_SLAB_ANGLE_DEGREES: float = 0.0
DEFAULT_SLAB_LOAD_CASE: int = 1

DEFAULT_SLAB_VOLUME: float = 6.0192