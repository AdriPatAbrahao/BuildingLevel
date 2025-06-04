from pathlib import Path

class BuildingConfig:
    NAME = "OptimizedBuilding"
    COORDINATES = (
        (10., 10.), (360., 10.),
        (10., 410.), (360., 410.), (710., 410.),
        (360., 810.), (710., 810.)
    )
    NUM_SAMPLES = 1000 # Increased to 100 buildings
    IMAGE_PATH = "images" 
    RESULTS_PATH = Path(r"C:\TQS\OptimizedBuilding\ESPACIAL\RESDES.HTM")
    USE_VECTOR_INPUT = True  # Flag to switch between input methods
    USE_GEOMETRIC_VOLUME_ESTIMATE = True # Mude para False para usar TQS

    # Output directories
    RESULTS_DIR = Path("results")
    PLOTS_DIR = RESULTS_DIR / "plots"
    
    # Ensure directories exist on startup
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    
    # Variation parameters
    MIN_LENGTH_CHANGE = 50.0  # Minimum change in length (cm)
    MAX_LENGTH_CHANGE = 150.0  # Maximum change in length (cm)
    VARIATION_PROBABILITY = 0.4  # 40% chance to modify each segment
    
    # Constraints
    MIN_SEGMENT_LENGTH = 200.0  # Minimum allowed segment length (cm)
    MAX_SEGMENT_LENGTH = 1000.0  # Maximum allowed segment length (cm)


