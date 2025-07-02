from pathlib import Path

class BuildingConfig:
    NAME = "OptimizedBuilding"
    COORDINATES = (
        (180., 105.), (540., 105.),
        (180., 615.), (540., 615.),
    )
    NUM_SAMPLES = 20 # Increased to 100 buildings
    IMAGE_PATH = "images" 
    RESULTS_PATH = Path(r"C:\TQS\OptimizedBuilding\ESPACIAL\RESDES.HTM")
    USE_VECTOR_INPUT = True  # Flag to switch between input methods
    USE_GEOMETRIC_VOLUME_ESTIMATE = False # Mude para False para usar TQS

    # Output directories
    RESULTS_DIR = Path("results")
    PLOTS_DIR = RESULTS_DIR / "plots"
    
    # Ensure directories exist on startup
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    
    


