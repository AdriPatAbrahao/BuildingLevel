from typing import List, Tuple, Dict

class VectorConfig:
    # Building walls definition (counterclockwise from bottom-left)
    WALL_SEGMENTS: List[Dict] = [
        {"start": (0, 10), "end": (720, 10)},     
        {"start": (0, 410), "end": (720, 410)},
        {"start": (0, 810), "end": (720, 810)},
        {"start": (10, 0), "end": (10, 820)}, 
        {"start": (360, 0), "end": (360, 820)}, 
        {"start": (710, 0), "end": (710, 820)} 
    ]
    
    # CSV column names
    VECTOR_CSV_COLUMNS = ["x", "y", "dx", "dy", "length"]