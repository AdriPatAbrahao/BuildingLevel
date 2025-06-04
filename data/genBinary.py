import random
from typing import List, Dict

def generate_new_binary_vector(segments: List[Dict], mutation_rate: float = 0.05) -> List[Dict]:
    """
    Generates a new binary vector by randomly mutating the existing one.
    
    Args:
        segments: List of segment dictionaries with 'start', 'end', and 'binary' keys
        mutation_rate: Probability of flipping each segment's binary value
        
    Returns:
        List of new segment dictionaries with potentially mutated binary values
    """
    new_segments = []
    for segment in segments:
        if random.random() < mutation_rate:
            # Create new dictionary with flipped binary value
            new_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "binary": 1 - segment["binary"]  # Flip binary value
            })
        else:
            # Create new dictionary with same values
            new_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "binary": segment["binary"]
            })
    return new_segments