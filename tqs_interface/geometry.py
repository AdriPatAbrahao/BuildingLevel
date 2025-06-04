from dataclasses import dataclass
from typing import Tuple

@dataclass
class Segment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    binary: int

    @property
    def is_horizontal(self) -> bool:
        return self.start[1] == self.end[1]

    @property
    def is_vertical(self) -> bool:
        return self.start[0] == self.end[0]