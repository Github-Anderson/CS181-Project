from typing import Tuple
from enum import Enum

class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


DIRECTION_OFFSET = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),

    Direction.UP_LEFT: (-1, -1),
    Direction.UP_RIGHT: (-1, 1),
    Direction.DOWN_LEFT: (1, -1),
    Direction.DOWN_RIGHT: (1, 1),
}
