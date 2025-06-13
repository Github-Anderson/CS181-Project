from typing import Tuple
from enum import Enum

MAX_TURNS = 100
MIN_JUMP_SCALAR = 0.0
MAX_JUMP_SCALAR = 10.0
MAX_TURN_DRAW_FRAC_THRESHOLD = 0.50
BINARY_SEARCH_STEPS = 10
DEFAULT_ROUND = 100
jump_scalar = 1.0
jump_scalars = [1.0, 1.2, 1.5, 2.0, 5.0, 10.0]

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
