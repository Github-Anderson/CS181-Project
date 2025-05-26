# utils.py
import math

COLOR_RED = "RED"
COLOR_GREEN = "GREEN"
COLOR_BLACK = "BLACK"

BOARD_SIZES = [8, 10, 16]

PAWN_NONE = 0
PAWN_GREEN = 1
PAWN_RED = 2

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def in_board(x, y, board_size):
    return 1 <= x <= board_size and 1 <= y <= board_size
