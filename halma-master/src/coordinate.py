# coordinate.py
from utils import COLOR_BLACK, PAWN_NONE

class Coordinate:
    def __init__(self, x, y, color=COLOR_BLACK, pawn=PAWN_NONE):
        self.x = x
        self.y = y
        self.color = color
        self.pawn = pawn

    def setColor(self, color):
        self.color = color

    def setPawn(self, pawn):
        self.pawn = pawn

    def printCoordinate(self):
        print(str(self.x) + str(self.y) + self.color + str(self.pawn))
