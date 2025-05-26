# game.py
import sys
import argparse
from engine import Engine

def readCommand(argv):
    parser = argparse.ArgumentParser(description='CS181 Final Project: Halma AI')
    parser.add_argument('--boardsize', type=int, choices=[8, 10, 16], default=8,
                        help='Board size: 8, 10, or 16.')
    parser.add_argument('--timelimit', type=int, default=30,
                        help='Time limit per move in seconds.')
    parser.add_argument('--gamesystem', type=str, choices=['CMD', 'GUI'], default='CMD',
                        help='Game system: CMD (command line) or GUI (graphical interface).')
    parser.add_argument('--player1', type=str, choices=['M', 'MLS', 'G', 'R'], default='M',
                        help='Player 1 type: M, MLS, G, or R.')
    parser.add_argument('--player2', type=str, choices=['M', 'MLS', 'G', 'R'], default='M',
                        help='Player 2 type: M, MLS, G, or R.')
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = readCommand(sys.argv[1:])

    boardsize = args.boardsize
    timelimit = args.timelimit
    system = args.gamesystem.upper()
    player1 = args.player1.upper()
    player2 = args.player2.upper()

    # Validate command-line arguments
    if boardsize not in [8, 10, 16]:
        print("Error: boardsize should be 8, 10, or 16.")
        sys.exit(1)

    if not isinstance(boardsize, int) or not isinstance(timelimit, int):
        print("Error: boardsize and timelimit should be integers.")
        sys.exit(1)

    if system not in ["CMD", "GUI"]:
        print("Error: gamesystem should be CMD or GUI.")
        sys.exit(1)

    supported_agents = ["M", "MLS", "G", "R"]
    if player1 not in supported_agents or player2 not in supported_agents:
        print("Error: player1 and player2 should be one of M, MLS, G, or R.")
        sys.exit(1)

    game = Engine(boardsize, timelimit, None, system, (player1, player2))
