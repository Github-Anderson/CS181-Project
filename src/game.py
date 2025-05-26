# game.py
import sys
import argparse
from engine import Engine, BoardGUI
from board import Board
from agents import *
from utils import *

def readCommand(argv):
    parser = argparse.ArgumentParser(description='CS181 Final Project: Halma AI')
    parser.add_argument('-s', '--boardsize', type=int, choices=[4, 8, 10, 12], default=8,
                        help='Board size: 4, 8, 10, or 12.')
    parser.add_argument('-n', '--numplayers', type=int, choices=[2, 4], default=2,
                        help='Number of players: 2 or 4.')
    parser.add_argument('-p1', '--player1', type=str, choices=['H', 'M', 'MLS', 'G', 'R', "AQL", "MCTS"], default='H',
                        help='Player 1 type: H, M, MLS, G, R, AQL, or MCTS.')
    parser.add_argument('-p2', '--player2', type=str, choices=['H', 'M', 'MLS', 'G', 'R', "AQL", "MCTS"], default='H',
                        help='Player 2 type: H, M, MLS, G, R, AQL, or MCTS.')
    parser.add_argument('-p3', '--player3', type=str, choices=['H', 'M', 'MLS', 'G', 'R', "AQL", "MCTS"], default='H',
                        help='Player 3 type: H, M, MLS, G, R, AQL, or MCTS.')
    parser.add_argument('-p4', '--player4', type=str, choices=['H', 'M', 'MLS', 'G', 'R', "AQL", "MCTS"], default='H',
                        help='Player 4 type: H, M, MLS, G, R, AQL, or MCTS.')
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = readCommand(sys.argv[1:])

    boardsize = args.boardsize
    numplayers = args.numplayers

    player1 = args.player1.upper()
    player2 = args.player2.upper()

    if numplayers == 4:
        player3 = args.player3.upper()
        player4 = args.player4.upper()

    if player1 == "H":
        player1 = HumanPlayer("RED")
    elif player1 == "M":
        player1 = MinimaxPlayer("RED")
    elif player1 == "MLS":
        player1 = MinimaxPlayer("RED", use_local_search=True)
    elif player1 == "G":
        player1 = GreedyPlayer("RED")
    elif player1 == "R":
        player1 = RandomPlayer("RED")
    elif player1 == "MCTS":
        player1 = MCTSPlayer("RED")
    elif player1 == "AQL":
        player1 = ApproximateQLearningPlayer("RED")

    if player2 == "H":
        player2 = HumanPlayer("GREEN")
    elif player2 == "M":
        player2 = MinimaxPlayer("GREEN")
    elif player2 == "MLS":
        player2 = MinimaxPlayer("GREEN", use_local_search=True)
    elif player2 == "G":
        player2 = GreedyPlayer("GREEN")
    elif player2 == "R":
        player2 = RandomPlayer("GREEN")
    elif player2 == "MCTS":
        player2 = MCTSPlayer("GREEN")
    elif player2 == "AQL":
        player2 = ApproximateQLearningPlayer("RED")
    
    if numplayers == 4:
        if player3 == "H":
            player3 = HumanPlayer("BLUE")
        elif player3 == "M":
            player3 = MinimaxPlayer("BLUE")
        elif player3 == "MLS":
            player3 = MinimaxPlayer("BLUE", use_local_search=True)
        elif player3 == "G":
            player3 = GreedyPlayer("BLUE")
        elif player3 == "R":
            player3 = RandomPlayer("BLUE")
        elif player3 == "MCTS":
            player3 = MCTSPlayer("BLUE")
        elif player3 == "AQL":
            player3 = ApproximateQLearningPlayer("RED")

        if player4 == "H":
            player4 = HumanPlayer("YELLOW")
        elif player4 == "M":
            player4 = MinimaxPlayer("YELLOW")
        elif player4 == "MLS":
            player4 = MinimaxPlayer("YELLOW", use_local_search=True)
        elif player4 == "G":
            player4 = GreedyPlayer("YELLOW")
        elif player4 == "R":
            player4 = RandomPlayer("YELLOW")
        elif player4 == "MCTS":
            player4 = MCTSPlayer("YELLOW")
        elif player4 == "AQL":
            player4 = ApproximateQLearningPlayer("RED")

    if numplayers == 2:
        board = Board(boardsize, (player1, player2))
    elif numplayers == 4:
        board = Board(boardsize, (player1, player2, player3, player4))

    if not isinstance(player1, HumanPlayer) and not isinstance(player1, RandomPlayer):
        player1.set_board(board)
    if not isinstance(player2, HumanPlayer) and not isinstance(player2, RandomPlayer):
        player2.set_board(board)
        
    if numplayers == 4:
        if not isinstance(player3, HumanPlayer) and not isinstance(player3, RandomPlayer):
            player3.set_board(board)
        if not isinstance(player4, HumanPlayer) and not isinstance(player4, RandomPlayer):
            player4.set_board(board)

    engine = Engine(board)
    engine.start()
