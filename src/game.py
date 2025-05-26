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
    parser.add_argument('-p1', '--player1', type=str, choices=['H', 'M', 'MLS', 'G', 'R', "AQL", "MCTS", "NAQL"], default='H',
                        help='Player 1 type: H, M, MLS, G, R, AQL, MCTS, or NAQL.')
    parser.add_argument('-p2', '--player2', type=str, choices=['H', 'M', 'MLS', 'G', 'R', "AQL", "MCTS", "NAQL"], default='H',
                        help='Player 2 type: H, M, MLS, G, R, AQL, MCTS, or NAQL.')
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = readCommand(sys.argv[1:])

    boardsize = args.boardsize
    player1 = args.player1.upper()
    player2 = args.player2.upper()

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
    elif player1 == "NAQL":
        player1 = Neural_ApproximateQLearningPlayer("RED", if_train=False)
        player1.load_model("sente_agent_ep300.pth")
        print("Weight successfully loaded!")

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
        player2 = ApproximateQLearningPlayer("GREEN")
    elif player2 == "NAQL":
        player2 = Neural_ApproximateQLearningPlayer("GREEN", if_train=False)
        player2.load_model("gote_agent_ep300.pth")
        print("Weight successfully loaded!")

    board = Board(boardsize, (player1, player2))

    if not isinstance(player1, HumanPlayer) and not isinstance(player1, RandomPlayer):
        player1.set_board(board)
    if not isinstance(player2, HumanPlayer) and not isinstance(player2, RandomPlayer):
        player2.set_board(board)
    
    engine = Engine(board)
    engine.start()
