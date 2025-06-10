# game.py
import sys
import argparse
from engine import Engine, BoardGUI
from board import Board
from agents import *
from utils import *

def create_player(player_type_str, color, board_size=8, model_path=None):
    player_type_str = player_type_str.upper()
    player = None

    if player_type_str == "H":
        player = HumanPlayer(color)
    elif player_type_str == "M":
        player = MinimaxPlayer(color)
    elif player_type_str == "MLS": # Minimax with Local Search
        player = MinimaxPlayer(color, use_local_search=True)
    elif player_type_str == "G":
        player = GreedyPlayer(color)
    elif player_type_str == "R":
        player = RandomPlayer(color)
    elif player_type_str == "MCTS":
        player = MCTSPlayer(color)
    elif player_type_str == "AQL":
        player = ApproximateQLearningPlayer(color)
    elif player_type_str == "NAQL":
        player = Neural_ApproximateQLearningPlayer(color, board_size=board_size, if_train=False)
        
        model_to_load = model_path # User-specified path takes precedence
        if not model_to_load: # If no specific path, try defaults based on color
            if color == "RED":
                model_to_load = "sente_agent_ep300.pth"
            elif color == "GREEN":
                model_to_load = "gote_agent_ep300.pth"
            elif color == "BLUE": # Placeholder for default blue model
                model_to_load = "blue_naql_model.pth" 
            elif color == "YELLOW": # Placeholder for default yellow model
                model_to_load = "yellow_naql_model.pth"
        
        if model_to_load:
            try:
                player.load_model(model_to_load)
                print(f"NAQL {color} loaded model: {model_to_load}")
            except FileNotFoundError:
                print(f"Warning: Model file not found for NAQL {color}: {model_to_load}. Player will use initial/random weights.")
            except Exception as e:
                print(f"Warning: Error loading model for NAQL {color} from {model_to_load}: {e}. Player will use initial/random weights.")
        else:
            print(f"Warning: NAQL {color} initialized without a pre-trained model path specified and no default for its color. Player will use initial/random weights.")
    else:
        raise ValueError(f"Unknown player type: {player_type_str}")
    
    if player is None:
        raise ValueError(f"Failed to create player for type: {player_type_str}")
    return player

def read_command(argv):
    parser = argparse.ArgumentParser(description='CS181 Final Project: Halma AI')
    parser.add_argument('-s', '--boardsize', type=int, choices=[4, 8, 10, 12], default=8,
                        help='Board size: 4, 8, 10, or 12.')
    parser.add_argument('-m', '--mode', type=str, choices=['classic', 'score'], default='classic',
                        help='Game mode: classic or score.')
    parser.add_argument('-n', '--numplayers', type=int, choices=[2, 4], default=2,
                        help='Number of players: 2 or 4.')
    parser.add_argument('--max_turns', type=int, default=500, 
                        help='Maximum turns per game before draw/score evaluation.')
    
    player_choices = ['H', 'M', 'MLS', 'G', 'R', "AQL", "MCTS", "NAQL"]
    
    parser.add_argument('-p1', '--player1', type=str, choices=player_choices, default='H',
                        help='Player 1 (RED) type: H, M, MLS, G, R, AQL, MCTS, or NAQL.')
    parser.add_argument('--p1_model', type=str, default=None, help="Path to Player 1's NAQL model.")
    
    parser.add_argument('-p2', '--player2', type=str, choices=player_choices, default='H',
                        help='Player 2 (GREEN) type: H, M, MLS, G, R, AQL, MCTS, or NAQL.')
    parser.add_argument('--p2_model', type=str, default=None, help="Path to Player 2's NAQL model.")

    parser.add_argument('-p3', '--player3', type=str, choices=player_choices, default='H',
                        help='Player 3 (BLUE) type (for 4-player games): H, M, MLS, G, R, AQL, MCTS, NAQL.')
    parser.add_argument('--p3_model', type=str, default=None, help="Path to Player 3's NAQL model.")
    
    parser.add_argument('-p4', '--player4', type=str, choices=player_choices, default='H',
                        help='Player 4 (YELLOW) type (for 4-player games): H, M, MLS, G, R, AQL, MCTS, NAQL.')
    parser.add_argument('--p4_model', type=str, default=None, help="Path to Player 4's NAQL model.")
    
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = read_command(sys.argv[1:])

    boardsize = args.boardsize
    mode = args.mode
    numplayers = args.numplayers
    max_turns = args.max_turns

    player1 = create_player(args.player1, "RED", boardsize, args.p1_model)
    player2 = create_player(args.player2, "GREEN", boardsize, args.p2_model)
    
    players_list = [player1, player2]

    if numplayers == 4:
        player3 = create_player(args.player3, "BLUE", boardsize, args.p3_model)
        player4 = create_player(args.player4, "YELLOW", boardsize, args.p4_model)
        players_list.extend([player3, player4])
    
    board = Board(boardsize, mode, tuple(players_list))

    for player_obj in players_list:
        if hasattr(player_obj, 'set_board') and not isinstance(player_obj, (HumanPlayer, RandomPlayer)):
             player_obj.set_board(board)
        
    engine = Engine(board, max_turns)
    engine.start()