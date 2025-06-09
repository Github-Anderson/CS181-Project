import sys
from engine import *
import argparse
from collections import defaultdict
import time
import random

from board import Board 
from agents import (
    Player, MinimaxPlayer, GreedyPlayer, RandomPlayer, MCTSPlayer, 
    ApproximateQLearningPlayer, Neural_ApproximateQLearningPlayer, DefaultPlayer # DefaultPlayer is used by board.clone()
)

# Helper function to create player instances from strings
def create_player(player_type_str, color, board_size=8, model_path=None):
    player_type_str = player_type_str.upper()
    player = None

    if player_type_str == "M":
        player = MinimaxPlayer(color)
    elif player_type_str == "MLS":
        player = MinimaxPlayer(color, use_local_search=True)
    elif player_type_str == "G":
        player = GreedyPlayer(color)
    elif player_type_str == "R":
        player = RandomPlayer(color)
    elif player_type_str == "MCTS":
        # You might want to configure MCTS simulations/time_limit via args too
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
            # Add more defaults if other colors use NAQL and have standard models, e.g., for 4-player games
        
        if model_to_load:
            try:
                player.load_model(model_to_load)
                print(f"NAQL {color} loaded model: {model_to_load}")
            except FileNotFoundError:
                print(f"Warning: Model file not found for NAQL {color}: {model_to_load}")
            except Exception as e:
                print(f"Warning: Error loading model for NAQL {color} from {model_to_load}: {e}")
        else:
            print(f"Warning: NAQL {color} initialized without a pre-trained model path specified and no default for its color.")
    else:
        raise ValueError(f"Unknown player type: {player_type_str}")
    
    if player is None:
        raise ValueError(f"Failed to create player for type: {player_type_str}")
    return player

def run_single_match(p1_config, p2_config, p3_config, p4_config, board_size, game_mode, num_players):
    players_for_board = []
    
    # Create Player 1 (RED)
    p1 = create_player(p1_config['type'], "RED", board_size, p1_config.get('model'))
    players_for_board.append(p1)

    # Create Player 2 (GREEN)
    p2 = create_player(p2_config['type'], "GREEN", board_size, p2_config.get('model'))
    players_for_board.append(p2)

    if num_players == 4:
        # Create Player 3 (BLUE)
        p3 = create_player(p3_config['type'], "BLUE", board_size, p3_config.get('model'))
        players_for_board.append(p3)
        # Create Player 4 (YELLOW)
        p4 = create_player(p4_config['type'], "YELLOW", board_size, p4_config.get('model'))
        players_for_board.append(p4)
    
    board = Board(board_size, game_mode, players_for_board)

    # Set board for AI players that need it (after board and players are fully initialized)
    for player_obj in players_for_board:
        if hasattr(player_obj, 'set_board') and not isinstance(player_obj, RandomPlayer):
             player_obj.set_board(board) # RandomPlayer's get_action doesn't use self.board

    engine = HeadlessEngine(board)
    winner_player_object = engine.run_game()

    if winner_player_object:
        return winner_player_object.color # "RED", "GREEN", etc.
    return "DRAW"

def read_command(argv):
    parser = argparse.ArgumentParser(description="CS181 Halma Autograder")
    parser.add_argument('-r', '--repetitions', type=int, default=10, help="Number of games to run for each matchup.")
    parser.add_argument('-s', '--boardsize', type=int, choices=[4, 8, 10, 12], default=8, help='Board size.')
    parser.add_argument('-m', '--mode', type=str, choices=['classic', 'score'], default='classic', help='Game mode.')
    parser.add_argument('--max_turns', type=int, default=500, help='Maximum turns per game before draw.')

    player_choices = ['M', 'MLS', 'G', 'R', 'MCTS', 'AQL', 'NAQL']
    parser.add_argument('-p1', '--player1_type', type=str, choices=player_choices, default='G', help='Player 1 (RED) type.')
    parser.add_argument('--p1_model', type=str, default=None, help="Path to Player 1's NAQL model.")
    
    parser.add_argument('-p2', '--player2_type', type=str, choices=player_choices, default='G', help='Player 2 (GREEN) type.')
    parser.add_argument('--p2_model', type=str, default=None, help="Path to Player 2's NAQL model.")

    args = parser.parse_args()

def main():
    args = read_command(sys.argv[1:])

    # Prepare player configurations
    p1_config = {'type': args.player1_type, 'model': args.p1_model, 'name': f"{args.player1_type}(RED)"}
    p2_config = {'type': args.player2_type, 'model': args.p2_model, 'name': f"{args.player2_type}(GREEN)"}
    p3_config = {'type': args.player3_type, 'model': args.p3_model, 'name': f"{args.player3_type}(BLUE)"}
    p4_config = {'type': args.player4_type, 'model': args.p4_model, 'name': f"{args.player4_type}(YELLOW)"}

    win_counts = defaultdict(int)
    total_game_time = 0

    print(f"--- Autograder Setup ---")
    print(f"Board Size: {args.boardsize}, Mode: {args.mode}, Num Players: {args.numplayers}")
    print(f"Repetitions: {args.repetitions}, Max Turns: {args.max_turns}")
    print(f"Player 1 (RED): {p1_config['type']} {'(Model: ' + p1_config['model'] + ')' if p1_config['model'] else ''}")
    print(f"Player 2 (GREEN): {p2_config['type']} {'(Model: ' + p2_config['model'] + ')' if p2_config['model'] else ''}")
    if args.numplayers == 4:
        print(f"Player 3 (BLUE): {p3_config['type']} {'(Model: ' + p3_config['model'] + ')' if p3_config['model'] else ''}")
        print(f"Player 4 (YELLOW): {p4_config['type']} {'(Model: ' + p4_config['model'] + ')' if p4_config['model'] else ''}")
    print("-------------------------")

    for i in range(args.repetitions):
        print(f"\nStarting Game {i+1} / {args.repetitions}...")
        start_time = time.time()
        
        winner_color = run_single_match(
            p1_config, p2_config, p3_config, p4_config,
            args.boardsize, args.mode, args.numplayers
        )
        
        game_duration = time.time() - start_time
        total_game_time += game_duration
        
        print(f"Game {i+1} finished. Winner: {winner_color}. Duration: {game_duration:.2f}s")
        win_counts[winner_color] += 1

    print("\n--- Overall Results ---")
    print(f"Total games played: {args.repetitions}")
    
    player_names_map = {
        "RED": p1_config['name'],
        "GREEN": p2_config['name'],
        "BLUE": p3_config['name'],
        "YELLOW": p4_config['name'],
        "DRAW": "Draws"
    }

    for color_key, count in win_counts.items():
        player_name = player_names_map.get(color_key, color_key) # Use descriptive name or color if not mapped
        percentage = (count / args.repetitions) * 100
        print(f"{player_name}: {count} wins ({percentage:.2f}%)")
    
    if args.repetitions > 0:
        print(f"Average game duration: {total_game_time / args.repetitions:.2f}s")
    print("-------------------------")

if __name__ == "__main__":
    main()