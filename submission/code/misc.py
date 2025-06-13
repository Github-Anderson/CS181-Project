import argparse
from collections import defaultdict
import time
import random # Added for random choice in case of agent failure
import sys
import os

from engine import HeadlessEngine
from board import Board
from agents import (
    Player, MinimaxPlayer, GreedyPlayer, RandomPlayer, MCTSPlayer,
    ApproximateQLearningPlayer, Neural_ApproximateQLearningPlayer, DefaultPlayer
)
import utils

# Helper function to create player instances from strings (adapted from autograder.py)
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
        player = MCTSPlayer(color)
    elif player_type_str == "AQL":
        player = ApproximateQLearningPlayer(color)
    elif player_type_str == "NAQL":
        player = Neural_ApproximateQLearningPlayer(color, board_size=board_size, if_train=False)
        model_to_load = model_path
        if not model_to_load:
            if color == "RED":
                model_to_load = "sente_agent_ep300.pth"
            elif color == "GREEN":
                model_to_load = "gote_agent_ep300.pth"
        if model_to_load:
            try:
                player.load_model(model_to_load)
                print(f"NAQL {color} loaded model: {model_to_load}")
            except FileNotFoundError:
                print(f"Warning: Model file not found for NAQL {color}: {model_to_load}")
            except Exception as e:
                print(f"Warning: Error loading model for NAQL {color} from {model_to_load}: {e}")
        else:
            print(f"Warning: NAQL {color} initialized without a pre-trained model path.")
    else:
        raise ValueError(f"Unknown player type: {player_type_str}")

    if player is None:
        raise ValueError(f"Failed to create player for type: {player_type_str}")
    return player

def run_single_match(p1_config, p2_config, board_size, game_mode, max_turns_for_game):
    players_for_board = []

    p1 = create_player(p1_config['type'], "RED", board_size, p1_config.get('model'))
    players_for_board.append(p1)
    p2 = create_player(p2_config['type'], "GREEN", board_size, p2_config.get('model'))
    players_for_board.append(p2)

    current_board = Board(board_size, game_mode, players_for_board)

    for player_obj in players_for_board:
        if hasattr(player_obj, 'set_board') and not isinstance(player_obj, RandomPlayer):
            player_obj.set_board(current_board)

    engine = HeadlessEngine(current_board, max_turns=max_turns_for_game)
    winner_player_object = engine.run_game()

    ended_by_max_turns = engine.game_turn_count >= engine.max_turns

    if ended_by_max_turns:
        winner_color = "DRAW"  # If max turns are reached, this script considers it a draw
    else:
        # Game ended before reaching max turns
        if winner_player_object:
            winner_color = winner_player_object.color
        else:
            # Game ended before max turns, but engine did not determine a winner (e.g., stalemate not due to max turns)
            winner_color = "DRAW"
    
    return winner_color, ended_by_max_turns, engine.game_turn_count

def read_command(argv):
    parser = argparse.ArgumentParser(description="CS181 Halma Misc Tester for jump_scalar")
    parser.add_argument('-p1', '--player1_type', type=str, choices=['M', 'MLS', 'G'], default='G', help='Player 1 (RED) type.')
    parser.add_argument('-p2', '--player2_type', type=str, choices=['M', 'MLS', 'G'], default='G', help='Player 2 (GREEN) type.')
    parser.add_argument('-r', '--repetitions', type=int, default=utils.DEFAULT_ROUND, help="Number of games to run for each test phase.")
    
    # Adding other arguments similar to autograder.py for completeness
    parser.add_argument('-s', '--boardsize', type=int, choices=[4, 8, 10, 12], default=8, help='Board size.')
    parser.add_argument('-m', '--mode', type=str, choices=['classic', 'score'], default='score', help='Game mode.')
    parser.add_argument('--max_turns', type=int, default=utils.MAX_TURNS, help='Maximum turns per game before draw.')

    parser.add_argument('-j', '--jump_score_mode', type=str, choices=['list', 'search'], default='search', help='Mode for testing jump_scalar: "list" to iterate through utils.jump_scalars, "search" for binary search.')

    args = parser.parse_args()
    return args

def main():
    args = read_command(sys.argv[1:])

    p1_config = {'type': args.player1_type, 'name': f"{args.player1_type}(RED)"}
    p2_config = {'type': args.player2_type, 'name': f"{args.player2_type}(GREEN)"}

    player_names_map = {
        "RED": p1_config['name'],
        "GREEN": p2_config['name'],
        "DRAW": "Draws"
    }
    
    print("--- Test Setup ---")
    print(f"Player 1 (RED): {p1_config['type']}")
    print(f"Player 2 (GREEN): {p2_config['type']}")
    print(f"Board Size: {args.boardsize}, Mode: {args.mode}")
    print(f"Repetitions per test: {args.repetitions}, Max Turns per game: {args.max_turns}")
    print(f"Jump Score Mode: {args.jump_score_mode}")
    print(f"IMPORTANT: jump_scalar is now read from 'utils.py' by 'board.py'.")
    print("------------------\n")

    if args.jump_score_mode == "list":
        print("--- Testing jump_scalar values from list ---")
        print(f"List of jump_scalars to test: {utils.jump_scalars}")
        print(f"Repetitions per scalar: {args.repetitions}")
        print("--------------------------------------------\n")

        for scalar_value in utils.jump_scalars:
            utils.jump_scalar = scalar_value
            print(f"--- Testing with jump_scalar = {utils.jump_scalar:.4f} ---")
            
            win_counts_current_scalar = defaultdict(int)
            total_game_time_current_scalar = 0
            total_turns_current_scalar = 0
            num_max_turn_draws_current_scalar = 0

            for i in range(args.repetitions):
                # print(f"  Game {i+1}/{args.repetitions} for jump_scalar = {utils.jump_scalar:.4f}...")
                start_time = time.time()
                winner_color, ended_by_max_turns, game_turns = run_single_match(
                    p1_config, p2_config,
                    args.boardsize, args.mode, args.max_turns
                )
                game_duration = time.time() - start_time
                total_game_time_current_scalar += game_duration
                total_turns_current_scalar += game_turns
                win_counts_current_scalar[winner_color] += 1
                if ended_by_max_turns:
                    num_max_turn_draws_current_scalar +=1
            
            print(f"\n  Results for jump_scalar = {utils.jump_scalar:.4f}:")
            for color_key, count in win_counts_current_scalar.items():
                player_name = player_names_map.get(color_key, color_key)
                percentage = (count / args.repetitions) * 100 if args.repetitions > 0 else 0
                print(f"  {player_name}: {count} wins/draws ({percentage:.2f}%)")
            if args.repetitions > 0:
                print(f"  Games ended by max_turns: {num_max_turn_draws_current_scalar}/{args.repetitions}")
                avg_duration = total_game_time_current_scalar / args.repetitions
                avg_turns = total_turns_current_scalar / args.repetitions
                print(f"  Average game duration: {avg_duration:.2f}s")
                print(f"  Average game turns: {avg_turns:.2f}")
            print("---------------------------------------------------\n")

    elif args.jump_score_mode == "search":
        # --- Binary search for jump_scalar ---
        print("--- Binary search for optimal jump_scalar ---")
        print(f"Range: [{utils.MIN_JUMP_SCALAR}, {utils.MAX_JUMP_SCALAR}]")
        
        low = utils.MIN_JUMP_SCALAR
        high = utils.MAX_JUMP_SCALAR
        optimal_jump_scalar = (low + high) / 2 # Default

        for step in range(utils.BINARY_SEARCH_STEPS):
            mid = (low + high) / 2
            if mid == low or mid == high: # Converged or stuck
                optimal_jump_scalar = low # Prefer lower if stuck
                break
            
            utils.jump_scalar = mid # Update utils.jump_scalar for the Board
            num_max_turn_draws = 0
            current_run_total_time = 0
            current_run_total_turns = 0

            print(f"\nBinary Search Iteration {step+1}/{utils.BINARY_SEARCH_STEPS}: Testing jump_scalar = {utils.jump_scalar:.4f}")

            for i in range(args.repetitions):
                # print(f"  Game {i+1}/{args.repetitions} for jump_scalar = {utils.jump_scalar:.4f}...")
                start_time = time.time()
                winner_color, ended_by_max_turns, game_turns = run_single_match(
                    p1_config, p2_config,
                    args.boardsize, args.mode, args.max_turns
                )
                current_run_total_time += (time.time() - start_time)
                current_run_total_turns += game_turns
                if ended_by_max_turns:
                    num_max_turn_draws += 1
            
            avg_duration_iter = current_run_total_time / args.repetitions if args.repetitions > 0 else 0
            avg_turns_iter = current_run_total_turns / args.repetitions if args.repetitions > 0 else 0
            print(f"  Result: {num_max_turn_draws}/{args.repetitions} games ended by max_turns. Avg duration: {avg_duration_iter:.2f}s. Avg turns: {avg_turns_iter:.2f}.")

            if num_max_turn_draws > utils.MAX_TURN_DRAW_FRAC_THRESHOLD * args.repetitions:
                print(f"  Too many max-turn draws. Adjusting jump_scalar downwards (high = {mid:.4f}).")
                high = mid
            else:
                print(f"  Max-turn draws acceptable. Adjusting jump_scalar upwards (low = {mid:.4f}).")
                low = mid
            optimal_jump_scalar = (low + high) / 2 # Keep track of the middle of the current best range

        print(f"\n--- Binary Search Complete ---") # Changed title for clarity
        print(f"Final search range for jump_scalar: [{low:.4f}, {high:.4f}]")
        print(f"Determined optimal jump_scalar (midpoint of final range): {optimal_jump_scalar:.4f}")
        print(f"(Using threshold: >{utils.MAX_TURN_DRAW_FRAC_THRESHOLD*100:.0f}% games ending by max_turns leads to smaller scalar)")
        print("-------------------------------------------------\n")

        # --- Optional: Validation run with optimal_jump_scalar ---
        print(f"--- Validation: Testing with optimal jump_scalar = {optimal_jump_scalar:.4f} ---")
        utils.jump_scalar = optimal_jump_scalar # Set utils.jump_scalar for the Board for validation runs
        win_counts_validation = defaultdict(int)
        total_game_time_validation = 0
        total_turns_validation = 0
        num_max_turn_draws_validation = 0

        for i in range(args.repetitions):
            # print(f"Starting Game {i+1}/{args.repetitions} for optimal jump_scalar = {utils.jump_scalar:.4f}...")
            start_time = time.time()
            winner_color, ended_by_max_turns, game_turns = run_single_match(
                p1_config, p2_config,
                args.boardsize, args.mode, args.max_turns
            )
            game_duration = time.time() - start_time
            total_game_time_validation += game_duration
            total_turns_validation += game_turns
            win_counts_validation[winner_color] += 1
            if ended_by_max_turns:
                num_max_turn_draws_validation +=1
            # print(f"Game {i+1} finished. Winner: {winner_color}. Duration: {game_duration:.2f}s. Turns: {game_turns}")

        print(f"\n--- Validation Results (jump_scalar = {optimal_jump_scalar:.4f}) ---")
        for color_key, count in win_counts_validation.items():
            player_name = player_names_map.get(color_key, color_key)
            percentage = (count / args.repetitions) * 100
            print(f"{player_name}: {count} wins/draws ({percentage:.2f}%)")
        if args.repetitions > 0:
            print(f"Games ended by max_turns: {num_max_turn_draws_validation}/{args.repetitions}")
            avg_duration_val = total_game_time_validation / args.repetitions
            avg_turns_val = total_turns_validation / args.repetitions
            print(f"Average game duration: {avg_duration_val:.2f}s")
            print(f"Average game turns: {avg_turns_val:.2f}")
        print("---------------------------------------------------\n")

    print("Script finished.")

if __name__ == "__main__":
    main()