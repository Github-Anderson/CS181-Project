import sys
import os
import random
import copy
from agents import ApproximateQLearningPlayer, MinimaxPlayer
from board import Board

def train_approx_vs_minimax(
    num_episodes=200,
    minimax_depth=2,
    print_every=20
):
    # 初始化玩家，ApproximateQLearningPlayer为后手
    minimax_agent = MinimaxPlayer(color="RED", depth=minimax_depth)
    approx_agent = ApproximateQLearningPlayer(color="BLUE", alpha=0.1, gamma=0.9, epsilon=0.2)
    players = [minimax_agent, approx_agent]  # 先手Minimax，后手Approx

    # 记录胜负
    win_count = {approx_agent: 0, minimax_agent: 0}

    for episode in range(1, num_episodes + 1):
        board = Board(players = players, mode="classic", boardsize=8)
        for p in players:
            p.set_board(board)
        board.reset()

        current_player_idx = 0
        state = None
        step = 0

        while True:
            player = players[current_player_idx]
            actions = board.get_actions(player)
            if not actions:
                current_player_idx = (current_player_idx + 1) % len(players)
                continue

            action = player.get_action(actions)
            board.apply_action(action)
            state = board.get_state()
            step += 1

            if state is not None:
                break
            current_player_idx = (current_player_idx + 1) % len(players)

        # 统计胜负
        if state == approx_agent:
            win_count[approx_agent] += 1
        elif state == minimax_agent:
            win_count[minimax_agent] += 1

        if episode % print_every == 0:
            print(f"Episode {episode}: Approx wins {win_count[approx_agent]}, Minimax wins {win_count[minimax_agent]}")
            print("Current weights:")
            for k, v in approx_agent.weights.items():
                print(f"  {k}: {v:.2f}")
            print("-" * 40)

    print("\n=== Training Finished ===")
    print(f"Total: Approx wins {win_count[approx_agent]}, Minimax wins {win_count[minimax_agent]}")
    print("Final weights:")
    for k, v in approx_agent.weights.items():
        print(f"  {k}: {v:.2f}")

if __name__ == "__main__":
    train_approx_vs_minimax(num_episodes=200, minimax_depth=2, print_every=20)