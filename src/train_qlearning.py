from board import Board
from agents import MinimaxPlayer, QLearningAgent
from tqdm import tqdm

def train_qlearning(episodes=10):
    agent = QLearningAgent("RED", alpha=0.1, gamma=0.9, epsilon=0.1)
    board_size = 8
    player1 = agent
    player2 = MinimaxPlayer("GREEN", depth=2)  # 对手为 Minimax 玩家，设置搜索深度为 2
    board = Board(board_size, (player1, player2))
    player1.set_board(board)  # 设置棋盘引用
    player2.set_board(board)
    for episode in tqdm(range(episodes)):
        board.initialize_board()
        state = str(board)  # 将棋盘状态序列化为字符串
        done = False

        while not done:
            # Player 1 (Q-learning 玩家) 的回合
            if board.turn % 2 == 1:  # 假设奇数回合是 player1 的回合
                actions = board.get_actions(player1)
                if not actions:
                    break

                action = player1.get_action(actions)  # 使用 Q-learning 玩家选择动作
                board.apply_action(action)
                reward = 1 if board.get_state() and board.get_state()["winner"] == player1 else 0
                next_state = str(board)
                next_actions = board.get_actions(player1)
                player1.update_q_value(state, action, reward, next_state, next_actions)
                state = next_state

            # Player 2 (Minimax 玩家) 的回合
            else:  # 偶数回合是 player2 的回合
                actions = board.get_actions(player2)
                if not actions:
                    break

                action = player2.get_action(actions)  # 使用 Minimax 玩家选择动作
                board.apply_action(action)

            # 检查游戏是否结束
            if board.get_state():
                done = True

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")

    player1.save_model("qlearning_model.txt")
    print("Training completed and model saved.")

if __name__ == "__main__":
    train_qlearning()