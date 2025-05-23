from board import Board
from agents import QLearningAgent, evaluation
from tqdm import tqdm

def train_qlearning(episodes=50):
    board_size = 8
    agent_red = QLearningAgent("RED", alpha=0.1, gamma=0.9, epsilon=0.1)
    agent_green = QLearningAgent("GREEN", alpha=0.1, gamma=0.9, epsilon=0.1)
    board = Board(board_size, (agent_red, agent_green))
    agent_red.set_board(board)
    agent_green.set_board(board)

    for episode in tqdm(range(episodes)):
        board.initialize_board()
        state_red = agent_red.serialize_board(board)
        state_green = agent_green.serialize_board(board)
        done = False
        eval_red = evaluation(board, agent_red)
        eval_green = evaluation(board, agent_green)

        while not done:
            # RED 回合
            if board.turn % 2 == 1:
                actions = board.get_actions(agent_red)
                if not actions:
                    break
                action = agent_red.get_action(actions)
                board.apply_action(action)
                eval_red_new = evaluation(board, agent_red)
                reward = eval_red_new - eval_red
                eval_red = eval_red_new
                # 终局奖励
                state_info = board.get_state()
                if state_info:
                    winner = state_info.get("winner")
                    if winner == agent_red:
                        reward += 100
                    elif winner == agent_green:
                        reward -= 100
                next_state = agent_red.serialize_board(board)
                next_actions = board.get_actions(agent_red)
                agent_red.update_q_value(state_red, action, reward, next_state, next_actions)
                state_red = next_state
            # GREEN 回合
            else:
                actions = board.get_actions(agent_green)
                if not actions:
                    break
                action = agent_green.get_action(actions)
                board.apply_action(action)
                eval_green_new = evaluation(board, agent_green)
                reward = eval_green_new - eval_green
                eval_green = eval_green_new
                state_info = board.get_state()
                if state_info:
                    winner = state_info.get("winner")
                    if winner == agent_green:
                        reward += 100
                    elif winner == agent_red:
                        reward -= 100
                next_state = agent_green.serialize_board(board)
                next_actions = board.get_actions(agent_green)
                agent_green.update_q_value(state_green, action, reward, next_state, next_actions)
                state_green = next_state

            if board.get_state():
                done = True

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed.")

    agent_red.save_model("qlearning_model_red.txt")
    agent_green.save_model("qlearning_model_green.txt")
    print("Training completed and models saved.")

if __name__ == "__main__":
    train_qlearning()