from board import Board
from agents import QLearningAgent, HumanPlayer

def play_with_qlearning():
    agent = QLearningAgent("RED")
    agent.load_model("qlearning_model.json")  # 加载训练好的模型

    board_size = 8
    player1 = agent
    player2 = HumanPlayer("GREEN")  # 人类玩家
    board = Board(board_size, (player1, player2))
    player1.set_board(board)  # 设置棋盘引用

    board.initialize_board()
    done = False

    while not done:
        print(board)
        if board.turn % 2 == 1:  # Q-learning 玩家回合
            actions = board.get_actions(player1)
            action = player1.get_action(actions)  # 使用 get_action 方法选择动作
            board.apply_action(action)
        else:  # 人类玩家回合
            print("Your turn!")
            action = input("Enter your move (e.g., 0 0 1 1): ")
            action = tuple(map(int, action.split()))
            board.apply_action(action)

        if board.get_state():
            done = True
            print(board.get_state()["message"])

if __name__ == "__main__":
    play_with_qlearning()