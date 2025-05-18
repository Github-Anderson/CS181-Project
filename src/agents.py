# agents.py
from abc import ABC, abstractmethod
import random
import time
import json
from collections import defaultdict
import math
import copy
import time
class Player(ABC):
    def __init__(self, color):
        self.index = None
        self.color = color
    
    @abstractmethod
    def get_action(self, actions : list[int, int, int, int]) -> tuple[int, int, int, int]:
        pass

    def set_index(self, index):
        self.index = index


class HumanPlayer(Player):
    def get_action(self, actions : list):
        return None


class AgentPlayer(Player):
    pass


class RandomPlayer(AgentPlayer):
    def get_action(self, actions):
        return random.choice(actions)
    

class GreedyPlayer(AgentPlayer):
    def __init__(self, color):
        super().__init__(color)

    def set_board(self, board):
        self.board = board
    
    def get_action(self, actions):
        if not actions:
            return None
            
        best_action = None
        best_value = float("-inf")
        
        # 对于每个可能的移动，计算移动后的评估值
        for action in actions:
            # 保存当前状态
            start_x, start_y, end_x, end_y = action
            pawn = self.board.board[start_x][start_y]
            temp_x, temp_y = pawn.x, pawn.y
            
            # 临时移动
            self.board.board[end_x][end_y] = pawn
            self.board.board[start_x][start_y] = None
            pawn.x, pawn.y = end_x, end_y
            
            # 评估移动后的状态
            value = evaluation(self.board, self)
            
            # 恢复状态
            pawn.x, pawn.y = temp_x, temp_y
            self.board.board[start_x][start_y] = pawn
            self.board.board[end_x][end_y] = None
            
            # 更新最佳动作
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action
        

class MinimaxPlayer(AgentPlayer):
    def __init__(self, color, depth=2, use_local_search=False):
        super().__init__(color)
        self.depth = depth
        self.use_local_search = use_local_search
        self.time_limit = 3.0
    
    def set_board(self, board):
        self.board = board
    
    def get_action(self, actions):
        if not actions:
            return None
            
        # 设置时间限制
        time_limit = time.time() + self.time_limit
        
        # 找出对手
        opponent_idx = (self.index + len(self.board.players) // 2) % len(self.board.players)
        opponent = self.board.players[opponent_idx]
        
        # 调用minimax算法
        best_value, best_action = self.minimax(
            self.depth,
            self,  # 当前玩家(最大化玩家)
            opponent,  # 对手(最小化玩家)
            time_limit,
            float("-inf"),  # alpha
            float("inf"),   # beta
            True  # 当前层是否为最大化层
        )
        
        return best_action

    def minimax(self, depth, max_player, min_player, time_limit, alpha, beta, is_max):
        # 终止条件：达到搜索深度或时间限制
        if depth == 0 or time.time() > time_limit:
            return evaluation(self.board, max_player), None
        
        # 确定当前玩家
        current_player = max_player if is_max else min_player
        # 获取当前玩家可用的动作
        actions = self.board.get_actions(current_player)
        
        if not actions:
            return evaluation(self.board, max_player), None
            
        best_action = None
        if is_max:
            best_value = float("-inf")
            for action in actions:
                # 保存当前状态
                start_x, start_y, end_x, end_y = action
                pawn = self.board.board[start_x][start_y]
                temp_x, temp_y = pawn.x, pawn.y
                
                # 临时移动
                self.board.board[end_x][end_y] = pawn
                self.board.board[start_x][start_y] = None
                pawn.x, pawn.y = end_x, end_y
                
                # 递归调用minimax
                value, _ = self.minimax(depth-1, max_player, min_player, time_limit, alpha, beta, False)
                
                # 恢复状态
                pawn.x, pawn.y = temp_x, temp_y
                self.board.board[start_x][start_y] = pawn
                self.board.board[end_x][end_y] = None
                
                # 更新最佳值和动作
                if value > best_value:
                    best_value = value
                    best_action = action
                    
                # 更新alpha值
                alpha = max(alpha, best_value)
                # Alpha-Beta剪枝
                if beta <= alpha:
                    break
        else:
            best_value = float("inf")
            for action in actions:
                # 保存当前状态
                start_x, start_y, end_x, end_y = action
                pawn = self.board.board[start_x][start_y]
                temp_x, temp_y = pawn.x, pawn.y
                
                # 临时移动
                self.board.board[end_x][end_y] = pawn
                self.board.board[start_x][start_y] = None
                pawn.x, pawn.y = end_x, end_y
                
                # 递归调用minimax
                value, _ = self.minimax(depth-1, max_player, min_player, time_limit, alpha, beta, True)
                
                # 恢复状态
                pawn.x, pawn.y = temp_x, temp_y
                self.board.board[start_x][start_y] = pawn
                self.board.board[end_x][end_y] = None
                
                # 更新最佳值和动作
                if value < best_value:
                    best_value = value
                    best_action = action
                    
                # 更新beta值
                beta = min(beta, best_value)
                # Alpha-Beta剪枝
                if beta <= alpha:
                    break
                    
        return best_value, best_action
    

# TODO: Implement more agents
    
class QLearningAgent(AgentPlayer):
    def __init__(self, color, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(color)
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q 表
        self.board = None  # 棋盘引用

    def choose_action(self, state, actions):
        """根据 ε-greedy 策略选择动作"""
        if random.random() < self.epsilon:
            return random.choice(actions)  # 探索
        else:
            # 确保所有动作都有默认的 Q 值
            for action in actions:
                if action not in self.q_table[state]:
                    self.q_table[state][action] = 0  # 初始化为 0
            return max(actions, key=lambda action: self.q_table[state][action])  # 利用

    def update_q_value(self, state, action, reward, next_state, next_actions):
        """更新 Q 值"""
        # 确保所有动作都有默认的 Q 值
        for next_action in next_actions:
            if next_action not in self.q_table[next_state]:
                self.q_table[next_state][next_action] = 0  # 初始化为 0
        max_next_q = max([self.q_table[next_state][a] for a in next_actions], default=0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[state][action])

class QLearningAgent(AgentPlayer):
    def __init__(self, color, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(color)
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q 表
        self.board = None  # 棋盘引用

    def choose_action(self, state, actions):
        """根据 ε-greedy 策略选择动作"""
        if random.random() < self.epsilon:
            return random.choice(actions)  # 探索
        else:
            # 确保所有动作都有默认的 Q 值
            for action in actions:
                if action not in self.q_table[state]:
                    self.q_table[state][action] = 0  # 初始化为 0
            return max(actions, key=lambda action: self.q_table[state][action])  # 利用

    def update_q_value(self, state, action, reward, next_state, next_actions):
        """更新 Q 值"""
        # 确保所有动作都有默认的 Q 值
        for next_action in next_actions:
            if next_action not in self.q_table[next_state]:
                self.q_table[next_state][next_action] = 0  # 初始化为 0
        max_next_q = max([self.q_table[next_state][a] for a in next_actions], default=0)
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[state][action])

    def save_model(self, filepath):
        """将 Q 表保存为纯文本文件（动作用 JSON，状态多行用 \\n）"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for state, actions in self.q_table.items():
                state_str = state.replace('\n', '\\n')
                for action, q_value in actions.items():
                    action_str = json.dumps(action)
                    f.write(f"{state_str}\t{action_str}\t{q_value}\n")
        print(f"Q-table saved to {filepath} as plain text.")

    def load_model(self, filepath):
        """从纯文本文件加载 Q 表"""
        self.q_table = defaultdict(lambda: defaultdict(float))
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                state_str, action_str, q_value = line.strip().split('\t')
                state = state_str.replace('\\n', '\n')
                action = tuple(json.loads(action_str))
                self.q_table[state][action] = float(q_value)
        print(f"Q-table loaded from {filepath} as plain text.")

    def get_action(self, actions):
        """
        实现抽象方法 get_action，用于与游戏框架交互。
        :param actions: 当前可用的动作列表
        :return: 选择的动作
        """
        if not actions:
            return None
        state = self.serialize_board(self.board)  # 将棋盘状态序列化为字符串
        return self.choose_action(state, actions)

    def serialize_board(self, board):
        """
        将棋盘状态序列化为字符串，确保训练和推理阶段的表示一致。
        :param board: 当前棋盘对象
        :return: 棋盘的字符串表示
        """
        return str(board)  # 或者使用自定义的序列化逻辑

    def set_board(self, board):
        """
        设置棋盘引用，用于访问棋盘状态。
        :param board: 当前棋盘对象
        """
        self.board = board

    def end_game(self, reward):
        """
        游戏结束时的处理（可选）。
        :param reward: 游戏结束时的奖励
        """
        pass

    def get_action(self, actions):
        """
        实现抽象方法 get_action，用于与游戏框架交互。
        :param actions: 当前可用的动作列表
        :return: 选择的动作
        """
        if not actions:
            return None
        state = self.serialize_board(self.board)  # 将棋盘状态序列化为字符串
        return self.choose_action(state, actions)

    def serialize_board(self, board):
        """
        将棋盘状态序列化为字符串，确保训练和推理阶段的表示一致。
        :param board: 当前棋盘对象
        :return: 棋盘的字符串表示
        """
        return str(board)  # 或者使用自定义的序列化逻辑

    def set_board(self, board):
        """
        设置棋盘引用，用于访问棋盘状态。
        :param board: 当前棋盘对象
        """
        self.board = board

    def end_game(self, reward):
        """
        游戏结束时的处理（可选）。
        :param reward: 游戏结束时的奖励
        """
        pass

class MCTSNode:
    def __init__(self, board, parent=None, action=None, player=None):
        self.board = board  # 当前棋盘状态（深拷贝）
        self.parent = parent
        self.action = action  # 到达该节点的动作
        self.player = player  # 当前节点的玩家
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None  # 延迟初始化

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / (child.visits + 1e-6)) + c_param * math.sqrt(2 * math.log(self.visits + 1) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self, player):
        if self.untried_actions is None:
            self.untried_actions = self.board.get_actions(player)
        action = self.untried_actions.pop()
        next_board = copy.deepcopy(self.board)
        next_board.apply_action(action)
        child_node = MCTSNode(next_board, parent=self, action=action, player=player)
        self.children.append(child_node)
        return child_node

    def most_visited_child(self):
        return max(self.children, key=lambda child: child.visits)

class MCTSPlayer(AgentPlayer):
    def __init__(self, color, simulations=300, time_limit=0.3):
        super().__init__(color)
        self.simulations = simulations
        self.board = None
        self.time_limit = time_limit

    def set_board(self, board):
        self.board = board

    def get_action(self, actions):
        if not actions:
            return None
        root = MCTSNode(copy.deepcopy(self.board), player=self)
        root.untried_actions = actions.copy()
        start_time = time.time()
        simulations = 0
        while simulations < self.simulations and (time.time() - start_time) < self.time_limit:
            node = root
            board_copy = copy.deepcopy(self.board)
            player = self

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                board_copy.apply_action(node.action)
                player = self._next_player(board_copy, player)

            # Expansion
            if node.untried_actions is None:
                node.untried_actions = board_copy.get_actions(player)
            if node.untried_actions:
                node = node.expand(player)
                board_copy.apply_action(node.action)
                player = self._next_player(board_copy, player)

            # Simulation
            reward = self._simulate_with_eval_noise(board_copy, player)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent
                reward = 1 - reward  # 对手视角

            simulations += 1

        best_child = root.most_visited_child()
        return best_child.action

    def _simulate_with_eval_noise(self, board, player):
        """模拟阶段：90%概率贪心选evaluation最优动作，10%概率随机，终局胜负优先，否则用evaluation归一化分数。"""
        current_player = player
        max_steps = 20
        steps = 0
        while steps < max_steps:
            actions = board.get_actions(current_player)
            if not actions:
                break

            # 90%概率贪心，10%概率随机
            if random.random() < 0.9:
                best_action = None
                best_value = float('-inf')
                for action in actions:
                    start_x, start_y, end_x, end_y = action
                    pawn = board.board[start_x][start_y]
                    temp_x, temp_y = pawn.x, pawn.y
                    board.board[end_x][end_y] = pawn
                    board.board[start_x][start_y] = None
                    pawn.x, pawn.y = end_x, end_y

                    value = evaluation_MCTS(board, current_player)

                    pawn.x, pawn.y = temp_x, temp_y
                    board.board[start_x][start_y] = pawn
                    board.board[end_x][end_y] = None

                    if value > best_value:
                        best_value = value
                        best_action = action
            else:
                best_action = random.choice(actions)

            board.apply_action(best_action)
            current_player = self._next_player(board, current_player)
            state = board.get_state()
            if state and state.get("winner") is not None:
                # 赢了返回1，输了返回0
                return 1 if state["winner"] == self else 0
            steps += 1

        # 超步数未分胜负，用evaluation归一化分数
        my_score = evaluation_MCTS(board, self)
        norm_score = (my_score + 100) / 200
        return norm_score

    def _next_player(self, board, current_player):
        current_color = current_player.color
        players = board.players
        idx = None
        for i, p in enumerate(players):
            if hasattr(p, "color") and p.color == current_color:
                idx = i
                break
        if idx is None:
            raise ValueError("Current player not found in board.players by color!")
        next_idx = (idx + 1) % len(players)
        return players[next_idx]

def evaluation_MCTS(board, player):
    val = 0
    goal_area = board.get_goal_area(player)
    for i in range(board.boardsize):
        for j in range(board.boardsize):
            pawn = board.board[i][j]
            if pawn and pawn.player == player:
                # 在目标区内奖励+50
                if (i, j) in goal_area:
                    val += 50
                else:
                    # 距离目标区最近点的负距离
                    min_dist = min(((goal_x - i) ** 2 + (goal_y - j) ** 2) ** 0.5 for goal_x, goal_y in goal_area)
                    val -= min_dist
    return val

def evaluation(board, player):
    val = 0
    
    goal_area = board.get_goal_area(player)
    
    # 遍历所有棋子计算得分
    for i in range(board.boardsize):
        for j in range(board.boardsize):
            pawn = board.board[i][j]
            if pawn and pawn.player == player:
                # 计算到目标区域中空位置的最大距离
                goal_distances = []
                for goal_x, goal_y in goal_area:
                    if board.board[goal_x][goal_y] is None or board.board[goal_x][goal_y].player != player:
                        # 计算欧几里得距离
                        distance = ((goal_x - i) ** 2 + (goal_y - j) ** 2) ** 0.5
                        goal_distances.append(distance)
                
                if goal_distances:
                    val += max(goal_distances)
                else:
                    # 所有目标位置都被占用，给予惩罚
                    val -= 20
    
    # 取反以便最大化评估值（距离越小越好）
    val *= -1
    return val

# TODO: Implement more evaluation functions