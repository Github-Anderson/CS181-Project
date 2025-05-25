# agents.py
from abc import ABC, abstractmethod
import random
import time
import json
from collections import defaultdict
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class Player(ABC):
    def __init__(self, color):
        self.index = None
        self.color = color
    
    @abstractmethod
    def get_action(self, actions : list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        pass

    def set_index(self, index):
        self.index = index

class HumanPlayer(Player):
    def get_action(self, actions : list):
        return None

class AgentPlayer(Player):
    pass

def evaluation(board, player):
    val = 0
    goal_area = board.get_goal_area(player)
    
    for i in range(board.boardsize):
        for j in range(board.boardsize):
            pawn = board.board[i][j]
            if pawn and pawn.player == player:
                goal_distances = []
                for goal_x, goal_y in goal_area:
                    if board.board[goal_x][goal_y] is None or board.board[goal_x][goal_y].player != player:
                        distance = ((goal_x - i) ** 2 + (goal_y - j) ** 2) ** 0.5
                        goal_distances.append(distance)
                
                if goal_distances:
                    val += max(goal_distances)
                else:
                    val -= 20
    
    val *= -1
    return val

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
        
        for action in actions:
            start_x, start_y, end_x, end_y = action
            pawn = self.board.board[start_x][start_y]
            temp_x, temp_y = pawn.x, pawn.y
            
            self.board.board[end_x][end_y] = pawn
            self.board.board[start_x][start_y] = None
            pawn.x, pawn.y = end_x, end_y
            
            value = evaluation(self.board, self)
            
            pawn.x, pawn.y = temp_x, temp_y
            self.board.board[start_x][start_y] = pawn
            self.board.board[end_x][end_y] = None
            
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
            
        time_limit = time.time() + self.time_limit
        
        opponent_idx = (self.index + len(self.board.players) // 2) % len(self.board.players)
        opponent = self.board.players[opponent_idx]
        
        best_value, best_action = self.minimax(
            self.depth,
            self,
            opponent,
            time_limit,
            float("-inf"),
            float("inf"),
            True
        )
        
        return best_action

    def minimax(self, depth, max_player, min_player, time_limit, alpha, beta, is_max):
        if depth == 0 or time.time() > time_limit:
            return evaluation(self.board, max_player), None
        
        current_player = max_player if is_max else min_player
        actions = self.board.get_actions(current_player)
        
        if not actions:
            return evaluation(self.board, max_player), None
            
        best_action = None
        if is_max:
            best_value = float("-inf")
            for action in actions:
                start_x, start_y, end_x, end_y = action
                pawn = self.board.board[start_x][start_y]
                temp_x, temp_y = pawn.x, pawn.y
                
                self.board.board[end_x][end_y] = pawn
                self.board.board[start_x][start_y] = None
                pawn.x, pawn.y = end_x, end_y
                
                value, _ = self.minimax(depth-1, max_player, min_player, time_limit, alpha, beta, False)
                
                pawn.x, pawn.y = temp_x, temp_y
                self.board.board[start_x][start_y] = pawn
                self.board.board[end_x][end_y] = None
                
                if value > best_value:
                    best_value = value
                    best_action = action
                    
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
        else:
            best_value = float("inf")
            for action in actions:
                start_x, start_y, end_x, end_y = action
                pawn = self.board.board[start_x][start_y]
                temp_x, temp_y = pawn.x, pawn.y
                
                self.board.board[end_x][end_y] = pawn
                self.board.board[start_x][start_y] = None
                pawn.x, pawn.y = end_x, end_y
                
                value, _ = self.minimax(depth-1, max_player, min_player, time_limit, alpha, beta, True)
                
                pawn.x, pawn.y = temp_x, temp_y
                self.board.board[start_x][start_y] = pawn
                self.board.board[end_x][end_y] = None
                
                if value < best_value:
                    best_value = value
                    best_action = action
                    
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
                    
        return best_value, best_action

def evaluation_MCTS(board, player):
    """添加调试信息的评估函数"""
    my_pieces = []
    goal_area = set(board.get_goal_area(player))
    home_area = set(board.get_home_area(player))
    
    if player.color == "RED":
        goal_center = (board.boardsize - 1, board.boardsize - 1)
    else:
        goal_center = (0, 0)
    
    # 基础评分项
    pieces_in_goal = 0
    normalized_dist_sum = 0
    max_possible_dist = 2 * board.boardsize
    
    # 统计棋子位置和距离
    for i in range(board.boardsize):
        for j in range(board.boardsize):
            pawn = board.board[i][j]
            if pawn and pawn.player.color == player.color:  # 修改这里：比较颜色而不是player对象
                my_pieces.append((i, j))
                
                if (i, j) in goal_area:
                    pieces_in_goal += 1
                else:
                    # 计算到目标的曼哈顿距离
                    dist = abs(goal_center[0] - i) + abs(goal_center[1] - j)
                    normalized_dist = dist / max_possible_dist
                    normalized_dist_sum += normalized_dist
                
                # ... 其余代码保持不变 ...
    

    # 1. 目标区域棋子数量奖励（总分值占比40%）
    goal_progress = pieces_in_goal / 4  # 归一化到[0,1]
    goal_score = 0.4 * (1000 * goal_progress)  # 最高400分
    
    # 2. 距离评分（总分值占比30%）
    distance_score = 0.3 * (300 * (1 - normalized_dist_sum / len(my_pieces)))  # 最高300分
    
    # 3. 每增加一个到达目标的棋子的额外奖励（递进式）
    stage_bonus = 0
    if pieces_in_goal >= 1:
        stage_bonus += 50   # 第一个棋子
    if pieces_in_goal >= 2:
        stage_bonus += 100  # 第二个棋子
    if pieces_in_goal >= 3:
        stage_bonus += 200  # 第三个棋子
    if pieces_in_goal >= 4:
        stage_bonus += 400  # 第四个棋子（胜利）
    
    # 4. 家中棋子惩罚（归一化）
    pieces_at_home = sum(1 for (i, j) in my_pieces 
                        if (i, j) in home_area and not board.board[i][j].has_left_home)
    home_penalty = -100 * (pieces_at_home / len(my_pieces))  # 归一化惩罚
    
    # 最终评分（归一化到[-1000, 1000]范围）
    final_score = (goal_score + 
                  distance_score + 
                  stage_bonus + 
                  home_penalty)
    
    # 确保评分在合理范围内
    return max(-1000, min(1000, final_score))
def is_going_backwards(board_size, action, player):
    start_x, start_y, end_x, end_y = action
    if player.color == "RED":
        return end_x < start_x or end_y < start_y
    else:
        return end_x > start_x or end_y > start_y

def is_jump_move(board, action):
    """Check if the action is a jump move (not a single step)"""
    start_x, start_y, end_x, end_y = action
    return abs(end_x - start_x) > 1 or abs(end_y - start_y) > 1

# Here I want to designa an agent utilizing Monte Carlo Tree Search
# Please provide a full code with all procedures

class MCTSNode:
    def __init__(self, board, parent=None, action=None, player_index=None):
        self.board = copy.deepcopy(board)
        self.parent = parent
        self.action = action
        # 确保player_index在有效范围内
        self.player_index = player_index % len(board.players) if player_index is not None else None
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.last_moved_piece = (action[0], action[1]) if action else None
        self.terminal = False
        self.winner = None

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0
    
    def expand(self, mcts_player):
        """修复索引越界问题的expand方法"""
        if self.untried_actions is None:
            # 确保player_index在有效范围内
            self.player_index = self.player_index % len(self.board.players)
            current_player = self.board.players[self.player_index]
            self.untried_actions = self.board.get_actions(current_player)
            
            # Prioritize actions that move pieces toward goal
            if self.untried_actions:
                self.untried_actions = self.prioritize_actions(self.untried_actions, current_player)
            
            if not self.untried_actions:
                self.terminal = True
                return None

        action = self.untried_actions.pop()
        next_board = copy.deepcopy(self.board)
        next_board.apply_action(action)
        
        # Check if this action leads to a terminal state
        state = next_board.get_state()
        if state:
            self.terminal = True
            self.winner = state["winner"]
        
        # 确保下一个玩家索引在有效范围内
        next_player_index = (self.player_index + 1) % len(next_board.players)
        child_node = MCTSNode(next_board, parent=self, action=action, player_index=next_player_index)
        self.children.append(child_node)
        return child_node

    def prioritize_actions(self, actions, player):
        """Prioritize actions that move pieces toward goal area"""
        goal_area = set(self.board.get_goal_area(player))
        
        def action_score(action):
            start_x, start_y, end_x, end_y = action
            start_dist = min(((gx - start_x)**2 + (gy - start_y)**2) for gx, gy in goal_area)
            end_dist = min(((gx - end_x)**2 + (gy - end_y)**2) for gx, gy in goal_area)
            
            # Prefer moves that reduce distance to goal
            distance_improvement = start_dist - end_dist
            
            # Bonus for jumps
            is_jump = is_jump_move(self.board, action)
            
            # Penalty for moving backwards (away from goal)
            backwards_penalty = 1 if is_going_backwards(self.board.boardsize, action, player) else 0
            
            return (distance_improvement * 10 + is_jump * 5 - backwards_penalty * 15, 
                    -end_dist)  # Secondary sort by distance to goal

        return sorted(actions, key=action_score, reverse=True)

    def best_child(self, mcts_player, c_param=1.4):
        """改进的UCB选择"""
        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            
            exploit = child.value / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
            
            # 策略引导
            strategy_score = 0
            if child.action:
                start_x, start_y, end_x, end_y = child.action
                player = mcts_player.board.players[child.player_index]
                
                # 方向性评估
                if player.color == "RED":
                    direction_score = (end_x - start_x + end_y - start_y) / 2
                else:
                    direction_score = (start_x - end_x + start_y - end_y) / 2
                strategy_score += direction_score * 0.2
                
                # 跳跃奖励
                if is_jump_move(self.board, child.action):
                    strategy_score += 0.3
            
            return exploit + explore + strategy_score

        return max(self.children, key=ucb_score)

    def most_visited_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.visits)


class MCTSPlayer(AgentPlayer):
    def __init__(self, color, simulations=3000, time_limit=5.0):
        super().__init__(color)
        self.simulations = simulations
        self.time_limit = time_limit
        self.board = None
    
    def set_board(self, board):
        self.board = board

    def get_action(self, actions):
        if not actions:
            return None
        
        start_time = time.time()
        root = MCTSNode(self.board, player_index=self.index)
        current_simulations = 0
        
        while (current_simulations < self.simulations and 
               time.time() - start_time < self.time_limit):
            # 1. Selection
            node = root
            board_copy = copy.deepcopy(self.board)
            
            # 选择阶段 - 一直选到叶子节点
            while node.is_fully_expanded() and not node.terminal:
                node = node.best_child(self, c_param=self._get_ucb_param(current_simulations))
                if node.action:
                    board_copy.apply_action(node.action)
            
            # 2. Expansion
            if not node.terminal and not node.is_fully_expanded():
                node = node.expand(self)
                if node and node.action:
                    board_copy.apply_action(node.action)
            
            # 3. Simulation
            if node:
                value = self.simulate(board_copy, node.player_index)
            else:
                value = 0
            
            # 4. Backpropagation
            while node:
                node.visits += 1
                node.value += value
                node = node.parent
            
            current_simulations += 1
        
        # 最终动作选择
        best_action = self._select_final_action(root)
        return best_action
    
    def _select_final_action(self, root):
        """添加调试信息的最终动作选择"""
        if not root.children:
            return None
        
        best_score = float('-inf')
        best_action = None
        
        print("\n=== MCTS Final Selection Debug ===")
        print(f"Total simulations: {root.visits}")
        print(f"Current pieces in goal: {self._count_pieces_in_goal()}")
        
        for child in root.children:
            if child.visits == 0:
                continue
            
            action = child.action
            start_x, start_y, end_x, end_y = action
            
            # 计算各项得分
            visit_ratio = child.visits / root.visits
            win_ratio = child.value / child.visits if child.visits > 0 else 0
            
            # 方向性得分
            direction_score = 0
            if self.color == "RED":
                direction_score = (end_x - start_x) + (end_y - start_y)
            else:
                direction_score = (start_x - end_x) + (start_y - end_y)
            
            # 计算该动作的evaluation值
            temp_board = copy.deepcopy(self.board)
            temp_board.apply_action(action)
            eval_score = evaluation_MCTS(temp_board, self)
            
            # 综合评分
            score = (
                win_ratio * 0.4 +
                visit_ratio * 0.2 +
                direction_score * 0.4
            )
            
            print(f"\nAction {action}:")
            print(f"- Visits: {child.visits} ({visit_ratio:.2f})")
            print(f"- Value: {child.value:.2f} (Win ratio: {win_ratio:.2f})")
            print(f"- Direction score: {direction_score}")
            print(f"- Evaluation score: {eval_score}")
            print(f"- Final score: {score}")
            
            if score > best_score:
                best_score = score
                best_action = action
                
        print(f"\nSelected action: {best_action} with score {best_score}")
        return best_action

    def _get_ucb_param(self, simulations):
        """获取UCB参数"""
        pieces_in_goal = self._count_pieces_in_goal()
        if pieces_in_goal < 2:  # 前期更多探索
            return 1.8
        elif pieces_in_goal < 4:  # 中期平衡
            return 1.4
        else:  # 后期更多利用
            return 1.0
    
    def _count_pieces_in_goal(self):
        """计算在目标区域的棋子数"""
        count = 0
        goal_area = set(self.board.get_goal_area(self))
        for i in range(self.board.boardsize):
            for j in range(self.board.boardsize):
                pawn = self.board.board[i][j]
                if pawn and pawn.player == self and (i, j) in goal_area:
                    count += 1
        return count

    def simulate(self, board, current_player_index):
        """更有目标性的模拟策略"""
        depth = 0
        max_depth = 30
        
        while depth < max_depth:
            state = board.get_state()
            if state:
                if state["winner"] == self:
                    pieces_in_goal = sum(1 for i, j in board.get_goal_area(self)
                                    if board.board[i][j] and 
                                    board.board[i][j].player == self)
                    # 归一化的胜利奖励
                    return min(1.0, 0.6 + 0.1 * pieces_in_goal)  # 0.7 到 1.0
            
            current_player = board.players[current_player_index]
            actions = board.get_actions(current_player)
            
            if not actions:
                break
            
            # 提高启发式选择的概率
            if random.random() < 0.9:  # 90%使用启发式
                valid_actions = []
                for action in actions:
                    start_x, start_y, end_x, end_y = action
                    # 严格筛选动作，优先选择向目标方向移动的动作
                    if current_player.color == "RED":
                        if end_x >= start_x and end_y >= start_y:
                            valid_actions.append(action)
                    else:
                        if end_x <= start_x and end_y <= start_y:
                            valid_actions.append(action)
                
                if valid_actions:
                    action = self._select_simulation_action(board, valid_actions, current_player)
                else:
                    action = random.choice(actions)
            else:
                action = random.choice(actions)
            
            board.apply_action(action)
            current_player_index = (current_player_index + 1) % len(board.players)
            depth += 1
        
        eval_score = evaluation_MCTS(board, self)
        return eval_score / 1000.0  # 归一化到[-1, 1]范围

    def _select_simulation_action(self, board, actions, player):
        """改进的模拟动作选择"""
        best_action = None
        best_score = float('-inf')
        goal_area = set(board.get_goal_area(player))
        
        for action in actions:
            start_x, start_y, end_x, end_y = action
            
            # 计算到目标的距离改善
            start_dist = min(abs(gx - start_x) + abs(gy - start_y) 
                           for gx, gy in goal_area)
            end_dist = min(abs(gx - end_x) + abs(gy - end_y) 
                          for gx, gy in goal_area)
            
            # 显著增加距离改善的权重
            score = (start_dist - end_dist) * 4
            
            # 方向性评估
            if player.color == "RED":
                direction_score = (end_x - start_x + end_y - start_y)
            else:
                direction_score = (start_x - end_x + start_y - end_y)
            score += direction_score * 2
            
            # 跳跃奖励
            if is_jump_move(board, action):
                score += 3
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
class FeatureExtractor:
    """特征提取器"""
    def get_features(self, board, player, action=None):
        features = {}
        goal_area = set(board.get_goal_area(player))
        home_area = set(board.get_home_area(player))
        
        if player.color == "RED":
            goal_center = (board.boardsize - 1, board.boardsize - 1)
        else:
            goal_center = (0, 0)
        
        # 状态特征
        pieces_positions = []
        pieces_in_goal = 0
        total_dist = 0
        max_possible_dist = 2 * board.boardsize
        
        # 统计所有棋子
        for i in range(board.boardsize):
            for j in range(board.boardsize):
                pawn = board.board[i][j]
                if pawn and pawn.player.color == player.color:
                    pieces_positions.append((i, j))
                    if (i, j) in goal_area:
                        pieces_in_goal += 1
                    else:
                        dist = abs(goal_center[0] - i) + abs(goal_center[1] - j)
                        total_dist += dist
        
        features["pieces_in_goal"] = pieces_in_goal / 4
        features["avg_distance"] = total_dist / (4 * max_possible_dist)
        
        # 动作特征
        if action:
            start_x, start_y, end_x, end_y = action
            
            # 距离特征
            start_dist = abs(goal_center[0] - start_x) + abs(goal_center[1] - start_y)
            end_dist = abs(goal_center[0] - end_x) + abs(goal_center[1] - end_y)
            features["distance_improvement"] = (start_dist - end_dist) / max_possible_dist
            
            # 方向特征
            if player.color == "RED":
                direction_score = (end_x - start_x + end_y - start_y)
            else:
                direction_score = (start_x - end_x + start_y - end_y)
            features["direction"] = direction_score / (2 * board.boardsize)
            
            # 跳跃特征
            features["is_jump"] = 1.0 if is_jump_move(board, action) else 0.0
            
            # 目标特征
            features["reaches_goal"] = 1.0 if (end_x, end_y) in goal_area else 0.0
            
            # 后退惩罚
            features["is_backwards"] = 1.0 if is_going_backwards(board.boardsize, action, player) else 0.0
            
            # 离开家特征
            features["leaves_home"] = 1.0 if (start_x, start_y) in home_area and not (end_x, end_y) in home_area else 0.0
        
        return features

class ApproximateQLearningPlayer(AgentPlayer):
    def __init__(self, color, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(color)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.weights = defaultdict(float)
        self.feature_extractor = FeatureExtractor()
        self.previous_state = None
        self.previous_action = None
        self.board = None
        
        # 初始化权重
        self.weights.update({
            'pieces_in_goal': 2000,      # 提高目标达成权重
            'avg_distance': -800,        # 加大距离惩罚
            'distance_improvement': 500,  # 增加距离改善奖励
            'direction': 300,            # 增加方向性权重
            'is_jump': 100,             # 适度增加跳跃奖励
            'reaches_goal': 1500,        # 显著提高到达目标奖励
            'is_backwards': -1000,       # 加大后退惩罚
            'leaves_home': 200          # 增加离家奖励
        })
    
    def set_board(self, board):
        self.board = board
    
    def get_qvalue(self, board, action):
        features = self.feature_extractor.get_features(board, self, action)
        return sum(self.weights[f] * v for f, v in features.items())
    
    def get_action(self, actions):
        if not actions:
            return None
        
        # 根据游戏阶段调整探索策略
        pieces_in_goal = sum(1 for i, j in self.board.get_goal_area(self)
                            if self.board.board[i][j] and 
                            self.board.board[i][j].player == self)
        
        # 后期更倾向于利用
        if pieces_in_goal >= 3:
            exploration_prob = 0.05
        elif pieces_in_goal >= 2:
            exploration_prob = 0.1
        else:
            exploration_prob = self.epsilon
        
        if random.random() < exploration_prob:
            # 智能随机：优先选择向目标移动的动作
            valid_actions = [a for a in actions if not is_going_backwards(self.board.boardsize, a, self)]
            if valid_actions:
                best_action = random.choice(valid_actions)
            else:
                best_action = random.choice(actions)
        else:
            # 计算所有动作的Q值
            q_values = []
            for action in actions:
                q_value = self.get_qvalue(self.board, action)
                q_values.append((action, q_value))
            best_action = max(q_values, key=lambda x: x[1])[0]
        
        # 保存当前状态
        self.previous_state = copy.deepcopy(self.board)
        self.previous_action = best_action
        
        # 应用动作并获取奖励
        next_board = copy.deepcopy(self.board)
        next_board.apply_action(best_action)
        reward = self.get_reward(next_board, best_action)
        
        # 更新Q值
        self.update(reward, next_board)
        
        # 衰减探索率
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return best_action
    
    def update(self, reward, next_board):
        if self.previous_state is None or self.previous_action is None:
            return
        
        # 获取特征
        current_features = self.feature_extractor.get_features(
            self.previous_state, 
            self, 
            self.previous_action
        )
        
        # 计算当前Q值
        current_q = sum(self.weights[f] * v for f, v in current_features.items())
        
        # 计算下一状态的最大Q值
        next_max_q = float('-inf')
        next_actions = next_board.get_actions(self)
        if next_actions:
            next_q_values = [self.get_qvalue(next_board, a) for a in next_actions]
            next_max_q = max(next_q_values)
        else:
            next_max_q = 0
        
        # TD更新
        target = reward + self.gamma * next_max_q
        difference = target - current_q
        
        # 更新权重
        for feature, value in current_features.items():
            self.weights[feature] += self.alpha * difference * value
        
        # 清除状态
        self.previous_state = None
        self.previous_action = None
    
    def get_reward(self, board, action):
        reward = 0
        state = board.get_state()
        
        # 1. 胜利奖励大幅提高
        if state and state["winner"] == self:
            pieces_in_goal = sum(1 for i, j in board.get_goal_area(self)
                            if board.board[i][j] and 
                            board.board[i][j].player == self)
            return 3000 + pieces_in_goal * 500  # 基础奖励 + 额外奖励
        
        old_features = self.feature_extractor.get_features(self.previous_state, self)
        new_features = self.feature_extractor.get_features(board, self)
        
        # 2. 目标进展奖励（指数增长）
        goal_progress = new_features["pieces_in_goal"] - old_features["pieces_in_goal"]
        if goal_progress > 0:
            current_pieces = int(new_features["pieces_in_goal"] * 4)
            reward += 300 * (2 ** current_pieces)  # 指数增长奖励
        
        # 3. 距离改善奖励（根据棋子数调整）
        dist_improvement = old_features["avg_distance"] - new_features["avg_distance"]
        pieces_in_goal = int(new_features["pieces_in_goal"] * 4)
        if pieces_in_goal >= 2:
            reward += dist_improvement * 200  # 后期加大距离改善奖励
        else:
            reward += dist_improvement * 100
        
        # 4. 整体距离惩罚
        if new_features["avg_distance"] > old_features["avg_distance"]:
            reward -= 300
        
        # 5. 跳跃奖励（动态调整）
        if is_jump_move(board, action):
            if pieces_in_goal >= 3:
                reward += 0  # 接近胜利时不奖励跳跃
            elif pieces_in_goal >= 2:
                reward += 50  # 中期降低跳跃奖励
            else:
                reward += 200  # 前期鼓励跳跃
                
        # 6. 后退惩罚（动态调整）
        if is_going_backwards(board.boardsize, action, self):
            if pieces_in_goal >= 2:
                reward -= 500  # 后期加大后退惩罚
            else:
                reward -= 200
        
        return reward

class NeuralFeatureExtractor(nn.Module):
    def __init__(self, board_size=8):
        super().__init__()
        self.board_size = board_size
        
        # 1. 棋盘状态编码网络
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 2. 动作编码网络
        self.action_fc = nn.Linear(4, 32)
        
        # 3. 组合网络
        board_features = 64 * board_size * board_size
        combined_features = board_features + 32
        
        self.fc1 = nn.Linear(combined_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, board_state, action):
        # 1. 处理棋盘状态
        x1 = self.relu(self.conv1(board_state))
        x1 = self.relu(self.conv2(x1))
        x1 = self.relu(self.conv3(x1))
        x1 = x1.view(x1.size(0), -1)
        
        # 2. 处理动作
        x2 = self.relu(self.action_fc(action))
        
        # 3. 特征组合
        combined = torch.cat([x1, x2], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        return self.fc3(x)
    
    def encode_board(self, board, player):
        """将棋盘编码为4通道张量"""
        device = next(self.parameters()).device
        channels = torch.zeros((4, self.board_size, self.board_size), device=device)
        
        # Channel 0: 己方棋子
        # Channel 1: 对方棋子
        # Channel 2: 目标区域
        # Channel 3: 初始区域
        
        goal_area = set(board.get_goal_area(player))
        home_area = set(board.get_home_area(player))
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                pawn = board.board[i][j]
                if pawn:
                    if pawn.player.color == player.color:
                        channels[0, i, j] = 1
                    else:
                        channels[1, i, j] = 1
                        
                if (i, j) in goal_area:
                    channels[2, i, j] = 1
                if (i, j) in home_area:
                    channels[3, i, j] = 1
                    
        return channels.unsqueeze(0)  # 添加batch维度

    def encode_action(self, action):
        """将动作编码为向量"""
        device = next(self.parameters()).device
        return torch.tensor([action], dtype=torch.float32, device=device)

class Neural_ApproximateQLearningPlayer(AgentPlayer):
    def __init__(self, color, board_size=8, lr=0.001, gamma=0.99, epsilon=0.1):
        super().__init__(color)
        self.board_size = board_size
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 初始化网络和优化器
        self.network = NeuralFeatureExtractor(board_size)
        self.target_network = NeuralFeatureExtractor(board_size)
        self.target_network.load_state_dict(self.network.state_dict())
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.target_update = 1000
        self.steps = 0
        
        self.board = None
        
    def set_board(self, board):
        self.board = board
        
    def get_action(self, actions, training=True):
        if not actions:
            return None
            
        # 根据游戏阶段动态调整探索
        pieces_in_goal = sum(1 for i, j in self.board.get_goal_area(self)
                            if self.board.board[i][j] and 
                            self.board.board[i][j].player == self)
        
        exploration_prob = self.epsilon
        if not training:
            exploration_prob = 0.05
        elif pieces_in_goal >= 3:
            exploration_prob = 0.05
        elif pieces_in_goal >= 2:
            exploration_prob = 0.1
            
        if random.random() < exploration_prob:
            valid_actions = [a for a in actions if not is_going_backwards(self.board.boardsize, a, self)]
            return random.choice(valid_actions) if valid_actions else random.choice(actions)
        
        # 计算所有动作的Q值
        state_tensor = self.network.encode_board(self.board, self)
        max_q = float('-inf')
        best_action = None
        
        self.network.eval()
        with torch.no_grad():
            for action in actions:
                action_tensor = self.network.encode_action(action)
                q_value = self.network(state_tensor, action_tensor).item()
                
                if q_value > max_q:
                    max_q = q_value
                    best_action = action
        
        self.network.train()
        return best_action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def update_network(self):
        if len(self.memory) < self.batch_size:
            return
            
        # 随机采样批次
        batch = random.sample(self.memory, self.batch_size)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        
        for state, action, reward, next_state, done in batch:
            state_batch.append(self.network.encode_board(state, self))
            action_batch.append(self.network.encode_action(action))
            reward_batch.append(reward)
            next_state_batch.append(self.network.encode_board(next_state, self))
            done_batch.append(done)
            
        state_batch = torch.cat(state_batch)
        action_batch = torch.cat(action_batch)
        reward_batch = torch.tensor(reward_batch, device=self.device)
        next_state_batch = torch.cat(next_state_batch)
        done_batch = torch.tensor(done_batch, device=self.device)
        
        # 计算当前Q值
        current_q = self.network(state_batch, action_batch).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = torch.zeros_like(reward_batch)
            for idx, next_state in enumerate(next_state_batch):
                if not done_batch[idx]:
                    next_actions = self.board.get_actions(self)
                    if next_actions:
                        max_next_q = float('-inf')
                        for action in next_actions:
                            action_tensor = self.network.encode_action(action)
                            q = self.target_network(next_state.unsqueeze(0), action_tensor).item()
                            max_next_q = max(max_next_q, q)
                        next_q[idx] = max_next_q
                        
            target_q = reward_batch + self.gamma * next_q
        
        # 更新网络
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            
    def save_model(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']