# agents.py
from abc import ABC, abstractmethod
import random
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