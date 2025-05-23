# agents.py
from abc import ABC, abstractmethod
import random
import time
import json
from collections import defaultdict
import math
import copy

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
    my_pieces = []
    goal_area = set(board.get_goal_area(player))
    home_area = set(board.get_home_area(player))
    for i in range(board.boardsize):
        for j in range(board.boardsize):
            pawn = board.board[i][j]
            if pawn and pawn.player == player:
                my_pieces.append((i, j))
    
    # Define goal center based on player (simplified for two players; extend as needed)
    if player.index == 0:  # Top-left player, goal bottom-right
        goal_center = (board.boardsize - 1, board.boardsize - 1)
    elif player.index == 3:  # Bottom-right player, goal top-left
        goal_center = (0, 0)
    
    # Distance sum to goal center for pieces not in goal
    dist_sum = 0
    for (i, j) in my_pieces:
        if (i, j) not in goal_area:
            dist = math.hypot(goal_center[0] - i, goal_center[1] - j)
            dist_sum += dist
    
    # Rewards and penalties
    goal_bonus = 100 * sum(1 for (i, j) in my_pieces if (i, j) in goal_area)
    home_penalty = -50 * sum(1 for (i, j) in my_pieces if (i, j) in home_area and not board.board[i][j].has_left_home)
    
    # Jump bonus
    jump_bonus = 0
    directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    for (i, j) in my_pieces:
        for dx, dy in directions:
            mx, my = i + dx, j + dy
            jx, jy = i + 2 * dx, j + 2 * dy
            if (0 <= mx < board.boardsize and 0 <= my < board.boardsize and 
                0 <= jx < board.boardsize and 0 <= jy < board.boardsize):
                if board.board[mx][my] is not None and board.board[jx][jy] is None:
                    jump_bonus += 5
    
    # Lone wolf penalty
    lone_wolf_penalty = 0
    for (i, j) in my_pieces:
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ni, nj = i + dx, j + dy
                if 0 <= ni < board.boardsize and 0 <= nj < board.boardsize:
                    pawn2 = board.board[ni][nj]
                    if pawn2 and pawn2.player == player:
                        neighbors += 1
        if neighbors == 0:
            lone_wolf_penalty -= 2
    
    return -dist_sum + goal_bonus + home_penalty + jump_bonus + lone_wolf_penalty

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
        self.player_index = player_index
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
        if self.untried_actions is None:
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
        
        next_player_index = (self.player_index + 1) % len(mcts_player.board.players)
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

    def best_child(self, mcts_player, c_param=1.4, recent_moves=None):
        """Select child with highest UCB score"""
        def ucb_score(child):
            if child.visits == 0:
                return float('inf')
            
            exploit = child.value / child.visits
            explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
            
            # Add small bonuses/penalties to encourage good strategies
            strategy_bonus = 0
            
            # Bonus for moving toward goal
            if child.action:
                start_x, start_y, end_x, end_y = child.action
                goal_area = set(self.board.get_goal_area(mcts_player.board.players[child.player_index]))
                start_dist = min(((gx - start_x)**2 + (gy - start_y)**2) for gx, gy in goal_area)
                end_dist = min(((gx - end_x)**2 + (gy - end_y)**2) for gx, gy in goal_area)
                distance_improvement = start_dist - end_dist
                strategy_bonus += distance_improvement * 0.1
                
                # Bonus for jumps
                if is_jump_move(self.board, child.action):
                    strategy_bonus += 0.5
                
                # Penalty for moving backwards
                if is_going_backwards(self.board.boardsize, child.action, 
                                    mcts_player.board.players[child.player_index]):
                    strategy_bonus -= 1.0
            
            return exploit + explore + strategy_bonus

        return max(self.children, key=ucb_score)

    def most_visited_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.visits)


class MCTSPlayer(AgentPlayer):
    def __init__(self, color, simulations=2000, time_limit=5.0):
        super().__init__(color)
        self.simulations = simulations
        self.time_limit = time_limit
        self.board = None
        self.move_history = []

    def set_board(self, board):
        self.board = board

    def get_action(self, actions):
        if not actions:
            return None

        # If only one action, no need for MCTS
        if len(actions) == 1:
            return actions[0]

        root = MCTSNode(self.board, player_index=self.index)
        start_time = time.time()
        simulations = 0
        
        # Run simulations until time or count limit
        while simulations < self.simulations and (time.time() - start_time) < self.time_limit:
            node = root
            board_copy = copy.deepcopy(self.board)
            
            # Selection
            while not node.terminal and node.is_fully_expanded() and node.children:
                node = node.best_child(self)
                board_copy.apply_action(node.action)
            
            # Expansion
            if not node.terminal and not node.is_fully_expanded():
                node = node.expand(self)
                if node:
                    board_copy.apply_action(node.action)
            
            # Simulation
            if node and not node.terminal:
                reward = self.simulate(board_copy, node.player_index)
            else:
                # Terminal node reached during selection/expansion
                reward = 1000 if node.winner == self else -1000 if node.winner else 0
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward if node.player_index == self.index else -reward
                node = node.parent
            
            simulations += 1

        # Select best action based on most visited child
        best_child = root.most_visited_child()
        if not best_child:
            return random.choice(actions)

        # Update move history for diversity
        self.move_history.append(best_child.action)
        if len(self.move_history) > 5:
            self.move_history.pop(0)

        return best_child.action

    def simulate(self, board, current_player_index):
        """Fast simulation with heuristic evaluation"""
        # Use a combination of random moves and heuristic evaluation
        max_steps = 20  # Limit simulation depth
        current_player = board.players[current_player_index]
        opponent_idx = (current_player_index + 1) % len(board.players)
        opponent = board.players[opponent_idx]
        
        for _ in range(max_steps):
            state = board.get_state()
            if state:
                winner = state["winner"]
                if winner == self:
                    return 1000
                elif winner == opponent:
                    return -1000
                return 0
            
            current_player = board.players[current_player_index]
            actions = board.get_actions(current_player)
            if not actions:
                break
                
            # Use heuristic to select moves during simulation
            if current_player_index == self.index:
                # For our player, use greedy moves
                best_action = None
                best_value = float('-inf')
                for action in actions:
                    temp_board = copy.deepcopy(board)
                    temp_board.apply_action(action)
                    value = evaluation(temp_board, self)
                    if value > best_value:
                        best_value = value
                        best_action = action
            else:
                # For opponent, mix random and greedy moves
                if random.random() < 0.7:  # 70% chance for greedy move
                    best_action = None
                    best_value = float('inf')
                    for action in actions:
                        temp_board = copy.deepcopy(board)
                        temp_board.apply_action(action)
                        value = evaluation(temp_board, self)
                        if value < best_value:
                            best_value = value
                            best_action = action
                else:
                    best_action = random.choice(actions)
            
            board.apply_action(best_action)
            current_player_index = (current_player_index + 1) % len(board.players)
        
        # Final evaluation
        return evaluation(board, self) - evaluation(board, opponent)

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