from agents import *
from utils import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents import Player
    from src.board import Board

class Pawn:
    def __init__(self, board: 'Board', player: 'Player', x, y):
        self.board = board
        self.player = player
        self.color = player.color
        self.x = x
        self.y = y
        self.is_in_goal = False
        self.has_left_home = False

    def move(self, new_x, new_y):
        home_area = set(self.board.get_home_area(self.player))
        goal_area = set(self.board.get_goal_area(self.player))
        
        # 更新是否在目标区域
        if (new_x, new_y) in goal_area:
            self.is_in_goal = True
        
        # 更新是否离开家
        if not self.has_left_home and (self.x, self.y) in home_area and (new_x, new_y) not in home_area:
            self.has_left_home = True
        
        # 更新位置
        self.x = new_x
        self.y = new_y


class Board:
    def __init__(self, boardsize, mode, players : list[Player]):
        self.boardsize = boardsize
        self.mode = mode
        self.players = players
        self.board = [[None for _ in range(boardsize)] for _ in range(boardsize)]
        self.initialize_board()

    def __str__(self):
        rows = []
        for i in range(self.boardsize):
            row = []
            for j in range(self.boardsize):
                if self.board[i][j] is None:
                    row.append(".")
                else:
                    row.append(self.board[i][j].color[0])
            rows.append(" ".join(row))
        return "\n".join(rows)
    
    def reset(self):
        """重置棋盘到初始状态"""
        # 清空棋盘
        self.board = [[None for _ in range(self.boardsize)] for _ in range(self.boardsize)]
        self.initialize_board()

    def clone(self):
        """创建棋盘的深度复制"""
        import copy
        new_board = Board(self.boardsize, self.mode, self.players)
        new_board.board = [[None for _ in range(self.boardsize)] for _ in range(self.boardsize)]
        
        # 复制棋子
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                if self.board[i][j]:
                    pawn = self.board[i][j]
                    new_pawn = Pawn(new_board, pawn.player, pawn.x, pawn.y)
                    new_pawn.is_in_goal = pawn.is_in_goal
                    new_pawn.has_left_home = pawn.has_left_home
                    new_board.board[i][j] = new_pawn
        
        new_board.turn = self.turn
        return new_board

    def initialize_board(self):
        """初始化棋盘"""
        for player in self.players:
            player.score = 0

        self.turn = 1
        if len(self.players) == 2:
            self.players[0].set_index(0)
            self.players[1].set_index(3)
            
            size = min(6, self.boardsize // 2)
            for i in range(size):
                for j in range(size):
                    if (i + j) < (size) and i < 6 and j < 6:
                        self.board[i][j] = Pawn(self, self.players[0], i, j)

            for i in range(size):
                for j in range(size):
                    if (i + j) < (size):
                        self.board[self.boardsize - i - 1][self.boardsize - j - 1] = Pawn(self, self.players[1], self.boardsize - i - 1, self.boardsize - j - 1)

        elif len(self.players) == 4:
            self.players[0].set_index(0)
            self.players[1].set_index(1)
            self.players[2].set_index(2)
            self.players[3].set_index(3)

            size = min(6, self.boardsize // 2)

            # 左上角玩家
            for i in range(size):
                for j in range(size):
                    if (i + j) < size and i < 6 and j < 6:
                        self.board[i][j] = Pawn(self, self.players[0], i, j)
            # 右上角玩家
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        self.board[i][self.boardsize - j - 1] = Pawn(self, self.players[1], i, self.boardsize - j - 1)
            # 左下角玩家
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        self.board[self.boardsize - i - 1][j] = Pawn(self, self.players[2], self.boardsize - i - 1, j)
            # 右下角玩家
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        self.board[self.boardsize - i - 1][self.boardsize - j - 1] = Pawn(self, self.players[3], self.boardsize - i - 1, self.boardsize - j - 1)
        
    def get_home_area(self, player): 
        """获取玩家的家区域"""
        home = []
        size = min(6, self.boardsize // 2)
        if player.index == 0:  # 左上角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        home.append((i, j))
        elif player.index == 1:  # 右上角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        home.append((i, self.boardsize - j - 1))
        elif player.index == 2:  # 左下角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        home.append((self.boardsize - i - 1, j))
        elif player.index == 3:  # 右下角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        home.append((self.boardsize - i - 1, self.boardsize - j - 1))

        return home
    
    def get_goal_area(self, player):
        """获取玩家的目标区域"""
        goal = []
        size = min(6, self.boardsize // 2)
        
        # 根据玩家的索引确定目标区域
        if player.index == 0:  # 左上角玩家的目标是右下角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        goal.append((self.boardsize - i - 1, self.boardsize - j - 1))
        elif player.index == 1:  # 右上角玩家的目标是左下角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        goal.append((self.boardsize - i - 1, j))
        elif player.index == 2:  # 左下角玩家的目标是右上角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        goal.append((i, self.boardsize - j - 1))
        elif player.index == 3:  # 右下角玩家的目标是左上角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        goal.append((i, j))
    
        return goal

    def get_actions(self, player : Player) -> list[tuple[int, int, int, int]]:
        """获取当前玩家的所有合法动作"""
        actions = set()
        directions = list(DIRECTION_OFFSET.values())
        
        home_area = set(self.get_home_area(player))
        goal_area = set(self.get_goal_area(player))

        def in_board(x, y):
            return 0 <= x < self.boardsize and 0 <= y < self.boardsize

        def jump_moves(x, y, visited):
            results = []
            for dx, dy in directions:
                mx, my = x + dx, y + dy
                jx, jy = x + 2*dx, y + 2*dy
                # check hop over occupied and land on empty
                if (
                    in_board(jx, jy)
                    and self.board[mx][my] is not None
                    and self.board[jx][jy] is None
                    and (jx, jy) not in visited
                ):
                    results.append((jx, jy))
                    # recursive hops
                    results.extend(jump_moves(jx, jy, visited | {(jx, jy)}))
            return results

        for i in range(self.boardsize):
            for j in range(self.boardsize):
                pawn = self.board[i][j]
                if pawn and pawn.player == player:
                    # simple one-step moves
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if in_board(ni, nj) and self.board[ni][nj] is None:
                            actions.add((i, j, ni, nj))
                    # multi-hop jumps
                    for nx, ny in jump_moves(i, j, {(i, j)}):
                        actions.add((i, j, nx, ny))

        # return sorted list for deterministic ordering
        filtered_actions = set()
        for action in actions:
            start_x, start_y, end_x, end_y = action
            pawn = self.board[start_x][start_y]
            
            # 已到达目标区域的棋子不能离开目标区域
            if hasattr(pawn, 'is_in_goal') and pawn.is_in_goal and (end_x, end_y) not in goal_area:
                continue
            
            # 已离开家的棋子不能回到家
            if hasattr(pawn, 'has_left_home') and pawn.has_left_home and (end_x, end_y) in home_area:
                continue
            
            filtered_actions.add(action)
        
        # 返回过滤后的动作
        return sorted(filtered_actions)
    
    def get_actions_with_jump(self, player: Player) -> list[tuple[tuple[int, int, int, int], int]]:
        """获取当前玩家的所有合法动作及其连跳次数"""
        actions_with_jumps = []
        directions = list(DIRECTION_OFFSET.values())
        
        home_area = set(self.get_home_area(player))
        goal_area = set(self.get_goal_area(player))

        def in_board(x, y):
            return 0 <= x < self.boardsize and 0 <= y < self.boardsize

        def jump_moves(x, y, visited, jump_count=0):
            results = []
            for dx, dy in directions:
                mx, my = x + dx, y + dy
                jx, jy = x + 2*dx, y + 2*dy
                # check hop over occupied and land on empty
                if (
                    in_board(jx, jy)
                    and self.board[mx][my] is not None
                    and self.board[jx][jy] is None
                    and (jx, jy) not in visited
                ):
                    current_jump_count = jump_count + 1
                    results.append((jx, jy, current_jump_count))
                    # recursive hops
                    results.extend(jump_moves(jx, jy, visited | {(jx, jy)}, current_jump_count))
            return results

        for i in range(self.boardsize):
            for j in range(self.boardsize):
                pawn = self.board[i][j]
                if pawn and pawn.player == player:
                    # simple one-step moves (jump_count = 0)
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if in_board(ni, nj) and self.board[ni][nj] is None:
                            actions_with_jumps.append(((i, j, ni, nj), 0))
                    
                    # multi-hop jumps
                    for nx, ny, jump_count in jump_moves(i, j, {(i, j)}):
                        actions_with_jumps.append(((i, j, nx, ny), jump_count))

        # 过滤不合法的动作
        filtered_actions = []
        for action_tuple in actions_with_jumps:
            action, jump_count = action_tuple
            start_x, start_y, end_x, end_y = action
            pawn = self.board[start_x][start_y]
            
            # 已到达目标区域的棋子不能离开目标区域
            if hasattr(pawn, 'is_in_goal') and pawn.is_in_goal and (end_x, end_y) not in goal_area:
                continue
            
            # 已离开家的棋子不能回到家
            if hasattr(pawn, 'has_left_home') and pawn.has_left_home and (end_x, end_y) in home_area:
                continue
            
            filtered_actions.append(action_tuple)
        
        # 返回过滤后的动作（按动作排序以保证确定性）
        return sorted(filtered_actions, key=lambda x: x[0])
    
    def apply_action(self, action):
        start_x, start_y, end_x, end_y = action
        pawn = self.board[start_x][start_y]
        if pawn is not None:
            # 检查棋子移动前是否已经在目标区域
            goal_area = set(self.get_goal_area(pawn.player))
            was_in_goal = hasattr(pawn, 'is_in_goal') and pawn.is_in_goal
            
            # 检查连跳次数
            jump_count = self._get_jump_count(action, pawn.player)
            
            # 移动棋子
            self.board[end_x][end_y] = pawn
            self.board[start_x][start_y] = None
            pawn.move(end_x, end_y)
            
            # 检查棋子移动后是否进入目标区域（首次进入）
            is_now_in_goal = (end_x, end_y) in goal_area
            if is_now_in_goal and not was_in_goal:
                # 棋子首次进入目标区域，玩家得5分
                pawn.player.score += 5
            
            # 连跳得分：连跳n次得n分
            if jump_count > 0:
                pawn.player.score += jump_count
            
            self.turn += 1
        else:
            raise ValueError("Invalid move: No pawn at starting position.")
    
    def _get_jump_count(self, action, player):
        """计算指定动作的连跳次数"""
        actions_with_jumps = self.get_actions_with_jump(player)
        for action_tuple in actions_with_jumps:
            stored_action, jump_count = action_tuple
            if stored_action == action:
                return jump_count
        return 0
    
    def get_max_score(self) -> int:
        """获取当前游戏模式下的最大分数"""
        max_score = 0
        for player in self.players:
            max_score = max(max_score, player.score)

        return max_score
    
    def get_max_player(self) -> Player:
        """获取当前游戏模式下分数最高的玩家"""
        max_score = self.get_max_score()
        for player in self.players:
            if player.score == max_score:
                return player
            
        return None
    
    def get_state(self) -> Player:
        """获取当前棋盘状态"""
        if self.mode == 'score':
            all_players_are_finished = True
            if not self.players: # 如果没有玩家，游戏无法根据玩家状态结束
                return None

            for player_idx, player in enumerate(self.players):
                goal_area = self.get_goal_area(player)

                player_pawns = []
                for i in range(self.boardsize):
                    for j in range(self.boardsize):
                        if self.board[i][j] and self.board[i][j].player == player:
                            player_pawns.append((i, j))
                
                # 检查所有棋子是否都在目标区域
                all_in_goal_area = all((x, y) in goal_area for x, y in player_pawns)

                # 如果所有棋子都在目标区域，玩家获胜
                if not all_in_goal_area and len(player_pawns) > 0:
                    all_players_are_finished = False

            # 如果所有玩家都完成了，游戏结束
            if all_players_are_finished:
                max_player = self.get_max_player()
                return max_player
            else:
                return None

        elif self.mode == 'classic':
            # 检查每个玩家的棋子是否全部进入对方的家
            for player_idx, player in enumerate(self.players):
                # 获取玩家的目标区域
                goal_area = self.get_goal_area(player)

                # 获取玩家的所有棋子位置
                player_pawns = []
                for i in range(self.boardsize):
                    for j in range(self.boardsize):
                        if self.board[i][j] and self.board[i][j].player == player:
                            player_pawns.append((i, j))

                # 检查所有棋子是否都在目标区域
                all_in_goal_area = all((x, y) in goal_area for x, y in player_pawns)
                
                # 如果所有棋子都在目标区域，玩家获胜
                if all_in_goal_area and len(player_pawns) > 0:
                    return player
            
            # 如果没有玩家赢，游戏继续
            return None
