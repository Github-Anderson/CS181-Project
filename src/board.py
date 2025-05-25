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
    def __init__(self, boardsize, players):
        self.boardsize = boardsize
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
        new_board = Board(self.boardsize, self.players)
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

        # TODO: Implement for more players
        elif len(self.players) == 4:
            pass
    
    def get_home_area(self, player): 
        home = []
        size = min(6, self.boardsize // 2)
        if player.index == 0:
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        home.append((i, j))
        elif player.index == 1:
            pass
        elif player.index == 2:
            pass
        elif player.index == 3:
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        home.append((self.boardsize - i - 1, self.boardsize - j - 1))
        return home
    
    def get_goal_area(self, player):
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
                        goal.append((i, self.boardsize - j - 1))
        elif player.index == 2:  # 左下角玩家的目标是右上角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        goal.append((self.boardsize - i - 1, j))
        elif player.index == 3:  # 右下角玩家的目标是左上角
            for i in range(size):
                for j in range(size):
                    if (i + j) < size:
                        goal.append((i, j))
    
        return goal

    def get_actions(self, player : Player) -> list[tuple[int, int, int, int]]:
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
    
    def apply_action(self, action):
        start_x, start_y, end_x, end_y = action
        pawn = self.board[start_x][start_y]
        if pawn is not None:
            self.board[end_x][end_y] = pawn
            self.board[start_x][start_y] = None
            pawn.move(end_x, end_y)
            self.turn += 1
        else:
            raise ValueError("Invalid move: No pawn at starting position.")
    
    def get_state(self):
        # 检查每个玩家的棋子是否全部进入对方的家
        for player_idx, player in enumerate(self.players):
            # 获取对方的家
            goal_area = self.get_goal_area(player)
            
            # 获取当前玩家的所有棋子位置
            player_pawns = []
            for i in range(self.boardsize):
                for j in range(self.boardsize):
                    if self.board[i][j] and self.board[i][j].player == player:
                        player_pawns.append((i, j))
            
            # 检查是否所有棋子都在对方家中
            all_in_goal_area = all((x, y) in goal_area for x, y in player_pawns)
            
            if all_in_goal_area and len(player_pawns) > 0:
                return {
                    "winner": player,
                    "message": f"{player.color} 获胜!"
                }
        
        # 如果没有玩家赢，游戏继续
        return None
