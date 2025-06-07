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
        self.players_finished_goal = [] # 新增：记录完成目标区的玩家顺序
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
        self.players_finished_goal = [] # 重置完成列表
        self.initialize_board()

    def clone(self):
        """创建棋盘的深度复制，确保玩家分数等状态独立"""
        new_board = object.__new__(Board)

        # 2. 复制棋盘的基本属性
        new_board.boardsize = self.boardsize
        new_board.mode = self.mode
        new_board.turn = self.turn  # 复制当前回合数
        # 复制已完成目标区的玩家列表（列表本身是新的，颜色字符串是不可变的）
        new_board.players_finished_goal = list(self.players_finished_goal) 

        # 3. 创建玩家对象的独立副本，并保留其当前状态（包括分数）
        cloned_players_list = []
        # player_map 用于将原始玩家实例映射到其克隆副本，以便在复制棋子时使用
        player_map = {} 

        for original_player in self.players:
            cloned_player = DefaultPlayer(original_player.color)
            cloned_player.score = original_player.score  # 复制当前分数
            cloned_player.index = original_player.index  # 复制索引
            
            # 需要设置其 board 属性为 new_board
            if hasattr(cloned_player, 'set_board'):
                cloned_player.set_board(new_board)
            
            cloned_players_list.append(cloned_player)
            player_map[original_player] = cloned_player # 存储原始玩家到克隆玩家的映射
        
        new_board.players = cloned_players_list # 将克隆的玩家列表赋给新棋盘

        # 4. 初始化并复制棋盘上的棋子
        # 创建一个新的二维列表来存储棋盘上的棋子
        new_board.board = [[None for _ in range(new_board.boardsize)] for _ in range(new_board.boardsize)]
        for r in range(self.boardsize):
            for c in range(self.boardsize):
                original_pawn = self.board[r][c]
                if original_pawn:
                    # 获取此棋子对应的克隆玩家实例
                    owner_cloned_player = player_map[original_pawn.player]
                    
                    # 为新棋盘创建一个新的 Pawn 实例
                    # 新棋子将引用 new_board 和 owner_cloned_player
                    new_pawn = Pawn(new_board, owner_cloned_player, original_pawn.x, original_pawn.y)
                    # 复制棋子的特定状态
                    new_pawn.is_in_goal = original_pawn.is_in_goal
                    new_pawn.has_left_home = original_pawn.has_left_home
                    
                    new_board.board[r][c] = new_pawn # 将新棋子放置在新棋盘上
                
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
        
    def get_mode(self):
        """获取当前游戏模式"""
        return self.mode
    
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
    
    def get_jump(self, action) -> int:
        """获取当前动作的连跳次数"""
        start_x, start_y, end_x, end_y = action
        pawn = self.board[start_x][start_y]
        if pawn is None:
            return 0
        
        # 获取所有合法动作
        actions_with_jumps = self.get_actions_with_jump(pawn.player)
        
        # 查找当前动作对应的连跳次数
        for action_tuple in actions_with_jumps:
            stored_action, jump_count = action_tuple
            if stored_action == action:
                return jump_count
        
        return 0
    
    def get_action_score(self, action):
        """获取当前动作的得分"""
        start_x, start_y, end_x, end_y = action
        pawn = self.board[start_x][start_y]
        if pawn is None:
            raise ValueError("Invalid move: No pawn at starting position.")
        
        player_obj = pawn.player
        goal_area = set(self.get_goal_area(player_obj))
        home_area = set(self.get_home_area(player_obj))  # 获取家区域
        
        # 检查棋子移动前是否已经在目标区域
        was_in_goal = hasattr(pawn, 'is_in_goal') and pawn.is_in_goal

        # 检查连跳次数
        jump_count = self._get_jump_count(action, player_obj)

        score = 0
        # 棋子首次进入目标区域
        is_now_in_goal = (end_x, end_y) in goal_area
        if is_now_in_goal and not was_in_goal:
            score += 10

        # 连跳得分：连跳n次得n分，但前提是棋子移动前不在目标区域且不在家区域
        is_in_home_area = (start_x, start_y) in home_area
        if jump_count > 0 and not was_in_goal and not is_in_home_area:
            score += jump_count * 10

        # 检查此动作是否导致玩家所有棋子都进入目标区域
        # 首先，统计玩家棋子总数，并检查在此动作之前是否所有棋子都已在目标区
        all_pawns_in_goal_before_this_action_flag = True
        num_player_pawns_on_board = 0
        for r_before in range(self.boardsize):
            for c_before in range(self.boardsize):
                p_before = self.board[r_before][c_before]
                if p_before and p_before.player == player_obj:
                    num_player_pawns_on_board += 1
                    if (r_before, c_before) not in goal_area:
                        all_pawns_in_goal_before_this_action_flag = False
        
        # 然后，检查此动作之后是否所有棋子都在目标区
        if not all_pawns_in_goal_before_this_action_flag and num_player_pawns_on_board > 0:
            all_pawns_in_goal_after_this_action_flag = True
            current_pawns_in_goal_after_action_count = 0
            for r_after in range(self.boardsize):
                for c_after in range(self.boardsize):
                    p_after_obj = self.board[r_after][c_after] # Original pawn object on board
                    
                    # Determine the state of this cell *after* the action
                    current_cell_pawn = None
                    current_cell_pos_is_goal = False

                    if r_after == end_x and c_after == end_y: # This is the destination of the moved pawn
                        current_cell_pawn = pawn 
                        current_cell_pos_is_goal = (end_x, end_y) in goal_area
                    elif r_after == start_x and c_after == start_y: # This is the start, will be empty
                        current_cell_pawn = None
                    else: # Other cells
                        current_cell_pawn = p_after_obj
                        current_cell_pos_is_goal = (r_after, c_after) in goal_area
                    
                    if current_cell_pawn and current_cell_pawn.player == player_obj:
                        if current_cell_pos_is_goal:
                            current_pawns_in_goal_after_action_count += 1
                        else:
                            all_pawns_in_goal_after_this_action_flag = False
                            break
                if not all_pawns_in_goal_after_this_action_flag: # break outer loop
                    pass # Handled by the break from inner loop

            if all_pawns_in_goal_after_this_action_flag and current_pawns_in_goal_after_action_count == num_player_pawns_on_board:
                if player_obj.color not in self.players_finished_goal:
                    if len(self.players_finished_goal) == 0:
                        score += 200  # 第一个完成所有棋子进入目标区的玩家
                    elif len(self.players_finished_goal) == 1:
                        score += 100  # 第二个完成所有棋子进入目标区的玩家

        return score
    
    def apply_action(self, action):
        start_x, start_y, end_x, end_y = action
        pawn = self.board[start_x][start_y]
        if pawn is not None:
            player_obj = pawn.player # 保存玩家对象
            # 获取此动作将产生的得分
            score_for_this_action = self.get_action_score(action)
            
            # 移动棋子
            self.board[end_x][end_y] = pawn
            self.board[start_x][start_y] = None
            pawn.move(end_x, end_y) # pawn.move 会更新 pawn.is_in_goal
            
            # 更新玩家分数
            player_obj.score += score_for_this_action

            # 检查玩家是否因为此动作完成了所有棋子到目标区，并且尚未记录
            if player_obj.color not in self.players_finished_goal:
                all_pawns_now_in_goal = True
                num_player_pawns_on_board_now = 0
                player_goal_area = set(self.get_goal_area(player_obj))
                for r in range(self.boardsize):
                    for c in range(self.boardsize):
                        p_check = self.board[r][c]
                        if p_check and p_check.player == player_obj:
                            num_player_pawns_on_board_now +=1
                            if (r, c) not in player_goal_area:
                                all_pawns_now_in_goal = False
                                break
                    if not all_pawns_now_in_goal:
                        break
                
                if all_pawns_now_in_goal and num_player_pawns_on_board_now > 0:
                    self.players_finished_goal.append(player_obj.color)
            
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
        if self.mode == "classic":
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
                    max_player = self.get_max_player()
                    return max_player
            
            # 如果没有玩家赢，游戏继续
            return None
    
        elif self.mode == "score":
            """计分模式：当所有玩家都将其所有棋子移动到目标区域时，游戏结束。"""
            
            for player in self.players:
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
                if not all_in_goal_area:
                    return None
            
            max_player = self.get_max_player()
            return max_player