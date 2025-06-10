from agents import HumanPlayer
from utils import *
from board import *
import tkinter as tk

class Engine:
    def __init__(self, board : Board, max_turns : int = 500):
        self.board = board
        self.players = board.players
        self.current_player_index = 0
        self.current_player = None
        self.game_over = False
        self.waiting_for_human = False
        self.human_action = None
        self.turn_count = 0
        self.max_turns = max_turns

    def start(self):
        self.gui = BoardGUI(self.board, self)
        # time.sleep(4)
        self.gui.after(10, self.game_loop)
        self.gui.mainloop()

    def game_loop(self):
        # 检查是否达到最大回合数
        if self.turn_count >= self.max_turns:
            self.game_over = True
            message = f"游戏达到最大回合数 ({self.max_turns}). "
            winner = None # 初始化winner变量

            # 检查是否有玩家已经通过常规方式获胜 (e.g. classic mode goal reached)
            game_state_winner = self.board.get_state()
            if game_state_winner:
                winner = game_state_winner
                winner_color_name = self.gui.color_names.get(winner.color, winner.color)
                message = f"**{winner_color_name}** 玩家获胜!" # 覆盖最大回合数消息
            else:
                # 按目标区域棋子数决定胜负
                pawns_in_goal_counts = {}
                for player_obj in self.players: # 直接迭代 self.players
                    goal_area = set(self.board.get_goal_area(player_obj))
                    count = 0
                    for r in range(self.board.boardsize):
                        for c in range(self.board.boardsize):
                            pawn = self.board.board[r][c]
                            if pawn and pawn.player == player_obj and (r, c) in goal_area:
                                count += 1
                    pawns_in_goal_counts[player_obj] = count
                
                if not pawns_in_goal_counts:
                    message += "平局!"
                    # winner 保持 None
                else:
                    max_pawns = -1
                    # Find max count first
                    for player_obj in self.players: # 确保迭代 self.players
                         if pawns_in_goal_counts[player_obj] > max_pawns:
                            max_pawns = pawns_in_goal_counts[player_obj]

                    # Identify all players with that max count
                    winners_by_pawns = [p for p, count in pawns_in_goal_counts.items() if count == max_pawns and max_pawns > -1]

                    if len(winners_by_pawns) == 1 and max_pawns > 0:
                        winner = winners_by_pawns[0]
                        winner_color_name = self.gui.color_names.get(winner.color, winner.color)
                        message += f"目标区棋子最多者: **{winner_color_name}** ({max_pawns}个棋子)"
                    elif len(winners_by_pawns) > 1 and max_pawns > 0:
                        message += "平局! (多名玩家目标区棋子数相同: "
                        winner_names = [self.gui.color_names.get(w.color, w.color) for w in winners_by_pawns]
                        message += ", ".join(winner_names) + f" 均为{max_pawns}个)"
                        # winner 保持 None
                    else: # No player has any pawns in goal (max_pawns is 0 or -1)
                        message += "平局! (无玩家在目标区有棋子或棋子数均为0)"
                        # winner 保持 None
            
            self.gui.update_status(message)
            return

        game_state = self.board.get_state() # get_state() 返回 Player 对象或 None

        if game_state:
            self.game_over = True
            winner = game_state # 返回获胜玩家

            message = ""
            if self.board.mode == 'score':
                winner_color_name = self.gui.color_names.get(winner.color, winner.color)
                message = f"分数最高者: **{winner_color_name}** ({winner.score}分)"
            elif self.board.mode == 'classic':
                winner_color_name = self.gui.color_names.get(winner.color, winner.color)
                message = f"**{winner_color_name}** 玩家获胜!"

            self.gui.update_status(message)
            return
        
        for player in self.players:
            if self.current_player_index == player.index:
                self.current_player = player
                break
            
        # 获取当前玩家可用的动作
        actions = self.board.get_actions(self.current_player)
        if not actions:
            # 如果当前玩家没有可用动作，直接进入下一个玩家的回合
            self.gui.update_status(f"**{self.current_player.color}** 玩家无可用动作，跳过回合")
            self.next_turn()
            self.gui.after(10, self.game_loop)
            return
        
        # 如果是人类玩家，等待GUI输入
        if isinstance(self.current_player, HumanPlayer):
            if not self.waiting_for_human:
                self.waiting_for_human = True
                color_name = self.gui.color_names.get(self.current_player.color, self.current_player.color)
                self.gui.update_status(f"**{color_name}** 玩家回合 - 请选择棋子")
            elif self.human_action:
                # 人类玩家已经选择了动作
                self.turn_count += 1
                self.board.apply_action(self.human_action)
                self.human_action = None
                self.waiting_for_human = False
                self.next_turn()
        else:
            # AI玩家自动行动
            color_name = self.gui.color_names.get(self.current_player.color, self.current_player.color)
            self.gui.update_status(f"**{color_name}** 玩家回合 - AI思考中...")
            self.gui.update()  # 刷新界面显示
            
            # 添加短暂延迟，让玩家能看到AI行动
            # time.sleep(0.5)
            
            action = self.current_player.get_action(actions)
            self.turn_count += 1
            self.board.apply_action(action)
            self.next_turn()
        
        # 重绘棋盘
        self.gui.draw_pawns()
        
        # 更新分数显示
        self.gui.update_scores()
        
        self.gui.after(10, self.game_loop)

    def next_turn(self):
        """进入下一个玩家的回合"""
        current_pos = 0
        for i, player in enumerate(self.players):
            if player.index == self.current_player_index:
                current_pos = i
                break
        
        # 切换到列表中的下一个玩家
        next_pos = (current_pos + 1) % len(self.players)
        self.current_player_index = self.players[next_pos].index
        
    def restart_game(self):
        """重新开始游戏"""
        # 重置引擎状态
        self.stop_game = False
        self.game_over = False
        self.current_player_index = 0
        # self.current_player = self.players[0] # current_player 会在 game_loop 开始时设置
        self.current_player = None # 重置 current_player
        self.waiting_for_human = False
        self.human_action = None
        self.turn_count = 0
        
        # 重置棋盘
        self.board.reset()
        
        # 重新设置AI玩家的棋盘引用
        for player in self.players:
            if hasattr(player, 'set_board'):
                player.set_board(self.board)
        
        # 重新开始游戏循环
        self.gui.after(100, self.game_loop)  # 稍微延迟确保停止完成
    
    def human_move(self, action):
        """接收来自GUI的人类玩家动作"""
        if self.waiting_for_human and isinstance(self.current_player, HumanPlayer):
            self.human_action = action


class HeadlessEngine:
    def __init__(self, board: Board, max_turns=MAX_TURNS):
        self.board = board
        self.players = board.players # This is the list of player objects
        
        if not self.players:
            raise ValueError("Player list is empty in HeadlessEngine.")
        # Start with the canonical index of the first player in the board's player list
        self.current_player_canonical_index = self.players[0].index 
        
        self.game_over = False
        self.winner = None # Stores the winning Player object
        self.game_turn_count = 0 # Counts number of moves made
        self.max_turns = max_turns # To prevent infinite games

    def run_game(self):
        while not self.game_over and self.game_turn_count < self.max_turns:
            # Check for game end condition based on board state
            game_state_winner = self.board.get_state() 
            if game_state_winner:
                self.game_over = True
                self.winner = game_state_winner
                print(f"Game over. Winner by game rules: {self.winner.color}")
                break

            current_player_obj = None
            for p in self.players:
                if p.index == self.current_player_canonical_index:
                    current_player_obj = p
                    break
            
            if current_player_obj is None:
                print(f"Error: Could not find player with canonical index {self.current_player_canonical_index}")
                self.game_over = True # End game if error
                break

            actions = self.board.get_actions(current_player_obj)
            
            if not actions:
                # print(f"Player {current_player_obj.color} has no actions. Skipping turn.")
                self.next_turn()
                continue

            action = current_player_obj.get_action(actions)
            
            if action is None:
                if actions: # Agent failed to pick an action but actions were available
                    print(f"Warning: Player {current_player_obj.color} ({type(current_player_obj).__name__}) returned None action despite available actions. Choosing random.")
                    action = random.choice(actions)
                else: # No actions available, already handled by the 'if not actions:' block
                    self.next_turn()
                    continue
            
            self.board.apply_action(action)
            self.game_turn_count += 1
            # print(f"Turn {self.game_turn_count}: Player {current_player_obj.color} ({type(current_player_obj).__name__}) takes action {action}")
            # print(self.board) # Optional: print board state for debugging
            self.next_turn()

        # After loop: game ended either by win condition or max_turns
        if not self.game_over and self.game_turn_count >= self.max_turns:
            # print(f"Game ended due to max turns ({self.max_turns}).")
            # Try to get winner via get_state() one last time (e.g. if last move was winning)
            final_winner_check = self.board.get_state()
            if final_winner_check:
                self.winner = final_winner_check
                print(f"Winner after max turns (by game rules): {self.winner.color}")
            else: # No winner by game rules yet, decide by pawns in goal
                pawns_in_goal_counts = {}
                for player_obj in self.players:
                    goal_area = set(self.board.get_goal_area(player_obj))
                    count = 0
                    for r in range(self.board.boardsize):
                        for c in range(self.board.boardsize):
                            pawn = self.board.board[r][c]
                            if pawn and pawn.player == player_obj and (r, c) in goal_area:
                                count += 1
                    pawns_in_goal_counts[player_obj] = count
                
                if not pawns_in_goal_counts:
                    print("Max turns reached. No players found for pawn count. Declaring draw.")
                    self.winner = None
                else:
                    max_pawns = -1
                    # Find max count first
                    for player_obj in self.players:
                        if pawns_in_goal_counts[player_obj] > max_pawns:
                            max_pawns = pawns_in_goal_counts[player_obj]
                    
                    # Identify all players with that max count
                    winners_by_pawns = [p for p, count in pawns_in_goal_counts.items() if count == max_pawns and max_pawns > -1]

                    if len(winners_by_pawns) == 1 and max_pawns > 0:
                        self.winner = winners_by_pawns[0]
                        print(f"Winner by most pawns in goal ({max_pawns}): {self.winner.color}")
                    elif len(winners_by_pawns) > 1 and max_pawns > 0: # Multiple players with same max pawns
                        winner_colors = [w.color for w in winners_by_pawns]
                        print(f"Draw by most pawns in goal. Players {', '.join(winner_colors)} all have {max_pawns} pawns.")
                        self.winner = None # Draw
                    else: # No player has pawns in goal (max_pawns is 0 or -1)
                        print("Max turns reached. No player has pawns in their goal area or pawn counts are all zero. Declaring draw.")
                        self.winner = None # Draw
            self.game_over = True

        return self.winner # Returns Player object or None (for a draw)

    def next_turn(self):
        current_player_list_position = -1
        # Find current player in the self.players list (which is ordered) by its canonical index
        for i, p_in_list in enumerate(self.players):
            if p_in_list.index == self.current_player_canonical_index:
                current_player_list_position = i
                break
        
        if current_player_list_position == -1:
            # This should not happen if current_player_canonical_index is always valid
            print(f"Error: Could not find current player with canonical index {self.current_player_canonical_index} in player list.")
            # Fallback: cycle through the list directly, though this might break turn order logic if indices are complex
            # For safety, just move to the first player's index if lost.
            self.current_player_canonical_index = self.players[0].index
            return

        # Get the next player from the list by cycling list position
        next_player_list_position = (current_player_list_position + 1) % len(self.players)
        # Update current_player_canonical_index to the canonical index of the next player
        self.current_player_canonical_index = self.players[next_player_list_position].index


class BoardGUI(tk.Tk):
    def __init__(self, board : Board, engine : Engine, *args, **kwargs):
        # initialize parent tk class
        tk.Tk.__init__(self, *args, **kwargs)

        # 颜色名称映射
        self.color_names = {
            'RED': '红色',
            'GREEN': '绿色', 
            'BLUE': '蓝色',
            'YELLOW': '黄色'
        }

        # 颜色名称到实际颜色值的映射
        self.color_values = {
            'RED': '#FF5252',
            'GREEN': '#4CDF50',
            'BLUE': '#00BFFF', 
            'YELLOW': '#FFDC35'
        }

        # 自定义颜色方案
        self.colors = {
            'bg': '#F0F0F0',
            'frame': '#E0E0E0',
            
            'neutral_dark': '#ECCB96',
            'neutral_light': '#BAA077',

            'player1_pawn': '#FF8899',
            'player1_pawn_border': '#AA4477',

            'player1_light': '#D0352E',
            'player1_dark': '#AC352E',

            'player2_pawn': '#77DD77',
            'player2_pawn_border': '#779977',

            'player2_light': '#12C47A',
            'player2_dark': '#0FA868', 

            'player3_pawn': '#77BBDD',
            'player3_pawn_border': '#7799CC',

            'player3_light': "#25A3F2",
            'player3_dark': "#1F7FDF",

            'player4_pawn': '#FFDD88',
            'player4_pawn_border': "#BB9955",

            'player4_light': "#F1C02F",
            'player4_dark': "#CFA028",

            'highlight': '#FFDC35',
            'valid_move': '#8A7047',
            'text': '#212121'
        }

        # metadata
        self.title('Halma')
        self.resizable(True, True)
        self.configure(bg=self.colors['bg'])  # 更柔和的背景色
        
        # 设置图标和字体
        self.default_font = ('Helvetica', 12)
        self.title_font = ('Helvetica', 16, 'bold')

        # variables
        self.board = board
        self.engine = engine
        self.board_size = board.boardsize
        self.selected_pawn = None
        self.tiles = {}
        self.valid_moves = []
        
        # 状态显示变量
        self.status_var = tk.StringVar()
        self.status_var.set("游戏初始化...")

        # 游戏信息变量
        self.turn_var = tk.StringVar()
        self.turn_var.set("步数: 0")
        
        # 分数显示变量
        self.score_vars = {}
        for player in self.board.players:
            color_name = self.color_names.get(player.color, player.color)
            # 分别存储颜色名称和分数
            self.score_vars[player.color] = {
                'name': tk.StringVar(),
                'score': tk.StringVar()
            }
            self.score_vars[player.color]['name'].set(color_name)
            self.score_vars[player.color]['score'].set(": 0分")

        # 创建界面框架
        self.create_ui_frame()
        
        # 初始绘制
        self.draw_tiles()

    def create_ui_frame(self):
        """创建UI基本框架"""
        # 主框架
        main_frame = tk.Frame(self, bg=self.colors['bg'], padx=20, pady=20)
        main_frame.pack(expand=True, fill='both')
        
        # 标题栏
        title_frame = tk.Frame(main_frame, bg=self.colors['frame'], padx=10, pady=10)
        title_frame.pack(fill='x', pady=(0, 15))
        
        title_label = tk.Label(title_frame, text="Halma 跳棋", font=self.title_font, 
                               bg=self.colors['frame'], fg=self.colors['text'])
        title_label.pack(side='left')
        
        # 信息栏
        info_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        info_frame.pack(fill='x', pady=(0, 10))
        
        turn_label = tk.Label(info_frame, textvariable=self.turn_var, font=self.default_font,
                             bg=self.colors['bg'], fg=self.colors['text'])
        turn_label.pack(side='left')
        
        # 分数显示区域
        score_frame = tk.Frame(info_frame, bg=self.colors['bg'])
        score_frame.pack(side='right')
        
        self.score_labels = {}
        for i, player in enumerate(self.board.players):
            # 创建每个玩家的分数显示框架
            player_score_frame = tk.Frame(score_frame, bg=self.colors['bg'])
            player_score_frame.pack(side='left', padx=(10, 0))
            
            # 玩家颜色名称标签（使用对应颜色）
            color = self.color_values.get(player.color, self.colors['text'])
            name_label = tk.Label(player_score_frame, textvariable=self.score_vars[player.color]['name'], 
                                font=self.default_font, bg=self.colors['bg'], fg=color)
            name_label.pack(side='left')
            
            # 分数标签（使用默认颜色）
            score_label = tk.Label(player_score_frame, textvariable=self.score_vars[player.color]['score'], 
                                 font=self.default_font, bg=self.colors['bg'], fg=self.colors['text'])
            score_label.pack(side='left')
            
            # 存储标签引用
            self.score_labels[player.color] = {
                'name': name_label,
                'score': score_label
            }

        # 创建游戏区域
        game_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        game_frame.pack(expand=True)
        
        # 棋盘标签框架
        board_frame = tk.Frame(game_frame, bg=self.colors['bg'], padx=10, pady=10,
                              highlightbackground=self.colors['frame'],
                              highlightthickness=2)
        board_frame.pack(expand=True)
        
        # 行列标签
        for i in range(self.board_size):
            row_label = tk.Label(board_frame, text=str(i+1), font=self.default_font, 
                                bg=self.colors['bg'], fg=self.colors['text'],
                                width=2)
            row_label.grid(row=i+1, column=0)

            col_label = tk.Label(board_frame, text=chr(i+97), font=self.default_font, 
                                bg=self.colors['bg'], fg=self.colors['text'],
                                height=1)
            col_label.grid(row=0, column=i+1)

        # 网格画布 - 使用稍大的尺寸
        canvas_size = 640
        self.canvas = tk.Canvas(board_frame, width=canvas_size, height=canvas_size, 
                              bg=self.colors['bg'], highlightthickness=0)
        self.canvas.grid(row=1, column=1, columnspan=self.board_size, rowspan=self.board_size)

        # 状态栏
        status_frame = tk.Frame(main_frame, bg=self.colors['frame'], padx=10, pady=10)
        status_frame.pack(fill='x', pady=(15, 0))
        
        self.status_label = tk.Label(status_frame, textvariable=self.status_var, 
                              font=self.default_font, bg=self.colors['frame'], 
                              fg=self.colors['text'])
        self.status_label.pack(fill='x')
        
        # 按钮区域
        button_frame = tk.Frame(main_frame, bg=self.colors['bg'], pady=10)
        button_frame.pack(fill='x')
        
        new_game_btn = tk.Button(button_frame, text="新游戏", 
                       font=self.default_font,
                       bg=self.colors['bg'], fg=self.colors['text'],
                       padx=10, pady=5,
                       relief=tk.RAISED,
                       bd=0,  # 设置边框宽度为0，去除黑色边缘
                       highlightthickness=0,  # 去除高亮边框
                       command=self.new_game)
        new_game_btn.pack(side='left', padx=5)
        
        exit_btn = tk.Button(button_frame, text="退出", 
                     font=self.default_font,
                     bg=self.colors['bg'], fg=self.colors['text'],
                     padx=10, pady=5,
                     relief=tk.RAISED,
                     bd=0,  # 设置边框宽度为0
                     highlightthickness=0,  # 去除高亮边框
                     command=self.quit)
        exit_btn.pack(side='right', padx=5)

        # 设置事件绑定
        self.canvas.bind("<Configure>", self.draw_tiles)

    def new_game(self):
        """重新开始游戏"""
        self.engine.restart_game()
        
        # 重置选择状态
        self.selected_pawn = None
        self.valid_moves = []
        
        # 重绘棋盘和棋子
        self.draw_tiles()
        self.draw_pawns()
        
        # 更新回合显示
        self.turn_var.set("步数: 0")
        
        # 更新分数显示
        self.update_scores()
        
        # 更新状态
        self.update_status("新游戏开始！")

    def update_status(self, message):
        """更新状态栏显示，支持加粗格式和颜色"""
        # 检查是否包含加粗标记
        if '**' in message:
            # 创建富文本标签来显示加粗效果
            parts = message.split('**')
            if hasattr(self, 'status_label'):
                self.status_label.destroy()
            
            # 创建新的状态标签框架
            status_frame = self.status_label.master
            self.status_label = tk.Frame(status_frame, bg=self.colors['frame'])
            self.status_label.pack(fill='x')
            
            # 添加文本部分
            for i, part in enumerate(parts):
                if part:  # 跳过空字符串
                    if i % 2 == 1:  # 加粗的部分
                        # 检查是否是颜色名称，设置对应颜色
                        color = self.colors['text']  # 默认颜色
                        for eng_color, cn_color in self.color_names.items():
                            if cn_color in part:
                                color = self.color_values[eng_color]
                                break
                        
                        font = (self.default_font[0], self.default_font[1], 'bold')
                        label = tk.Label(self.status_label, text=part, font=font,
                                       bg=self.colors['frame'], fg=color)
                    else:
                        font = self.default_font
                        label = tk.Label(self.status_label, text=part, font=font,
                                       bg=self.colors['frame'], fg=self.colors['text'])
                    label.pack(side='left')
        else:
            self.status_var.set(message)
        
        # 更新回合计数
        if hasattr(self.engine, 'turn_count'):
            self.turn_var.set(f"步数: {self.engine.turn_count}")
            
        # 更新分数显示
        self.update_scores()

    def update_scores(self):
        """更新分数显示"""
        for player in self.board.players:
            self.score_vars[player.color]['score'].set(f": {player.score}分")

    def draw_tiles(self, event=None):
        """绘制棋盘格"""
        self.canvas.delete("tile")
        
        # 重新计算尺寸以适应画布
        width = self.canvas.winfo_width() or 640
        height = self.canvas.winfo_height() or 640
        cell = min(width, height) / self.board_size  # 每个格子大小
        
        # 获取玩家的家区域
        if len(self.board.players) == 2:
            player1_home = set(self.board.get_home_area(self.board.players[0]))
            player2_home = set(self.board.get_home_area(self.board.players[1]))
        elif len(self.board.players) == 4:
            player1_home = set(self.board.get_home_area(self.board.players[0]))
            player2_home = set(self.board.get_home_area(self.board.players[1]))
            player3_home = set(self.board.get_home_area(self.board.players[2]))
            player4_home = set(self.board.get_home_area(self.board.players[3]))
        
        # 先绘制整个棋盘背景
        # board_bg = self.canvas.create_rectangle(0, 0, width, height, fill=self.colors['frame'], width=0, tags="tile")
        
        # 棋盘格间隙
        border_size = 1
        
        for col in range(self.board_size):
            for row in range(self.board_size):
                x1 = col * cell + border_size
                y1 = row * cell + border_size
                x2 = (col + 1) * cell - border_size
                y2 = (row + 1) * cell - border_size

                # 根据位置确定颜色
                if len(self.board.players) == 2:
                    if (row, col) in player1_home:
                        # 玩家1区域
                        color = self.colors['player1_dark'] if (row + col) % 2 == 0 else self.colors['player1_light']
                    elif (row, col) in player2_home:
                        # 玩家2区域
                        color = self.colors['player2_dark'] if (row + col) % 2 == 0 else self.colors['player2_light']
                    else:
                        # 中立区域
                        color = self.colors['neutral_dark'] if (row + col) % 2 == 0 else self.colors['neutral_light']
                elif len(self.board.players) == 4:
                    if (row, col) in player1_home:
                        # 玩家1区域
                        color = self.colors['player1_dark'] if (row + col) % 2 == 0 else self.colors['player1_light']
                    elif (row, col) in player2_home:
                        # 玩家2区域
                        color = self.colors['player2_dark'] if (row + col) % 2 == 0 else self.colors['player2_light']
                    elif (row, col) in player3_home:
                        # 玩家3区域
                        color = self.colors['player3_dark'] if (row + col) % 2 == 0 else self.colors['player3_light']
                    elif (row, col) in player4_home:
                        # 玩家4区域
                        color = self.colors['player4_dark'] if (row + col) % 2 == 0 else self.colors['player4_light']
                    else:
                        # 中立区域
                        color = self.colors['neutral_dark'] if (row + col) % 2 == 0 else self.colors['neutral_light']
                
                # 创建格子 - 添加轻微的阴影效果
                tile = self.canvas.create_rectangle(x1, y1, x2, y2, tags="tile", width=0,fill=color)
                self.tiles[col, row] = tile
                self.canvas.tag_bind(tile, "<1>", lambda event, r=row, c=col: self.clicked(r, c))

        # 绘制棋子
        self.draw_pawns()

    def draw_pawns(self):
        """绘制所有棋子"""
        # 获取画布尺寸
        width = self.canvas.winfo_width() or 640
        height = self.canvas.winfo_height() or 640
        cell = min(width, height) / self.board_size
        
        # 删除之前的棋子
        self.canvas.delete('pawn')
        
        # 棋子边距
        border_size = cell * 0.16
        
        # 遍历棋盘绘制棋子
        for row in range(self.board_size):
            for col in range(self.board_size):
                pawn = self.board.board[row][col]
                if pawn:
                    x1 = col * cell + border_size
                    y1 = row * cell + border_size
                    x2 = (col + 1) * cell - border_size
                    y2 = (row + 1) * cell - border_size
                    
                    # 棋子中心点
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    radius = (x2 - x1) / 2
                    
                    # 根据棋子颜色绘制
                    if pawn.color == "RED":
                        fill_color = self.colors['player1_pawn']
                        border_color = self.colors['player1_pawn_border']
                    elif pawn.color == "GREEN":
                        fill_color = self.colors['player2_pawn']
                        border_color = self.colors['player2_pawn_border']
                    elif pawn.color == "BLUE":
                        fill_color = self.colors['player3_pawn']
                        border_color = self.colors['player3_pawn_border']
                    elif pawn.color == "YELLOW":
                        fill_color = self.colors['player4_pawn']
                        border_color = self.colors['player4_pawn_border']

                    # 创建光晕效果
                    pawn_border = self.canvas.create_oval(
                        cx-radius-2, cy-radius-2, 
                        cx+radius+2, cy+radius+2,
                        tags="pawn", width=1, 
                        fill=border_color,
                        outline=""
                    )
                    
                    # 创建棋子主体
                    pawn_oval = self.canvas.create_oval(
                        x1, y1, x2, y2, 
                        tags="pawn", width=0, fill=fill_color
                    )
                    
                    # 添加高光效果
                    # highlight = self.canvas.create_oval(
                    #     x1+radius*0.45, y1+radius*0.45, 
                    #     x1+radius*0.85, y1+radius*0.85,
                    #     tags="pawn", width=0,
                    #     fill=highlight_color  # 使用高亮颜色
                    # )
                    
                    # 绑定事件
                    self.canvas.tag_bind(pawn_oval, "<1>", lambda event, r=row, c=col: self.clicked(r, c))
                    self.canvas.tag_bind(pawn_border, "<1>", lambda event, r=row, c=col: self.clicked(r, c))
                    # self.canvas.tag_bind(highlight, "<1>", lambda event, r=row, c=col: self.clicked(r, c))

    def highlight_valid_moves(self):
        """高亮显示可能的移动位置"""
        # 首先高亮当前选中的棋子位置
        sx, sy = self.selected_pawn.x, self.selected_pawn.y
        
        # 获取当前选中棋子格子的坐标
        x1, y1, x2, y2 = self.canvas.coords(self.tiles[sy, sx])
        
        # 计算圆的参数
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # 中心点
        radius = min(x2 - x1, y2 - y1) * 0.4  # 直径约为格子大小的90%
        
        # 创建高亮空心圆
        highlight_circle = self.canvas.create_oval(
            cx - radius, cy - radius,
            cx + radius, cy + radius,
            tags="highlight_marker",  # 添加特殊标签，方便后续删除
            width=3,
            outline=self.colors['highlight'],
            fill=""  # 空心圆，没有填充色
        )
    
        
        # 然后高亮所有可能的目标位置
        for move in self.valid_moves:
            _, _, ex, ey = move
            
            # 获取格子属性
            x1, y1, x2, y2 = self.canvas.coords(self.tiles[ey, ex])
            
            # 创建动态的目标位置标记
            marker_size = min(x2-x1, y2-y1) * 0.1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 创建移动标记
            marker = self.canvas.create_oval(
                cx - marker_size, cy - marker_size,
                cx + marker_size, cy + marker_size,
                tags="move_marker",
                fill=self.colors['valid_move'],
                outline=self.colors['valid_move'],
            )
            
            # 绑定点击事件
            self.canvas.tag_bind(marker, "<1>", lambda event, r=ex, c=ey: self.clicked(r, c))
    
    def reset_selection(self):
        """重置选择状态"""
        # 清除所有高亮
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.canvas.itemconfigure(self.tiles[j, i], outline="#00000020", width=1)
        
        # 删除所有移动标记
        self.canvas.delete("move_marker")
        
        # 重置选择状态
        self.selected_pawn = None
        self.valid_moves = []

    def clicked(self, row, col):
        """处理点击事件 - 仅当是HumanPlayer回合时有效"""
        # 确认当前是否为人类玩家回合
        current_player = self.engine.current_player
        if not isinstance(current_player, HumanPlayer) or not self.engine.waiting_for_human:
            return
            
        # 获取当前格子上的棋子
        pawn = self.board.board[row][col]
        
        # 如果已经选择了棋子，尝试移动
        if self.selected_pawn:
            # 检查是否是有效移动
            for move in self.valid_moves:
                if move[2] == row and move[3] == col:  # 找到对应的目标位置
                    # 将动作传递给引擎
                    self.engine.human_move(move)
                    # 重置选择状态
                    self.reset_selection()
                    return
            
            # 点击了无效位置，重置选择
            self.reset_selection()
            
        # 如果点击了一个棋子，选择它
        elif pawn and pawn.player == current_player:
            self.selected_pawn = pawn
            # 获取该棋子的有效移动
            all_moves = self.board.get_actions(current_player)
            # 筛选出当前棋子的有效移动
            self.valid_moves = [move for move in all_moves if move[0] == row and move[1] == col]
            # 高亮显示所有可能的移动位置
            self.highlight_valid_moves()
            # 更新状态信息
            color_name = self.color_names.get(current_player.color, current_player.color)
            self.update_status(f"**{color_name}** 玩家回合 - 请选择目标位置")
    
    def reset_selection(self):
        """重置选择状态"""
        current_player = self.engine.current_player
        # 清除所有高亮
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.canvas.itemconfigure(self.tiles[j, i], outline="black", width=0)

        # 删除高亮选中标记
        self.canvas.delete("highlight_marker")
        
        # 删除所有移动标记
        self.canvas.delete("move_marker")
        
        color_name = self.color_names.get(current_player.color, current_player.color)
        self.update_status(f"**{color_name}** 玩家回合 - 请选择棋子")
        # 重置选择状态
        self.selected_pawn = None
        self.valid_moves = []
