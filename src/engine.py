from agents import HumanPlayer
from utils import *
from board import *
import tkinter as tk

class Engine:
    def __init__(self, board : Board):
        self.board = board
        self.players = board.players
        self.current_player_index = 0
        self.current_player = None
        self.game_over = False
        self.waiting_for_human = False
        self.human_action = None
        self.turn_count = 0

    def start(self):
        self.gui = BoardGUI(self.board, self)
        self.gui.after(10, self.game_loop)
        self.gui.mainloop()

    def game_loop(self):
        state = self.board.get_state()
        if state:
            winner = state["winner"]
            message = state["message"]
            self.game_over = True
            self.gui.update_status(message)
            print(message)
            return
        
        for player in self.players:
            if self.current_player_index == player.index:
                self.current_player = player
                break
            
        # 获取当前玩家可用的动作
        actions = self.board.get_actions(self.current_player)
        
        # 如果是人类玩家，等待GUI输入
        if isinstance(self.current_player, HumanPlayer):
            if not self.waiting_for_human:
                self.waiting_for_human = True
                self.gui.update_status(f"{self.current_player.color} 玩家回合 - 请选择棋子")
            elif self.human_action:
                # 人类玩家已经选择了动作
                self.turn_count += 1
                self.board.apply_action(self.human_action)
                self.human_action = None
                self.waiting_for_human = False
                self.next_turn()
        else:
            # AI玩家自动行动
            self.gui.update_status(f"{self.current_player.color} 玩家回合 - AI思考中...")
            self.gui.update()  # 刷新界面显示
            
            # 添加短暂延迟，让玩家能看到AI行动
            #time.sleep(0.5)
            
            action = self.current_player.get_action(actions)
            self.turn_count += 1
            self.board.apply_action(action)
            self.next_turn()
        
        # 重绘棋盘
        self.gui.draw_pawns()
        
        # 继续游戏循环
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
    
    def human_move(self, action):
        """接收来自GUI的人类玩家动作"""
        if self.waiting_for_human and isinstance(self.current_player, HumanPlayer):
            self.human_action = action


class BoardGUI(tk.Tk):
    def __init__(self, board : Board, engine : Engine, *args, **kwargs):
        # initialize parent tk class
        tk.Tk.__init__(self, *args, **kwargs)

        # 自定义颜色方案
        self.colors = {
            'bg': '#F0F0F0',
            'frame': '#E0E0E0',

            'player1_dark': '#AC352E',
            'player1_light': '#D0352E',
            'player2_dark': '#0FA868', 
            'player2_light': '#12C47A',

            'neutral_dark': '#ECCB96',
            'neutral_light': '#BAA077',

            'player1_pawn': '#FF5252', 
            'player1_pawn_highlight': '#FFB0B0',
            'player1_pawn_border': '#8C150E',

            'player2_pawn': '#4CDF50',
            'player2_pawn_highlight': '#B0FFB0',
            'player2_pawn_border': '#0F8858',

            'highlight': '#FFEC45',
            'valid_move': '#8A7047',
            'text': '#212121'
        }

        # metadata
        self.title('Halma - 跳棋游戏')
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
        
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                              font=self.default_font, bg=self.colors['frame'], 
                              fg=self.colors['text'])
        status_label.pack(fill='x')
        
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
        self.update_status("开始新游戏...")
        # 重置棋盘
        self.board.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.board.initialize_board()
        
        # 重置引擎状态
        self.engine.current_player_index = 0
        self.engine.current_player = self.engine.players[0]
        self.engine.waiting_for_human = False
        self.engine.human_action = None
        self.engine.turn_count = 0
        
        # 重置选择状态
        self.selected_pawn = None
        self.valid_moves = []
        
        # 重绘棋盘和棋子
        self.draw_tiles()
        self.draw_pawns()
        
        # 更新回合显示
        self.turn_var.set("回合: 0")
        
        # 若游戏循环已停止，重新启动
        if self.engine.game_over:
            self.engine.game_over = False
            self.after(10, self.engine.game_loop)
    
    def update_status(self, message):
        """更新状态栏显示"""
        self.status_var.set(message)
        # 更新回合计数 - 实际应用中需要从游戏逻辑获取回合数
        if hasattr(self.engine, 'turn_count'):
            self.turn_var.set(f"步数: {self.engine.turn_count}")

    def draw_tiles(self, event=None):
        """绘制棋盘格"""
        self.canvas.delete("tile")
        
        # 重新计算尺寸以适应画布
        width = self.canvas.winfo_width() or 640
        height = self.canvas.winfo_height() or 640
        cell = min(width, height) / self.board_size  # 每个格子大小
        
        # 获取玩家的家区域
        player1_home = set(self.board.get_home_area(self.board.players[0]))
        player2_home = set(self.board.get_home_area(self.board.players[1]))
        
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
                if (row, col) in player1_home:
                    # 玩家1区域
                    color = self.colors['player1_dark'] if (row + col) % 2 == 0 else self.colors['player1_light']
                elif (row, col) in player2_home:
                    # 玩家2区域
                    color = self.colors['player2_dark'] if (row + col) % 2 == 0 else self.colors['player2_light']
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
                        outline_color = self.colors['player1_dark']
                        highlight_color = self.colors['player1_pawn_highlight']
                        border_color = self.colors['player1_pawn_border']
                    else:  # "GREEN"
                        fill_color = self.colors['player2_pawn']
                        outline_color = self.colors['player2_dark']
                        highlight_color = self.colors['player2_pawn_highlight']
                        border_color = self.colors['player2_pawn_border']
                    
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
            self.update_status(f"{current_player.color} 玩家回合 - 请选择目标位置")
    
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
        
        self.update_status(f"{current_player.color} 玩家回合 - 请选择棋子")
        # 重置选择状态
        self.selected_pawn = None
        self.valid_moves = []
