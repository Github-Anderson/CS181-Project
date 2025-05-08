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

    def start(self):
        self.gui = BoardGUI(self.board, self)
        self.gui.after(100, self.game_loop)
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
    def __init__(self, board, engine, *args, **kwargs):
        # initialize parent tk class
        tk.Tk.__init__(self, *args, **kwargs)

        # metadata
        self.title('Halma')
        self.resizable(True, True)
        self.configure(bg='#fff')

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

        # 创建界面框架
        self.create_ui_frame()
        
        # 初始绘制
        self.draw_tiles()

    def create_ui_frame(self):
        """创建UI基本框架"""
        # 行列标签
        for i in range(self.board_size):
            row_label = tk.Label(self, text=i+1, font='Times', bg='#fff', fg='#000')
            row_label.grid(row=i+1, column=0)

            col_label = tk.Label(self, text=chr(i+97), font='Times', bg='#fff', fg='#000')
            col_label.grid(row=0, column=i+1)

        # 网格画布
        self.canvas = tk.Canvas(self, width=600, height=600, bg="#fff", highlightthickness=0)
        self.canvas.grid(row=1, column=1, columnspan=self.board_size, rowspan=self.board_size)

        # 配置列和行
        self.columnconfigure(0, minsize=50)
        self.rowconfigure(0, minsize=50)
        self.columnconfigure(self.board_size + 1, minsize=50)
        self.rowconfigure(self.board_size + 1, minsize=50)
        
        # 添加状态栏
        status_label = tk.Label(self, textvariable=self.status_var, font=('Times', 12), bg='#fff', fg='#000')
        status_label.grid(row=self.board_size + 1, column=1, columnspan=self.board_size, sticky="w")
        
        # 设置事件绑定
        self.canvas.bind("<Configure>", self.draw_tiles)
    
    def update_status(self, message):
        """更新状态栏显示"""
        self.status_var.set(message)

    def draw_tiles(self, event=None):
        """绘制棋盘格"""
        self.canvas.delete("tile")
        cell = int(600 / self.board_size)  # 每个格子大小
        border_size = 1
        
        # 获取玩家的家区域
        player1_home = set(self.board.get_home_area(self.board.players[0]))
        player2_home = set(self.board.get_home_area(self.board.players[1]))
        
        for col in range(self.board_size):
            for row in range(self.board_size):
                x1 = col * cell + border_size / 2
                y1 = row * cell + border_size / 2
                x2 = (col + 1) * cell - border_size / 2
                y2 = (row + 1) * cell - border_size / 2

                # 根据位置确定颜色
                if (row, col) in player1_home:
                    # 玩家1区域
                    color = '#AC352E' if (row + col) % 2 == 0 else '#D0352E'
                elif (row, col) in player2_home:
                    # 玩家2区域
                    color = '#12C47A' if (row + col) % 2 == 0 else '#0FA868'
                else:
                    # 中立区域
                    color = '#ECCB96' if (row + col) % 2 == 0 else '#BAA077'
                
                # 创建格子
                tile = self.canvas.create_rectangle(x1, y1, x2, y2, tags="tile", width=0, fill=color)
                self.tiles[col, row] = tile
                self.canvas.tag_bind(tile, "<1>", lambda event, r=row, c=col: self.clicked(r, c))

        # 绘制棋子
        self.draw_pawns()

    def draw_pawns(self):
        """绘制所有棋子"""
        cell = int(600 / self.board_size)
        border_size = 10
        # 删除之前的棋子
        self.canvas.delete('pawn')
        
        # 遍历棋盘绘制棋子
        for row in range(self.board_size):
            for col in range(self.board_size):
                pawn = self.board.board[row][col]
                if pawn:
                    x1 = col * cell + border_size
                    y1 = row * cell + border_size
                    x2 = (col + 1) * cell - border_size
                    y2 = (row + 1) * cell - border_size
                    
                    # 根据棋子颜色绘制
                    if pawn.color == "RED":
                        fill_color = "#CF6E67"
                    else:  # "GREEN"
                        fill_color = "#67BF9B"
                    
                    # 创建棋子
                    pawn_oval = self.canvas.create_oval(x1, y1, x2, y2, tags="pawn", width=0, fill=fill_color)
                    self.canvas.tag_bind(pawn_oval, "<1>", lambda event, r=row, c=col: self.clicked(r, c))

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
    
    def highlight_valid_moves(self):
        """高亮显示可能的移动位置"""
        # 首先高亮当前选中的棋子位置
        sx, sy = self.selected_pawn.x, self.selected_pawn.y
        self.canvas.itemconfigure(self.tiles[sy, sx], outline="yellow", width=2)
        
        # 然后高亮所有可能的目标位置
        for move in self.valid_moves:
            _, _, ex, ey = move
            self.canvas.itemconfigure(self.tiles[ey, ex], outline="blue", width=2)
    
    def reset_selection(self):
        """重置选择状态"""
        # 清除所有高亮
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.canvas.itemconfigure(self.tiles[j, i], outline="black", width=0)
        
        # 重置选择状态
        self.selected_pawn = None
        self.valid_moves = []
