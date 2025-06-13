import os
from agents import Neural_ApproximateQLearningPlayer, MinimaxPlayer, ApproximateQLearningPlayer
from board import Board
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def self_play_episode(agent, opponent, board):
    """进行一局自我对弈"""
    board.reset()
    total_reward = 0
    step = 0
    max_steps = 200
    
    while step < max_steps:
        current_state = board.clone()
        actions = board.get_actions(agent)
        
        if not actions:
            break
            
        action = agent.get_action(actions, training=True)
        board.apply_action(action)
        
        # 获取奖励
        reward = get_reward(board, agent)
        total_reward += reward
        
        # 检查游戏是否结束
        state = board.get_state()
        done = state is not None
        
        # 存储经验
        agent.store_transition(current_state, action, reward, board.clone(), done)
        
        # 更新网络
        agent.update_network()
        
        if done:
            break
            
        # 对手行动
        opp_actions = board.get_actions(opponent)
        if opp_actions:
            opp_action = opponent.get_action(opp_actions)
            if opp_action:
                board.apply_action(opp_action)
        
        step += 1
        
    return total_reward, step, state if state else None

def get_reward(board, player):
    """计算奖励"""
    state = board.get_state()
    if state:
        if state == player:
            return 1000
        else:
            return -1000
            
    # 计算棋子到目标的进展
    goal_area = set(board.get_goal_area(player))
    pieces_in_goal = 0
    total_distance = 0
    
    if player.color == "RED":
        goal_center = (board.boardsize-1, board.boardsize-1)
    else:
        goal_center = (0, 0)
        
    for i in range(board.boardsize):
        for j in range(board.boardsize):
            pawn = board.board[i][j]
            if pawn and pawn.player == player:
                if (i, j) in goal_area:
                    pieces_in_goal += 1
                else:
                    distance = abs(goal_center[0]-i) + abs(goal_center[1]-j)
                    total_distance += distance
                    
    # 归一化奖励
    goal_reward = pieces_in_goal * 50
    distance_penalty = -total_distance
    
    return goal_reward + distance_penalty

def train(episodes=150, save_interval=15):
    agent = Neural_ApproximateQLearningPlayer("RED")
    opponent = MinimaxPlayer("GREEN", depth = 2)  # 使用Minimax作为对手
    players = [agent, opponent]
    
    # 正确初始化棋盘
    board = Board(boardsize=8, mode="classic", players=players)
    
    # 设置玩家的棋盘引用
    agent.set_board(board)
    opponent.set_board(board)
    
    # 训练记录
    rewards_history = []
    win_rate_history = []
    steps_history = []
    
    # 创建保存目录
    os.makedirs("models_sente", exist_ok=True)
    
    for episode in tqdm(range(episodes)):
        reward, steps, winner = self_play_episode(agent, opponent, board)
        
        rewards_history.append(reward)
        steps_history.append(steps)
        
        # 计算胜率
        if (episode+1) % 10 == 0:
            recent_wins = sum(1 for i in range(episode-100, episode)
                            if rewards_history[i] > 0)
            win_rate = recent_wins / 100
            win_rate_history.append(win_rate)
            
            print(f"\nEpisode {episode}")
            print(f"Recent win rate: {win_rate:.2f}")
            print(f"Average reward: {np.mean(rewards_history[-100:]):.2f}")
            print(f"Average steps: {np.mean(steps_history[-100:]):.2f}")
            
        # 保存模型
        if (episode+1) % save_interval == 0:
            agent.save_model(f"models/neural_agent_ep{episode+1}.pth")
            
        # 动态调整探索率
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(rewards_history)
    plt.title('Rewards')
    
    plt.subplot(132)
    plt.plot(win_rate_history)
    plt.title('Win Rate')
    
    plt.subplot(133)
    plt.plot(steps_history)
    plt.title('Steps per Episode')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    
if __name__ == "__main__":
    train()