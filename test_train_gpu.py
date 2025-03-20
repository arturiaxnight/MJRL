from src.environment.mahjong_env import MahjongEnv
from src.models.dqn_agent import DQNAgent
import torch
import time
import numpy as np

def test_train():
    print("開始測試GPU訓練...")
    
    # 創建環境
    env = MahjongEnv()
    
    # 獲取狀態和動作空間大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"狀態空間大小: {state_size}")
    print(f"動作空間大小: {action_size}")
    
    # 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    
    # 創建代理
    agent = DQNAgent(state_size, action_size, device)
    
    # 進行少量訓練來測試GPU使用
    episodes = 20
    max_steps = 100
    
    for episode in range(episodes):
        start_time = time.time()
        
        # 重置環境
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # 選擇動作
            action = agent.act(state)
            
            # 執行動作
            next_state, reward, done, _ = env.step(action)
            
            # 記憶經驗
            agent.remember(state, action, reward, next_state, done)
            
            # 更新狀態和獎勵
            state = next_state
            total_reward += reward
            
            # 從經驗中學習
            if len(agent.memory) >= agent.batch_size:
                agent.replay()
            
            # 如果遊戲結束，退出循環
            if done:
                break
        
        # 計算每個episode的訓練時間
        episode_time = time.time() - start_time
        
        print(f"Episode {episode+1}/{episodes} 完成")
        print(f"  總獎勵: {total_reward:.2f}")
        print(f"  耗時: {episode_time:.2f} 秒")
        
        # 如果有CUDA，打印記憶體使用情況
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
            print(f"  GPU記憶體使用: 已分配 {memory_allocated:.2f} MB, 已保留 {memory_reserved:.2f} MB")
    
    print("\n測試完成!")

if __name__ == "__main__":
    test_train() 