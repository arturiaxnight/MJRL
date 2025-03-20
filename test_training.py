from src.environment.mahjong_env import MahjongEnv
from src.models.dqn_agent import DQNAgent
import numpy as np
import torch
import time

def test_training():
    """
    測試強化學習訓練功能
    """
    print("=== 開始測試強化學習訓練 ===")
    
    # 創建環境
    env = MahjongEnv(enable_logging=True, log_level=2)
    
    # 創建 DQN 代理
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # 訓練參數
    n_episodes = 10  # 測試用較少的回合數
    max_steps = 1000
    batch_size = 32
    
    # 記錄訓練數據
    scores = []
    times = []
    
    # 開始訓練
    for episode in range(n_episodes):
        start_time = time.time()
        state = env.reset()
        total_reward = 0
        
        print(f"\n開始第 {episode + 1} 回合訓練")
        
        for step in range(max_steps):
            # 選擇動作
            action = agent.act(state)
            
            # 執行動作
            next_state, reward, done, info = env.step(action)
            
            # 儲存經驗
            agent.remember(state, action, reward, next_state, done)
            
            # 更新總獎勵
            total_reward += reward
            
            # 如果記憶庫中有足夠的經驗，進行學習
            if len(agent.memory) > batch_size:
                agent.replay()
            
            # 更新狀態
            state = next_state
            
            # 如果回合結束，跳出循環
            if done:
                break
        
        # 記錄本回合的得分和時間
        scores.append(total_reward)
        times.append(time.time() - start_time)
        
        # 輸出本回合的統計信息
        print(f"回合 {episode + 1} 完成:")
        print(f"總獎勵: {total_reward}")
        print(f"耗時: {times[-1]:.2f} 秒")
        print(f"平均獎勵: {np.mean(scores):.2f}")
        
        # 每5回合保存一次模型
        if (episode + 1) % 5 == 0:
            agent.save(f"models/dqn_model_episode_{episode + 1}.pth")
            print(f"模型已保存: models/dqn_model_episode_{episode + 1}.pth")
    
    # 輸出訓練總結
    print("\n=== 訓練總結 ===")
    print(f"平均獎勵: {np.mean(scores):.2f}")
    print(f"最高獎勵: {np.max(scores):.2f}")
    print(f"最低獎勵: {np.min(scores):.2f}")
    print(f"平均回合時間: {np.mean(times):.2f} 秒")
    
    # 保存最終模型
    agent.save("models/dqn_model_final.pth")
    print("\n最終模型已保存: models/dqn_model_final.pth")

if __name__ == "__main__":
    # 設定隨機種子以確保可重現性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 創建模型目錄
    import os
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # 開始測試
    test_training() 