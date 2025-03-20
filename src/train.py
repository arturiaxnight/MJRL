import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from environment.mahjong_env import MahjongEnv
from models.dqn_agent import DQNAgent
import time
import gc

# 確保隨機性的可重現性
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_gpu_memory_usage():
    """打印GPU記憶體使用情況"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_memory_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            
            print(f"GPU {i} - 已分配: {memory_allocated:.3f} GB, 已保留: {memory_reserved:.3f} GB, 最大分配: {max_memory_allocated:.3f} GB")

def train_agent(episodes=1000, max_steps=1000, target_update=10, save_freq=100, debug_freq=10):
    """
    訓練DQN代理
    """
    # 創建環境
    env = MahjongEnv()
    
    # 獲取狀態和動作空間大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"狀態空間大小: {state_size}")
    print(f"動作空間大小: {action_size}")
    
    # 檢測設備並創建代理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練使用設備: {device}")
    
    agent = DQNAgent(state_size, action_size, device)
    
    # 創建模型保存目錄
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # 追踪訓練進度
    scores = []
    episodes_x = []
    losses = []
    replay_times = []
    
    # 開始訓練
    total_start_time = time.time()
    
    for episode in range(episodes):
        episode_start_time = time.time()
        # 重置環境
        state = env.reset()
        total_reward = 0
        replay_count = 0
        
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
                replay_count += 1
            
            # 如果遊戲結束，退出循環
            if done:
                break
        
        # 記錄每個episode的回放次數
        replay_times.append(replay_count)
        
        # 定期更新目標網絡
        if episode % target_update == 0:
            agent.update_target_network()
        
        # 定期保存模型
        if episode % save_freq == 0 and episode > 0:
            agent.save(f'checkpoints/mahjong_dqn_episode_{episode}.pth')
            
            # 繪製到目前為止的學習曲線並保存
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(episodes_x, scores)
            plt.title('DQN學習曲線')
            plt.xlabel('集數')
            plt.ylabel('總獎勵')
            
            plt.subplot(2, 1, 2)
            plt.plot(episodes_x, replay_times)
            plt.title('每個Episode的回放次數')
            plt.xlabel('集數')
            plt.ylabel('回放次數')
            
            plt.tight_layout()
            plt.savefig(f'learning_curve_{episode}.png')
            plt.close()
        
        # 記錄分數
        scores.append(total_reward)
        episodes_x.append(episode)
        
        # 計算每個episode的訓練時間
        episode_time = time.time() - episode_start_time
        
        # 輸出進度
        if episode % debug_freq == 0:
            avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
            avg_replay = np.mean(replay_times[-10:]) if len(replay_times) >= 10 else np.mean(replay_times)
            
            elapsed_time = time.time() - total_start_time
            estimated_total_time = elapsed_time / (episode + 1) * episodes
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"Episode: {episode}/{episodes}")
            print(f"  平均分數: {avg_score:.2f}")
            print(f"  探索率: {agent.epsilon:.4f}")
            print(f"  平均回放次數: {avg_replay:.2f}")
            print(f"  本集耗時: {episode_time:.2f} 秒")
            print(f"  估計剩餘時間: {remaining_time/60:.2f} 分鐘")
            print_gpu_memory_usage()
            print("-" * 50)
            
            # 強制執行垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 保存最終模型
    agent.save('checkpoints/mahjong_dqn_final.pth')
    
    # 繪製最終學習曲線
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(episodes_x, scores)
    plt.title('DQN學習曲線')
    plt.xlabel('集數')
    plt.ylabel('總獎勵')
    
    plt.subplot(2, 1, 2)
    plt.plot(episodes_x, replay_times)
    plt.title('每個Episode的回放次數')
    plt.xlabel('集數')
    plt.ylabel('回放次數')
    
    plt.tight_layout()
    plt.savefig('learning_curve_final.png')
    plt.close()
    
    total_time = time.time() - total_start_time
    print(f"總訓練時間: {total_time/60:.2f} 分鐘")
    
    return agent, scores

def evaluate_agent(agent, env, episodes=100):
    """
    評估訓練好的代理
    """
    total_rewards = []
    win_count = 0
    
    eval_start_time = time.time()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, epsilon=0)  # 無探索
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            
        total_rewards.append(total_reward)
        
        # 假設獲得正獎勵意味著贏了遊戲
        if total_reward > 0:
            win_count += 1
            
        if (episode + 1) % 10 == 0:
            print(f"已評估 {episode+1}/{episodes} 局")
    
    eval_time = time.time() - eval_start_time
    avg_reward = np.mean(total_rewards)
    win_rate = win_count / episodes
    
    print(f"\n評估結果:")
    print(f"平均總獎勵: {avg_reward:.2f}")
    print(f"勝率: {win_rate:.2%}")
    print(f"評估耗時: {eval_time:.2f} 秒")
    
    return avg_reward, win_rate

if __name__ == "__main__":
    print("開始訓練麻將AI...")
    print("=" * 50)
    print("系統資訊:")
    
    # 檢測CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"檢測到 {gpu_count} 個CUDA設備:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("未檢測到CUDA設備，將使用CPU訓練(速度會很慢)")
    
    print("=" * 50)
    
    start_time = time.time()
    
    # 訓練代理
    agent, scores = train_agent(episodes=500, debug_freq=5)
    
    training_time = time.time() - start_time
    print(f"訓練完成！總耗時: {training_time/60:.2f} 分鐘")
    
    # 創建新環境進行評估
    eval_env = MahjongEnv()
    avg_reward, win_rate = evaluate_agent(agent, eval_env)
    
    print("訓練與評估完成!") 