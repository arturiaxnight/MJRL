from src.environment.mahjong_env import MahjongEnv
from src.utils.mahjong_utils import id_to_string, tile_to_id
import numpy as np
import os
import time

def test_logging_functionality():
    print("測試麻將記錄功能...")
    
    # 創建環境（啟用記錄功能，設置適當的日誌級別）
    env = MahjongEnv(enable_logging=True, log_level=1)
    print("麻將環境創建成功")
    
    # 設定隨機種子以確保可重現性
    np.random.seed(42)
    
    # 運行幾個回合的遊戲
    num_episodes = 2
    max_steps_per_episode = 10
    
    for episode in range(num_episodes):
        print(f"\n===== 測試回合 {episode+1}/{num_episodes} =====")
        
        # 重置環境
        observation = env.reset()
        current_episode = env.episode_count
        
        # 打印初始狀態
        player_id = env.current_player
        hand = env.players_hands[player_id]
        readable_hand = [id_to_string(tile_to_id(tile)) for tile in hand]
        print(f"當前玩家: {player_id}")
        print(f"初始手牌: {', '.join(readable_hand)}")
        
        total_reward = 0
        done = False
        
        # 執行多個步驟
        step_count = 0
        for step in range(max_steps_per_episode):
            # 確保動作是有效的（丟棄手中的牌）
            current_player = env.current_player
            valid_tiles = [tile_to_id(tile) for tile in env.players_hands[current_player]]
            
            if not valid_tiles:
                print(f"玩家 {current_player} 沒有有效牌可打，跳過")
                break
                
            # 隨機選擇一張牌打出
            action = valid_tiles[np.random.randint(0, len(valid_tiles))]
            
            # 執行動作
            observation, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                print(f"回合 {episode+1} 在 {step+1} 步後自然結束")
                break
        
        # 如果回合沒有自然結束，強制結束並記錄
        if not done:
            # 手動結束回合並記錄
            env.logger.end_episode(env.total_scores, total_reward, is_win=(total_reward > 0))
            print(f"回合 {episode+1} 在執行 {step_count} 步後手動結束")
        
        # 等待一秒，確保文件有時間寫入
        time.sleep(1)
        
        # 生成文本記錄
        log_file = env.logger.generate_text_log(current_episode)
        
        if log_file and os.path.exists(log_file):
            print(f"成功生成回合 {current_episode} 的文本記錄")
            
            # 顯示文件內容預覽
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # 如果內容太長，只顯示開頭的部分
                if len(content) > 500:
                    preview = content[:500] + "..."
                else:
                    preview = content
                    
                print("\n記錄文件預覽:")
                print("-" * 50)
                print(preview)
                print("-" * 50)
                
                # 檢查是否包含牌山記錄
                if "牌山:" in content:
                    print("成功記錄了牌山！")
                else:
                    print("警告：未找到牌山記錄")
        else:
            print(f"無法找到或生成回合 {current_episode} 的文本記錄")
    
    # 獲取記錄目錄
    log_dir = env.logger.session_dir
    print("\n記錄測試完成！")
    print(f"記錄文件保存在: {log_dir}")
    
    # 列出記錄目錄中的文件
    try:
        log_files = os.listdir(log_dir)
        print("\n記錄目錄中的文件:")
        for file in sorted(log_files):
            file_path = os.path.join(log_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # 轉換為KB
            print(f"  - {file} ({file_size:.2f} KB)")
    except Exception as e:
        print(f"無法列出記錄文件: {e}")

if __name__ == "__main__":
    test_logging_functionality() 