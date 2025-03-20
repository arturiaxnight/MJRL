from src.environment.mahjong_env import MahjongEnv
from src.utils.mahjong_utils import id_to_string, id_to_tile, tile_to_id, hand_to_counts, check_win, calculate_score, id_to_emoji, RED_FIVE_MAN, RED_FIVE_PIN, RED_FIVE_SOU
import numpy as np
import random
import time

def simulate_full_game():
    """
    模擬一局完整的麻將遊戲並測試計分功能
    """
    # 創建環境，啟用記錄功能
    env = MahjongEnv(enable_logging=True, log_level=2)
    
    # 設定隨機種子以確保可重現性
    np.random.seed(42)
    random.seed(42)
    
    # 重置環境開始新遊戲
    observation = env.reset()
    current_episode = env.episode_count
    
    # 記錄遊戲參數
    dealer = env.dealer
    round_wind = env.round_wind
    wind_emoji = id_to_emoji(tile_to_id((3, round_wind)))
    env.logger.log_action(0, "game_info", f"場風: {wind_emoji} {id_to_string(tile_to_id((3, round_wind)))}")
    env.logger.log_action(0, "game_info", f"莊家: 玩家{dealer}")
    
    # 顯示每位玩家的初始手牌
    for player in range(4):
        hand = env.players_hands[player]
        # 首先將手牌從(suit, value)轉換為tile_id
        hand_ids = [tile_to_id(tile) for tile in hand]
        readable_hand = [id_to_string(tile_id) for tile_id in hand_ids]
        emoji_hand = [id_to_emoji(tile_id) for tile_id in hand_ids]
        env.logger.log_action(0, "game_info", f"玩家{player}初始手牌: {''.join(emoji_hand)}")
    
    # 創建一個理想的和牌手牌用於演示和牌判斷，添加赤寶牌
    winning_hand = create_winning_hand_with_red()
    
    # 繼續遊戲直到和牌或流局
    max_turns = 4  # 減少回合數
    current_turn = 0
    done = False
    
    # 記錄每個玩家的牌河
    discard_piles = [[] for _ in range(4)]
    
    while not done and current_turn < max_turns:
        current_turn += 1
        env.logger.log_action(0, "game_info", f"\n--- 回合 {current_turn} ---")
        
        # 取得當前玩家
        player_id = env.current_player
        
        # 顯示當前玩家信息
        hand = env.players_hands[player_id]
        # 首先將手牌從(suit, value)轉換為tile_id
        hand_ids = [tile_to_id(tile) for tile in hand]
        emoji_hand = [id_to_emoji(tile_id) for tile_id in hand_ids]
        env.logger.log_action(0, "game_info", f"玩家{player_id}的手牌: {''.join(emoji_hand)}")
        
        # 為了演示計分系統，在最後一回合創造和牌情況
        if current_turn == max_turns:
            env.logger.log_action(0, "game_info", "\n=== 創造和牌情況進行得分計算 ===")
            
            # 使用預設的和牌手牌進行測試
            winning_hand_ids = winning_hand
            counts = hand_to_counts(winning_hand_ids)
            
            # 顯示和牌手牌
            winning_hand_emoji = [id_to_emoji(tile_id) for tile_id in winning_hand_ids]
            env.logger.log_action(0, "game_info", f"和牌手牌: {''.join(winning_hand_emoji)}")
            
            # 模擬自摸
            is_self_draw = True
            # 假設是門前清
            is_open = False
            # 假設玩家是莊家
            is_dealer = (player_id == dealer)
            # 假設當前回合風為東
            round_wind = 0
            # 玩家自風
            player_wind = player_id
            # 假設有一張寶牌
            dora_count = 1
            # 假設立直
            is_riichi = True
            
            # 計算和展示得分
            win_tile = winning_hand_ids[-1]  # 使用最後一張牌作為和牌
            win_tile_emoji = id_to_emoji(win_tile)
            
            # 顯示赤寶牌的數量（這更直接地展示赤寶牌的影響）
            red_fives_count = len([tile_id for tile_id in winning_hand_ids if tile_id in [RED_FIVE_MAN, RED_FIVE_PIN, RED_FIVE_SOU]])
            if red_fives_count > 0:
                env.logger.log_action(0, "game_info", f"赤寶牌數量: {red_fives_count}")
            
            score, yaku_list = calculate_score(counts, win_tile, is_self_draw, is_open,
                                              is_dealer, round_wind, player_wind,
                                              dora_count, is_riichi)
            
            env.logger.log_action(0, "game_info", "和牌信息：")
            env.logger.log_action(0, "game_info", f"和牌: {win_tile_emoji} {id_to_string(win_tile)}")
            env.logger.log_action(0, "game_info", f"自摸: {'是' if is_self_draw else '否'}")
            env.logger.log_action(0, "game_info", f"門前清: {'是' if not is_open else '否'}")
            env.logger.log_action(0, "game_info", f"莊家: {'是' if is_dealer else '否'}")
            env.logger.log_action(0, "game_info", f"立直: {'是' if is_riichi else '否'}")
            env.logger.log_action(0, "game_info", f"寶牌數: {dora_count}")
            
            env.logger.log_action(0, "game_info", f"役種列表: {yaku_list}")
            env.logger.log_action(0, "game_info", f"和牌得分: {score}點")
            
            # 檢查和牌是否有效
            if check_win(counts):
                env.logger.log_action(0, "game_info", "有效的和牌")
            else:
                env.logger.log_action(0, "game_info", "無效的和牌，請檢查手牌結構")
            
            # 直接處理贏家手牌和和牌
            winning_tiles = []
            for tile_id in winning_hand_ids[:-1]:
                if isinstance(tile_id, int):
                    winning_tiles.append(id_to_tile(tile_id))
                else:
                    winning_tiles.append(tile_id)
                
            if isinstance(win_tile, int):
                winning_tile = id_to_tile(win_tile)
            else:
                winning_tile = win_tile
                
            # 更新玩家得分
            env.total_scores[player_id] += score
            
            # 使用新的log_win方法記錄和牌
            if hasattr(env, 'logger') and env.logger:
                env.logger.log_win(player_id, winning_tiles, winning_tile, "自摸", score, yaku_list)
            
            # 結束遊戲
            done = True
            
            # 結束回合並生成記錄
            env.logger.end_episode(env.total_scores, score, is_win=True)
            
            break
        
        # 否則執行簡單的隨機動作
        valid_actions = []
        
        # 使用hand_ids而不是手牌對象
        for tile_id in hand_ids:
            # 只考慮有效的丟牌動作
            if 0 <= tile_id < 37:  # 考慮赤寶牌
                valid_actions.append(tile_id)
        
        # 隨機選擇一個有效的丟牌動作
        if valid_actions:
            action = random.choice(valid_actions)
            tile_emoji = id_to_emoji(action)
            env.logger.log_action(0, "game_info", f"玩家{player_id}丟棄: {tile_emoji} {id_to_string(action)}")
            
            # 記錄丟棄的牌到牌河
            discard_piles[player_id].append(action)
            
            _, reward, done, _ = env.step(action)
        else:
            env.logger.log_action(0, "game_info", f"玩家{player_id}無牌可丟")
            done = True
    
    # 顯示每個玩家的牌河
    env.logger.log_action(0, "game_info", "\n=== 玩家牌河 ===")
    for player_id, discard_pile in enumerate(discard_piles):
        if discard_pile:
            discard_emoji = [id_to_emoji(tile_id) for tile_id in discard_pile]
            env.logger.log_action(0, "game_info", f"玩家{player_id}牌河: {''.join(discard_emoji)}")
    
    # 生成並顯示遊戲記錄
    log_text = env.generate_text_log(current_episode)
    if log_text:
        env.logger.log_action(0, "game_info", "\n=== 遊戲記錄摘要 ===")
        with open(log_text, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
        # 檢查記錄中是否有赤寶牌的信息
        if "赤五" in log_content or "\u2764" in log_content:  # 紅心unicode
            env.logger.log_action(0, "game_info", "赤寶牌已正確記錄！")
        else:
            env.logger.log_action(0, "game_info", "警告：記錄中沒有找到赤寶牌信息")
            
        # 檢查記錄中是否有牌河信息
        if "牌河記錄" in log_content:
            env.logger.log_action(0, "game_info", "牌河已正確記錄！")
        else:
            env.logger.log_action(0, "game_info", "警告：記錄中沒有找到牌河信息")
            
        # 檢查記錄中是否有和牌信息
        if "和牌信息" in log_content:
            env.logger.log_action(0, "game_info", "和牌信息已正確記錄！")
        else:
            env.logger.log_action(0, "game_info", "警告：記錄中沒有找到和牌信息")
            
        # 顯示部分記錄內容
        log_lines = log_content.split('\n')
        if len(log_lines) > 20:
            env.logger.log_action(0, "game_info", '\n'.join(log_lines[:10]))
            env.logger.log_action(0, "game_info", "...")
            env.logger.log_action(0, "game_info", '\n'.join(log_lines[-10:]))
        else:
            env.logger.log_action(0, "game_info", log_content)
    
    env.logger.log_action(0, "game_info", "\n=== 遊戲結束 ===")

def create_winning_hand_with_red():
    """
    創建一個包含赤寶牌的示例和牌手牌
    """
    # 創建一個包含赤寶牌的"平和"和牌形式
    # 例如：一二三萬 + 四赤五六萬 + 六七八筒 + 二三四索 + 赤五五筒(雀頭)
    # 使用直接的牌ID表示
    winning_hand = [
        0, 1, 2,  # 一二三萬
        3, RED_FIVE_MAN, 5,  # 四赤五六萬
        15, 16, 17,  # 六七八筒
        19, 20, 21,  # 二三四索
        RED_FIVE_PIN, 13  # 赤五五筒(雀頭)
    ]
    
    return winning_hand

if __name__ == "__main__":
    print("開始完整麻將遊戲測試...")
    simulate_full_game() 