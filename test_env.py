from src.environment.mahjong_env import MahjongEnv
from src.utils.mahjong_utils import id_to_string

def test_environment():
    # 創建環境
    env = MahjongEnv()
    print("麻將環境創建成功")
    
    # 重置環境
    observation = env.reset()
    print(f"初始觀察: 形狀={observation.shape}")
    
    # 查看手牌
    player_id = env.current_player
    hand = env.players_hands[player_id]
    print(f"當前玩家: {player_id}")
    print(f"手牌數量: {len(hand)}")
    
    # 轉換為可讀形式並顯示
    try:
        readable_hand = [id_to_string(tile_to_id(tile)) for tile in hand]
        print(f"手牌: {readable_hand}")
    except Exception as e:
        print(f"無法轉換手牌: {e}")
        print(f"原始手牌數據: {hand}")
    
    # 執行幾個隨機動作
    for step in range(5):
        action = env.action_space.sample()  # 隨機選擇動作
        print(f"\n步驟 {step+1}")
        print(f"執行動作: {action}")
        
        observation, reward, done, info = env.step(action)
        print(f"獎勵: {reward}")
        print(f"遊戲結束: {done}")
        
        if done:
            print("遊戲結束!")
            break
    
    print("\n環境測試完成")

if __name__ == "__main__":
    from src.utils.mahjong_utils import tile_to_id
    print("開始測試麻將環境...")
    test_environment() 