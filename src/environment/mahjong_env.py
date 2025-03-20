import gym
import numpy as np
from gym import spaces
from src.utils.logger import MahjongLogger
from src.utils.mahjong_utils import check_win, hand_to_counts, tile_to_id, calculate_score

class MahjongEnv(gym.Env):
    """
    日本麻將強化學習環境
    """
    
    def __init__(self, enable_logging=True, log_level=1):
        super(MahjongEnv, self).__init__()
        
        # 定義動作空間
        # 例如：丟棄一張牌(34種), 吃(n種), 碰(n種), 槓(n種), 立直, 自摸, 榮和等
        self.action_space = spaces.Discrete(100)  # 臨時數值，需要根據實際動作數量調整
        
        # 定義觀察空間
        # 包括：自己的手牌，場上的狀態(河牌，立直狀態，場風等)
        self.observation_space = spaces.Box(low=0, high=1, shape=(200,), dtype=np.float32)  # 臨時設定
        
        # 遊戲記錄設置
        self.enable_logging = enable_logging
        if enable_logging:
            self.logger = MahjongLogger(debug_level=log_level)
        
        # 遊戲統計數據
        self.episode_count = 0
        self.total_scores = [0, 0, 0, 0]  # 四位玩家的總分
        
        # 初始化遊戲狀態
        self.reset()
    
    def reset(self):
        """
        重置環境到初始狀態
        """
        # 更新episode計數
        self.episode_count += 1
        
        # 開始新一輪的記錄
        if self.enable_logging:
            self.logger.start_episode(self.episode_count)
        
        # 初始化牌山
        self.tiles = self._init_tiles()
        
        # 記錄完整牌山（僅用於記錄，之後不再修改）
        if self.enable_logging:
            original_tiles = self.tiles.copy()
            self.logger.log_tiles(original_tiles)
        
        # 初始化玩家手牌
        self.players_hands = [[] for _ in range(4)]
        
        # 初始化河牌
        self.discards = [[] for _ in range(4)]
        
        # 初始化其他遊戲狀態
        self.current_player = 0
        self.dora_indicators = []
        self.round_wind = 0  # 東風場
        self.dealer = 0  # 起始莊家
        
        # 發牌
        self._deal_tiles()
        
        # 記錄初始手牌
        if self.enable_logging:
            for player in range(4):
                self.logger.log_initial_hand(player, self.players_hands[player])
        
        # 返回觀察
        return self._get_observation()
    
    def step(self, action):
        """
        執行一個動作並返回下一個狀態，獎勵和是否結束
        """
        # 執行動作前的狀態
        pre_action_hand = self.players_hands[self.current_player].copy() if self.enable_logging else None
        
        # 執行動作
        reward = self._execute_action(action)
        
        # 檢查遊戲是否結束
        done = self._check_game_end()
        
        # 獲取新的觀察
        observation = self._get_observation()
        
        # 額外信息
        info = {}
        
        # 記錄本輪遊戲結果
        if done and self.enable_logging:
            self.logger.end_episode(self.total_scores, reward, is_win=(reward > 0))
        
        return observation, reward, done, info
    
    def render(self, mode='human'):
        """
        渲染當前遊戲狀態
        """
        # 打印遊戲狀態
        print(f"當前玩家: {self.current_player}")
        print(f"手牌: {self.players_hands[self.current_player]}")
        print(f"河牌: {self.discards}")
        
    def _init_tiles(self):
        """
        初始化並洗牌
        """
        # 創建一副完整的麻將牌
        # 數牌 (萬子, 筒子, 索子): 1-9 各4張
        # 字牌 (東南西北白發中): 各4張
        tiles = []
        
        # 數牌
        for suit in range(3):  # 萬子, 筒子, 索子
            for value in range(1, 10):
                for _ in range(4):
                    tiles.append((suit, value))
        
        # 字牌
        for value in range(7):  # 東南西北白發中
            for _ in range(4):
                tiles.append((3, value))
        
        # 洗牌
        np.random.shuffle(tiles)
        
        return tiles
    
    def _deal_tiles(self):
        """
        發牌
        """
        # 每位玩家13張牌
        for i in range(13):
            for player in range(4):
                self.players_hands[player].append(self.tiles.pop())
        
        # 莊家額外摸一張
        self.players_hands[self.dealer].append(self.tiles.pop())
    
    def _execute_action(self, action):
        """
        執行動作並返回獎勵
        """
        # 這裡需要實現具體的麻將規則和動作處理
        # 例如：丟棄牌，吃碰槓，立直等
        
        # 臨時實現：隨機丟棄一張牌
        if action < 37:  # 支持丟棄所有牌，包括赤寶牌
            # 檢查玩家手牌中是否有這張牌
            tile_to_discard = None
            player_hand = self.players_hands[self.current_player]
            
            # 如果action直接是牌ID，先轉換為(suit, value)格式
            for tile in player_hand:
                if isinstance(tile, tuple) and tile_to_id(tile) == action:
                    tile_to_discard = tile
                    break
                elif tile == action:  # 如果已經是ID格式
                    tile_to_discard = action
                    break
            
            if tile_to_discard is not None:
                # 記錄打牌前的狀態（如果啟用記錄）
                if self.enable_logging:
                    # 摸牌動作通常在玩家回合開始時
                    # 假設每輪回合都會從牌山摸一張牌
                    if len(self.tiles) > 0:
                        drawn_tile = self.tiles.pop()
                        self.players_hands[self.current_player].append(drawn_tile)
                        self.logger.log_draw(self.current_player, 
                                             self.players_hands[self.current_player], 
                                             drawn_tile)
                
                # 執行丟棄動作
                self.players_hands[self.current_player].remove(tile_to_discard)
                self.discards[self.current_player].append(tile_to_discard)
                
                # 記錄丟牌
                if self.enable_logging:
                    self.logger.log_discard(self.current_player, 
                                          self.players_hands[self.current_player], 
                                          tile_to_discard)
                
                # 下一位玩家摸牌（在實際遊戲中下一位玩家摸牌會在下一個step中處理）
                self.current_player = (self.current_player + 1) % 4
        
        # 這裡簡單返回0獎勵，實際應該根據牌型計算
        reward = 0
        
        # 記錄獎勵
        if self.enable_logging and reward != 0:
            self.logger.log_reward(self.current_player, reward, "action outcome")
        
        return reward
    
    def _check_game_end(self):
        """
        檢查遊戲是否結束
        """
        # 牌山耗盡
        if len(self.tiles) <= 14:  # 保留14張牌(雙北)
            return True
        
        # 檢查和牌
        # 從mahjong_utils引入check_win函數
        from src.utils.mahjong_utils import check_win, hand_to_counts, tile_to_id, calculate_score
        
        # 判斷當前玩家是否和牌
        player_hand = self.players_hands[self.current_player]
        
        # 將手牌轉換為tile_id
        hand_ids = [tile_to_id(tile) for tile in player_hand]
        counts = hand_to_counts(hand_ids)
        
        # 檢查是否和牌
        if check_win(counts):
            # 如果和牌，記錄和牌信息
            if self.enable_logging:
                # 假設最後摸到的牌是和牌
                win_tile = player_hand[-1]
                win_tile_id = tile_to_id(win_tile)
                
                # 計算和牌得分
                score, yaku_list = calculate_score(counts, win_tile_id, 
                                                is_self_draw=True, 
                                                is_open=False,
                                                is_dealer=(self.current_player == self.dealer),
                                                round_wind=self.round_wind,
                                                player_wind=self.current_player,
                                                dora_count=1,  # 假設有一張寶牌
                                                is_riichi=True)  # 假設立直
                
                # 記錄和牌
                self.logger.log_win(self.current_player, player_hand, win_tile, "自摸", score, yaku_list)
                
                # 記錄獎勵
                self.logger.log_reward(self.current_player, score, "和牌")
            
            return True
        
        return False
    
    def _get_observation(self):
        """
        獲取當前遊戲狀態的觀察
        """
        # 將遊戲狀態轉換為神經網絡可用的格式
        # 這裡只是一個占位實現，實際應該設計一個合理的狀態表示
        
        # 臨時返回一個隨機觀察
        return np.random.rand(200).astype(np.float32)
    
    def generate_text_log(self, episode_num=None):
        """
        生成特定回合的文本格式記錄
        """
        if not self.enable_logging:
            print("未啟用記錄功能")
            return None
        
        # 如果未指定回合，則使用當前回合
        episode_to_log = episode_num if episode_num is not None else self.episode_count
        
        return self.logger.generate_text_log(episode_to_log) 