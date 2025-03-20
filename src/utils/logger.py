import os
import json
import datetime
from src.utils.mahjong_utils import id_to_string, id_to_emoji, tile_to_id, sort_hand, analyze_winning_hand, calculate_score, hand_to_counts

class MahjongLogger:
    """
    麻將訓練記錄器，用於記錄訓練過程中的牌局情況
    """
    def __init__(self, log_dir="logs", debug_level=1):
        """
        初始化記錄器
        
        參數:
            log_dir: 記錄文件存放目錄
            debug_level: 調試級別，0-不記錄，1-僅記錄關鍵動作，2-記錄所有動作
        """
        self.log_dir = log_dir
        self.debug_level = debug_level
        self.current_episode = 0
        self.current_game_log = []
        self.initial_tiles = None
        self.player_discard_tiles = [[] for _ in range(4)]  # 記錄每位玩家的牌河
        self.winning_player = None  # 記錄胡牌的玩家
        self.winning_tiles = None  # 記錄胡牌的牌型
        self.winning_tile = None  # 記錄和了的牌
        
        # 創建日誌目錄
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 創建會話目錄（以時間戳命名）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(log_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir)
        
        # 初始化會話摘要文件
        self.summary_file = os.path.join(self.session_dir, "summary.json")
        self.episodes_summary = []
    
    def start_episode(self, episode_num):
        """開始記錄新的episode"""
        self.current_episode = episode_num
        self.current_game_log = []
        self.episode_start_time = datetime.datetime.now()
        self.initial_tiles = None
        self.player_discard_tiles = [[] for _ in range(4)]  # 重置所有玩家的牌河
        self.winning_player = None
        self.winning_tiles = None
        self.winning_tile = None
        
        if self.debug_level > 0:
            print(f"開始記錄第 {episode_num} 輪遊戲")
    
    def log_tiles(self, tiles):
        """記錄整個牌山"""
        if self.debug_level == 0:
            return
            
        try:
            # 轉換牌ID為可讀形式和Emoji
            readable_tiles = [id_to_string(tile_to_id(tile)) for tile in tiles]
            emoji_tiles = [id_to_emoji(tile_to_id(tile)) for tile in tiles]
            
            # 記錄牌山
            self.initial_tiles = readable_tiles
            
            entry = {
                "action_type": "tiles",
                "tiles": readable_tiles,
                "emoji_tiles": emoji_tiles,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            
            self.current_game_log.append(entry)
            
            if self.debug_level > 1:
                print(f"本局牌山: {''.join(emoji_tiles[:20])}... (共{len(emoji_tiles)}張)")
        except Exception as e:
            print(f"記錄牌山時出錯: {e}")
    
    def log_initial_hand(self, player_id, hand):
        """記錄初始手牌"""
        if self.debug_level == 0:
            return
            
        try:
            # 轉換牌ID為可讀形式和Emoji
            readable_hand = [id_to_string(tile_to_id(tile)) for tile in hand]
            emoji_hand = [id_to_emoji(tile_to_id(tile)) for tile in hand]
            
            # 對手牌進行排序
            sorted_tile_ids = [tile_to_id(tile) for tile in hand]
            sorted_tile_ids = sort_hand(sorted_tile_ids)
            sorted_emoji_hand = [id_to_emoji(tile_id) for tile_id in sorted_tile_ids]
            
            entry = {
                "action_type": "initial_hand",
                "player_id": player_id,
                "hand": readable_hand,
                "emoji_hand": emoji_hand,
                "sorted_emoji_hand": sorted_emoji_hand,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            
            self.current_game_log.append(entry)
            
            if self.debug_level > 1:
                print(f"玩家 {player_id} 初始手牌: {''.join(sorted_emoji_hand)}")
        except Exception as e:
            print(f"記錄初始手牌時出錯: {e}")
    
    def log_draw(self, player_id, hand, drawn_tile):
        """記錄摸牌"""
        if self.debug_level == 0:
            return
            
        try:
            # 轉換牌ID為可讀形式和Emoji
            readable_hand = [id_to_string(tile_to_id(tile)) for tile in hand]
            readable_drawn = id_to_string(tile_to_id(drawn_tile))
            
            # 對手牌進行排序
            sorted_tile_ids = [tile_to_id(tile) for tile in hand]
            sorted_tile_ids = sort_hand(sorted_tile_ids)
            sorted_emoji_hand = [id_to_emoji(tile_id) for tile_id in sorted_tile_ids]
            
            emoji_hand = [id_to_emoji(tile_to_id(tile)) for tile in hand]
            emoji_drawn = id_to_emoji(tile_to_id(drawn_tile))
            
            entry = {
                "action_type": "draw",
                "player_id": player_id,
                "hand": readable_hand,
                "drawn_tile": readable_drawn,
                "emoji_hand": emoji_hand,
                "emoji_drawn": emoji_drawn,
                "sorted_emoji_hand": sorted_emoji_hand,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            
            self.current_game_log.append(entry)
            
            if self.debug_level > 1:
                print(f"玩家 {player_id} 摸牌: {emoji_drawn}, 手牌: {''.join(sorted_emoji_hand)}")
        except Exception as e:
            print(f"記錄摸牌時出錯: {e}")
    
    def log_discard(self, player_id, hand, discarded_tile):
        """記錄打牌"""
        if self.debug_level == 0:
            return
            
        try:
            # 轉換牌ID為可讀形式和Emoji
            readable_hand = [id_to_string(tile_to_id(tile)) for tile in hand]
            readable_discarded = id_to_string(tile_to_id(discarded_tile))
            
            # 對手牌進行排序
            sorted_tile_ids = [tile_to_id(tile) for tile in hand]
            sorted_tile_ids = sort_hand(sorted_tile_ids)
            sorted_emoji_hand = [id_to_emoji(tile_id) for tile_id in sorted_tile_ids]
            
            emoji_hand = [id_to_emoji(tile_to_id(tile)) for tile in hand]
            emoji_discarded = id_to_emoji(tile_to_id(discarded_tile))
            
            # 添加到玩家的牌河
            tile_id = tile_to_id(discarded_tile)
            self.player_discard_tiles[player_id].append(tile_id)
            
            entry = {
                "action_type": "discard",
                "player_id": player_id,
                "hand": readable_hand,
                "discarded_tile": readable_discarded,
                "emoji_hand": emoji_hand,
                "emoji_discarded": emoji_discarded,
                "sorted_emoji_hand": sorted_emoji_hand,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            
            self.current_game_log.append(entry)
            
            if self.debug_level > 1:
                print(f"玩家 {player_id} 打牌: {emoji_discarded}, 手牌: {''.join(sorted_emoji_hand)}")
        except Exception as e:
            print(f"記錄打牌時出錯: {e}")
    
    def log_action(self, player_id, action_type, action_detail, hand=None):
        """記錄其他動作（如吃、碰、槓、立直等）"""
        if self.debug_level == 0:
            return
            
        try:
            entry = {
                "action_type": action_type,
                "player_id": player_id,
                "detail": action_detail,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            
            if hand is not None:
                # 轉換牌ID為可讀形式和Emoji
                readable_hand = [id_to_string(tile_to_id(tile)) for tile in hand]
                emoji_hand = [id_to_emoji(tile_to_id(tile)) for tile in hand]
                
                # 對手牌進行排序
                sorted_tile_ids = [tile_to_id(tile) for tile in hand]
                sorted_tile_ids = sort_hand(sorted_tile_ids)
                sorted_emoji_hand = [id_to_emoji(tile_id) for tile_id in sorted_tile_ids]
                
                entry["hand"] = readable_hand
                entry["emoji_hand"] = emoji_hand
                entry["sorted_emoji_hand"] = sorted_emoji_hand
            
            self.current_game_log.append(entry)
            
            if self.debug_level > 1:
                hand_display = f", 手牌: {''.join(sorted_emoji_hand)}" if hand is not None else ""
                print(f"玩家 {player_id} {action_type}: {action_detail}{hand_display}")
        except Exception as e:
            print(f"記錄動作時出錯: {e}")
    
    def log_win(self, player_id, hand, win_tile, win_type="和牌", score=None, yaku_list=None):
        """記錄和牌"""
        if self.debug_level == 0:
            return
            
        try:
            # 轉換牌ID
            hand_ids = [tile_to_id(tile) for tile in hand]
            win_tile_id = tile_to_id(win_tile)
            
            # 記錄胡牌玩家和牌型
            self.winning_player = player_id
            self.winning_tiles = analyze_winning_hand(hand_ids, win_tile_id)
            self.winning_tile = win_tile_id
            
            # 轉換為可讀格式和Emoji
            readable_hand = [id_to_string(tile_id) for tile_id in hand_ids]
            emoji_hand = [id_to_emoji(tile_id) for tile_id in hand_ids]
            
            readable_win_tile = id_to_string(win_tile_id)
            emoji_win_tile = id_to_emoji(win_tile_id)
            
            # 排序手牌並轉換為Emoji
            winning_emoji_tiles = [id_to_emoji(tile_id) for tile_id in self.winning_tiles]
            
            # 計算詳細的役種點數
            yaku_details = []
            total_han = 0
            if yaku_list:
                for yaku in yaku_list:
                    if "寶牌" in yaku:
                        han = int(yaku.split()[1])
                        yaku_details.append(f"{yaku} (+{han}翻)")
                        total_han += han
                    elif "赤寶牌" in yaku:
                        han = int(yaku.split()[1])
                        yaku_details.append(f"{yaku} (+{han}翻)")
                        total_han += han
                    elif yaku == "立直":
                        yaku_details.append(f"{yaku} (+1翻)")
                        total_han += 1
                    elif yaku == "門前清自摸和":
                        yaku_details.append(f"{yaku} (+1翻)")
                        total_han += 1
                    else:
                        yaku_details.append(f"{yaku} (+1翻)")
                        total_han += 1
            
            # 計算滿貫等級
            if total_han >= 13:
                limit = "役滿"
            elif total_han >= 11:
                limit = "三倍滿"
            elif total_han >= 8:
                limit = "倍滿"
            elif total_han >= 6:
                limit = "跳滿"
            elif total_han >= 5:
                limit = "滿貫"
            else:
                limit = f"{total_han}翻"
            
            entry = {
                "action_type": win_type,
                "player_id": player_id,
                "hand": readable_hand,
                "win_tile": readable_win_tile,
                "emoji_hand": emoji_hand,
                "emoji_win_tile": emoji_win_tile,
                "sorted_winning_tiles": winning_emoji_tiles,
                "score": score,
                "yaku_list": yaku_list,
                "yaku_details": yaku_details,
                "total_han": total_han,
                "limit": limit,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            }
            
            self.current_game_log.append(entry)
            
            if self.debug_level > 0:
                print(f"玩家 {player_id} {win_type}: {emoji_win_tile}")
                print(f"胡牌牌型: {''.join(winning_emoji_tiles)}")
                if yaku_details:
                    print("\n役種詳情:")
                    for detail in yaku_details:
                        print(f"  {detail}")
                    print(f"\n總翻數: {total_han}翻")
                    print(f"滿貫等級: {limit}")
                if score is not None:
                    print(f"最終得分: {score}點")
        except Exception as e:
            print(f"記錄和牌時出錯: {e}")
    
    def log_reward(self, player_id, reward, reason=""):
        """記錄獎勵"""
        if self.debug_level == 0:
            return
            
        entry = {
            "action_type": "reward",
            "player_id": player_id,
            "reward": reward,
            "reason": reason,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        }
        
        self.current_game_log.append(entry)
        
        if self.debug_level > 1:
            print(f"玩家 {player_id} 獲得獎勵: {reward}, 原因: {reason}")
    
    def end_episode(self, scores, total_reward, is_win=False):
        """結束當前episode的記錄並保存"""
        if self.debug_level == 0:
            return
            
        # 計算遊戲時長
        episode_end_time = datetime.datetime.now()
        duration = (episode_end_time - self.episode_start_time).total_seconds()
        
        # 創建摘要信息
        summary = {
            "episode": self.current_episode,
            "duration": duration,
            "total_reward": total_reward,
            "scores": scores,
            "is_win": is_win,
            "winning_player": self.winning_player,
            "timestamp": episode_end_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 將摘要添加到會話摘要中
        self.episodes_summary.append(summary)
        
        # 將當前遊戲記錄保存到文件
        if len(self.current_game_log) > 0:
            game_file = os.path.join(self.session_dir, f"episode_{self.current_episode}.json")
            
            # 添加玩家牌河信息
            discard_info = []
            for player_id, discard_tiles in enumerate(self.player_discard_tiles):
                discard_emoji = [id_to_emoji(tile_id) for tile_id in discard_tiles]
                discard_info.append({
                    "player_id": player_id,
                    "discard_tiles": [id_to_string(tile_id) for tile_id in discard_tiles],
                    "discard_emoji": discard_emoji
                })
                
            # 添加胡牌信息
            winning_info = None
            if self.winning_player is not None and self.winning_tiles is not None:
                winning_emoji = [id_to_emoji(tile_id) for tile_id in self.winning_tiles]
                winning_info = {
                    "player_id": self.winning_player,
                    "winning_tiles": [id_to_string(tile_id) for tile_id in self.winning_tiles],
                    "winning_emoji": winning_emoji,
                    "winning_tile": id_to_string(self.winning_tile) if self.winning_tile else None,
                    "winning_tile_emoji": id_to_emoji(self.winning_tile) if self.winning_tile else None
                }
            
            with open(game_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": summary,
                    "discard_info": discard_info,
                    "winning_info": winning_info,
                    "actions": self.current_game_log
                }, f, ensure_ascii=False, indent=2)
        
        # 更新會話摘要文件
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.episodes_summary, f, ensure_ascii=False, indent=2)
        
        if self.debug_level > 0:
            print(f"第 {self.current_episode} 輪遊戲記錄已保存, 總獎勵: {total_reward}, 遊戲時長: {duration:.2f}秒")
    
    def generate_text_log(self, episode_num):
        """生成特定回合的文本格式記錄"""
        game_file = os.path.join(self.session_dir, f"episode_{episode_num}.json")
        text_file = os.path.join(self.session_dir, f"episode_{episode_num}_text.log")
        
        if not os.path.exists(game_file):
            print(f"找不到第 {episode_num} 輪遊戲的記錄")
            return
        
        try:
            # 讀取JSON記錄
            with open(game_file, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
            
            # 檢查是否已經存在文本記錄
            is_new_file = not os.path.exists(text_file)
            
            # 根據文件是否存在決定使用哪種模式
            mode = 'w' if is_new_file else 'a'
            
            # 按照 [手上的牌] [摸到的牌] ===>[打出的牌] 格式生成文本記錄
            with open(text_file, mode, encoding='utf-8') as f:
                # 如果是新文件，寫入標題
                if is_new_file:
                    f.write(f"第 {episode_num} 輪遊戲記錄\n")
                    f.write(f"遊戲時間: {game_data['summary']['timestamp']}\n")
                    f.write(f"總獎勵: {game_data['summary']['total_reward']}\n")
                    f.write(f"遊戲結果: {'勝利' if game_data['summary']['is_win'] else '失敗'}\n")
                    f.write("-" * 50 + "\n\n")
                    
                    # 先輸出牌山信息（如果有）
                    for action in game_data['actions']:
                        if action['action_type'] == 'tiles':
                            emoji_tiles = action.get('emoji_tiles', [])
                            if emoji_tiles:
                                # 每行顯示20張牌，方便閱讀
                                for i in range(0, len(emoji_tiles), 20):
                                    f.write(f"牌山: {''.join(emoji_tiles[i:i+20])}\n")
                                f.write("\n")
                            else:
                                tiles = action['tiles']
                                f.write(f"牌山: {', '.join(tiles)}\n\n")
                            break
                    
                    # 輸出初始手牌
                    for action in game_data['actions']:
                        if action['action_type'] == 'initial_hand':
                            player_id = action['player_id']
                            sorted_emoji_hand = action.get('sorted_emoji_hand', [])
                            if sorted_emoji_hand:
                                f.write(f"玩家 {player_id} 初始手牌: {''.join(sorted_emoji_hand)}\n")
                            else:
                                hand = action['hand']
                                f.write(f"玩家 {player_id} 初始手牌: {', '.join(hand)}\n")
                    
                    f.write("\n")
                    
                    # 輸出牌河信息
                    f.write("===== 牌河記錄 =====\n")
                    if 'discard_info' in game_data:
                        for discard in game_data['discard_info']:
                            player_id = discard['player_id']
                            emoji_tiles = discard.get('discard_emoji', [])
                            
                            if emoji_tiles:
                                # 每行顯示20張牌，方便閱讀
                                f.write(f"玩家 {player_id} 牌河: ")
                                for i in range(0, len(emoji_tiles), 20):
                                    if i > 0:
                                        f.write("          ")
                                    f.write(f"{''.join(emoji_tiles[i:i+20])}\n")
                            else:
                                tiles = discard['discard_tiles']
                                f.write(f"玩家 {player_id} 牌河: {', '.join(tiles)}\n")
                    f.write("\n")
                
                # 輸出和牌信息
                if 'winning_info' in game_data and game_data['winning_info']:
                    winning = game_data['winning_info']
                    player_id = winning['player_id']
                    f.write("===== 和牌信息 =====\n")
                    f.write(f"和牌者: 玩家 {player_id}\n")
                    
                    emoji_tiles = winning.get('winning_emoji', [])
                    win_tile_emoji = winning.get('winning_tile_emoji', None)
                    
                    if emoji_tiles:
                        f.write(f"和牌牌型: {''.join(emoji_tiles)}\n")
                    else:
                        tiles = winning['winning_tiles']
                        f.write(f"和牌牌型: {', '.join(tiles)}\n")
                    
                    if win_tile_emoji:
                        f.write(f"和了牌: {win_tile_emoji}\n")
                    elif 'winning_tile' in winning and winning['winning_tile']:
                        f.write(f"和了牌: {winning['winning_tile']}\n")
                    
                    # 添加役種和計分過程
                    for action in game_data['actions']:
                        if action['action_type'] in ['和牌', 'ron', 'tsumo']:
                            if 'yaku_details' in action:
                                f.write("\n役種詳情:\n")
                                for detail in action['yaku_details']:
                                    f.write(f"  {detail}\n")
                                if 'total_han' in action:
                                    f.write(f"\n總翻數: {action['total_han']}翻\n")
                                if 'limit' in action:
                                    f.write(f"滿貫等級: {action['limit']}\n")
                            if 'score' in action:
                                f.write(f"最終得分: {action['score']}點\n")
                    
                    # 添加計分信息
                    if 'summary' in game_data:
                        summary = game_data['summary']
                        f.write(f"總獎勵: {summary['total_reward']}\n")
                        
                        f.write("\n玩家得分:\n")
                        for i, score in enumerate(summary['scores']):
                            f.write(f"玩家 {i}: {score}分\n")
                    
                    f.write("\n")
                
                # 跟踪遊戲進程，特別關注AI玩家的動作
                f.write("===== 遊戲過程 =====\n")
                current_player = None
                current_hand_emoji = None
                drawn_tile_emoji = None
                
                for action in game_data['actions']:
                    if action['action_type'] == 'initial_hand':
                        player_id = action['player_id']
                        
                        if player_id == 0:  # 假設玩家0是AI代理
                            current_player = player_id
                            current_hand_emoji = action.get('sorted_emoji_hand', action.get('emoji_hand', []))
                    
                    elif action['action_type'] == 'draw':
                        player_id = action['player_id']
                        if player_id == current_player:
                            current_hand_emoji = action.get('sorted_emoji_hand', action.get('emoji_hand', []))
                            drawn_tile_emoji = action.get('emoji_drawn', None)
                            
                            if drawn_tile_emoji:
                                f.write(f"[{''.join(current_hand_emoji)}] [摸到: {drawn_tile_emoji}] ")
                            else:
                                current_hand = action['hand']
                                drawn_tile = action['drawn_tile']
                                f.write(f"[{', '.join(current_hand)}] [摸到: {drawn_tile}] ")
                    
                    elif action['action_type'] == 'discard':
                        player_id = action['player_id']
                        if player_id == current_player and drawn_tile_emoji is not None:
                            discarded_emoji = action.get('emoji_discarded', None)
                            
                            if discarded_emoji:
                                f.write(f"===>[打出: {discarded_emoji}]\n")
                            else:
                                discarded = action['discarded_tile']
                                f.write(f"===>[打出: {discarded}]\n")
                                
                            drawn_tile_emoji = None
                    
                    elif action['action_type'] in ['和牌', 'ron', 'tsumo']:
                        player_id = action['player_id']
                        win_tile_emoji = action.get('emoji_win_tile', None)
                        winning_tiles = action.get('sorted_winning_tiles', [])
                        
                        if win_tile_emoji and winning_tiles:
                            f.write(f"玩家 {player_id} {action['action_type']}: {win_tile_emoji}\n")
                            f.write(f"和牌牌型: {''.join(winning_tiles)}\n")
                        else:
                            detail = action.get('detail', '')
                            f.write(f"玩家 {player_id} {action['action_type']}: {detail}\n")
                    
                    elif action['action_type'] == 'reward':
                        player_id = action['player_id']
                        if player_id == current_player:
                            reward = action['reward']
                            reason = action['reason']
                            f.write(f"獲得獎勵: {reward}, 原因: {reason}\n")
                    
                    elif action['action_type'] in ['chi', 'pon', 'kan', 'riichi']:
                        player_id = action['player_id']
                        if player_id == current_player:
                            f.write(f"玩家 {player_id} {action['action_type']}: {action['detail']}\n")
                    
                    elif action['action_type'] == 'game_info':
                        f.write(f"{action['detail']}\n")
                
                f.write("\n" + "-" * 50 + "\n")
                f.write(f"遊戲結束，總獎勵: {game_data['summary']['total_reward']}\n")
            
            print(f"第 {episode_num} 輪遊戲的文本記錄已生成: {text_file}")
            return text_file
            
        except Exception as e:
            print(f"生成文本記錄時出錯: {e}")
            return None 