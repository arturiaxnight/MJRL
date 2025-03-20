import numpy as np

# 牌型定義
SUITS = ['萬', '筒', '索', '字']
VALUES = ['一', '二', '三', '四', '五', '六', '七', '八', '九']
HONORS = ['東', '南', '西', '北', '白', '發', '中']

# 牌的總數目
TOTAL_TILES = 34 * 4  # 34種牌，每種4張

# 赤寶牌ID：使用額外ID來表示赤寶牌（紅色的五）
RED_FIVE_MAN = 34  # 赤五萬
RED_FIVE_PIN = 35  # 赤五筒
RED_FIVE_SOU = 36  # 赤五索

# 役種定義
class YakuType:
    # 1 番役
    RIICHI = "立直"                # 門前清限定，宣告聽牌
    IPPATSU = "一發"              # 立直後一巡內和牌
    TSUMO = "門前清自摸和"         # 門前清限定，自摸和牌
    PINFU = "平和"                # 門前清限定，無番牌的雀頭，全順子，聽牌形狀為兩面聽
    TANYAO = "斷么九"             # 沒有么九牌（1,9及字牌）的和牌
    IIPEIKOU = "一盃口"            # 門前清限定，兩組相同順子
    HAKU = "役牌 白"              # 包含白龍刻子
    HATSU = "役牌 發"             # 包含發財刻子
    CHUN = "役牌 中"              # 包含紅中刻子
    EAST = "役牌 東"              # 包含圈風或自風的東刻子
    SOUTH = "役牌 南"             # 包含圈風或自風的南刻子
    WEST = "役牌 西"              # 包含圈風或自風的西刻子
    NORTH = "役牌 北"             # 包含圈風或自風的北刻子
    AKADORA = "赤寶牌"            # 包含赤五（赤寶牌）
    
    # 2 番役
    DOUBLE_RIICHI = "雙立直"      # 門前清限定，首巡宣告聽牌
    CHANTAIYAO = "混全帶么九"      # 所有順子、刻子都包含么九牌
    SANSHOKU_DOUJUN = "三色同順"   # 三種花色相同數字的順子
    ITTSU = "一氣通貫"            # 同一花色的123、456、789順子
    TOITOI = "對對和"             # 全刻子和牌
    SANANKOU = "三暗刻"           # 三組暗刻
    SANKANTSU = "三槓子"          # 三組槓子
    CHIITOITSU = "七對子"         # 七組對子構成的和牌
    HONROUTOU = "混老頭"          # 全么九牌構成的和牌
    SHOUSANGEN = "小三元"          # 兩組三元牌的刻子及一組雀頭
    DOUBLE_WIND = "ダブル東場"     # 場風和自風相同
    
    # 3 番役
    HONITSU = "混一色"            # 同一花色加字牌的和牌
    JUNCHAN = "純全帶么九"         # 所有順子、刻子都包含么九牌，不含字牌
    RYANPEIKOU = "兩盃口"          # 門前清限定，兩組一盃口
    
    # 4 番役（字一色以下在食斷時減一番）
    CHINITSU = "清一色"            # 僅包含同一花色的和牌
    
    # 6 番役
    # 在某些規則中，這些役被視為日本麻將的"六番役"
    
    # 役滿
    KOKUSHI_MUSOU = "國士無雙"     # 13種么九牌加其中一種
    SUUANKOU = "四暗刻"            # 四組暗刻
    DAISANGEN = "大三元"           # 三組三元牌的刻子
    SHOUSUUSHII = "小四喜"         # 三組風牌的刻子加一組風牌的雀頭
    DAISUUSHII = "大四喜"          # 四組風牌的刻子
    TSUUIISOU = "字一色"           # 全部都是字牌的和牌
    CHINROUTOU = "清老頭"          # 全部都是么九數牌的和牌
    RYUUIISOU = "綠一色"           # 全部都是23468索和發的和牌
    CHUUREN_POUTOU = "九連寶燈"    # 同一花色1112345678999加任一張同花色
    SUUKANTSU = "四槓子"           # 四組槓子
    
    # 雙倍役滿
    TENHOU = "天和"               # 莊家第一輪自摸和牌
    CHIIHOU = "地和"              # 閒家第一輪自摸和牌
    DAISUURIN = "大數隣"          # 索子的22334455667788
    KOKUSHI_MUSOU_13 = "國士無雙十三面待ち" # 國士無雙13面聽，獲勝牌是14張不同的么九牌中的任意一張

# 麻將牌的Emoji表示
TILE_EMOJIS = {
    # 萬子 (Characters)
    0: "🀇", 1: "🀈", 2: "🀉", 3: "🀊", 4: "🀋", 5: "🀌", 6: "🀍", 7: "🀎", 8: "🀏",
    # 筒子 (Dots)
    9: "🀐", 10: "🀑", 11: "🀒", 12: "🀓", 13: "🀔", 14: "🀕", 15: "🀖", 16: "🀗", 17: "🀘",
    # 索子 (Bamboos)
    18: "🀙", 19: "🀚", 20: "🀛", 21: "🀜", 22: "🀝", 23: "🀞", 24: "🀟", 25: "🀠", 26: "🀡",
    # 風牌 (Winds)
    27: "🀀", 28: "🀁", 29: "🀂", 30: "🀃",
    # 三元牌 (Dragons)
    31: "🀆", 32: "🀅", 33: "🀄",
    # 赤寶牌 (Red tiles)
    34: "🀋\u2764", 35: "🀔\u2764", 36: "🀝\u2764"  # 紅心表示赤寶牌
}

def tile_to_id(tile):
    """
    將麻將牌轉換為唯一id
    tile: (suit, value)
    返回: 0-36的整數，34-36為赤寶牌
    """
    suit, value = tile
    # 檢查是否為赤寶牌
    if hasattr(tile, 'is_red') and tile.is_red:
        if suit == 0 and value == 5:  # 赤五萬
            return RED_FIVE_MAN
        elif suit == 1 and value == 5:  # 赤五筒
            return RED_FIVE_PIN
        elif suit == 2 and value == 5:  # 赤五索
            return RED_FIVE_SOU
            
    if suit < 3:  # 萬子, 筒子, 索子
        return suit * 9 + (value - 1)
    else:  # 字牌
        return 27 + value

def id_to_tile(tile_id):
    """
    將ID轉換回麻將牌
    tile_id: 0-36的整數
    返回: (suit, value)
    """
    # 處理赤寶牌
    if tile_id == RED_FIVE_MAN:
        return (0, 5)  # 赤五萬
    elif tile_id == RED_FIVE_PIN:
        return (1, 5)  # 赤五筒
    elif tile_id == RED_FIVE_SOU:
        return (2, 5)  # 赤五索
        
    if tile_id < 27:  # 數牌
        suit = tile_id // 9
        value = (tile_id % 9) + 1
        return (suit, value)
    else:  # 字牌
        return (3, tile_id - 27)

def id_to_string(tile_id):
    """
    將ID轉換為可讀字符串
    tile_id: 0-36的整數
    返回: 例如 "一萬", "三筒", "東", "赤五萬"
    """
    # 處理赤寶牌
    if tile_id == RED_FIVE_MAN:
        return f"赤五萬"
    elif tile_id == RED_FIVE_PIN:
        return f"赤五筒"
    elif tile_id == RED_FIVE_SOU:
        return f"赤五索"
    
    suit, value = id_to_tile(tile_id)
    if suit < 3:  # 數牌
        return f"{VALUES[value-1]}{SUITS[suit]}"
    else:  # 字牌
        return HONORS[value]

def id_to_emoji(tile_id):
    """
    將ID轉換為Emoji表示
    tile_id: 0-36的整數
    返回: 麻將牌的Emoji符號
    """
    if 0 <= tile_id <= 36 and tile_id in TILE_EMOJIS:
        return TILE_EMOJIS[tile_id]
    else:
        return "❓"  # 未知牌

def string_to_id(tile_string):
    """
    將可讀字符串轉換為ID
    """
    # 處理赤寶牌
    if tile_string == "赤五萬":
        return RED_FIVE_MAN
    elif tile_string == "赤五筒":
        return RED_FIVE_PIN
    elif tile_string == "赤五索":
        return RED_FIVE_SOU
    
    # 字牌處理
    if tile_string in HONORS:
        value = HONORS.index(tile_string)
        return tile_to_id((3, value))
    
    # 數牌處理
    for suit_idx, suit in enumerate(SUITS[:3]):
        if tile_string.endswith(suit):
            value_str = tile_string[:-1]
            value_idx = VALUES.index(value_str) + 1
            return tile_to_id((suit_idx, value_idx))
    
    raise ValueError(f"無法解析牌: {tile_string}")

def is_red_five(tile_id):
    """
    檢查是否為赤寶牌
    """
    return tile_id in [RED_FIVE_MAN, RED_FIVE_PIN, RED_FIVE_SOU]

def normalize_red_five(tile_id):
    """
    將赤寶牌ID轉換為普通牌ID，用於計算牌型
    """
    if tile_id == RED_FIVE_MAN:
        return 4  # 普通五萬
    elif tile_id == RED_FIVE_PIN:
        return 13  # 普通五筒
    elif tile_id == RED_FIVE_SOU:
        return 22  # 普通五索
    return tile_id

def sort_hand(hand_ids):
    """
    對手牌進行排序，順序為：萬子、筒子、索子、字牌
    hand_ids: 手牌ID列表
    返回: 排序後的ID列表
    """
    # 將赤寶牌ID轉換為對應普通牌ID用於排序
    normalized_hand = [(normalize_red_five(id), id) for id in hand_ids]
    
    # 排序: 先按照普通牌ID排序
    sorted_hand = sorted(normalized_hand, key=lambda x: x[0])
    
    # 返回原始ID
    return [original_id for _, original_id in sorted_hand]

def hand_to_counts(hand):
    """
    將手牌轉換為計數數組
    hand: 牌的ID列表
    返回: 長度為37的數組，表示每種牌的數量（包括赤寶牌）
    """
    counts = np.zeros(37, dtype=np.int32)
    for tile_id in hand:
        counts[tile_id] += 1
    return counts

def normalize_counts(counts):
    """
    將包含赤寶牌的counts轉換為標準counts（赤寶牌計入對應的普通牌）
    用於和牌判斷等需要忽略赤寶牌的情況
    """
    normalized = counts.copy()
    if RED_FIVE_MAN < len(counts) and counts[RED_FIVE_MAN] > 0:
        normalized[4] += counts[RED_FIVE_MAN]
        normalized[RED_FIVE_MAN] = 0
    if RED_FIVE_PIN < len(counts) and counts[RED_FIVE_PIN] > 0:
        normalized[13] += counts[RED_FIVE_PIN]
        normalized[RED_FIVE_PIN] = 0
    if RED_FIVE_SOU < len(counts) and counts[RED_FIVE_SOU] > 0:
        normalized[22] += counts[RED_FIVE_SOU]
        normalized[RED_FIVE_SOU] = 0
    return normalized

def is_valid_hand(counts):
    """
    檢查手牌是否合法（總數14張）
    """
    return np.sum(counts[:34]) == 14  # 只檢查標準牌的數量，不包括赤寶牌

def check_win(counts):
    """
    檢查是否和牌
    counts: 長度為37的數組，表示每種牌的數量
    返回: 是否和牌
    """
    # 首先將赤寶牌計入對應的普通牌
    normalized_counts = normalize_counts(counts)
    
    # 檢查總牌數
    if np.sum(normalized_counts[:34]) != 14:
        return False
    
    # 檢查雀頭+順子+刻子的形式
    for pair_tile in range(34):
        if normalized_counts[pair_tile] >= 2:
            # 嘗試將這個牌作為雀頭
            test_counts = normalized_counts.copy()
            test_counts[pair_tile] -= 2
            
            # 檢查剩餘牌是否可以組成4組順子或刻子
            if check_sets(test_counts):
                return True
    
    # 檢查特殊和牌：七對子
    pairs_count = np.sum(normalized_counts[:34] == 2)
    if pairs_count == 7:
        return True
    
    # 檢查特殊和牌：國士無雙
    if check_kokushi_musou(normalized_counts):
        return True
    
    return False

def check_sets(counts):
    """
    檢查剩餘牌是否可以組成4組順子或刻子
    """
    # 如果沒有牌了，說明成功組合
    if np.sum(counts[:34]) == 0:
        return True
    
    # 嘗試組成刻子
    for tile in range(34):
        if counts[tile] >= 3:
            test_counts = counts.copy()
            test_counts[tile] -= 3
            if check_sets(test_counts):
                return True
    
    # 嘗試組成順子(僅對數牌)
    for suit in range(3):  # 萬子, 筒子, 索子
        for start_value in range(7):  # 1-7作為起始點
            idx1 = suit * 9 + start_value
            idx2 = suit * 9 + start_value + 1
            idx3 = suit * 9 + start_value + 2
            
            if counts[idx1] > 0 and counts[idx2] > 0 and counts[idx3] > 0:
                test_counts = counts.copy()
                test_counts[idx1] -= 1
                test_counts[idx2] -= 1
                test_counts[idx3] -= 1
                if check_sets(test_counts):
                    return True
    
    # 無法組成有效的組合
    return False

def check_kokushi_musou(counts):
    """
    檢查是否為國士無雙
    """
    # 國士無雙要求手牌有全部13種么九牌，其中一種有2張
    yaochuuhai = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]  # 所有么九牌的ID
    
    # 檢查么九牌出現的數量
    yaochuu_counts = counts[yaochuuhai]
    
    # 所有么九牌必須至少出現一次，其中一種出現兩次
    return np.all(yaochuu_counts >= 1) and np.sum(yaochuu_counts) == 14

def count_red_fives(hand_ids):
    """
    計算手牌中赤寶牌的數量
    """
    count = 0
    for tile_id in hand_ids:
        if is_red_five(tile_id):
            count += 1
    return count

def is_mentsu(counts, tile_id):
    """
    檢查給定的牌是否能形成刻子
    """
    return counts[tile_id] >= 3

def is_terminal(tile_id):
    """
    檢查是否為么九牌
    """
    # 將赤寶牌轉換為普通牌
    tile_id = normalize_red_five(tile_id)
    
    suit, value = id_to_tile(tile_id)
    if suit < 3:  # 數牌
        return value == 1 or value == 9
    else:  # 字牌
        return True

def is_honor(tile_id):
    """
    檢查是否為字牌
    """
    # 將赤寶牌轉換為普通牌
    tile_id = normalize_red_five(tile_id)
    return tile_id >= 27

def is_dragon(tile_id):
    """
    檢查是否為三元牌（白發中）
    """
    # 將赤寶牌轉換為普通牌
    tile_id = normalize_red_five(tile_id)
    return tile_id in [31, 32, 33]

def is_wind(tile_id):
    """
    檢查是否為風牌（東南西北）
    """
    # 將赤寶牌轉換為普通牌
    tile_id = normalize_red_five(tile_id)
    return 27 <= tile_id <= 30

def calculate_fu(hand, win_tile, is_open, is_self_draw, has_waiting, is_pinfu=False):
    """
    計算符數
    """
    # 基本符
    fu = 20
    
    # 門前清自摸加符
    if not is_open and is_self_draw:
        fu += 2
    
    # 如果是平和形，且門前清榮和，總共只有30符
    if is_pinfu and not is_open and not is_self_draw:
        return 30
    
    # 自摸非平和加符
    if is_self_draw and not is_pinfu:
        fu += 2
    
    # 七對子固定25符
    normalized_hand = normalize_counts(hand)
    pairs_count = np.sum(normalized_hand[:34] == 2)
    if pairs_count == 7:
        return 25
    
    # 待以後實現更複雜的符數計算
    # 包括雀頭是否為役牌、刻子是否為么九牌、是否為明刻暗刻等計算
    
    # 進位到10符
    fu = ((fu + 9) // 10) * 10
    
    return fu

def calculate_han(hand, win_tile, is_open, is_self_draw, round_wind=0, player_wind=0, dora_count=0, is_riichi=False):
    """
    計算翻數（番數）
    """
    han = 0
    yakus = []
    
    # 先加入寶牌（dora）的翻數
    if dora_count > 0:
        han += dora_count
        yakus.append(f"寶牌 {dora_count}")
    
    # 立直
    if is_riichi:
        han += 1
        yakus.append(YakuType.RIICHI)
    
    # 自摸
    if is_self_draw and not is_open:
        han += 1
        yakus.append(YakuType.TSUMO)
    
    # 檢查赤寶牌
    red_five_count = 0
    for i in range(34, 37):
        if i < len(hand) and hand[i] > 0:
            red_five_count += hand[i]
    
    if red_five_count > 0:
        han += red_five_count
        yakus.append(f"{YakuType.AKADORA} {red_five_count}")
    
    # 待以後實現更多役種判斷
    
    return han, yakus

def analyze_winning_hand(hand_ids, win_tile_id):
    """
    分析和牌結構，返回整理後的牌型
    這裡僅做簡單的排序，未來可以擴展為識別具體的牌型（如順子、刻子等）
    
    參數:
        hand_ids: 手牌ID列表，不包括和牌
        win_tile_id: 和牌ID
    
    返回:
        sorted_tiles: 排序後的全部手牌（包括和牌）
    """
    # 添加和牌到手牌中
    full_hand = hand_ids + [win_tile_id]
    
    # 排序手牌
    sorted_tiles = sort_hand(full_hand)
    
    return sorted_tiles

def calculate_score(hand, win_tile, is_self_draw=False, is_open=False, is_dealer=False, round_wind=0, player_wind=0, dora_count=0, is_riichi=False):
    """
    計算和牌得分
    基於日本麻將（東大式）的計分規則
    
    參數:
        hand: 手牌的計數數組
        win_tile: 和牌的ID
        is_self_draw: 是否自摸
        is_open: 是否副露（開門）
        is_dealer: 是否為莊家
        round_wind: 場風（0=東, 1=南, 2=西, 3=北）
        player_wind: 自風（0=東, 1=南, 2=西, 3=北）
        dora_count: 寶牌數量
        is_riichi: 是否立直
    
    返回:
        score: 分數
        yaku_list: 役列表
    """
    # 計算赤寶牌的數量
    akadora_count = 0
    if len(hand) > 34:
        for i in range(34, min(len(hand), 37)):
            akadora_count += hand[i]
    
    # 將赤寶牌計入對應的普通牌，用於和牌判斷和符數計算
    normalized_hand = normalize_counts(hand)
    
    # 計算符數
    fu = calculate_fu(normalized_hand, win_tile, is_open, is_self_draw, has_waiting=True)
    
    # 計算翻數
    han, yaku_list = calculate_han(hand, win_tile, is_open, is_self_draw, round_wind, player_wind, dora_count, is_riichi)
    
    # 如果沒有役，則無法和牌
    if han == 0 or (han == dora_count + akadora_count and not is_riichi and not is_self_draw):
        return 0, []
    
    # 計算基本點數
    if han >= 13:  # 役滿
        base_points = 8000
    elif han >= 11:  # 三倍滿
        base_points = 6000
    elif han >= 8:  # 倍滿
        base_points = 4000
    elif han >= 6:  # 跳滿
        base_points = 3000
    elif han >= 5:  # 滿貫
        base_points = 2000
    else:
        base_points = fu * (2 ** (han + 2))
        # 限制上限
        if base_points > 2000:
            base_points = 2000
    
    # 根據自摸/榮和和是否為莊家計算最終得分
    if is_self_draw:
        if is_dealer:  # 莊家自摸
            score = base_points * 6  # 閒家各支付2倍基本分
        else:  # 子家自摸
            score = base_points * 4  # 莊家支付2倍基本分，閒家各支付基本分
    else:
        if is_dealer:  # 莊家榮和
            score = base_points * 2  # 放銃者支付2倍基本分
        else:  # 子家榮和
            score = base_points * 1  # 放銃者支付1倍基本分
    
    return score, yaku_list 