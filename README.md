# 麻將強化學習環境 (Mahjong RL Environment)

這是一個基於 Python 的麻將強化學習環境，專為訓練和測試麻將 AI 代理而設計。

## 功能特點

### 1. 完整的麻將遊戲邏輯
- 支持所有基本麻將規則
- 包含特殊牌型（赤寶牌）
- 支持所有役種判定
- 完整的計分系統

### 2. 詳細的日誌記錄系統
- 遊戲基本信息記錄
  - 回合數
  - 遊戲時間
  - 總獎勵
  - 遊戲結果

- 牌局信息記錄
  - 完整牌山（使用表情符號顯示）
  - 每位玩家的初始手牌
  - 牌河記錄

- 和牌信息記錄
  - 和牌者
  - 和牌牌型
  - 和了牌
  - 役種詳情（包含每個役種的翻數）
  - 總翻數
  - 滿貫等級
  - 最終得分

- 遊戲進程記錄
  - AI玩家的手牌變化
  - 摸牌和打牌記錄
  - 其他動作（吃、碰、槓、立直等）
  - 獎勵獲得情況

- 計分信息記錄
  - 總獎勵
  - 每位玩家的最終得分

### 3. 多種記錄格式
- JSON 格式：包含完整的結構化數據
- 文本格式：易讀的遊戲記錄
- 控制台輸出：即時遊戲進程顯示

### 4. 靈活的配置選項
- 可配置的日誌級別
- 可選擇的記錄內容
- 自定義的記錄格式

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from src.environment.mahjong_env import MahjongEnv

# 創建環境
env = MahjongEnv(enable_logging=True, log_level=2)

# 重置環境
observation = env.reset()

# 執行動作
action = your_agent.select_action(observation)
next_observation, reward, done, info = env.step(action)
```

### 查看遊戲記錄

遊戲記錄會自動保存在 `logs` 目錄下，包含以下文件：
- `session_YYYYMMDD_HHMMSS/`：每個遊戲會話的目錄
  - `episode_X.json`：結構化的遊戲數據
  - `episode_X_text.log`：易讀的文本記錄

### 記錄格式示例

```text
第 2 輪遊戲記錄
遊戲時間: 2025-03-20 17:34:25
總獎勵: 8000
遊戲結果: 勝利
--------------------------------------------------

牌山: 🀙🀒🀖🀑🀅🀖🀊🀛🀋🀗🀒🀓🀍🀐🀎🀑🀄🀡🀈🀁
...

玩家 0 初始手牌: 🀇🀌🀐🀒🀕🀘🀚🀛🀜🀠🀀🀁🀂🀅
...

===== 和牌信息 =====
和牌者: 玩家 3
和牌牌型: 🀇🀈🀉🀊🀋🀌🀔🀔🀖🀗🀘🀚🀛🀜
和了牌: 🀔

役種詳情:
  平和 1翻
  立直 1翻
  寶牌 1翻
  赤寶牌 1翻

總翻數: 4翻
滿貫等級: 滿貫
最終得分: 8000點

玩家得分:
玩家 0: 0分
玩家 1: 0分
玩家 2: 0分
玩家 3: 8000分
```

## 開發計劃

- [ ] 添加更多役種支持
- [ ] 優化計分系統
- [ ] 添加更多遊戲模式
- [ ] 改進日誌記錄格式
- [ ] 添加遊戲回放功能

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 授權

MIT License 