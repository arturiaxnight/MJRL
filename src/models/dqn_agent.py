import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# 定義神經網絡模型
class MahjongDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MahjongDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# 定義DQN代理
class DQNAgent:
    def __init__(self, state_size, action_size, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # 經驗回放記憶體
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        
        # 明確檢測CUDA可用性
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"使用設備: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU名稱: {torch.cuda.get_device_name(0)}")
            print(f"GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        
        self.model = MahjongDQN(state_size, action_size).to(self.device)
        self.target_model = MahjongDQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.update_target_network()
        
    def update_target_network(self):
        """更新目標網絡參數"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存儲經驗到記憶體"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None, epsilon=None):
        """選擇動作"""
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
            
        # 使用提供的epsilon或者實例的epsilon
        current_epsilon = self.epsilon if epsilon is None else epsilon
            
        if random.random() <= current_epsilon:
            return random.choice(valid_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.model(state_tensor).cpu().detach().numpy()[0]
        
        # 只考慮有效動作
        valid_q_values = {a: q_values[a] for a in valid_actions}
        return max(valid_q_values, key=valid_q_values.get)
    
    def replay(self):
        """從記憶體中隨機抽取批次經驗進行學習"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 計算當前Q值
        curr_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 計算下一個狀態的最大Q值
        next_q_values = self.target_model(next_states).max(1)[0]
        
        # 計算目標Q值
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 計算損失並更新網絡
        loss = F.smooth_l1_loss(curr_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        """載入模型權重"""
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        self.update_target_network()
        
    def save(self, name):
        """保存模型權重"""
        torch.save(self.model.state_dict(), name) 