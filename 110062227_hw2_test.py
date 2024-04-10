from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

state_deque = deque(maxlen=4)
for _ in range(4):
    state_deque.append(torch.zeros(64, 64))

def process_observation(observation):
    # 將numpy數組轉換為torch張量
    observation = np.array(observation).copy()
    observation = torch.tensor(observation, dtype=torch.float32)
    observation = observation.permute(2, 0, 1)
    
    # 定義轉換操作
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 將張量轉換為PIL圖像
        transforms.Grayscale(),   # 轉換為灰階
        transforms.Resize((64, 64)),  # 調整大小
        transforms.ToTensor(),    # 轉回張量
    ])
    
    # 應用轉換
    observation = transform(observation)
    
    # 增加一個批次大小維度和通道維度，使其形狀變為(1, 1, 256, 64)
    observation = observation.squeeze(0)
    #print(f'observation shape2: {observation.shape}')
    
    return observation #(64,64)

def update_state(observation):
    # 處理觀察值，將其轉換為模型需要的形狀
    processed_observation = process_observation(observation)
    
    # 更新deque
    state_deque.append(processed_observation)
    
    # 將deque中的狀態連接起來
    state = state = torch.cat(list(state_deque), dim=0)
    state = state.unsqueeze(0).unsqueeze(0)
    
    return state

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
      samples = random.sample(self.buffer, batch_size)
      states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))

      return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DuelingDQN(nn.Module):
    def __init__(self, input_channels, action_size=12):
        super(DuelingDQN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels,16, kernel_size=4, stride=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)

        self.flatten = nn.Flatten()

        # Adjust the size for the fully connected layer according to the new conv output
        self.fc_val = nn.Linear(2560, 512)  # Updated size
        self.fc_adv = nn.Linear(2560, 512)
        self.fc_val1 = nn.Linear(512, 32)  # Updated size
        self.fc_adv1 = nn.Linear(512, 32)

        self.value_stream = nn.Linear(32, 1)
        self.advantage_stream = nn.Linear(32, action_size)

    def forward(self, x):
        #cannot use cuda
        
        #if not torch.is_tensor(x):
        #  x = torch.FloatTensor(x)
        if len(x.shape) == 3:
          x = x.unsqueeze(1)
        #x = x.cuda()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.flatten(x)

        val = F.relu(self.fc_val(x))
        adv = F.relu(self.fc_adv(x))
        val = F.relu(self.fc_val1(val))
        adv = F.relu(self.fc_adv1(adv))

        value = self.value_stream(val)
        advantage = self.advantage_stream(adv)
        advAvg = torch.mean(advantage,dim=0, keepdim=True)

        # Combine value and advantage to get Q-values
        q_values = value + advantage - advAvg

        return q_values

class Agent():
    def __init__(self):
        self.state_shape = (240, 256, 3)  # Channels, Height, Width
        self.action_size = 12  # Define based on your environment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DuelingDQN(input_channels=1,action_size=self.action_size).to(self.device)
        self.model.load_state_dict(torch.load("mario_model_850.pth", map_location=torch.device('cpu')))
        #self.model.eval() 
        #self.target_model = DuelingDQN(input_channels=1,action_size=self.action_size).to(self.device)
        #self.target_model.load_state_dict(self.model.state_dict())
        #self.target_model.eval()
        self.loss_func = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters())
        self.memory_size = 20000
        self.memory = ReplayBuffer(self.memory_size)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        self.update_target_every = 10  # Target network update frequency
        self.learn_step_counter = 0
        self.update_rate = 10
        self.cur_state = []

    def act(self, observation):
        state = observation
        #print(f'Observation shape: {state.shape}')
        if np.random.rand() <= 0.5:
            return random.randrange(1,12)
        #print("Use model to choose action")
        state = update_state(observation)
        with torch.no_grad():
            #print("Feed to model success")
            Q = self.model.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return 3
    
