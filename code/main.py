import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from collections import deque, namedtuple
import pickle
import logging

# Define constants and hyperparameters
BATCH_SIZE = 1024
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200000
TAU = 0.00005
LR = 1e-4
NUM_EPISODES = 256
LEN_EPISODES = 4096

ACTION = namedtuple('action', ['m', 'n', 'frames'])
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
MN_STEP = 4
DECAY_TIME = 800
MAX_Q_LEN = 100
LOSS_PENALTY = 0
IDLE_REWARD = 0.5
SUCC_REWARD = 2
PHOTONS_PER_SLOT = 1000
BIG_SLOT = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logging
logging.basicConfig(filename="train.log", filemode="w")

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 512)
        self.layer4 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

# Define the Channel Environment
class ChannelENV:
    def __init__(self, L0, n_hop, m_range, n_range, n_photons, income):
        self.m_range = m_range
        self.n_range = n_range
        self.n_photons = n_photons
        self.device = device
        self.avg_incoming_frames = income
        self.action_space = [ACTION(0, 0, 0)]
        self.generate_actions()
        self.L0 = L0
        self.queue_time = deque(maxlen=MAX_Q_LEN)
        self.max_frames_per_slot = n_photons // (m_range[0] * n_range[0])
        self.Nhop = n_hop
        self.p_loss = [0.0, 0.05]
        self.p_depo = [0.004, 0.008, 0.012, 0.016, 0.02]
        self.p_losstrans = [0.8, 0.2]
        self.p_depotrans = [0.6, 0.5, 0.4, 0.3, 0.2]
        self.ch_state = [0, 0]
        self.slot_count = 0
        self.qber_dict = self.load_qber_dict()

    def load_qber_dict(self):
        fname = f"STEP{MN_STEP}LOSS{self.p_loss}DEPO{self.p_depo}L0{self.L0}NHOP{self.Nhop}.dict"
        with open(fname, "rb") as f:
            return pickle.load(f)

    def loss_transition(self, rand_num):
        if rand_num >= self.p_losstrans[self.ch_state[0]]:
            if self.ch_state[0] == 0:
                self.ch_state[0] = 1
            elif self.ch_state[0] == len(self.p_loss) - 1:
                self.ch_state[0] = len(self.p_loss) - 2
            else:
                self.ch_state[0] = self.ch_state[0] - 1 if random.uniform(0, 1) > 0.5 else self.ch_state[0] + 1

    def depo_transition(self, rand_num):
        if rand_num >= self.p_depotrans[self.ch_state[1]]:
            if self.ch_state[1] == 0:
                self.ch_state[1] = 1
            elif self.ch_state[1] == len(self.p_depo) - 1:
                self.ch_state[1] = len(self.p_depo) - 2
            else:
                self.ch_state[1] = self.ch_state[1] - 1 if random.uniform(0, 1) > 0.5 else self.ch_state[1] + 1

    def next_state(self):
        self.loss_transition(random.uniform(0, 1))
        self.depo_transition(random.uniform(0, 1))
        self.slot_count = (self.slot_count + 1) % BIG_SLOT

    def feedback(self, m: int, n: int, s: int):
        if s == 0:
            reward = IDLE_REWARD
            feedback = [-1] * self.max_frames_per_slot
        else:
            n_frames_to_be_send = min(s, len(self.queue_time))
            tmp = self.qber_dict[(m, n, self.p_loss[self.ch_state[0]], self.p_depo[self.ch_state[1]])]
            feedback = [random.uniform(0, 1) > tmp[1] for _ in range(n_frames_to_be_send)]
            reward = sum(tmp[0]/(1-tmp[1]) * SUCC_REWARD if fb == 1 and t <= DECAY_TIME else -LOSS_PENALTY 
                         for fb, t in zip(feedback, self.queue_time))
            feedback.extend([-1] * (self.max_frames_per_slot - n_frames_to_be_send))

        feedback.extend(self.queue_time)
        feedback.extend([-1] * (MAX_Q_LEN - len(self.queue_time)))
        self.next_state()
        return reward, feedback

    def reset(self):
        self.ch_state = [0, 0]
        self.slot_count = 0
        ret = [1] + [0] * (self.max_frames_per_slot + MAX_Q_LEN)
        return ret

    def step(self, act_index: int):
        for _ in range(self.avg_incoming_frames):
            self.queue_time.append(0)
        penalty = LOSS_PENALTY * max(0, self.avg_incoming_frames + len(self.queue_time) - MAX_Q_LEN)
        act = self.action_space[act_index]
        rwd, fdbk = self.feedback(act.m, act.n, act.frames)
        fdbk = torch.tensor(fdbk, device=self.device)
        act_onehot = torch.zeros(len(self.action_space), device=self.device)
        act_onehot[act_index] = 1
        fdbk = torch.cat((act_onehot, fdbk))
        return fdbk, rwd - penalty

    def generate_actions(self):
        code_params = [(m, n) for m in range(self.m_range[0], self.m_range[1], MN_STEP)
                       for n in range(self.n_range[0], self.n_range[1], MN_STEP)]
        for m, n in code_params:
            possible_frames = min(self.n_photons // (m * n), self.avg_incoming_frames + MAX_Q_LEN)
            self.action_space.extend([ACTION(m, n, frames) for frames in range(1, possible_frames + 1)])

    def sample_action(self):
        return random.randint(0, len(self.action_space) - 1)

# Define ReplayMemory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Training class
class Train:
    def __init__(self):
        self.steps_done = 0
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(200000)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[environment.sample_action()]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        print(f"Loss: {loss.item()}")
        logging.info(f"Loss: {loss.item()}")

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        for i_episode in range(NUM_EPISODES):
            state = environment.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            print(f"Start of Episode: {i_episode}")
            logging.info(f"Start of Episode: {i_episode}")
            for t in range(LEN_EPISODES):
                action = self.select_action(state)
                observation, reward = environment.step(action.item())
                print(f"Reward: {reward}")
                logging.info(f"Reward: {reward}")
                reward = torch.tensor([reward], device=device)
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                self.target_net.load_state_dict(target_net_state_dict)

        torch.save(self.target_net.state_dict(), "tgt_net.pth")
        torch.save(self.policy_net.state_dict(), "pol_net.pth")
        print('Complete')

if __name__ == "__main__":
    n_observations = len(ChannelENV(1, 1, (1, 1), (1, 1), PHOTONS_PER_SLOT, 1).reset())
    n_actions = len(ChannelENV(1, 1, (1, 1), (1, 1), PHOTONS_PER_SLOT, 1).action_space)
    environment = ChannelENV(1, 1, (1, 1), (1, 1), PHOTONS_PER_SLOT, 1)
    trainer = Train()
    trainer.train()
