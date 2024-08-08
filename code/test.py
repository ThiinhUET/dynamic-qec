import random
import env
import DQN
import torch
import logging
from collections import namedtuple

Action = namedtuple('action', ['m', 'n', 'frames'])

logging.basicConfig(filename="test.log", filemode="w")
timeslots = 20000
random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environment = env.ChannelENV(1, 1, [3, 15], [3, 15], 1000, 10, device)
n_actions = len(environment.action_space)
# Get the number of state observations
n_observations = environment.max_frames_per_slot + env.max_q_len + n_actions
policy_net = DQN.DQN(n_observations, n_actions).to(device)
target_net = DQN.DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(torch.load("tgt_net.pth"))
policy_net.load_state_dict(torch.load("pol_net.pth"))

state = environment.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
ave_rate = 0.0
for t in range(timeslots):
    print("Channel Loss:", environment.p_loss[environment.ch_state[0]],
          "Channel Depo.: ", environment.p_depo[environment.ch_state[1]])
    action_idx = policy_net(state).max(1).indices.view(1, 1)
    action = environment.action_space[action_idx.item()]
    queue_prev = len(environment.queue_time)
    # action = Action(7, 11, 10)
    if action.m == 0 or action.n ==0:
        tmp = [0]
    else:
        tmp = environment.qber_dict[(action.m, action.n, environment.p_loss[environment.ch_state[0]], environment.p_depo[environment.ch_state[1]])]
    actual_s = action.frames if action.frames < 10 + queue_prev else 10 + queue_prev
    ave_rate += tmp[0] * actual_s
    print("Action s:", action.frames, "Action m: ", action.m, "Action n: ", action.n)
    print("R: " + str(tmp[0] * actual_s))
    logging.info("R: " + str(tmp[0]))
    # observation, reward = environment.step(0)
    observation, reward = environment.step(action_idx.item())
    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    print("Queue: ", environment.queue_time, "\n")

    # Move to the next state
    state = next_state
print(ave_rate/timeslots)
