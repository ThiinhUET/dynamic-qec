import math
import random
from collections import deque
import torch
from collections import namedtuple
import pickle

action = namedtuple('action', ['m', 'n', 'frames'])
mn_step = 4 #What does this for ?
decay_time = 800 #What is it unit ?
max_q_len = 100
LOSS_PENALTY = 0
IDLE_REWARD = 0.5
SUCC_REWARD = 2
PHOTONS_PER_SLOT = 1000
BIG_SLOT = 10


#  Photon interval = 5 us
#  Slot length = 5000 us = 5 ms
#  Big Slot = 200 Small Slots
#  Big Slot = 5 ms * 200 = 1 s


class ChannelENV:
    def __init__(self, L0, n_hop, m_range, n_range, n_photons, income, device):
        self.m_range = m_range
        self.n_range = n_range
        self.n_photons = n_photons
        self.device = device
        self.avg_incoming_frames = income
        self.action_space = [action(0, 0, 0)]
        self.generate_actions()
        self.L0 = L0
        self.queue_time = deque(maxlen=max_q_len)
        self.max_frames_per_slot = n_photons // (m_range[0] * n_range[0])
        self.Nhop = n_hop
        self.p_loss = [0.0, 0.05]
        # [0.0, 0.05]
        # [0.0, 0.1]
        # [0.0, 0.15000000000000002]
        # [0, 0.2]
        # [0.004, 0.008, 0.012, 0.016, 0.02]
        # [0.003, 0.006, 0.009000000000000001, 0.012, 0.015]
        # [0.002, 0.004, 0.006, 0.008, 0.01]
        # [0.001, 0.002, 0.003, 0.004, 0.005]
        self.p_depo = [0.004, 0.008, 0.012, 0.016, 0.02]
        self.p_losstrans = [0.8, 0.2]  # the chance of getting back to current state
        self.p_depotrans = [0.6, 0.5, 0.4, 0.3, 0.2]
        self.ch_state = [0, 0]  # first entry for loss state, and the second for depolarization state
        fname = ("STEP" + str(mn_step) + "LOSS" + str(self.p_loss) + "DEPO" +
                 str(self.p_depo) + "L0" + str(L0) + "NHOP" + str(n_hop) + ".dict")
        with open(fname, "rb") as f:
            self.qber_dict = pickle.load(f)
        self.slot_count = 0

    def loss_transition(self, rand_num):
        if rand_num < self.p_losstrans[self.ch_state[0]]:
            self.ch_state[0] = self.ch_state[0]  # do nothing if transit back
        else:  # transit to neighbor loss states (boundary check)
            if self.ch_state[0] == 0:
                self.ch_state[0] = 1
            elif self.ch_state[0] == len(self.p_loss) - 1:
                self.ch_state[0] = len(self.p_loss) - 2
            else:
                self.ch_state[0] = self.ch_state[0] - 1 if random.uniform(0, 1) > 0.5 else self.ch_state[0] + 1

    def depo_transition(self, rand_num):
        if rand_num < self.p_depotrans[self.ch_state[1]]:  # do nothing if transit back
            self.ch_state[1] = self.ch_state[1]
        else:  # transit to neighbor depolarization states (boundary check)
            if self.ch_state[1] == 0:
                self.ch_state[1] = 1
            elif self.ch_state[1] == len(self.p_depo) - 1:
                self.ch_state[1] = len(self.p_depo) - 2
            else:
                self.ch_state[1] = self.ch_state[1] - 1 if random.uniform(0, 1) > 0.5 else self.ch_state[1] + 1

    def next_state(self):
        depo = random.uniform(0, 1)
        loss = random.uniform(0, 1)
        if self.slot_count == 0:
            self.loss_transition(loss)
            self.depo_transition(depo)
        else:
            self.depo_transition(depo)
        self.slot_count = (self.slot_count + 1) % BIG_SLOT

    # todo: threshold reward (implemented)
    def feedback(self, m: int, n: int, s: int):
        if s == 0:
            # Reward is 0.5 if nothing to send, and the feedback will be an all -1 list
            reward = IDLE_REWARD
            feedback = [-1 for _ in range(self.max_frames_per_slot)]
        else:
            n_frames_to_be_send = s if s < len(self.queue_time) else len(self.queue_time)
            tmp = self.qber_dict[(m, n, self.p_loss[self.ch_state[0]], self.p_depo[self.ch_state[1]])]
            # list of frames loss for each frame, 0 for loss and 1 otherwise
            feedback = [random.uniform(0, 1) > tmp[1] for _ in range(n_frames_to_be_send)]
            # reward function decays exponentially as waiting time grows, and has a flat penalty for each lost frame
            reward = 0
            for i in range(n_frames_to_be_send):
                t = self.queue_time.popleft()
                if feedback[i] == 1 and t <= decay_time:
                    reward += tmp[0]/(1-tmp[1]) * SUCC_REWARD
                else:
                    reward -= LOSS_PENALTY
            # pad -1 if number of sent frames is less than max capacity
            feedback.extend([-1 for _ in range(self.max_frames_per_slot - n_frames_to_be_send)])

        # feedback consists of the list of frame lost information and queuing time information
        feedback.extend(self.queue_time)
        # pads -1 if the number of elements in queue less than max_q_len
        feedback.extend([-1 for _ in range(max_q_len - len(self.queue_time))])
        self.next_state()
        # feedback structure: [loss/receive info, queue_info]
        return reward, feedback

    def reset(self):
        self.ch_state[0] = 0
        self.ch_state[1] = 0
        self.slot_count = 0
        ret = [0 for _ in range(len(self.action_space))]
        ret[0] = 1  # default action for the beginning
        ret.extend([0 for _ in range(self.max_frames_per_slot)])  # default feedback from last slot
        ret.extend([-1 for _ in range(max_q_len)])  # default queue status
        return ret

    # todo: buffer overflow penalty when new frames arrive (implemented)
    def step(self, act_index: int):
        for i in range(len(self.queue_time)):
            i += 1
        # calculate penalty when overflow
        if self.avg_incoming_frames + len(self.queue_time) > max_q_len:
            penalty = LOSS_PENALTY * (self.avg_incoming_frames + len(self.queue_time) - max_q_len)
        else:
            penalty = 0
        for _ in range(self.avg_incoming_frames):
            self.queue_time.append(0)
        act = self.action_space[act_index]  # action_space is a list of tuples recording m, n, and frames
        rwd, fdbk = self.feedback(act.m, act.n, act.frames)
        fdbk = torch.tensor(fdbk, device=self.device)
        act_onehot = torch.zeros(len(self.action_space), device=self.device)
        act_onehot[act_index] = 1
        # fdbk contains the decision made by agent, the list of frames lost/received, the list of queuing time
        # frames lost/received and queening time are returned by self.feedback()
        # structure: [action info, loss/reception info, queue info]
        fdbk = torch.concatenate((act_onehot, fdbk))
        return fdbk, rwd - penalty

    def generate_actions(self):
        code_params = [(m, n) for m in range(self.m_range[0], self.m_range[1], mn_step)
                       for n in range(self.n_range[0], self.n_range[1], mn_step)]
        for m, n in code_params:
            possible_frames = self.n_photons // (m * n)
            possible_frames = possible_frames if possible_frames < self.avg_incoming_frames + max_q_len else (self.avg_incoming_frames + max_q_len)
            self.action_space.extend([action(m, n, frames) for frames in range(1, possible_frames + 1)])

    def sample_action(self):
        return random.randint(0, len(self.action_space) - 1)
