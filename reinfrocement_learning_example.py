"""
Example of reinforcement learning with NASBench-101.
The agent is based on the DDQN implementation by PyTorch
https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py



"""

from collections import deque, namedtuple
from copy import deepcopy
from itertools import count
import math
from multiprocessing import connection
import random
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

import torch
from nord.neural_nets import NASBench_101Evaluator, NeuralDescriptor
import torch.nn as nn
import torch.nn.functional as F

import time

import seaborn as sns
from nord.utils import assure_reproducibility


# Plotting parameters
sns.set_style("white")
sns.set_style("ticks")
sns.set_palette('colorblind')
font = {'family': 'normal',
        'weight': 'normal',
        'size': 16}

matplotlib.rc('font', **font)


# Instantiate the evaluator
evaluator = NASBench_101Evaluator()

assure_reproducibility()
# DDQN CODE

# Transition tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Replay memory with capacity


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


device = 'cuda:0'
# Policy and target network architecture


class DQN(nn.Module):

    # Inputs are 21 (upper triangular of 7x7 with 7th row empty) +
    # 6x4 (number of layers, one-hot encoded) = 45
    def __init__(self, outputs):
        super(DQN, self).__init__()
        input_sz = 45
        self.fc1 = nn.Linear(input_sz, input_sz*3)
        self.fc2 = nn.Linear(input_sz*3, input_sz*6)
        self.fc3 = nn.Linear(input_sz*6, 64)
        self.head = nn.Linear(64, outputs)

        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.bn3 = nn.BatchNorm1d(1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.float()
        x = x.to(device)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.head(x.view(x.size(0), -1))


# Training parameters
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# +1 as we can stop before maximum number of layers
n_actions = len(evaluator.get_available_ops())+1

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
# Target net does not train, only gets weights from policy net
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Our environment
class NASEnv:

    def __init__(self):
        self.avail_layers = evaluator.get_available_ops()
        self.layer_codes = {}

        for i in range(len(self.avail_layers)):
            code = np.zeros(4)
            code[i] = 1.0
            self.layer_codes[self.avail_layers[i]] = code

        self.layer_codes['input'] = np.array([0, 0, 0, 1])
        self.descriptor = NeuralDescriptor()
        self.descriptor.add_layer_sequential('input', {})

    def reset(self):

        self.descriptor = NeuralDescriptor()
        self.descriptor.add_layer_sequential('input', {})

    def get_state(self):

        tmp_desc = deepcopy(self.descriptor)
        tmp_desc.add_layer_sequential('output', {}, 'tmp_out')
        matrix, ops = evaluator.descriptor_to_matrix(tmp_desc)

        layers = np.zeros(6*4)
        # Don't need output layer
        for i in range(len(ops)-1):
            op = ops[i]
            code = np.array(self.layer_codes[op])
            layers[i*4:(i+1)*4] = code

        connections = np.zeros((7, 7))

        matrix = np.array(matrix)
        # Copy all but last column
        connections[0:matrix.shape[0], 0:matrix.shape[1]-1] = matrix[:, :-1]

        # Copy last column to last column of connections
        connections[0:matrix.shape[0], -1] = matrix[:, -1]

        inds_x, inds_y = np.triu_indices(6)
        # Move columns to the right as first column is zeros
        inds_y += 1
        connections = connections[(inds_x, inds_y)]

        return torch.tensor(np.concatenate([layers.reshape(1, -1), connections.reshape(1, -1)], axis=1)).unsqueeze(1)

    def step(self, action):

        reward = 0.0
        done = False
        # If terminal state or the agent chooses to stop
        if action >= len(self.avail_layers) or len(self.descriptor.layers) == 6:
            self.descriptor.add_layer_sequential('output', {})
            reward, time = evaluator.descriptor_evaluate(self.descriptor)
            done = True
        else:
            self.descriptor.add_layer_sequential(self.avail_layers[action], {})

        return reward, done


num_episodes = 500
# Get the available layers
env = NASEnv()
accs = []
start_t = time.time()
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()

    state = env.get_state()
    for t in count():
        # Select and perform an action
        action = select_action(state)
        reward, done = env.step(action.item())
        cpu_reward = reward
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = env.get_state()
        else:
            next_state = None
            accs.append(cpu_reward)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Total search time:', time.time() - start_t)
plt.plot(accs, label='Reinforcement Learning 34.1s')
plt.ylim((0.6, 0.95))
