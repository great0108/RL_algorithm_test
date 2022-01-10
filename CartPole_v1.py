import gym
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from collections import deque
import time
import copy

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 2)

    def forward(self, x):
        x = torch.nn.functional.elu(self.linear1(x))
        x = torch.nn.functional.elu(self.linear2(x))
        x = self.linear3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 50)
        self.l2 = nn.Linear(50, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = torch.nn.functional.elu(self.l1(x))
        x = torch.nn.functional.elu(self.l2(x))
        actor = torch.nn.functional.log_softmax(self.actor_lin1(x), dim=1)
        c = torch.nn.functional.elu(self.l3(x.detach()))
        critic = self.critic_lin1(c)
        return actor, critic

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

def epsilon_greedy_policy(qvalues, eps):
    if torch.rand(1) < eps:
        return np.random.randint(low=0, high=1)
    else:
        return int(torch.argmax(qvalues))

def softmax_policy(qvalues):
    return int(torch.multinomial(qvalues, num_samples=1))

def make_exp(replay, N, gamma, step):
    exps = []
    reward_batch = torch.Tensor([r for (s, a, r, s2, d) in replay])
    for i in range(0, len(replay), step):
        end_index = min(i+N, len(replay)-1)
        reward = reward_batch[i:end_index]
        reward = discount(reward, gamma=gamma)
        exps.append((replay[i][0], replay[i][1], reward, replay[end_index][3], replay[end_index][4]))

    return exps

def discount(rewards, gamma):
    disc_return = torch.pow(gamma, torch.arange(len(rewards)).float()) * rewards
    return sum(disc_return)

def get_minibatch(replay, size):
    batch_ids = np.random.randint(0, len(replay), size)
    batch = [replay[x] for x in batch_ids]
    state_batch = torch.Tensor([s for (s, a, r, s2, d) in batch]).cuda()
    action_batch = torch.Tensor([a for (s, a, r, s2, d) in batch]).long().cuda()
    reward_batch = torch.Tensor([r for (s, a, r, s2, d) in batch]).cuda()
    state2_batch = torch.Tensor([s2 for (s, a, r, s2, d) in batch]).cuda()
    done_batch = torch.Tensor([d for (s, a, r, s2, d) in batch]).cuda()
    return [state_batch, action_batch, reward_batch, state2_batch, done_batch]

MSE_loss = nn.MSELoss(reduction='sum')

def dqn_step(model, env, state, eps):
    pred = model(torch.from_numpy(state).float().cuda())
    action = epsilon_greedy_policy(pred, eps)
    state2, reward, done, info = env.step(action)
    return action, state2, reward, done, info

def dqn_loss(minibatch, model, N, eps, target_network, Tnet):
    state_batch, action_batch, reward_batch, state2_batch, done_batch = minibatch
    pred = model(state_batch.cuda())
    with torch.no_grad():
        if target_network:
            pred2 = Tnet(state2_batch.cuda())
        else:
            pred2 = model(state2_batch.cuda())

    actions = action_batch.unsqueeze(dim=1)
    state2_value = torch.mean(pred2, dim=1)*(0.2+eps*2) + torch.max(pred2, dim=1)[0]*(0.8-eps*2)
    Y = reward_batch + gamma**N * state2_value * (1 - done_batch)
    X = pred.gather(dim=1, index=actions).squeeze()
    return MSE_loss(X, Y.detach())

def a2c_step(model, env, state):
    policy, value = model(torch.from_numpy(state).unsqueeze(dim=0).float().cuda())
    policy, value = policy[0], value[0]
    action = softmax_policy(policy)
    state2, reward, done, info = env.step(action)
    return action, state2, reward, done, info

def a2c_loss(minibatch, model, N, clc, target_network, Tnet):
    state_batch, action_batch, reward_batch, state2_batch, done_batch = minibatch
    policy, value = model(state_batch)
    with torch.no_grad():
        if target_network:
            policy2, value2 = Tnet(state2_batch)
        else:
            policy2, value2 = model(state2_batch)

    actions = action_batch.unsqueeze(dim=1)
    action_policy = policy.gather(dim=1, index=actions).squeeze()
    N_value = (reward_batch + gamma**N * value2 * (1 - done_batch))
    actor_loss = -torch.log(action_policy) * (N_value - value.detach())
    critic_loss = torch.pow(value - N_value, 2)
    return actor_loss.mean() + clc * critic_loss.mean()

env = gym.make("CartPole-v1")

model = ActorCritic().cuda()
learning_rate = 0.001
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 2000
max_episode = 500
replay_size = 1000
batch_size = 50
gamma = 0.95
eps = 0.1
eps_down = 0.09
N = 4
target_network = True
sync_freq = 200
clc = 0.1

replay = deque(maxlen=replay_size)
start_time = time.time()
temp_replay = []
scores = []
losses = []
Tnet = copy.deepcopy(model)
Tnet.load_state_dict(model.state_dict())
sync_pass = 0

for i in range(epochs):
    state = env.reset()
    score = 0
    eps -= eps_down / epochs

    for j in range(max_episode):
        action, state2, reward, done, info = a2c_step(model, env, state=state)
        if done and j+1 < max_episode:
            reward = -20

        temp_replay.append((state, action, reward, state2, done))

        if done or j+1 >= max_episode:
            scores.append(score)
            break
        state = state2
        score += 1

        exp = make_exp(temp_replay, N=N, gamma=gamma, step=1)
        replay.extend(exp)
        temp_replay = []

        if len(replay) > batch_size:
            opt.zero_grad()
            minibatch = get_minibatch(replay, batch_size)
            loss = a2c_loss(minibatch, model=model, N=N, clc=clc, target_network=target_network, Tnet=Tnet)
            loss.backward()
            opt.step()
            losses.append(loss)

        if target_network:
            sync_pass += 1
            if sync_pass >= sync_freq:
                Tnet.load_state_dict(model.state_dict())
                sync_pass = 0

    if i % 10 == 0 and len(replay) > batch_size + 1:
        end_time = time.time()
        print("{}/{} loss: {:.4f} score: {}, time: {:.4f}".format(i, epochs, loss, scores[-1], end_time - start_time))
        start_time = end_time

plt.plot(running_mean(np.array(scores), 10), label="score", color="g")
plt.show()
plt.plot(losses, label="loss", color="r")
plt.ylim([0, 100])
plt.show()