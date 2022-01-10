import torch
from torch import nn, optim
import numpy as np
from torch.nn import functional as F
import gym
import matplotlib.pyplot as plt
import time
from collections import deque
import copy
import numpy as np

class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        y = F.elu(self.l1(x))
        y = F.elu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=1)
        c = F.elu(self.l3(y.detach()))
        critic = self.critic_lin1(c)
        return actor, critic

def running_mean(x, N=10):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

def epsilon_greedy_policy(qvalues, eps=0.1):
    if torch.rand(1) < eps:
        return np.random.randint(low=0, high=1)
    else:
        return int(torch.argmax(qvalues))

def softmax_policy(qvalues, eps=0):
    if torch.rand(1) < eps:
        return np.random.randint(low=0, high=1)
    else:
        return int(torch.multinomial(qvalues, num_samples=1))

def discount(rewards, gamma):
    disc_return = torch.pow(gamma, torch.arange(len(rewards)).float()) * rewards
    return sum(disc_return)

def get_minibatch(replay, size):
    batch_ids = np.random.randint(0, len(replay), size)
    batch = [replay[x] for x in batch_ids]
    state_batch = torch.Tensor([s for (s, a, r, s2, d) in batch])
    action_batch = torch.Tensor([a for (s, a, r, s2, d) in batch]).long()
    reward_batch = torch.Tensor([r for (s, a, r, s2, d) in batch])
    state2_batch = torch.Tensor([s2 for (s, a, r, s2, d) in batch])
    done_batch = torch.Tensor([d for (s, a, r, s2, d) in batch])
    return (state_batch, action_batch, reward_batch, state2_batch, done_batch)

def make_N_exp(exps, gamma):
    reward_exp = torch.Tensor([exp[2] for exp in exps])
    reward = discount(reward_exp, gamma)
    return (exps[0][0], exps[0][1], reward, exps[-1][3], exps[-1][4])

def train(model, batch):
    state_batch, action_batch, reward_batch, state2_batch, done_batch = batch
    opt.zero_grad()
    policy, value = model(state_batch)
    with torch.no_grad():
        policy2, value2 = model(state2_batch)
        next_value = (reward_batch + gamma**N * value2 * (1-done_batch))
            
    actions = action_batch.unsqueeze(dim=1)
    action_policy = policy.gather(dim=1, index=actions).squeeze()
    actor_loss = -action_policy * (next_value - value.detach())
    cricit_loss = torch.pow(value - next_value, 2)
    loss = actor_loss.mean() + clc * cricit_loss.mean()

    torch.nn.utils.clip_grad_norm(model.parameters(), 0.2)
    loss.backward()
    opt.step()
    losses.append(actor_loss.mean())
    losses2.append(clc * cricit_loss.mean())

env = gym.make("CartPole-v1")

model = ActorCritic()
learning_rate = 0.0003
opt = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 1000
max_episode = 500
gamma = 0.98
fail_reward = -50
clc = 0.1
N = 5
replay_size = 500
batch_size = 20
eps = 0.2
eps_down = 0.15

replay = deque(maxlen=replay_size)
start_time = time.time()
temp_replay = []
fail_replay = deque(maxlen=replay_size)
scores = []
test_scores = []
losses = [1]
losses2 = [1]

for i in range(epochs):
    state = env.reset()
    score = 1
    eps -= eps_down / epochs

    for j in range(max_episode):
        with torch.no_grad():
            policy, value = model(torch.Tensor([state]))
            policy, value = policy[0], value[0]
            action = softmax_policy(torch.exp(policy))
            state2, reward, done, info = env.step(action)

        if done and j+1 < max_episode:
            reward = fail_reward
        else:
            reward = 1

        temp_replay.append((state, action, reward, state2, done))
        if len(temp_replay) >= N:
            replay.append(make_N_exp(temp_replay, gamma=gamma))
            del temp_replay[0]

        if len(replay) >= batch_size:
            batch = get_minibatch(replay, batch_size)
            train(model, batch)

        if done or j+1 >= max_episode:
            scores.append(score)
            while len(temp_replay) > 0:
                exp = make_N_exp(temp_replay, gamma=gamma)
                replay.append(exp)
                if done:
                    fail_replay.append(exp)

                if len(replay) >= batch_size:
                    batch = get_minibatch(replay, batch_size)
                    train(model, batch)

                del temp_replay[0]
            break

        if len(fail_replay) >= batch_size and (j+1) % 100 == 0:
            batch = get_minibatch(fail_replay, batch_size)
            train(model, batch)
        state = state2
        score += 1

    if (i+1) % 25 == 0:
        end_time = time.time()
        print("{}/{} loss: {:.4f} loss2: {:.4f} score: {}, time: {:.4f}".format(i+1, epochs, losses[-1], losses2[-1], sum(scores[-10:])/10, end_time - start_time))
        start_time = end_time

        test_env = gym.make("CartPole-v1")
        test_score_mean = 0
        for i in range(10):
            state = env.reset()
            test_score = 1

            for j in range(500):
                with torch.no_grad():
                    policy, value = model(torch.Tensor([state]))
                    policy, value = policy[0], value[0]
                    action = softmax_policy(torch.exp(policy))
                    state2, reward, done, info = env.step(action)
                
                if done or j+1 >= 500:
                    test_score_mean += test_score
                    break

                state = state2
                test_score += 1

        test_scores.append(test_score_mean/10)
        print("test_score: {}".format(test_score_mean / 10))



plt.plot(running_mean(np.array(scores)), label="score", color="g")
plt.show()
plt.plot(np.array(test_scores), label="test_score", color="g")
plt.show()


