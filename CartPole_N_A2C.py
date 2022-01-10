import torch
from torch import nn, optim
import numpy as np
from torch.nn import functional as F
import gym
import matplotlib.pyplot as plt
import time

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        c = F.relu(self.l3(y.detach()))
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

def make_N_exp(exps, gamma):
    reward_exp = torch.Tensor([exp[2] for exp in exps])
    reward = discount(reward_exp, gamma)
    return (exps[0][0], exps[0][1], reward, exps[-1][3], exps[-1][4])

def discount(rewards, gamma):
    disc_return = torch.pow(gamma, torch.arange(len(rewards)).float()) * rewards
    return sum(disc_return)

env = gym.make("CartPole-v1")
model = ActorCritic()
learning_rate = 0.001
opt = optim.Adam(model.parameters(), lr=learning_rate)
epochs = 2500
max_episode = 500
gamma = 0.95
N = 5

start_time = time.time()
temp_replay = []
scores = []
losses = []

for i in range(epochs):
    state = env.reset()
    score = 0

    for j in range(max_episode):
        with torch.no_grad():
            policy, value = model(torch.from_numpy(state))
            action = int(torch.multinomial(torch.exp(policy), num_samples=1))
            state2, reward, done, info = env.step(action)

        if done and j+1 < max_episode:
            reward = -50

        temp_replay.append((state, action, reward, state2, done))
        if len(temp_replay) >= N:
            N_state, N_action, N_reward, N_state, N_done = make_N_exp(temp_replay, gamma=gamma)
            del temp_replay[0]

            opt.zero_grad()
            N_policy, N_value = model(torch.from_numpy(N_state))
            with torch.no_grad():
                next_value = (N_reward + gamma**N * value * (1-done))

            actor_loss = -N_policy[action] * (next_value - N_value.detach())
            cricit_loss = torch.pow(N_value - next_value, 2)
            loss = actor_loss + 0.1 * cricit_loss
            torch.nn.utils.clip_grad_norm(model.parameters(), 1)
            loss.backward()
            opt.step()
            losses.append(loss)

        if done or j+1 >= max_episode:
            scores.append(score)
            break
        state = state2
        score += 1

    if (i+1) % 25 == 0:
        end_time = time.time()
        print("{}/{} score: {}, time: {:.4f}".format(i+1, epochs, sum(scores[-10:])/10, end_time - start_time))
        start_time = end_time

plt.plot(running_mean(np.array(scores)), label="score", color="g")
plt.show()