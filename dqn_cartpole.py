import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--plot', type=bool, default=True)
parser.add_argument('--model', type=str, default='reinforce_cartpole/model.pt')
parser.add_argument('--runtype', type=str, default='train_run',
                    choices=('train', 'run', 'train_run'))
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=500)
parser.add_argument('--gamma', type=float, default=0.99)
args = parser.parse_args()

env = gym.make('CartPole-v1').unwrapped
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, in_size, out_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(in_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, out_size)
        self.dropout = nn.Dropout(0.7)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(F.relu(self.layer2(x)))
        return self.layer3(x)

class Runner():
    def __init__(self, dqn, loss, lr=0.001, eps_start=0.9, eps_end=0.1, eps_decay=200,
                 batch_size=128, target_update=40, logs="runs", gamma=0.999):
        self.writer = SummaryWriter(logs)
        self.logs = logs
        self.learner = dqn
        self.target = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
        self.target.load_state_dict(self.learner.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.learner.parameters(), lr=lr)
        self.loss = loss
        self.memory = ExperienceReplay(100000)
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.gamma = gamma
        self.steps = 0
        self.plots = {"Loss": [], "Reward": [], "Mean Reward": []}

    def select_action(self, state):
        self.steps += 1
        sample = random.random()
        eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps / self.eps_decay)
        if sample > eps_thresh:
            with torch.no_grad():
                return torch.argmax(self.learner(state)).view(1, 1)
        else:
            return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

    def train_inner(self):
        if len(self.memory) < self.batch_size:
            return 0
        sample_transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*sample_transitions))
        non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        pred_values = self.learner(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
        expected_values = (next_state_values * self.gamma) + reward_batch
        loss = self.loss(pred_values, expected_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.learner.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.detach()

    def env_step(self, action):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        return torch.FloatTensor(next_state).unsqueeze(0).to(device), torch.FloatTensor([reward]).to(device), done

    def train(self, episodes=100, smooth=10):
        smoothed_reward = []
        for episode in range(episodes):
            total_reward = 0
            state, _ = env.reset()
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            for t in range(500):
                action = self.select_action(state)
                next_state, reward, done = self.env_step(action.item())
                if done:
                    next_state = None
                self.memory.push(state, action, next_state, reward)
                state = next_state
                loss = self.train_inner()
                total_reward += reward.item()
                if done:
                    break
            smoothed_reward.append(total_reward)
            if len(smoothed_reward) > smooth:
                smoothed_reward = smoothed_reward[-smooth:]
            if isinstance(loss, torch.Tensor):
                loss_scalar = loss.detach().cpu().item()
            else:
                loss_scalar = float(loss)
            self.writer.add_scalar("Loss", loss_scalar, episode)
            self.writer.add_scalar("Reward", total_reward, episode)
            self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), episode)
            self.plots["Loss"].append(loss_scalar)
            self.plots["Reward"].append(total_reward)
            self.plots["Mean Reward"].append(np.mean(smoothed_reward))
            if episode % 20 == 0:
                print(f"\tEpisode {episode}\tFinal Reward: {total_reward:.2f}\tMean: {np.mean(smoothed_reward):.2f}")
            if episode % self.target_update == 0:
                self.target.load_state_dict(self.learner.state_dict())
        env.close()

    def plot(self):
        sns.set()
        sns.set_context("talk")  # smaller context
        plt.figure(figsize=(10, 6))
        plt.plot(self.plots["Loss"])
        plt.title("DQN Loss")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        loss_path = f"{self.logs}/plot_loss.png"
        plt.savefig(loss_path)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.plots["Reward"], label="Reward")
        plt.plot(self.plots["Mean Reward"], label="Mean Reward")
        plt.legend()
        plt.title("DQN Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        reward_path = f"{self.logs}/plot_rewards.png"
        plt.savefig(reward_path)
        plt.close()

        print("[Plot] Plots saved to:")
        print(f"  {loss_path}\n  {reward_path}")

    def save(self):
        torch.save(self.learner.state_dict(), f'{self.logs}/model.pt')

    def run(self):
        for ep in range(5):
            state, _ = env.reset()
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            total_reward = 0
            for t in range(500):
                with torch.no_grad():
                    action = torch.argmax(self.learner(state)).view(1,1)
                next_state, reward, done = self.env_step(action.item())
                state = next_state
                total_reward += reward.item()
                if done:
                    break
            print(f"[Run] Episode {ep} Total Reward: {total_reward:.1f}")
        env.close()

def main():
    device_name = f"cuda: {args.device}" if torch.cuda.is_available() else "cpu"
    print("[Device]\tDevice selected: ", device_name)
    dqn = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    if args.load:
        dqn.load_state_dict(torch.load(args.model))
    runner = Runner(dqn, nn.MSELoss(), lr=args.lr, gamma=args.gamma, logs=f"dqn_cartpole/{time.time()}")
    if "train" in args.runtype:
        print("[Train]\tTraining Beginning ...")
        runner.train(args.episodes)
        if args.plot:
            print("[Plot]\tPlotting Training Curves ...")
            runner.plot()
    if args.save:
        print("[Save]\tSaving Model ...")
        runner.save()
    if "run" in args.runtype:
        print("[Run]\tRunning Simulation ...")
        runner.run()
    print("[End]\tDone. Congratulations!")

if __name__ == '__main__':
    main()
