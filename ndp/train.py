import torch
import torch.nn as nn
import numpy as np
import itertools
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

def greedy(cost): # for comparison purposes only
    if cost.shape[0] == 0:
        return 0
    if cost.shape[0] == 1:
        return cost[0,0,0]
    immidiate_reward = cost.max()
    indexes = np.unravel_index(cost.argmax(), cost.shape)
    for axis, index in enumerate(indexes):
        cost = np.delete(cost, index, axis=axis)
    return immidiate_reward + greedy(cost)

class ValueNetwork(nn.Module):
    def __init__(self, n, hidden_1_k=4, hidden_2_k=8):
        super().__init__()
        self.n = n
        self.linear_1 = nn.Linear(n ** 3, hidden_1_k * n ** 2)
        self.linear_2 = nn.Linear(hidden_1_k * n ** 2, hidden_2_k * n)
        self.linear_3 = nn.Linear(hidden_2_k * n, 1)

    def forward(self, costs):
        features = costs.flatten(start_dim=1) #preserve batch dimension
        hidden_1 = nn.functional.relu(self.linear_1(features))
        hidden_2 = nn.functional.relu(self.linear_2(hidden_1))
        out = self.linear_3(hidden_2)
        return out.squeeze()

class ClassicalValueEstimator:
    def __init__(self, n):
        self.n = n

    def __call__(self, costs):
        if self.n == 0:
            return torch.zeros_like(costs).squeeze()
        if self.n == 1:
            return costs.squeeze()

class Agent:

    lr = 3e-4
    num_pretrain_epochs = 10
    num_pretrain_iters = 100
    batch_size = 64

    def __init__(self, n=2, hyper_params=None):
        self.n = 2 # Will be updated to user requested n later
        if hyper_params is not None:
            self.__dict__.update(hyper_params)

        self.value_networks = [ClassicalValueEstimator(n=0), ClassicalValueEstimator(n=1)]
        self.losses = []
        for i in range(2, n):
            loss = self.pretrain_new_network()
            self.losses.append(loss)

    def pretrain_new_network(self):
        new_network = ValueNetwork(n=self.n)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(new_network.parameters(), lr=self.lr)
        loss_history = []
        for epoch in trange(self.num_pretrain_epochs, desc=f"Pretraining network for size {self.n}"):
            for iteration in range(self.num_pretrain_iters):
                optimizer.zero_grad()
                costs = torch.rand((self.batch_size, self.n, self.n, self.n))
                pred_values = new_network(costs)
                best_values = torch.tensor(
                    [self.evaluate_position(cost)[0] for cost in costs])
                loss = criterion(pred_values, best_values)
                loss.backward()
                optimizer.step()
                loss_history.append(loss.item())
        np.save(open(f"./logs/losses_{self.n}.npy", "wb"), np.array(loss_history))
        self.value_networks.append(new_network)
        self.n += 1


    @torch.no_grad()
    def act(self, cost):
        reward = 0
        while cost.shape[0] > 1:
            _, best_action = self.evaluate_position(cost)
            j, k = best_action
            j_left = [i for i in range(cost.shape[0]) if i != j]
            k_left = [i for i in range(cost.shape[0]) if i != k]
            reward += cost[0, j, k]
            cost = cost[1:][:, j_left][:, :, k_left] # can't find a better way
        reward += cost.item()
        return reward
    
    @torch.no_grad()
    def evaluate_position(self, cost):
        size = cost.shape[0]
        if size == 1:
            return cost.item(), (0, 0)
        best_pos, best_reward = None, -1
        for j, k in itertools.product(range(size), repeat=2):
            j_left = [i for i in range(size) if i != j]
            k_left = [i for i in range(size) if i != k]
            cost_left = cost[1:][:, j_left][:, :, k_left] # can't find a better way
            reward = cost[0, j, k].item() + self.value_networks[size-1](cost_left.unsqueeze(dim=0)).item()
            if reward > best_reward:
                best_reward = reward
                best_pos = (j, k)
        return best_reward, best_pos


    def fine_tune(self):
        ...

def main():
    agent = Agent(5)
    instance = torch.rand((5, 5, 5))
    print(f"Greedy score: {greedy(instance.numpy()):.2f}")
    print(f"NDP score: {agent.act(instance):.2f}")


if __name__=="__main__":
    main()
