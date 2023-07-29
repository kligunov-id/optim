import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from model import Agent

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

def main():
    N = 12
    agent = Agent.load(n=N)
    
    attempts = 1000
    greedy_total_score = 0
    ndp_total_score = 0
    for _ in trange(attempts, desc="Testing"):
        instance = torch.rand((N, N, N))
        greedy_total_score += greedy(instance.numpy())
        ndp_total_score += agent.act(instance)
    print(f"Greedy mean score: {greedy_total_score / attempts:.2f}")
    print(f"NDP mean score: {ndp_total_score / attempts:.2f}")

if __name__=="__main__":
    main()
