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
    N = 20
    agent = Agent.load(n=N)
    
    attempts = 50
    greedy_scores = []
    ndp_scores = []
    for n in trange(1, 21, desc="Testing"):
        greedy_total_score = 0
        ndp_total_score = 0
        for _ in range(attempts):
            instance = torch.rand((n, n, n))
            greedy_total_score += greedy(instance.numpy())
            ndp_total_score += agent.act(instance)
        greedy_scores.append(greedy_total_score / attempts)
        ndp_scores.append(ndp_total_score / attempts)
    np.save(open(f"./logs/greedy_scores.npy", "wb"), np.array(greedy_scores))
    np.save(open(f"./logs/ndp_scores.npy", "wb"), np.array(ndp_scores))

    print('Estimated average scores on random 3AP instances:\n')
    print(f"{'n': <3}{'Greedy':^10}{'NDP':^10}")
    for n in range(1, 21):
        print(f"{n:<3}{greedy_scores[n-1]:^10.2f}{ndp_scores[n-1]:^10.2f}")

if __name__=="__main__":
    main()
