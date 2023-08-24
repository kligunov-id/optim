import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
import itertools as its

from model import Agent

def greedy(cost):
    if isinstance(cost, torch.Tensor):
        cost = cost.numpy()
    if cost.shape[0] <= 1:
        return cost.sum()
    immediate_reward = cost.max()
    indexes = np.unravel_index(cost.argmax(), cost.shape)
    for axis, index in enumerate(indexes):
        cost = np.delete(cost, index, axis=axis)
    return immediate_reward + greedy(cost)

dp_limit = 10

def dp(cost):
    n = cost.shape[0]
    if n <= 1:
        return cost.sum()
    if n > dp_limit:
        return float("NaN") # timeout
    m = 1 << n
    dp = np.zeros((m, m))
    for mask_j, mask_k in its.product(range(m), repeat=2):
        if mask_j.bit_count() != mask_k.bit_count():
            continue
        i = mask_j.bit_count()
        for j, k in its.product(range(n), repeat=2):
            new_mask_j = mask_j | (1 << j)
            new_mask_k = mask_k | (1 << k)
            if new_mask_k != mask_k and new_mask_j != mask_j:
                dp[new_mask_j][new_mask_k] = max(dp[new_mask_j, new_mask_k], 
                    dp[mask_j][mask_k] + cost[i, j, k])
    return dp[m - 1, m - 1]

def main():
    N = 20
    agent = Agent.load(n=N)
    
    attempts = 50
    greedy_scores = []
    ndp_scores = []
    optimal_scores = []
    for n in trange(1, N + 1, desc="Testing"):
        greedy_total_score = 0
        ndp_total_score = 0
        optimal_total_score = 0
        for _ in range(attempts):
            instance = torch.rand((n, n, n))
            greedy_total_score += greedy(instance.numpy())
            ndp_total_score += agent.act(instance)
            optimal_total_score += dp(instance.numpy())
        greedy_scores.append(greedy_total_score / attempts)
        ndp_scores.append(ndp_total_score / attempts)
        optimal_scores.append(optimal_total_score / attempts)
    np.save(open(f"./logs/greedy_scores.npy", "wb"), np.array(greedy_scores))
    np.save(open(f"./logs/ndp_scores.npy", "wb"), np.array(ndp_scores))
    np.save(open(f"./logs/optimal_scores.npy", "wb"), np.array(optimal_scores[:dp_limit]))

    print('Estimated average scores on random 3AP instances:\n')
    print(f"{'n': <3}{'Greedy':^10}{'NDP':^10}{'Optimal':^10}")
    for n in range(1, dp_limit + 1):
        print(f"{n:<3}{greedy_scores[n-1]:^10.2f}{ndp_scores[n-1]:^10.2f}{optimal_scores[n-1]:^10.2f}")
    for n in range(dp_limit, N + 1):
        print(f"{n:<3}{greedy_scores[n-1]:^10.2f}{ndp_scores[n-1]:^10.2f}{'--':^10}")

if __name__=="__main__":
    main()
