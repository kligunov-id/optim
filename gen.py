import numpy as np

def gen(n):
    return np.random.random((n, n, n))

def random(costs):
    diag_indexes = np.arange(costs.shape[0])
    return costs[(diag_indexes, diag_indexes, diag_indexes)].sum()

def greedy(costs):
    if costs.shape[0] == 0:
        return 0
    if costs.shape[0] == 1:
        return costs[0,0,0]
    immediate_reward = costs.max()
    indexes = np.unravel_index(costs.argmax(), costs.shape)
    for axis, index in enumerate(indexes):
        costs = np.delete(costs, index, axis=axis)
    return immediate_reward + greedy(costs)

def get_avg(n, reps=20):
    costs = [gen(n) for _ in range(reps)]
    avg_random = np.mean([random(c) for c in costs])
    avg_greedy = np.mean([greedy(c) for c in costs])
    return avg_random, avg_greedy
 
if __name__ == "__main__":
    print('Estimating average scores on random 3AP instances:\n')
    print(f"{'n': <3}{'Random':^10}{'Greedy':^10}")
    for i in range(5, 101, 5):
        avg_random, avg_greedy = get_avg(i)
        print(f"{i:<3}{avg_random:^10.2f}{avg_greedy:^10.2f}")
