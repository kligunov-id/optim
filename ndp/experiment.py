import torch
import torch.distributions
import numpy as np
from tqdm import trange
import itertools as its

from model import ValueNetwork, Agent, UniformGenerator
from test import dp, greedy

def large_value_network_generator(n):
    return ValueNetwork(n=n, hidden_1_k=128, hidden_2_k=256)

def random_method(cost):
    if isinstance(cost, torch.Tensor):
        cost = cost.numpy()
    diag_indexes = np.arange(cost.shape[0])
    return cost[(diag_indexes, diag_indexes, diag_indexes)].sum()

class BetaGenerator:
    def __init__(self, alpha=0.07, beta=0.17):
        self.dist = torch.distributions.Beta(alpha, beta)

    def get_batch(self, batch_size, problem_size):
        return self.dist.sample(
            sample_shape=(batch_size, problem_size, problem_size, problem_size))
    
    def get_instance(self, problem_size):
        return self.dist.sample(sample_shape=(problem_size, problem_size, problem_size))

class GeomGenerator:
    def __init__(self, coord_lim=1/3):
        self.coord_lim = coord_lim

    def get_instance(self, problem_size):
        points = torch.rand((2, 3, problem_size)) * self.coord_lim
        points_left = points.unsqueeze(dim=2).unsqueeze(dim=4)
        points_right = points.unsqueeze(dim=1).unsqueeze(dim=3)
        coord_diff = self.coord_lim - (points_left - points_right).abs()
        dist = (coord_diff ** 2).sum(dim=0) ** 0.5 # 3 x 3 x N x N
        return torch.Tensor([dist[0, 1, i, j] + dist[1, 2, j, k] + dist[2, 0, k, i]
            for i, j, k in its.product(range(problem_size), repeat=3)]).reshape((problem_size, problem_size, problem_size))

    def get_batch(self, batch_size, problem_size):
        return torch.stack([self.get_instance(problem_size) for _ in range(batch_size)])

def get_mean_score(algo, generator, problem_size, num_trials=5, tqdm_desc="Testing..."):
    results = []
    testing_range = trange(num_trials, desc=tqdm_desc) if tqdm_desc else range(num_trials)
    for _ in testing_range:
        results.append(algo(generator.get_instance(problem_size)))
    results = np.array(results)
    return results.mean(), results.std()

def str_from_val_error(val_error_pair):
    val, error = val_error_pair
    return f"{val:.2f} ± {error:.2f}"

def main():
    N = 10
    generators = {
        "uniform": UniformGenerator(),
        "beta": BetaGenerator(),
        "geom": GeomGenerator(),
    }
    print(f"Training {len(generators)} models on different data generators")
    agents = {
        f"NDP-{generator_name}": Agent(n=N,
            logs_folder=f"./untracked_logs/{generator_name}",
            generator=generators[generator_name],
            weights_folder=None,
            value_network_factory=large_value_network_generator,
            hyper_params=dict(num_pretrain_iters=100, num_finetune_iters=200)) for generator_name in generators
        }
    methods = { agent_name: agents[agent_name].act for agent_name in agents }
    methods.update({
        "Random": random_method,
        "Greedy": greedy,
        "Optimal": dp,
        })
    results = {}
    for method_name in methods:
        results[method_name] = [get_mean_score(
                algo=methods[method_name],
                generator=generators[generator_name],
                problem_size=N,
                num_trials=(20 if method_name=="Optimal" else 1000),
                tqdm_desc=f"Testing {method_name} on {generator_name}",
            ) for generator_name in generators]
    print(f"\n{'Method':<12}" + "".join([f"{generator_name:^15}" for generator_name in generators]))
    for method_name in results:
        print(f"{method_name:<12}" + "".join(f"{str_from_val_error(result):^15}" for result in results[method_name]))

if __name__ == "__main__":
    main()
