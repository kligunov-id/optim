import torch
import numpy
from tqdm import trange
from model import ValueNetwork, Agent, UniformGenerator
from experiment import get_mean_score, str_from_val_error

def main():
    N = 10
    scales = [2 ** i for i in range(5, 7)] # 1, ..., 256
    agents = [Agent(
        n=N,
        value_network_factory=lambda n: ValueNetwork(n, scale, scale * 4),
        logs_folder="./untracked_logs/scale",
        weights_folder=None,
        hyper_params=dict(num_pretrain_iters=50, num_finetune_iters=200)) for scale in scales]
    scores = [get_mean_score(
        algo=agent.act,
        generator=UniformGenerator(),
        problem_size=N,
        num_trials=1000,
        tqdm_desc=f"Testing network of scale {scale}") for agent, scale in zip(agents, scales)]
    for scale, score in zip(scales, scores):
        print(f"Scale {scale} mean score: {str_from_val_error(score)}")

if __name__ == "__main__":
    main()
