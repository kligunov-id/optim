import torch
import numpy as np
from model import UniformGenerator, Agent, ValueNetwork
from tqdm import trange

class UniformRepeater:
    def __init__(self, num_repeats, problem_size, total_volume=Agent.num_finetune_iters*Agent.finetune_batch_size):
        self.size = total_volume // num_repeats
        self.problem_size = problem_size
        self.data = torch.rand((self.size, problem_size, problem_size, problem_size))
        self.access_order = torch.randperm(self.size)
        self.i = -1
    
    def get_instance(self, problem_size):
        if problem_size != self.problem_size:
            return torch.rand((problem_size, problem_size, problem_size))
        self.i += 1
        if self.i == self.size:
            self.i = 0
            self.access_order = torch.randperm(self.size)
        return self.data[self.access_order[self.i]]

    def get_batch(self, batch_size, problem_size):
        return torch.stack([self.get_instance(problem_size) for _ in range(batch_size)])

def value_network_generator(n):
    return ValueNetwork(n=n, hidden_1_k=32, hidden_2_k=64)

N = 8
test = UniformGenerator()

def get_score_on_generator(agent, generator, attempts=100, tqdm_desc=None):
    total = 0
    attempts = 1000
    for _ in trange(attempts, desc=tqdm_desc):
        total += agent.act(generator.get_instance(N))
    return total / attempts

def get_scores(dataset_reps):
    train = UniformRepeater(dataset_reps, N, total_volume=300 * 50)
    agent = Agent(N,
        logs_folder=None,
        weights_folder=None,
        hyper_params=dict(num_pretrain_iters=50, num_finetune_iters=300),
        generator=train,
        value_network_factory=value_network_generator)
    score_train = get_score_on_generator(agent, train, tqdm_desc=f"Testing {dataset_reps} reps on train")
    score_test = get_score_on_generator(agent, test, tqdm_desc=f"Testing {dataset_reps} reps on test")
    return score_train, score_test

def main():
    save_folder = "./untracked_logs/overfit"
    reps = np.array([1, 50, 100, 500, 1000])
    scores = np.array([np.array(get_scores(rep)) for rep in reps])
    np.save(open(f"{save_folder}/reps.npy", "wb"), reps)
    np.save(open(f"{save_folder}/train.npy", "wb"), scores[:, 0])
    np.save(open(f"{save_folder}/test.npy", "wb"), scores[:, 1])

if __name__ == "__main__":
    main()
