# TQ optimization project

This repository contains source code for the implementations of various approaches to the (axial) three-dimensional linear sum assingment problem, as well as readily deployable models.

## 3AP formulation

In this work we'll consider a rather simple and canonical version of the three dimensional assignmet problem. It can be formulated as a 0-1 programming problem as follows:

Given matrix $C_{ijk}$ of size $N \times N \times N$, a set of values $x_{ijk}$ is being optimized to minimize total cost $C$:

$$ C = \sum_i \sum_j \sum_k c_{ijk} x_{ijk} $$

with $x_{ijk}$ subject to the constraint group <a name="(1)">(1)</a>, ensuring that each job, worker and place is assinged exactly once:

$$\sum_i \sum_j x_{ijk} = 1$$

$$\sum_i \sum_k x_{ijk} = 1$$

$$\sum_j \sum_k x_{ijk} = 1$$

and a constrain <a name="(2)">(2)</a> on the possible values of indicators $x_{ijk}$

$$x_{ijk} \in \lbrace 0, 1\rbrace $$

Condition (2) can sometimes be relaxed into condition <a name="(2*)">(2*)</a>:

$$ x_{ijk} \in \left[0, 1\right]$$

allowing for continous indicators instead of binary ones, though special steps must be taken to derive a solution compliant with the original constraint (2) from the solution compliant only with the (2*).

Some approaches also do not inherently produce feasible solutions and instead incorporate into the target cost special penalties for violating constraints (1) or utilize normalization techniques such as the softmax function to meat one of theese (although satisfying all 3 with such approach is quite problematic).

## Data preparation and benchmarking

All instances used in training are generated randomly with costs $c_{ijk}$ sampled from a given distribution which is assumed to be uniform from 0 to 1 unless otherwise stated. In order to increase complexity (and specifically make greedy approaches less competative) different distributions or generation schemes can be used; for a example, [[1]](#1) considers using a beta distribution with the parameters $\alpha$ and $\beta$ set to 0.07 and 0.17 respectively.

An alternative scheme is proposed in [[2]](#2). Three different sets $I$, $J$ and $K$ of $N$ points on a 2-dimesional plane are generated randomly and uniformly from a $[0, 1]^2$ square. Distances between each pair of points from different sets are then calculated and costs $c_{ijk}$ are set to be equal to $\text{dist}(I_i, J_j) + \text{dist}(J_j, K_k) + \text{dist}(K_k, I_i)$. Many different triangular inequalities arise as the consequnce of the procedure, so potential for some performance improvements or new strategy discoveries is created. An 18-instance dataset with relatively large N (33 and 66) is provided in the paper as well, alongside with the globaly optimal solutions. Another research paper [[3]](#3) reports perfomance on this dataset for various classical algorithms and heuristics, making it an interesting benchmark.

## Repository content

#### gen.py

A simple utility programm is provided under the name `gen.py` which calculates average scores for random and greedy assignments for some random 3AP instances. It presents them in the table with respect to the size of instances $N$. The scores are to be used as a quick reference (to provide typical total cost estimate, which might be usefull during training).

#### NDP

This folder contains implemetations of methods developed in [[1]](#1). Assignment problem is treated as a sequence of $N$ subsequent assignments of jobs and places to the $i$-th worker. Each step we choose exactly one pair $(0, j, k)$ and then solve the subproblem for the matrix of size $N-1 \times N-1 \times N-1$. Such problem interpretation naturally satisfies all of the constraints, which is very desirable. Two ways of choosing $(j, k)$ pair are proposed, using either policy or value networks. Under the former approach, the network directly predicts probabilities of choosing any given pair, while the latter one approach was inspired by dynamic programming and derives the policy from value predictions (hence the name neural network approximated dynamic programming, or NDP for short). The best move to be taken is the one maximazing the sum of the immediate reward and the maximum total reward for the subproblem. In the context of dynamical programming, obtaining the exact value for the total reward is very costly (exponential in time for the 3AP), so a neural network approximation is used instead.

This approach, however, comes with another difficulty. Each subproblem has different size (from $N \times N \times N$ all the way down to the $1\times 1 \times 1$), which makes it problematic to feed cost matricies to the same network. In case of the assignment problem the issue can be solved by filling absent columns and rows with zero values, but this can lead to some collisions in indicies (as network can sometimes mistakenly pick this buffer elements despite them not being optimal). A more general approach devoid of any collisions and suitable to other NP hard problems is proposed in the same paper [[1]](#1), and it is to create, fit and store a separate network for each problem size. This does not lead to increase in time, as other machine learning method are usually iterative anyway, and the disk space is not a concern as it is abundant in modern computers. One thing that is affected, however, is the learning procedure.

We will discuss it in the context of value networks, as policy ones are not used in this work for the reasons discussed later. Learning process is divided into two distinct stages, called pretraining and fine-tuning. During pretraining, for each size starting from the minimal one a new network is being trained to predict optimal cost of a matrix of given size. Target cost is etsimated using previously trained network, by trying every possible pair of indicies $(j, k)$ and estimating the residual reward of submatrix. The loss function is MSE value of these two.

$$ L(s, W) = \text{MSE}(\max_{a \in A_s} R(s, a) + Net_{i-1}(s'), Net_i(s, W))$$

where $Net_i$ is the current network, $W$ are its weights, $s$ is the current state (cost matrix), $a$ is the action i.e. pair of indices $(j, k)$, $A_s$ is the set of possible actions, $R$ is the immediate reward ($c_{0jk}$), $s'$ is the cost matrix with rows $0$, $j$, and $k$ removed along dimesions 0, 1 and 2; and $Net_{i-1}$ is the previous network for problems of size $i-1$. Note that while training $i-th$ network (network of size $i$), we freeze the weigths of all previous networks.

During inference, we iteratively choose pair $(j, k)$ which achieves maximum sum of immediate reward and the estimated reward. 

During fine-tuning stage we first solve each problem instance of size $N - 1\times N - 1\times N - 1$ by following the strategy for the inference stage (i.e. iteratively choosing the best move according to reward estimations of pretrained neural networks; their weights are frozen during instance processing), while also recording all chosen moves. Those moves result in positions $s_{n-1}$, $s_{n-1}$, ..., $s_1$ (where indicies correspond to problem sizes). For each one we compare actual rewards (trailing sums of immediate rewards) to the predictions of networks (now with weights unfrozen). The loss is MSE between these values.

Each network $Net_i$ is a simple fully connected feed-forward network with two hidden layers of sizes $4n^2$ and $8n$. 

## Results

For the pretraining stage, we used learning rate of 3e-4 and batch size was set to 64. Following graph shows the results of training for the first 100 iterations and is very much typicall for all sizes.

<p align="center">
    <img src="/images/pretrain.png" alt="Pretrain loss history"/>
</p>

During fine-tuning, learning rate was redused to 1e-4 and batch size down to 50. Results for only 20 iterations are given, as training took significantly longer time per step than during the pretraining.

<p align="center">
<img src="/images/finetune.png" alt="Fine tune loss history" align="middle"/>
</p>

## Discussions and future development

## References

<a name="1">[1]</a> Xu et al. (2020). Deep Neural Network Approximated Dynamic Programming for Combinatorial Optimization [pdf](https://cdn.aaai.org/ojs/5531/5531-13-8756-1-10-20200512.pdf)

<a name="2">[2]</a> Y. Crama, F. Spieksma (1992). Approximation algorithms for three-dimensional assignment problems with triangle inequalities, [pdf](https://www.win.tue.nl/~fspieksma/papers/EJOR1992paper.pdf), [dataset link](https://www.win.tue.nl/~fspieksma/instancesEJOR.htm)

<a name="3">[3]</a> Jiang et al. (2017). Approximate Muscle Guided Beam Search for Three-Index Assignment Problem, [arxiv.org](https://arxiv.org/abs/1703.01893)
