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

## Approach

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

During both stages, loss function shows relatively fast convergence despite large number of weights (as is expected for the network with only few hidden layers), and is relatively stable. The latter is the main reason that value networks where chosen over policy ones, as policy gradients methods can suffer from learning instability. Fast convergence also allows for training with very few computational power.

To assess results, 50 random instances are generated for each problem size from 1 to 20, and average rewards are calculated for the greedy and the proposed NDP method. The scores are then normalized by dividing by the problem size. 

<p align="center">
    <img src="/images/scores.png" alt="Fine tune loss history" align="middle"/>
</p>

As can be derived from the experimental results, our network performs slightly better than greedy approach for relatively small $N$ less than 10. For the intermediate values of $10 \le N \le 20$ they perform very similar to each other, with hints indicating that greedy approach will probably overtake ours at larger values. Such decline in NDP effectiveness could have been expected, and is the sign that we scale the network size too slowly (asymptotically slower). On the other hand, further in increase in model size is undesirable, as currently we use over $4n^5$ weights, which contribute to $O(n^8)$ time complexity during inference or finetunning stage.

## Model scaling

One of the advantages of neural approaches is their exceptional ability to scale. By varying number of the parametrs, one can easily adjust balance between time (and space) comlexity and perfomance. In this experiment, we test a number of models for the fixed problem size $N = 10$ and varying sizes of their internal networks, which a set to be $k n^2$ for the first hidden layer and $4k n$ for the second. Perfomance (expressed as gap from the optimal solution obtained by the classical dynamic programming means) as a function of this scale factor $k$ is depicted on the following graph (note its log scale along x axis).

<p align="center">
    <img src="/images/scale.png" alt="Perfomance as a function of model size" align="middle"/>
</p>

Just as expected, perfomance exhibites a steady rise when increasing number of the parameters, which is convenient for finding a trade off between learning (and running) times and perfomance

## Different datasets (populations)

One of the interesting features of neural approaches is their (probable) sensibility to the instance classes. When trained on one population of instances, networks can adapt and potentially overfit to better perform on this specific population, while have very limited ability to solve instances from different populations. To assess the effects of overfitting (which is not necessarily bad in our scenario, as creating ensembles of neural models trained on different populations can potentialy yield better results than training a singular model on all of the available data), we train three models with the same parameters, but on the different populations of instances, described in the introduction and named uniform, beta and geom (short fro geometrical). These three models are then tested an all of this populations against each other and against greedy and optimal solutions. The results are presented on the bar chart below. Problem size was chosen to be $N=10$, and network hidden layers sizes are set to be $128 n$ $256 n$.

<p align="center">
    <img src="/images/experiment_datasets.png" alt="Perfomance on different datasets" align="middle"/>
</p>

We can see an expected overall tendency of model, perfoming on its own datasets at least as good as model trained on any other dataset (and at least just as good as the greedy approach). Its interesting that models, trained on beta and uniform distributed instances show very similar perfomance, and both fall back slightly on geom dataset, while NDP-geom has minor difficulties on beta and uniform datasets. This is probably solely due to the ranges, that individual cost occupied, with geometrical ranging from 0 to 3 with 0.9 mean (and probably concentrated higly around that mean) while beta and uniform being between 0 and 1.

A relatively large gap between greedy and neural approaches is easily seen on the beta dataset. Its difficulty probably comes from the nature of cost distribution, which is close to binomial. On the truly binomial distribution, one has to maximize the number of non-zero costs taken, and actual values of them are not important (since they are all 1's). Greedy algorithms has no ability to count numbers in any way (which is essential to successful solution), while NDP approach has no such flaw.

Geometrical instances, on the other hand, turned out to be the easiest for the greedy method, probably because of how clustered cost values turned out to be. Under this scenario, the number of costs picked stays the same, so there is no need to cound them, which opens up the way for greedy approach.

Overall, overfitting largely is not present at this model sizes, apart from the ranges problem, which once again showcases the importance of data normalization. 


## Discussions and future work

First of all, as is known from the 2-dimensional version of the problem and can be suspected based on the normalized perfomance graph, greedy approach is extremely effective on random instances, with perfamance gap between it and the optimal solution being less than 10% for $N = 10$ and less than 5% for $N = 20$. This makes it very difficult to assess different approaches, as great variety in their intelligence or machine time consumption (from the most basic and the fastest greedy method, to the exponential in time optimal) leads to very few differences in the actual results. To overcome this problem, a method of obtaining hard instances should be developed. One can try to generate a large number of instances, and test each one to know whether it is difficult enough. This method, however, has two major downsides. It is computationally complex (as is obtaining true solutions), so it only allows to generate relatively small dataset, rather than on-the-go manner generation. The other problem is that in this case we do not really know instance distribution, which could be valuable since neural approaches could be very sensitive to the kind of distribution used (so stellar perfomance on certain distributions can not guarantee even mediocre on another one).

On the other hand, NDP approach is a very general tool, and can be applied to virtually all NP-hard problems. It also can be enhanced using reinforcment learning techniques, for example actor-critic methods can be utilized to obtain significant perfomance boost, as policy networks can eliminate iteration over action space currently used to obtain policy, which would lead to only $O(n^6)$ time complexity with almost identical model size. This perfomance boost, however, would come with increased learning instability, though it would not be as significant as in pure policy-based approaches. 

In general, inference stage time complexities of less than $O(n^6)$ should not be expected from neural network approaches regardless of exact arcitecture details. This is due to the fact the problem is global in nature, so every individual cost can impact final solution, so network size should be at least $O(n^5)$ to properly accomodate for all $n^3$ input values and their interactions and another $n$ in complexity arises from iterative nature of most neural approaches. If further complexity reduction is desired, methods to divide problem into smaller subproblems should be developed, or to reduce size $N$ before applying neural approaches. The latter can be probably implemented using continuous versions of Monte Carlo tree search or analogues.

## References

<a name="1">[1]</a> Xu et al. (2020). Deep Neural Network Approximated Dynamic Programming for Combinatorial Optimization [pdf](https://cdn.aaai.org/ojs/5531/5531-13-8756-1-10-20200512.pdf)

<a name="2">[2]</a> Y. Crama, F. Spieksma (1992). Approximation algorithms for three-dimensional assignment problems with triangle inequalities, [pdf](https://www.win.tue.nl/~fspieksma/papers/EJOR1992paper.pdf), [dataset link](https://www.win.tue.nl/~fspieksma/instancesEJOR.htm)

<a name="3">[3]</a> Jiang et al. (2017). Approximate Muscle Guided Beam Search for Three-Index Assignment Problem, [arxiv.org](https://arxiv.org/abs/1703.01893)
