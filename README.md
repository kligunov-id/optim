# TQ optimization project

This repository contains source code for the implementations of various approaches to the (axial) three-dimensional linear sum assingment problem, as well as readily deployable models.

## 3AP formulation

In this work we'll consider a rather simple and canonical version of the three dimensional problem. It can be formulated as a 0-1 programming problem as follows:

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

Some approaches also do not inherently produce feasible solutions and instead incorporate into the target cost function special penalties for violating constraints (1) or utilize normalizations such as softmax to meat one of theese (although satisfying all 3 with such approach is quite problematic).

## Data preparation and benchmarking

All instances used in training are generated randomly with costs $c_{ijk}$ sampled from a given distribution which is assumed to be uniform from 0 to 1 unless otherwise stated. In order to increase complexity (and specifically make greedy approaches less competative) different distributions or generation schemes can be used; for a example, [[1]](#1) considers using a beta distribution with the parameters $\alpha$ and $\beta$ set to 0.07 and 0.17 respectively.

An alternative scheme is proposed in [[2]](#2). Three different sets $I$, $J$ and $K$ of $N$ points on a 2-dimesional plane are generated randomly and uniformly from a $[0, 1]^2$ square. Distances between each pair of points from different sets are then calculated and costs $c_{ijk}$ are set to be equal to $\text{dist}(I_i, J_j) + \text{dist}(J_j, K_k) + \text{dist}(K_k, I_i)$. Many different triangular inequalities arise as the consequnce of the procedure, so potential for some performance improvements or new strategies is created. An 18-instance dataset with relatively large N (33 and 66) is provided in the paper as well, alongside with the globaly optimal solutions. Another paper [[3]](#3) reports perfomance on the dataset for various classical algorithms and heuristics, making it an interesting benchmark.

## References

<a name="1">[1]</a> Xu et al. (2020). Deep Neural Network Approximated Dynamic Programming for Combinatorial Optimization [pdf](https://cdn.aaai.org/ojs/5531/5531-13-8756-1-10-20200512.pdf)

<a name="2">[2]</a> Y. Crama, F. Spieksma (1992). Approximation algorithms for three-dimensional assignment problems with triangle inequalities, [pdf](https://www.win.tue.nl/~fspieksma/papers/EJOR1992paper.pdf), [dataset link](https://www.win.tue.nl/~fspieksma/instancesEJOR.htm)

<a name="3">[3]</a> Jiang et al. (2017). Approximate Muscle Guided Beam Search for Three-Index Assignment Problem, [arxiv.org](https://arxiv.org/abs/1703.01893)
