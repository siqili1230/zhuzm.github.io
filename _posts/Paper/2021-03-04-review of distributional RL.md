---
layout: post
title: Review of Distributional RL
date: 2021-03-04 14:37:00
categories: 强化学习
tags:  Distributional-RL Review
mathjax: true

---

* content
{:toc}

论文题目: [trust region policy optimization](http://proceedings.mlr.press/v37/schulman15.pdf)

## Introduction

### C51 & QR-DQN 
Paper: [A Distributional Perspectiveon Reinforcement Learning](https://arxiv.org/abs/1707.06887)
Paper: [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)

These two have been introduced in [former reading notes](https://siqili1230.github.io/2019/01/03/Implicit-Quantile-Networks-for-Distributional-Reinforcement-Learning/).


### IQN

Paper: [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/pdf/1806.06923.pdf) (ICML 2018)

A key improvement of IQN is the expanding from fixed quantiles $\{\tau_i\}_{i=1}^N$ to parameterized quantiles $\tau \sim U(0,1)$. The q-values of quantiles are updated by sampled $\tau$ and $\tau'$:

![image-1](\images\2021-03-04-review of distributional RL\IQN-formula-1.png)

With parameterized quantiles, IQN can approximate the whole continuous distribution.

What's more, another important improvement is that we can apply the utility function into distributional q-values, which can be achieved by distorted sample $\beta(\tau)$ where $\beta:[0,1]\to [0,1]$.

The choice of $\beta$ includes risk-neutral, risk-averse and risk-seeking. A special example is conditional value-at-risk (CVaR):

$$
CVaR(\eta,\tau)=\eta\tau,
$$

which changes $\tau \sim U(0,1)$ to $\tau \sim U(0,\eta)$ for guaranteeing performance on worse conditions.

![image-1](\images\2021-03-04-review of distributional RL\IQN-fig-1.png)

A special trick of embedding the quantile $\tau$ is widely used in Distributional RL:

![image-1](\images\2021-03-04-review of distributional RL\IQN-formula-2.png)

### FQF

Paper: [Fully Parameterized Quantile Function for Distributional Reinforcement Learning](https://arxiv.org/pdf/1911.02140.pdf)

The main contribution of this paper is the parameterized quantiles $\tau(\theta)$ while in IQN the quantiles are sampled from a distribution. FQF projects the quantile function into a staircase function:

![image-1](\images\2021-03-04-review of distributional RL\FQF-formula-1.png)

and find the staircase function with minimal 1-Wasserstein loss:

![image-1](\images\2021-03-04-review of distributional RL\FQF-formula-2.png).

![image-1](\images\2021-03-04-review of distributional RL\FQF-fig-1.png).

### Non-crossing QR-DQN

Paper: [Non-crossing quantile regression for deep reinforcement learning](https://proceedings.neurips.cc//paper/2020/file/b6f8dc086b2d60c5856e4ff517060392-Paper.pdf)

Experiments show that quantile regression cannot guarantee the non-decreasing property of learned quantiles.

The constrained optimization is:

![image-1](\images\2021-03-04-review of distributional RL\NC-formula-1.png).

To solve this problem, NC-QR-DQN search for a subspace of initial Z-space that $Z_Q\in Z_\Theta$. quantile functions $Z_q=\{\theta_i\}_{i=1}^N$ in $Z_Q$ are all satisfying the non-decreasing constraint. 

In practice, it uses a network to produce the $\phi_{i,a}$ (the i-th quantile value for action $a$ given a state) and re-computes the outputs by $\psi_{i,a}=\sum_{j=1}^i \phi_{i,a}$ where $\psi_{N,a}=1$ and $\psi_{i,a}$ is non-decreasing. Since $\psi\in [0,1]$, there is another network to scale 


![image-1](\images\2021-03-04-review of distributional RL\NC-fig-1.png).




















