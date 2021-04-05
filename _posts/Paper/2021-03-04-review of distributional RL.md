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

In another perspective, solving this optimization problem is equivalent to finding a projection operator $\Pi_{W_1}$ such that:

![image-1](\images\2021-03-04-review of distributional RL\NC-formula-2.png).

In practice, it uses a network to produce the $\phi_{i,a}$ (the i-th quantile value for action $a$ given a state) and then re-computes the outputs by $\psi_{i,a}=\sum_{j=1}^i \phi_{j,a}$ where $\psi_{N,a}=1$ and $\psi_{i,a}$ is non-decreasing. Since $\psi\in [0,1]$, there is another scale network to recover the logits in $[0,1]$ to the original range with :

$$
q_i(s,a) =\alpha(s,a)*\psi_{i,a}+\beta(s,a)
$$

And the modified TD error is 
$\delta_{i,j}=r+\gamma q_j(s',a^*)-q_i(s,a)$
where 
$a^*=\arg\max_{a'}\sum_{j=1}^N q_j(s',a')$.

![image-1](\images\2021-03-04-review of distributional RL\NC-fig-1.png).

By the way, the precise estimation of truncated variance by including the non-cross constraint can help measure the intrinsic uncertainty.

![image-1](\images\2021-03-04-review of distributional RL\NC-fig-2.png).

### DLTV

Decaying Left Truncated Varianc--a novel exploration strategy that more sufficientli utilize the distributional information. The key idea is that the quantile is usually assymmetric and the upper trail variability is more relevant. There is an upper truncated measure of the uncertainty:

$$
\sigma_+^2=\frac{1}{2N}\sum_{i=N/2}^N(\tilde{\theta}-\theta_i)^2 \\
c_t=c\sqrt{\log t/t}\\
a^*=\arg\max_{a'}(Q(s,a')+c_t\sqrt{\sigma_+^2})
$$

where $\tilde{\theta}$ is the median rather than mean.


### Expectile Distributional RL (EDRL)

Paper: [Statistics and Samples in Distributional Reinforcement Learning](https://arxiv.org/abs/1902.08102)

In this paper, the algorithms of distributional RL can be decomposed into following steps:

1. Find a series of statistics to discrib the distribution (e.g., the discrete or continuous quantiles)
2. Find a rule to update/re-compute those statistics and get losses.

Previous distributional RL can be divided into two types: 

1.Categorical DRL: $\eta(x,a)=\sum_{k=1}^Kp_k(x,a)\delta_{z_k}$ and learns weights with fixed quantiles $z_k$

2.Quantile DRL: $\eta(x,a)=\frac{1}{K}\sum_{k=1}^K\delta_{z_k(x,a)}$ and learns the new $\tau_k=\frac{2k-1}{2K}$ quantiles by minimizing the quantile regression loss:
$$
QR(q;\mu,\tau_k)=\mathbb{E}_{Z\sim\mu}[[\tau_k\mathbb{I}_{Z>q}+(1-\tau_k)\mathbb{I}_{Z\geq q}]\mid Z-q\mid]
$$

This paper proposes the ``expectation quantile'' (expectiles):

$$
ER(q;\mu,\tau_k)=\mathbb{E}_{Z\sim\mu}[[\tau_k\mathbb{I}_{Z>q}+(1-\tau_k)\mathbb{I}_{Z\geq q}]\mid Z-q\mid^2]
$$

![image-1](\images\2021-03-04-review of distributional RL\EDRL-fig-2.png).

CDRL overestimates the variance because the projection splits the probability mass across the discrete support. In contrast, EDRL(naive) only replaces the quantiles with expectiles and underestimates the variance.

There is a crucial problem that the learned $z_k(x,a)$ have the semantic of a statistic, but in updating, $z_k(x,a)$ in $(\Tau^\pi\eta)(x,a)$ have the semantic of both statistics and samples

To solve this problem, EDRL seperates the bellman updating process into two parts:

1.learn statistics (expectiles) from the recovered distribution/samples;

2.Recover the distribution from learned statistics.

![image-1](\images\2021-03-04-review of distributional RL\EDRL-formula-1.png).

![image-1](\images\2021-03-04-review of distributional RL\EDRL-formula-2.png).

![image-1](\images\2021-03-04-review of distributional RL\EDRL-formula-3.png).

### quantiles and expectiles
quantiles and expectiles correspond to the concept of median and mean.
In the regression perspective of the median and mean, it can be viewed as the solution of an optimization problem under different norm:

1.$\textbf{median}[y]=\arg\min_{m\in\mathbb{R}}[\frac{1}{n}\sum_{i=1}^n\mid y_i-m\mid]$

2.$\textbf{mean}[y]=\arg\min_{m\in\mathbb{R}}[\frac{1}{n}\sum_{i=1}^n(y_i-m)^2]$

Correspondingly, given risk functions:

1.$\mathcal{R}_\tau^q(u)=\mid u\mid\cdot [(1-\tau)\cdot\mathbb{I}(u<0)+\tau\cdot\mathbb{I}(u\geq0)]$

2.$\mathcal{R}_\tau^e(u)=\mid u\mid^2\cdot [(1-\tau)\cdot\mathbb{I}(u<0)+\tau\cdot\mathbb{I}(u\geq0)]$

which is equivalent to weighted meadian/mean.

### MMD-DQN

Paper: [Distributional Reinforcement Learning with Maximum Mean Discrepancy](https://www.researchgate.net/publication/343228735_Distributional_Reinforcement_Learning_with_Maximum_Mean_Discrepancy)











