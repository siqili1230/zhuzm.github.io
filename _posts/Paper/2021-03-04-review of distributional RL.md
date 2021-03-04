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

### FQF

Paper: [Fully Parameterized Quantile Function for Distributional Reinforcement Learning](https://arxiv.org/pdf/1911.02140.pdf)

The main contribution of this paper is the parameterized quantiles $\tau(\theta)$ while in IQN the quantiles are sampled from a distribution. FQF projects the quantile function into a staircase function:

![image-1](\images\2021-03-04-review of distributional RL\FQF-formula-1.png)

and find the staircase funtion with minimal 1-Wasserstein loss:

![image-1](\images\2021-03-04-review of distributional RL\FQF-formula-2.png).

![image-1](\images\2021-03-04-review of distributional RL\FQF-fig-1.png).























