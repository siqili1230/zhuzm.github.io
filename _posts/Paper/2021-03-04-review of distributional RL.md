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




















