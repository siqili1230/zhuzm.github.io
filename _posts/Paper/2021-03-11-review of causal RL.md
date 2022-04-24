---
layout: post
title: Review of Causal RL
date: 2021-03-11 14:37:00
categories: 强化学习
tags:  Cuasal-RL Review
mathjax: true

---

* content
{:toc}

### Causal Inference Q-network
[Causal Inference Q-network: Towards]()

This paper learns invariant representation of perturbated observations.


### Intrumental variables for offline RL

[Instrumental Variable Value Iteration for Causal Offline Reinforcement Learning]()

![image-1](\images\2021-03-11-review of causal RL\deconfound-3.png)

This paper supposes an observable $z_t$ as the instrumental variable that affects the action $a_t$ jointly with state $x_t$. The difference between $x_t$ and $x_t$ is that $z_t$ affects the $x_{t+1}$ only through $a_t$. For example, $a_t$ is the treatment, $x_t$ is the current health status and $z_t$ is the physician's preference for treatments.

But what's the advantage of introducing such structure? It calculates $\hat{x}'=f(x',a\mid z)$


### RL with confounders

[Deconfounding RL in observational settings]()

This paper considers an unobserved factor $u$(confounders in causal learning) that affects observations, actions and rewards.

![image-1](\images\2021-03-11-review of causal RL\deconfound-1.png)

And it uses VAE to build an inference model for predicting $x_{t+1}$ based on $(x_t,a_t)$ and $z_t$.

![image-1](\images\2021-03-11-review of causal RL\deconfound-2.png)

I am puzzled that how and why the $u$ can be a time-independent confounder. 

In experiments, it provides a confounding benchmark where the action space is divided and $u$ decides which partition of the action space is avaliable. It assumes that $u$ influences the reward function and observations but estimate the reward and action without $u$. So I am confused about the $u$.

[Markov Decision Processes with Unobserved Confounders: A Causal Approach]()

![image-1](\images\2021-03-11-review of causal RL\deconfound-4.png)

The main difference between MDP and MDPUC is that the value function is conditioned on the actual action $x'_t$ when evaluating the $v(s_t)$ and $q(s_t,x_t)$ ($x_t$ is the potential action).

![image-1](\images\2021-03-11-review of causal RL\deconfound-5.png)

[WOULDA, COULDA, SHOULDA: COUNTERFACTUALLY-GUIDED POLICY SEARCH]()

This paper introduces unobserved variables in POMDP by $s_{t+1}=f(s_t,a_t,u_t)$.

And Counterfactual Inference (CFI) uses following method:

![image-1](\images\2021-03-11-review of causal RL\deconfound-6.png)

It uses data to attain posterior distribution of $u$ and re-computer the target value under estimation $\hat{u}$.

[Counterfactual Off-Policy Evaluation with Gumbel-Max Structural Causal Models (ICML19)](https://arxiv.org/abs/1905.05824)

This paper introduces a class of SCMs to generate counterfactual trajectories in POMDP.

![image-1](\images\2021-03-11-review of causal RL\deconfound-7.png)

Cannot understand the example in section 3.1.

It defines the Counterfactual Stability, which means that in a categorical SCM if an intervention $I'$ increase the probability of oobserved state $i$ compared with the intervention $I$, then we can only observe $i$ under $I'$.

![image-1](\images\2021-03-11-review of causal RL\deconfound-8.png)


[Off-Policy Evaluation in Partially Observable Environments](https://arxiv.org/pdf/1909.03739.pdf)

This paper defines Decoupled POMDP, where $(u,z)$ represents observed and unobserved states.
