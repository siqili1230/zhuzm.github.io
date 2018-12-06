---
layout: post
title: A Unified View of Entropy-Regularized
Markov Decision Processes阅读笔记
date: 2018-12-06 22:11:00
categories: 强化学习
tags: ACML2018 强化学习理论 收敛性
mathjax: true

---

* content
{:toc}

论文题目: [A Unified View of Entropy-Regularized
Markov Decision Processes](https://arxiv.org/pdf/1705.07798.pdf)

## 简介

本文的主要工作是给出了在稳态分布下一些算法(如TRPO)的收敛性证明。

## 背景

首先笔者假定读者对于马尔可夫决策过程（MDP）都有了一定的了解。
如下图（assum_1）给出了马尔可夫单链的假设，这个假设在MDP是非退化的且非周期的条件下是满足的。
![assum_1](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\assum_1.png)
这篇文章全部都是围绕在稳态分布下讨论的，在$\pi(a|x)$的基础上，我们可以引导出$\nu_\pi(x)$和$\mu_\pi(x,a)$。同时，我们关心的平均回报
$$
\rho(\pi)=\lim_{T\to \infty}\mathbb{E}[\frac{1}{T}\sum_{t=1}^{T}r_t(x_t,a_t)]
$$
也可以写成
$$
\rho(\pi)=\sum_{x,a}\nu_\pi(x)\pi(a|x)r(x,a) = \sum_{x,a}\mu_\pi(x,a)r(x,a)
$$
所以原问题可以写成带约束的优化问题：
$$
\max_\mu \rho(\mu)= \sum_{x,a}\mu_\pi(x,a)r(x,a) \\
subject \ to \ \sum_b\mu(x,b) = \sum_{x,a} P(y|x,a)\mu(x,a) \ \forall y \\
\sum_{x,a} \mu(x,a) = 1 \\
\sum_{x,a} \mu(x,a) \geq 0
$$
同时也可以写成对偶形式：
$$
min \rho(\mu) \\
subject \ to \ \rho + V(x)-\sum_y P(y|x,a)V(y) \geq r(x,a) \ \forall (x,a)
$$

## 算法

## 实验

## 小结