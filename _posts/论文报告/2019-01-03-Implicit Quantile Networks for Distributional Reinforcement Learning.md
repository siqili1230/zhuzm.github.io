---
layout: post
title: Implicit Quantile Networks for Distributional Reinforcement Learning 阅读笔记
date: 2019-01-03 22:02:00
categories: 强化学习
tags: Distributional-RL DQN quantile-regression 
mathjax: true

---

* content
{:toc}

论文题目: [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)

## 简介

该文的主要工作是对于分布型强化学习问题的研究，作者引入了分位数回归的技巧，来对状态动作对的值函数$Q(x,a)$的分布进行重参数化，最终在Atari游戏上取得了不错的效果。






## 背景

在这篇文章之前，作者还发表了两篇文章（《A Distributional Perspective on Reinforcement Learning》和《Distributional Reinforcement Learning with Quantile Regression》）——其中第一篇引入了强化学习的分布型视角，第二篇引入了分位数回归的方法来解决分布型强化学习——也是这篇文章的铺垫工作，所以本文将尝试一起介绍三篇文章的工作，帮助读者厘清这三篇文章背后的思路。

### 分布型强化学习(Distributional RL/DisRL)基本定义

我们定义一个随机变量$Z^\pi(x,a)$来表示一条轨迹的累计回报：

$$
Z^\pi(x,a)=\sum_{t=0}^{\infty}\gamma^tR(x_t,a_t)
$$

其中$x_t \sim P(\cdot|x_{t-1},a_{t-1}), \ a_t \sim \pi(\cdot|x_t)$，$Z$的随机性也来自于转移函数$P$，随机策略$\pi$（即轨迹的随机性）和回报函数$R$。状态动作值函数则定义为$Q^\pi(x,a)=\mathbb{E}[Z^\pi(x,a)]$，那么一般的bellman equation就是：

$$
Q(x,a)=\mathbb{E}R(x,a)+\gamma \mathbb{E}Q(X',A')
$$

那么对于DisRL来说，基于$Z$的bellman equation就是：

$$
Z(x,a)\overset{D}{=}R(x,a)+\gamma Z(X',A')
$$

对照非分布型情况下的bellman operator和optimal operator：

$$
\mathcal{T}^\pi Q(x,a)=\mathbb{E}R(x,a)+\gamma \mathbb{E}_{P,\pi}Q(X',A') \\
\mathcal{T} Q(x,a)=\mathbb{E}R(x,a)+\gamma \mathbb{E}_{P}\max_{a'\in \mathcal{A}}Q(X',a')
$$

我们也可以类似地定义DisRL下的转移算子和bellman算子（其中optimal算子不能简单类比，之后会提，此处按下不表）：

$$
\mathcal{P}^\pi Z(x,a):\overset{D}{=}Z(X',A') \\
X' \sim P(\cdot|x,a), \ A'\sim \pi(\cdot|X') \\
\mathcal{T}^\pi Z(x,a):\overset{D}{=}R(x,a)+\gamma \mathcal{P}^\pi Z(x,a)
$$

接下来关于算子/映射$\mathcal{T}^\pi$有一个重要性质：

引理：在$\bar{d}_p$空间中，$\mathcal{T}^\pi:\mathcal{Z}\to \mathcal{Z} $是一个$\gamma$-压缩映射。

由Banach不动点定理，$\mathcal{T}^\pi$必有唯一不动点，又由于$\mathcal{Z}^\pi$满足不动点条件，所以$\mathcal{Z}^\pi$就是那个唯一不动点。

即假设$\mathcal{Z}^\pi$的任意阶矩都是有界的（后面会说明为什么要这个假设），当我们从任意一个$Z_0$开始迭代$Z_{k+1}=\mathcal{T}^\pi Z_k$，那么$\{Z_k\}$最终会收敛到$Z^\pi$。

接下来我们要介绍$\bar{d}_p$是个什么样的度量空间。

### P-Wasserstein 度量空间
