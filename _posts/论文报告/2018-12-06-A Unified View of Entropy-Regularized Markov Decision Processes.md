---
layout: post
title: A Unified View of Entropy-Regularized
Markov Decision Processes阅读笔记
date: 2018-12-06 22:11:00
categories: 强化学习
tags: NIPS2018 强化学习理论 收敛性
mathjax: true

---

* content
{:toc}

论文题目: [A Unified View of Entropy-Regularized
Markov Decision Processes](https://arxiv.org/pdf/1705.07798.pdf)

## 简介

本文的主要工作是从MDP平均回报的最优化问题角度看待目前强化学习的主流算法，并通过一些近似手段给出了这些算法的收敛性的证明（存疑）。

## 背景

### 优化问题与对偶问题
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
\min_\lambda \lambda \\
subject \ to \ \lambda + V(x)-\sum_y P(y|x,a)V(y) \geq r(x,a) \ \forall (x,a)
$$
PS:
1. 记原问题$\max_x f(x)$的拉格朗日函数为$\mathcal{L}(x;\lambda)$，则原问题等价于求解$\max_x \min_\lambda \mathcal{L}(x;\lambda)$，其对偶函数为$g(\lambda)=\max_x \mathcal{L}(x;\lambda)$，其对偶问题为$\min_\lambda \max_x \mathcal{L}(x;\lambda)$。
2. 弱对偶性：$\min_\lambda \max_x \mathcal{L}(x;\lambda)\geq \max_x \min_\lambda \mathcal{L}(x;\lambda)$，强对偶性则是取等号，本文讨论的内容均满足强对偶性。

### 正则化子的选择
对于实际问题（强化学习）来说，往往会出现
1. 过拟合：由于采样数据不够
2. 欠缺探索：由于采样到的数据太差

这两个问题，导致学不到好的策略。
为了解决这些问题，目前主流的做法是正则化。
具体的做法分为下面两类：
1. 松弛化目标函数
![soft_1](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\soft_1.png)
![soft_2](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\soft_2.png)
2. 添加正则化子
![regular_1](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\regular_1.png)
![regular_2](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\regular_2.png)

后面文章将说明这两类做法本质上是一致的！
现在我们从添加正则化子的角度切入。
将目标函数更改为
$$\max_\mu \widetilde{\rho}(\mu)=\max_\mu [\rho(\mu)-\frac{1}{\eta}R(\mu)]$$
则拉格朗日函数更改为
![](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\equa_2.png)

## 算法

## 实验

## 小结