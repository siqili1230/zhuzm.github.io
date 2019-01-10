---
layout: post
title: Implicit Quantile Networks for Distributional Reinforcement Learning 阅读笔记（一）
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

在这篇文章之前，作者还发表了两篇文章（《[A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)》和《[Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)》）——其中第一篇引入了强化学习的分布型视角，第二篇引入了分位数回归的方法来解决分布型强化学习——也是这篇文章的铺垫工作，所以本文将尝试一起介绍三篇文章的工作，帮助读者厘清这三篇文章背后的思路。

### 分布型强化学习(Distributional RL/DisRL)基本定义

我们定义一个随机变量$Z^\pi(x,a)$来表示一条轨迹的累计回报：

$$
Z^\pi(x,a)=\sum_{t=0}^{\infty}\gamma^tR(x_t,a_t)
$$

其中$x_t \sim P(\cdot|x_{t-1},a_{t-1}), \ a_t \sim \pi(\cdot|x_t)$，
$Z$的随机性也来自于转移函数$P$，随机策略$\pi$（即轨迹的随机性）和回报函数$R$。
状态动作值函数则定义为$Q^\pi(x,a)=\mathbb{E}[Z^\pi(x,a)]$，
那么一般的bellman equation就是：

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

我们也可以类似地定义DisRL下的转移算子和贝尔曼算子（其中贝尔曼最优算子不能简单类比，之后会提，此处按下不表）：

$$
\mathcal{P}^\pi Z(x,a):\overset{D}{=}Z(X',A') \\
X' \sim P(\cdot|x,a), \ A'\sim \pi(\cdot|X') \\
\mathcal{T}^\pi Z(x,a):\overset{D}{=}R(x,a)+\gamma \mathcal{P}^\pi Z(x,a)
$$

其中$\mathcal{T}^\pi Z(x,a)$的随机性来自于：
1. 回报函数$R$
2. 转移函数$P$
3. 下一个阶段的值分布$Z(X',A')$

接下来关于算子/映射$\mathcal{T}^\pi$有一个重要性质：

引理：在$\bar{d}_p$空间中，$\mathcal{T}^\pi:\mathcal{Z}\to \mathcal{Z} $是一个$\gamma$-压缩映射。

由Banach不动点定理，$\mathcal{T}^\pi$必有唯一不动点，又由于$\mathcal{Z}^\pi$满足不动点条件，所以$\mathcal{Z}^\pi$就是那个唯一不动点。

即假设$\mathcal{Z}^\pi$的任意阶矩都是有界的（后面会说明为什么要这个假设），当我们从任意一个$Z_0$开始迭代$Z_{k+1}=\mathcal{T}^\pi Z_k$，那么$\{Z_k\}$最终会收敛到$Z^\pi$。

接下来我们要介绍$\bar{d}_p$是个什么样的度量空间。

### P-Wasserstein 度量空间

对于两个累积分布函数（cdf）$F,G$来说，其p-Wasserstein度量（即$d_p$）为：

$$
d_p(F,G)=\inf_{U,V}||U-V||_p \\
=(\int_0^1|F^{-1}(u)-G^{-1}(u)|^pdu)^{1/p}  \ (\mathrm{when} \ p < \infty)
$$

P-W度量的直观含义：如果将一个分布的看成是在空间中的点的采样分布，那么P-W距离就是将一族样本点以最小的代价移动到另一个分布的样本点，代价等于移动的距离乘以该点的密度。

我们考虑带参数的随机变量$Z(x,a)$，则有

$$
\bar{d}_p(Z_1,Z_2):=\sup_{x,a}d_p(Z_1(x,a),Z_2(x,a))
$$

### 基于DisRL的控制（control）

所谓控制，就是找到回报最高的策略。对于一般的RL，就是基于最优值函数$Q^*$的策略。但在DisRL中，尽管$Q^*$唯一，但对应的满足$Q^* = \mathbb{E}Z^*$的$Z^*$却不一定唯一。

我们记$\Pi^*$为最优策略集合，则对应记最优值分布集合为

$$
\mathcal{Z}^*=\{Z^{\pi^*}:\pi^* \in \Pi^*\}
$$

这里要注意，并不是所有的满足$Q^* = \mathbb{E}Z^*$的都是最优的$Z^*$。

基于一个给定的值分布，可以定义其贪心策略$\pi$的集合：

$$
\mathcal{G}_Z:=\{\pi:\sum_a\pi(a|x)\mathbb{E}Z(x,a)=\max_{a'\in\mathcal{A}}\mathbb{E}Z(x,a')\}
$$

回忆贝尔曼最优算子$\mathcal{T}$:

$$
\mathcal{T}Q(x,a)=\mathbb{E}R(x,a)+\gamma\mathbb{E}_{P_{x,x'}^a}\max_{a'\in\mathcal{A}}Q(x',a')
$$

文章引导出了分布型贝尔曼最优算子:

$$
\mathcal{T}Z=\mathcal{T}^\pi Z \ \mathrm{for \ some } \ \pi \in \mathcal{G}_Z
$$

即我们将 "从 '基于当前的$Z$的最优策略集$\mathcal{G}_Z$ ' 挑出来的$\pi$" 组成的算子$\mathcal{T}^\pi$ 称为分布型贝尔曼最优算子。

性质：$\mathbb{E}Z_k \to Q^*$ 以指数级收敛。

虽然收敛速度很美好，但目前还是有三个巨大的困难：
1. 算子$\mathcal{T}$不是一个压缩映射。
2. 不是所有的最优算子都有固定点$Z^* $使得$Z^*=\mathcal{T}Z^*$
3. 即使最优算子有固定点$Z^* $，也不足以保证$\{Z_k\}$能收敛到$\mathcal{Z}^*$。（与前面的$\mathbb{E}Z$的收敛性质并不矛盾，这里指的是分布的收敛。）

### 基于值分布$Z(x,a)$的初代算法

首先将连续的值分布$Z$用离散点来表示：

$$
\{z_i=V_{MIN}+i\Delta z:0\leq i <N\}, \ \Delta z:=\frac{V_{MAX}-V_{MIN}}{N-1}
$$

而每个值的出现概率则用参数模型$\theta:\mathcal{X}\times \mathcal{A} \to \mathbb{R}^N$来表示：

$$
Z_{\theta}(x,a)=z_i \ \ w.p. \  \ p(z_i|x,a)=p_i(x,a):=\frac{e^{\theta_i(x,a)}}{\sum_j e^{\theta_j(x,a)}}
$$

不过进行贝尔曼迭代的时候，$z_{t+1,i}=\mathcal{T}z_{t,i}=r+\gamma z_{t,i}, \ \forall \ i$ 不一定落在上一步的几个划分点上，因此为了统一表示，文章在每一次迭代时都维持划分点在最初的$N$个划分点$z_i$上，方法是调整$p_i(x,a)$。 

$\mathcal{T}Z_{t,\theta}(x,a)$在各个划分点$z_i$上的概率分布按如下方法计算：

$$
(\Phi \mathcal{T}Z_\theta(x,a))_i=\sum_{j=0}^{N-1}[1-\frac{|[\mathcal{T}z_j]_{V_{MIN}}^{V_{MAX}}-z_i|}{\Delta z}]_0^1 p_j(x',\pi(x'))
$$

其中$[\cdot]_a^b$表示将数值约束在$[a,b]$之间。

最终采样损失为：

$$
\mathcal{L}_{x,a}(\theta)=D_{KL}(\Phi \mathcal{T}Z_\theta(x,a)||Z_\theta(x,a))
$$

算法的伪代码如下：

![](images\2019-01-03-Implicit Quantile Networks for Distributional Reinforcement Learning\algo_1.png)

### 小结

通过上一张的算法图可以发现，我们只有在需要$Q(s,a)$时才通过对$Z(s,a)$求期望来获得数值，而在$(s,a) \to (s',a')$的采样过程中时，都是以$Z$分布形式传递的——这样的方式能极大地保留回报的信息。

值得注意的是，在第一篇文章中用的是KL散度而不是P-W度量，这是因为P-W度量非常难算。

接下来的阅读报告（二）介绍的文章就将描述如何解决P-W度量的计算问题。

