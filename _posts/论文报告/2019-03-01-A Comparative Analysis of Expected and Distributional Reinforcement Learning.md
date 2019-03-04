---
layout: post
title: A Comparative Analysis of Expected and Distributional Reinforcement Learning 阅读笔记（二）
date: 2019-03-01 14:37:00
categories: 强化学习
tags: Distributional-RL 强化学习理论 AAAI2019
mathjax: true

---

* content
{:toc}

论文题目: [A Comparative Analysis of Expected and Distributional Reinforcement Learning](https://arxiv.org/pdf/1901.11084.pdf)

## 简介

这篇文章是分布型强化学习(Distributional RL)研究方向中综述型的一篇论文。主要贡献为从理论和实验两个角度分析了ERL(Expected RL)与DRL(Distributional RL)的异同，其中分类讨论了表格式模型、线性值函数近似和非线性值函数近似三种情景，结论为前两者情况下ERL与DRL没有区别，最后一种情况下有差异。





## 作者介绍

**第一作者：** Clare Lyle，牛津大学CS PhD在读，本科为加拿大麦吉尔大学(McGill University)数学计算机双学位，这篇文章是她在Google Brain暑期实习期间发表。她[个人博客](https://clarelyle.com/index.html)中有一篇博文对这篇文章有补充说明。

**第二作者：** Pablo Samuel Castro，麦吉尔大学CS PhD，在Google Brain工作，今年才开始有DRL相关论文发表，另一篇相关工作《Distributional reinforcement learning with linear function approximation 》在arxiv上可查到。

**第三作者：** Marc G. Bellemare，Google Brain的研究科学家(Reasearch Scientist), 麦吉尔大学的客座教授(Adjunct Professor), Canada CIFAR AI Chair. 他于2017年发表了《A distributional perspective on reinforcement learning》，是DRL的奠基人之一。

## DRL 发展历程

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\development.png)

可以看到Bellemare(红)、Dabney(绿)、Munos(蓝)最早于2017年发表相关文章，是DRL的奠基人；之后DRL的相关文章都有这三人的身影。（Dabney与Munos为DeepMind员工）


## DRL 背景介绍

ERL是将奖赏看成一个标量$Q(x,a)$，而DRL是将奖赏看成分布$Z(x,a)$，满足$Q(x,a)=\mathbb{E}Z(x,a)=\mathbb{E}[\sum_{t=0}^\infty \gamma^tR(x_t,a_t)]$。然后以分布的形式进行迭代，例如
$$
Z_{i+1}(x_t,a_t)=Z_i(x_t,a_t)+\alpha(R_t+\gamma Z_i(x_{t+1},a_{t+1})-Z_i(x_t,a_t))
$$

更多细节可以参考笔者之前的另一篇关于DRL的[阅读笔记](https://siqili1230.github.io/2019/01/03/Implicit-Quantile-Networks-for-Distributional-Reinforcement-Learning/)。

那么以分布的形式研究RL有什么意义呢？

目前普遍认为（并非证实）有以下三个方面的意义：
1. 降低方差：以分布的形式预测未来的回报，被认为能降低预测回报的方差。
2. 更好的优化表现：分布或许能作为一个更好更稳定的优化目标，在某些神经网络中或许能有正则化的效果。
3. 辅助任务：分布能提供更丰富的预测信息用于学习。

基于分布进行研究的必要工具：

#### **1 分布的距离**

在《Implicit Quantile Networks for Distributional Reinforcement Learning》一文中提到Wasserstein度量是一个很好的分布度量，但其在实际使用中难以计算和分析，因此本文中作者采用Cramer度量：

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\def_1.png)

#### **2 分布的表示**

用点支撑集表示，记$\mathbf{z}=\{z_1,\cdots,z_k\}$，其中$z_1 \leq z_2 \leq \cdots \leq z_k$，则一个用支撑集$\mathbf{z}$ 表示的分布$P$可以写成：

$$
P\in Z_z:=\{\sum_{i=1}^{k}\alpha_i\delta_{z_i}:\alpha_i\geq 0,\sum_{i=1}^{k}\alpha_i=1\}
$$

那么Cramer度量可以重新写为：

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\cramer_1.png)

## Cramer投射

为了解决迭代过程中支撑集变化的问题，即$Z_{i+1}(x,a)$与$Z_i(x,a)$的支撑集不一致，提出了Cramer Projection方法:

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\def_2.png)

另外Cramer Projection有一个性质：

$$
\mathbb{E}[\Pi_C(P)]=\mathbb{E}[P]
$$

即关于期望运算不变。

## 期望等价性(expectation-equivalent)

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\update_1.png)

首先，ERL与DRL的性能比较时，要采用成对的算法，比如ERL用bellman算子，DRL也要用相似的分布型的bellman算子。

其次，这里定义期望等价性。

记

$$
Z\overset{\mathbb{E}}= Q \Longleftrightarrow \mathbb{E}[Z(x,a)]=Q(x,a)  \ \ \ \forall (x,a)\in \mathcal{X}\times \mathcal{A}
$$

我们称两种更新规则$U_E$和$U_D$是期望等价的，如果有下式满足：

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\update_2.png)

在满足期望等价性的成对更新规则下，以下三点关于DRL可能存在的假设均被推翻：

1. DRL可以降低方差。
        因为$𝑉ar[\mathbb{E} Z_t (x,a)]=Var[Q_𝑡 (𝑥,𝑎)], ∀ (𝑥,𝑎)$ 
2. DRL有利于策略迭代。
        因为贪心策略基于$ \arg⁡max[⁡𝑄_𝑡 (𝑥,⋅)]  $和$\arg⁡max⁡[\mathbb{E} 𝑍_𝑡 (𝑥,⋅)]$,DRL在所有基于期望做决策的策略中都没有帮助。
3. DRL在值函数近似中更为稳定。
        下文会说明，在线性值函数近似情况下，DRL没有帮助。


## 理论分析

### tabular model

1. model-based

基于模型的情况，即可以利用到状态转移的信息。

首先考虑$Z$为一般的分布，即支撑集是实数域。

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_2.png) 

之后考虑$Z$为Dirac函数，即点支撑集。

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_3.png) 

2. sample-based

基于采样的情况，即只能根据采样得到的固定轨迹迭代优化。

依然先考虑$Z$为一般的分布。

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_4.png) 

之后考虑$Z$为Dirac函数。

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_5.png) 

**semi-gradient**

之前都是基于bellman迭代的，现在考虑半梯度方法。

首先定义Cramer度量下的梯度，可以分为对CDF求梯度和对PDF求梯度。

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\grad_1.png) 

以下分别是用CDF和PDF求梯度的结论。

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_6.png) 

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_7.png) 

### 函数线性近似

首先用线性函数近似分布函数，与ERL中的$Q$值对比如下：

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_8_1.png) 

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_8_2.png) 

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_8.png) 

### 函数非线性近似

![](\images\2019-03-01-A Comparative Analysis of Expected and Distributional Reinforcement Learning\prop_9.png) 

## 小结

虽然结论非常简洁明了，证明过程也基本没有问题，但这篇文章最主要的问题在于方向有偏颇。私以为既然DRL的特点在于用分布的角度看待回报，那就不应该用回报的期望去做决策，而是应该充分利用回报的分布来做决策，这才是DRL领域最重要的出路。





