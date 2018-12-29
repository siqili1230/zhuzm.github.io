---
layout: post
title: Universal Planning Networks 阅读笔记
date: 2018-09-08 22:11:00
categories: 强化学习
tags: ICML2018 状态抽象
mathjax: true

---

* content
{:toc}

论文题目: [Universal Planning Networks](https://arxiv.org/pdf/1804.00645v1.pdf)

## 论文简介与笔记大纲

本文的主要工作是给出了一个新的抽象表示的学习方式。




## 背景介绍
抽象表示（abstract representation）是解决高维数据的一种重要方法。在强化学习中，我们经常遇到一些基于像素级图像信息输入的环境，比如atari上的游戏。我们希望通过抽象表示压缩数据的维度，并提取关键信息，从而降低计算量，实现更高效的学习。

一般的，基于强化学习环境$<S,A,R,T,\gamma>$，我们希望能够得到一个状态抽象化函数 $f(s_t)=\hat{s}_t$ ，其中 $\hat{s}_t$ 是一个低维数据；同时得到抽象状态 $\hat{s}\_t$ 的转移函数 $\hat{T}(\hat{s}\_{t+1})$ 。这样，我们就可以在低维空间中使用强化学习的算法。

## 算法内容

算法分为两个部分，第一部分是训练GDP函数。GDP函数的输入为起始原始状态$o_t$，目标原始状态$o_g$，输出为一条实现从$o_t$到$o_g$的动作序列$\hat{a}_{t:t+T}$。

训练过程为：在固定抽象化函数$f_\phi$和抽象空间的转移函数$g_\theta$的情况下，获得一条动作序列$\hat{a}_{t:t+T}$，以目标抽象状态$x_g$和抽象空间的转移结果$\hat{x}_g$的距离为损失函数优化动作序列，具体算法如下：

![algo-1](\images\2018-9-8-Universal-Planning-Networks\2018-9-8-Universal-Planning-Networks-algo-1.png)

第二部分是在固定GDP函数的情况下，优化抽象化函数$f_\phi$和抽象空间的转移函数$g_\theta$。训练过程为：输入一条专家轨迹，以专家的动作序列$a_{t:t+T}$和通过GDP跑出来的动作序列 $\hat{a}\_{t:t+T}$ 的差值作为损失函数，同时优化 $f_\phi$和 $g_\theta$，具体算法如下：

![algo-2](\images\2018-9-8-Universal-Planning-Networks\2018-9-8-Universal-Planning-Networks-algo-2.png)

## 同类型算法
### RIL(reactive imitation learning)
RIL采用的结构为
$$
f_\phi(o_t,o_g)=x_t , \ \  
m(x_t)=\hat{a}_t
$$

### AIL(auto-regressive imitation learning)
AIL采用的结构为
$$
f_\phi(o_t,o_g)=x_t ,\ \ g_\theta(x_t,h_{t-1})=h_t, \ \ m(h_t)=\hat{a}_t
$$

### 比较实验
![experience-1](\images\2018-9-8-Universal-Planning-Networks\2018-9-8-Universal-Planning-Networks-exp-1.png)

## 小结
这篇文章是暑期研讨班陈雄辉同学所介绍的，因为最近在预测学习之余又在抽象表示上开展研究，所以近期关注了一些有关抽象表示的文章，希望能带来一些灵感（排掉一些错误方向）。


