---
layout: post
title: trust region policy optimization 阅读笔记
date: 2019-03-21 14:37:00
categories: 强化学习
tags:  Policy-Gradient Algorithm 
mathjax: true

---

* content
{:toc}

论文题目: [trust region policy optimization](http://proceedings.mlr.press/v37/schulman15.pdf)

## 简介







## 作者介绍

## 背景

考虑马尔可夫过程$(\mathcal{S},\mathcal{A},P,c,\rho_0,\gamma)$，其中$c$是代价函数，$c:\mathcal{S}\to \mathbb{R}$，类似于负回报函数，其余为常规含义。

定义$\pi$为随机策略$\pi:\mathcal{S}\times\mathcal{A}\to[0,1]$，定义$\eta(\pi)$为期望代价：

$$
\eta(\pi)=\mathbb{E}_{s_0,a_0,\cdots}\Big[\sum_{t=0}^\infty\gamma^tc(s_t)\Big]
$$

继而定义常规的$Q_\pi$，$V_\pi$和$A_\pi$。

$$
\begin{aligned}
Q_\pi(s_t,a_t)&=\mathbb{E}_{s_{t+1},a_{t+1},\cdots}\Big[\sum_{l=0}^\infty\gamma^lc(s_{t+l})\Big] \\
V_\pi(s_t)&=\mathbb{E}_{a_{t},s_{t+1},\cdots}\Big[\sum_{l=0}^\infty\gamma^lc(s_{t+l})\Big] \\
A_\pi(s,a)&=Q_\pi(s,a)-V_\pi(s) \\
\end{aligned}
$$

我们考虑一个新策略$\tilde{\pi}$的期望代价$\eta(\tilde{\pi})$与基准策略的期望代价$\eta(\pi)$的关系：

$$
\begin{aligned}
&\eta(\tilde{\pi})=\eta(\pi)+\mathbb{E}_{s_0,a_0,\cdots}\Big[\sum_{t=0}^\infty\gamma^tA_\pi(s_t,a_t)\Big],\mathrm{where} \\
&a_t\sim\tilde{\pi}(a_t\vert s_t)     \tag{1}
\end{aligned}
$$

证明：

$$
\begin{aligned}
\eta(\tilde{\pi})-\eta(\pi)&=\mathbb{E}_{(s_t,a_t)\sim\tilde{\pi}}\Big[\sum_{t=0}^\infty\gamma^tc(s_{t})\Big]-\mathbb{E}_{s_0\sim\rho_0}[V_\pi(s_0)] \\
&=\mathbb{E}_{(s_t,a_t)\sim\tilde{\pi}}\Big[\sum_{t=0}^\infty\gamma^tc(s_{t})-V_\pi(s_0)\Big]\\
&=\mathbb{E}_{(s_t,a_t)\sim\tilde{\pi}}\Big[\sum_{t=0}^\infty\gamma^tc(s_{t})+\sum_{t=1}^\infty\gamma^tV_\pi(s_t)-\sum_{t=0}^\infty\gamma^tV_\pi(s_t)\Big]\\
&=\mathbb{E}_{(s_t,a_t)\sim\tilde{\pi}}\Big[\sum_{t=0}^\infty\gamma^tc(s_{t})+\gamma\sum_{t=0}^\infty\gamma^tV_\pi(s_{t+1})-\sum_{t=0}^\infty\gamma^tV_\pi(s_t)\Big]\\
&=\mathbb{E}_{(s_t,a_t)\sim\tilde{\pi}}\Big[\sum_{t=0}^\infty\gamma^t[c(s_{t})+\gamma V_\pi(s_{t+1})-V_\pi(s_t)]\Big]\\
&=\mathbb{E}_{(s_t,a_t)\sim\tilde{\pi}}\Big[\sum_{t=0}^\infty\gamma^t[Q_\pi(s_t,a_t)-V_\pi(s_t)]\Big]\\
&=\mathbb{E}_{(s_t,a_t)\sim\tilde{\pi}}\Big[\sum_{t=0}^\infty\gamma^t A_\pi(s_t,a_t)\Big]\\
\end{aligned}
$$

为表示方便，记

$$
\rho_\pi(s)=(P_\pi(s_0=s)+\gamma P_\pi(s_1=s)+\gamma^2 P_\pi(s_2=s)+\cdots) 
$$

表示在策略$\pi$下出现状态$s$的折扣概率。其中

$$
\begin{aligned}
&P_\pi(s_t=s)\\
=&\sum_{s'\in\mathcal{S}}\sum_{a'\in\mathcal{A}} P_\pi(s_{t-1}=s')\pi(a_{t-1}=a'\vert s')P(s_t=s\vert a_{t-1}=a',s_{t-1}=s')
\end{aligned}
$$

我们对$(1)$进行重排，之前$(1)$时是以一条完整轨迹为求和单位，现在以任意一个状态动作元组$(s_t,a_t)$为求和单元，不难得到

$$
\eta(\tilde{\pi})=\eta(\pi)+\sum_{s}\rho_{\tilde{\pi}}(s)\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a)\tag{2}
$$

证明：

$$
\begin{aligned}
\eta(\tilde{\pi})&=\eta(\pi)+\sum_{t=0}^\infty\sum_{s,a}P(s_t=s,a_t=a\vert {\tilde{\pi}})\gamma^tA_\pi(s_t=s,a_t=a)\\
&=\eta(\pi)+\sum_{t=0}^\infty\sum_{s,a}P(s_t=s\vert {\tilde{\pi}})\tilde{\pi}(a\vert s)\gamma^tA_\pi(s,a)\\
&=\eta(\pi)+\sum_{t=0}^\infty\sum_{s}P(s_t=s\vert {\tilde{\pi}})\sum_a\tilde{\pi}(a\vert s)\gamma^tA_\pi(s,a)\\
&=\eta(\pi)+\sum_{s}\sum_{t=0}^\infty P(s_t=s\vert {\tilde{\pi}})\gamma^t\sum_a\tilde{\pi}(a\vert s)A_\pi(s,a)\\
&=\eta(\pi)+\sum_{s}\rho_{\tilde{\pi}}(s)\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a) 
\end{aligned}
$$

在$(2)$的基础上，我们可以发现，只要对任意的$s$，都有$\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a) \geq 0$，就可以保证$\eta(\tilde{\pi})\geq\eta(\pi)$，也就是实现了策略更新时，回报递增。这个结论同时也说明了贪心的策略迭代方法$\tilde{\pi}(s)=\arg\max_a A_\pi(s,a)$，由于

$$
\begin{aligned}
\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a) &=1\cdot A_\pi(s,\tilde{\pi}(s))\\
&=\max_a A(s,a) \\
&=\max_a \Big[Q(s,a)-V(s)\Big] \\
&=\max_a Q(s,a)-\mathbb{E}_aQ(s,a)\\
&\geq 0 
\end{aligned}
$$

总是保证回报不下降的。

那么看起来似乎贪心的策略迭代方法是一个很好的方法。
然而，由于估计（对$V,Q$的估计）和近似（对$\mathbb{E}$的计算不是严格穷举的，而是基于采样）的误差，总是存在某些状态$s$使得$\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a) < 0$，因此贪心策略迭代也不是那么好。










