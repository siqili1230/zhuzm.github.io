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

我们对$(1)$进行重排，以$(s_t,a_t)$的出现概率为求和单元，不难得到

$$
\eta(\tilde{\pi})=\eta(\pi)+\sum_{s}\rho_{\tilde{\pi}}(s)\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a)
$$

证明：

$$
\begin{aligned}
\eta(\tilde{\pi})&=\eta(\pi)+\sum_{t=0}^\infty\sum_{s,a}P_{\tilde{\pi}}(s_t=s,a_t=a)\gamma^tA_\pi(s_t=s,a_t=a)\\
&=\eta(\pi)+\sum_{t=0}^\infty\sum_{s,a}P_{\tilde{\pi}}(s_t=s)\tilde{\pi}(a\vert s)\gamma^tA_\pi(s,a)\\
&=\eta(\pi)+\sum_{t=0}^\infty\sum_{s}P_{\tilde{\pi}}(s_t=s)\sum_a\tilde{\pi}(a\vert s)\gamma^tA_\pi(s,a)\\
&=\eta(\pi)+\sum_{s}\sum_{t=0}^\infty P_{\tilde{\pi}}(s_t=s)\gamma^t\sum_a\tilde{\pi}(a\vert s)A_\pi(s,a)\\
&=\eta(\pi)+\sum_{s}\rho_{\tilde{\pi}}(s)\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a)
\end{aligned}
$$






