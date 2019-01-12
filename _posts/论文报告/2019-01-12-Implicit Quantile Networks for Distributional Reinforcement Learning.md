---
layout: post
title: Implicit Quantile Networks for Distributional Reinforcement Learning 阅读笔记（二）
date: 2019-01-12 15:02:00
categories: 强化学习
tags: Distributional-RL DQN quantile-regression 
mathjax: true

---

* content
{:toc}

论文题目: [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)

## 简介

上一篇文章（[阅读笔记一](https://siqili1230.github.io/2019/01/03/Implicit-Quantile-Networks-for-Distributional-Reinforcement-Learning/)）主要介绍了分布型强化学习（DisRL）的定义，以及值分布$Z(x,a)$的贝尔曼优化的方式和具体的算法；同时还介绍了P-Wasserstien度量，但却没有由于难以计算没有用上。本篇笔记则基于文章《[Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)》，主要介绍如何利用分位数回归来计算P-Wasserstein度量；和文章《[Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)》，主要是对于值分布引入了关于风险敏感的考量，这是在金融中常用的操作。






## P-W度量的必要性

与KL散度相比，P-W度量在衡量两个分布之间的距离时，能更充分地考虑两个分布之间的形状相似性。具体的例子笔者在博客中的另一篇文章（[Lipschitz Continuity in Model-based Reinforcement Learning 阅读笔记](https://siqili1230.github.io/2018/09/06/Lipschitz-Continuity-in-Model-based-Reinforcement-Learning/)中的“基础概念-W度量”和“论文推理部分——n步转移误差界”）里已讨论过，此处不再赘述。

## 用分位数重构

之前我们对离散点是采用等距采样，但这会导致难以计算P-W距离的，因此在第二篇文章中引入了等分位数采样，即取$\tau_i=\frac{i}{N}$和

$$
Z_\theta(x,a)=\frac{1}{N} \sum_{i=1}^N\delta_{\theta_i(x,a)} 
$$

其中$\delta_x$表示狄拉克函数，即仅在$x$这一点上概率密度不为零的概率密度函数。

为了求解$\theta_i$，根据P-W度量的定义（以$p=1$为例），$\theta$要满足最小化

$$
W_1(Z,Z_\theta)=\sum_{i=1}^N\int_{\tau_{i-1}}^{\tau_i}|F_{Z}^{-1}(\omega)-\theta_i|d\omega
$$

容易解得$\theta_i=F_Z^{-1}(\hat{\tau}_i)=F_Z^{-1}(\frac{\tau_{i-1}+\tau_i}{2})$。

下图给出简单的样例。

![](\images\2019-01-12-Implicit Quantile Networks for Distributional Reinforcement Learning\1.png)

那么如何求解随机变量$Z$的$\hat{\tau}_i$分位数呢？

利用分位数回归的知识，我们可以最小化损失函数

$$
\mathcal{L}_ {QR}(\theta) = \mathbb{E}_ {\hat{Z}\sim Z}[\rho_ \tau(\hat{Z}-\theta)] \\
\mathrm{where} \ \rho_ \tau(u)=u(\tau-\mathbb{I}\{u < 0\})
$$

来求解$Z$的$\tau$分位数。

在笔者之前写的文章：[分位数回归简介](https://siqili1230.github.io/2018/07/24/quantile-regression-1/)中有更为详细的解释和证明，欢迎读者移步。

再实际使用中，由于$\rho_ \tau(u)$在$u=0$处并不光滑，因此略微修改为

$$
\rho_ \tau ^\kappa(u)=|\tau-\mathbb{I}\{u<0\}|\mathcal{L} _\kappa(u) \\
\mathrm{where} \ \mathcal{L}_ \kappa(u)=\begin{cases}
        \frac{1}{2}u^2 ,& \mathrm{if} |u|<\kappa\\
        \kappa(|u|-\frac{1}{2}\kappa) ,& \mathrm{otherwise}&
\end{cases}
$$

记$\Pi_{W_1}Z:=\arg \min_{Z_\theta \in Z_Q} W_1(Z,Z_\theta)$，表示P-W度量空间中用等分位数离散点构造的最优解。

可以证明：$\Pi_{W_1}\mathcal{T}$是一个压缩映射，即

$$
\bar{d}_ \infty(\Pi_{W_1}\mathcal{T}Z_1,\Pi_{W_1}\mathcal{T}Z_2)\leq\gamma\bar{d}_ \infty(Z_1,Z_2)
$$

伪代码如下：

![](\images\2019-01-12-Implicit Quantile Networks for Distributional Reinforcement Learning\algo_1.png)

## 风险敏感

我们先介绍一个情景（阿莱悖论）：考虑一个选择，

A. 50%获得1000元，50%获得0元
B. 100%获得500元

大部分人都会选择后者，即使这两者的期望是一样的（$\mathbb{E}(A)=\mathbb{E}(B)$），这说明了两个选择对当事人的效用是不一样的，若我们记$U(A)$和$U(B)$分别表示对当事人的真实效用，那么应该有$\mathbb{E}(U(A)) < \mathbb{E}(U(B))$。

更一般的，如果$U$是一个凸函数，就有$\mathbb{E}(U(A)) > \mathbb{E}(U(B))$，这时我们称这是一个风险偏好者的效用;凹函数时则是风险厌恶者的效用。

同样的，当我们从$\pi(x)=\arg \max Q(x,a)$变成$\pi(x)=\arg \max \mathbb{E}Z(x,a)$时，也不得不考虑$Z(x,a)$的效用问题，即将上式修改为

$$
\begin{aligned}
\pi(x)&=\arg \max_a \mathbb{E}_{z\sim Z(x,a)}[U(z)] \\
&=\arg \max_a \int_{-\infty}^\infty U(z)p(z) dz \\
&=\arg \max_a \int_{-\infty}^\infty U(z)\frac{\partial}{\partial{z}}F_{Z(x,a)}(z) dz \\
&=\arg \max_a \int_{-\infty}^\infty z\frac{\partial}{\partial{z}}(h \circ F_{Z(x,a)})(z) dz
\end{aligned}
$$

最后一步是在金融的实际应用中常出现的变换，其中$h:[0,1]\to[0,1]$是失真度函数。















