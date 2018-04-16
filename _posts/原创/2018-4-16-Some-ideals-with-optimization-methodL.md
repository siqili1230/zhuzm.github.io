---
layout: post
title: 一个对优化算法等价于滑动平均的思考
date: 2018-04-16 22:43:00
categories: 深度学习
tags: optimization
mathjax: true
---

* content
{:toc}


# 一个关于优化算法等价于时间序列的思考

**FengHZ‘s Blog首发原创**

## 起因

Tensorflow官网上有一个Trick是这样的：

>当训练参数到一定的步数之后，可以考虑采用
>$$
>\hat{w_t}=mw_{t}+(1-m)w_{t-1}
>$$
>来让参数收敛到一个更好的效果，但是这个有时候会有作用，有时候并没有作用

我觉得这个策略等价于调整momentum与学习率，特此给出证明与分析




## momentum方法的等价形式

假设我们的参数初始化均值为0，采用momentum方法进行优化，momentum系数为$\gamma$，学习率为$\eta$。那么优化过程可以写成如下形式：
$$
v_0=0;w_0=0\\
v_1=(1-\gamma)v_0+\gamma \nabla_1J(\theta_1)=\gamma \nabla_1J(\theta_1)\\
w_1=-\eta v_1\\
v_2=(1-\gamma)v_1+\gamma\nabla_2J(\theta_2)\\
w_2=w_1-\eta v_2\\
v_3=(1-\gamma)v_2+\gamma\nabla_3J(\theta_3)\\
w_3=w_2-\eta v_3=w_2-(1-\gamma)\eta v_2-\eta\gamma\nabla_3J(\theta_3)\\
\because -v_2=\frac{w_2-w_1}{\eta}\\
w_3=w_2+(1-\gamma)(w_2-w_1)-\eta\gamma\nabla_3J(\theta_3)\\
=(2-\gamma)w_2-(1-\gamma)w_1-\eta\gamma\nabla_3J(\theta_3)\\
w_n=(1+(1-\gamma))w_{n-1}-(1-\gamma)w_{n-2}-\eta\gamma\nabla_nJ(\theta_n)\\
$$
这其实就是一个自回归（或者说关于时间序列$(\nabla_1J(\theta_1),\nabla_2J(\theta_2),...,\nabla_nJ(\theta_n))$的这样一个滑动平均）。

如果我们采用策略
$$
\hat{w_t}=mw_{t}+(1-m)w_{t-1}
$$
那么这等价于
$$
\hat{w_t}=mw_{t}+(1-m)w_{t-1}\\
=m(2-\gamma)w_{t-1}-(1-\gamma)w_{t-2}-\eta\gamma\nabla_tJ(\theta_t))+(1-m)w_{t-1}\\
=(1+m-m\gamma)w_{t-1}-m(1-\gamma)w_{t-2}-m\eta\gamma\nabla_tJ(\theta_t)\\
=(1+m(1-\gamma))w_{t-1}-m(1-\gamma)w_{t-2}-\eta(m\gamma)\nabla_tJ(\theta_t)
$$
这个本质还是一个滑动平均，但是我们可以看到这里相当于对所有参数都进行了弱化的过程。此时$(1-\gamma)+\gamma$变成了$m(1-\gamma)+m\gamma=m$，相当于加权参数变了，其实也变相相当于学习率的减弱。因此会出现这个trick。

## 一些关于时间序列的思考

在整个优化算法的过程中，随机梯度下降是一个很重要的思想。它用batch的梯度平均作为真实的梯度。如果我们把梯度看成是一组服从某种分布的时间序列参数呢？比如$\nabla_{n}J(\theta_n)-N(u_t,\sigma_t)$，那么我们能否研究出梯度的一些有用的信息，比如对梯度进行建模与评估呢？

用时间序列的角度分析整个梯度下降法我觉得是很有趣的。





