---
layout: post
title: A Unifide View of Entropy-Regularized Markov Decision Processes阅读笔记
date: 2018-12-06 22:11:00
categories: 强化学习
tags: NIPS2018 强化学习理论 收敛性
mathjax: true

---

* content
{:toc}

论文题目: [A Unified View of Entropy-Regularized Markov Decision Processes](https://arxiv.org/pdf/1705.07798.pdf)

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

由于在最优点处满足KKT约束，故有
![](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\equa_3.png)
这一步得到了在最优点处正则化子的梯度约束。

另一方面，我们具体地引入两类正则化函数：
![](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\regularizer_s.png)
![](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\regularizer_c.png)

同时计算两种熵的bregman divergence：
![](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\regularizer_s_1.png)
![](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\regularizer_c_1.png)

其中Bregman散度的定义如下：
![](\images\2018-12-06-A Unified View of Entropy-Regularized Markov Decision Processes\bregman_def.png)

接下来我们基于正则化子$R_s,R_c$分别讨论：
其中$$A(x,a)=r(x,a)+\sum_y P(y|x,a)V(y)-V(x)$$
1. 先讨论$R_s$的情况。我们首先列出结论：
    1. 最优分布$\mu^*_\eta$满足($A^*$对应$V^*$)：$$\mu^*_\eta(x,a) \varpropto \mu'(x,a)e^{\eta A^*_\eta(x,a)}$$
    2. 对偶函数如下：
    $$
    g(V)=\frac{1}{\eta}\log\sum_{x,a}\mu'(x,a)e^{\eta A(x,a)}
    $$
2. 对于$R_s$的情况，有类似的结论:
    1. 最优策略$\pi^*_\eta$满足：$$\pi^*_\eta(x,a) \varpropto \pi_{\mu'}(x,a)e^{\eta A^*_\eta(x,a)}$$
    2. 对偶问题如下：
    $$
    \min_{\lambda \in \mathbb{R}} \\
    subject \ to \ V(x)=\frac{1}{\eta}\log\sum_{x,a}\pi_{\mu'}(x,a)\exp(\eta(r(x,a)-\lambda+\sum_y P(y|x,a)V(y)))
    $$

因为这一步我在看附录的证明时遇到了问题过不去，所以此处特地列出我的困惑之处。
以$R_s$的对偶函数证明为例：
首先基于$R_s$的定义求梯度，有：
$$
\frac{\partial R(\mu)}{\partial\mu(x,a)}=\log\frac{\mu(x,a)}{\mu'(x,a)}+1
$$
结合上面求过的最优点处的梯度约束：
$$
\frac{\partial R(\mu)}{\partial\mu(x,a)}|_{\mu^*(x,a)}=\eta(A(x,a)-\lambda+\varphi(x,a))
$$
因此
$$
\mu^*_\eta(x,a)=\mu'(x,a)\exp(\eta(A(x,a)-\lambda+\varphi(x,a))-1)
$$
同时有KKT的约束，有：
$$
1=\sum_{x,a}\mu^*(x,a)
=\sum_{x,a}\mu'(x,a)\exp(\eta(A(x,a)-\lambda+\varphi(x,a))-1) \\
\lambda=\frac{1}{\eta}(\log\sum_{x,a}\mu'(x,a) \exp(\eta A(x,a))-1) \\
$$
将上式的$\lambda$代入拉格朗日函数可得：
$$
\begin{aligned}
\mathcal{L}(\mu^*_\eta;V,\lambda)&=\sum_{x,a}\mu^*_\eta(x,a)(A(x,a)-\lambda-\frac{1}{\eta}\log\frac{\mu^*_\eta(x,a)}{\mu'(x,a)})+\lambda \\
&=\frac{1}{\eta}+\lambda \ \ \ (*)\\
&=\frac{1}{\eta}\log\sum_{x,a}\mu'(x,a)\exp(\eta A(x,a))
\end{aligned}
$$
打星号$*$所在行的等号存疑。因为
$$
\begin{aligned}
&\sum_{x,a}\mu^*_\eta(x,a)(A(x,a)-\lambda-\frac{1}{\eta}\log\frac{\mu^*_\eta(x,a)}{\mu'(x,a)})+\lambda \\
=&\sum_{x,a}\mu^*_\eta(x,a)A(x,a)-\sum_{x,a}\mu^*_\eta(x,a)(\frac{1}{\eta}\log(\frac{\mu^*_\eta(x,a)}{\mu'_\eta(x,a)}\sum_{x',a'}\exp(\eta A(x',a'))))+\frac{1}{\eta}+\lambda \\
\end{aligned}
$$
若要证明上式等于$\frac{1}{\eta}+\lambda$，由于最优点$\mu^*_\eta$的任意性，可以通过证明
$$
\exp(\eta A(x,a))=\frac{\mu^*_\eta(x,a)}{\mu'_\eta(x,a)}\sum_{x',a'}\exp(\eta A(x',a')) \ \ \forall (x,a)
$$
但（我认为）上式显然不成立，或者说容易构造反例。
在$R_c$的证明中也有类似的困惑，此处不再赘述。
## 算法
此处我们考虑两类基于$\mu$的迭代算法：Mirror Decent 和 Dual Averaging。
### Mirror Decent
$$
\mu_{k+1}=\arg\max \{\rho(\mu)-\frac{1}{\eta}D_R(\mu || \mu_k)\}
$$
Mirror Decent的做法是固定$\eta$然后每次基于旧的$\mu_k$寻找一个最佳的$\mu_{k+1}$
下面将说明TRPO其实是这一类算法的变体。

## 实验

## 小结