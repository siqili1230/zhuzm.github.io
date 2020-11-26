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
然而，由于估计（对$V,Q$的估计）和近似（对$\mathbb{E}$的计算不是严格穷举的，而是基于采样）的误差，总是存在某些状态$s$使得$\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a) < 0$，因此贪心策略迭代也不能保证回报不下降。

另一方面，$(2)$对于$\rho_{\tilde{\pi}}(s)$的依赖也导致了基于$(2)$直接优化/求梯度是一件非常困难的事（因为我们没有$\tilde{\pi}$的轨迹信息）。因此我们考虑第一步近似操作，把$\rho_{\tilde{\pi}}(s)$替换为$\rho_{\pi}(s)$：

$$
L_{\pi}({\tilde{\pi}})=\eta(\pi)+\sum_{s}\rho_{\pi}(s)\sum_{a}\tilde{\pi}(a\vert s)A_\pi(s,a) \tag{3}
$$

此时我们需要注意到，（当我们把策略$\pi$看作参数化策略$\pi_\theta$时）$L_{\pi_0}(\cdot)$和$\eta(\cdot)$在$\pi_0$处的一阶泰勒展开是一样的，即具有一阶近似性：

$$
\begin{aligned}
L_{\pi_{\theta_0}}(\pi_{\theta_0})&=\eta(\pi_{\theta_0}) \\
\nabla_\theta L_{\pi_{\theta_0}}(\pi_{\theta})\vert_{\theta=\theta_0}&=\nabla_\theta \eta(\pi_{\theta})\vert_{\theta=\theta_0} \tag{4}
\end{aligned}
$$

这一性质暗示了，当我们用足够小的步长去执行$\pi_{\theta_0}\to\tilde{\pi}$时，如果$L_{\pi_{\theta_\mathrm{old}}}(\pi_{\theta_\mathrm{old}+\epsilon_1})>L_{\pi_{\theta_\mathrm{old}}}(\pi_{\theta_\mathrm{old}+\epsilon_2})$，那么就有$\eta(\pi_{\theta_\mathrm{old}+\epsilon_1})>\eta(\pi_{\theta_\mathrm{old}+\epsilon_2})$（一般情况下每次更新了策略，都会用新策略替换掉旧策略，即上式中$\epsilon_2$为零向量的情况）。

简单来说，上述结论为：在$\theta_0$的一个领域内，$L_{\pi_{\theta_0}}(\cdot)$与$\eta(\cdot)$同增同减。

但我们并不知道多小的领域才满足这个性质，或者说，$L_\pi(\cdot)$与$\eta(\cdot)$之间的差距与步长有什么关系。这里直接给出研究结论：

$$
\begin{aligned}
\mathrm{Note}& \ \pi^*=\arg\max_{\pi'}L_{\pi_\mathrm{old}}(\pi')\\
\mathrm{let} \ \ \ \ & \ \pi_{\mathrm{new}}(a\vert s)=(1-\alpha)\pi_{\mathrm{old}}(a\vert s)+\alpha \pi^*(a\vert s)\\
\mathrm{get}\ \ \ & \ \eta(\pi_\mathrm{new})\geq L_{\pi_\mathrm{old}}(\pi_\mathrm{new})-\frac{2\epsilon\gamma}{(1-\gamma)^2}\alpha^2 \\
&\mathrm{where} \ \epsilon=\max_s \vert \mathbb{E}_ {a\sim\pi'} [A_\pi(s,a)] \vert \tag{6}
\end{aligned}
$$

这里$\alpha$本质上是对$\pi_{\mathrm{new}}$和$\pi_{\mathrm{old}}$之间差距的刻画，即公式(6)是在说，$\pi_{\mathrm{new}}$的真实价值$\eta(\pi_{\mathrm{new}})$与作者给出的近似估计值$L_{\pi_\mathrm{old}}(\pi_\mathrm{new})$之间的差距，可以由$\alpha$控制住。虽然这里只给出$\eta$的下界，但是由于$\eta$是肯定小于$L_\pi$的（根据公式(3)），所以可以说差距已经被控制住了。但这个结论只适用于保守型（线性组合法）的策略更新规则，在实际中难以应用。

### 在一般随机策略中保持回报单调提升

先写结论：作者证明了$(6)$的结论稍作修改，在一般随机策略中也成立——只要修改步长$\alpha$（某种意义上也是邻域大小）为$\pi$和$\tilde{\pi}$的距离，再修正常数$\epsilon$即可。

首先，策略的距离选择上，作者采用全变分散度：$D_{TV}(p\vert \vert q)=\frac{1}{2}\sum_i \vert p_i-q_i\vert$，其中$p,q$是离散随机变量的分布。

$$
\begin{aligned}
D_{TV}^{\max}(\pi,\tilde{\pi})=\max_s D_{TV}(\pi(\cdot\vert s)\vert \vert \tilde{\pi}(\cdot\vert s)) \tag{7}
\end{aligned}
$$

![](\images\2019-03-21-trust region policy optimization\thm_1.png)

利用全变分散度和KL散度之间的关系：$D_{TV}(p\vert\vert q)^2\leq D_{KL}(p\vert\vert q)$，令$D_{KL}^{max}(\pi,\tilde{\pi})=\max_s D_{KL}(\pi(\cdot,s)\vert\vert \tilde{\pi}(\cdot,s))$，可以进一步修改定理一为：

![](\images\2019-03-21-trust region policy optimization\formal_9.png)

至此我们得到了一般随机策略下的$\eta$与$L_\pi$的关系，因此只要用公式(9)的右式代替$\eta$执行策略迭代算法，即算法一：

![](\images\2019-03-21-trust region policy optimization\algo_1.png)

容易证明算法一得到的策略序列$\{\pi_i\}$满足$\eta(\pi_i)\leq\eta(\pi_{i+1})$：

记$M_i(\pi)=L_{\pi_i}(\pi)-CD^{\mathrm{max}}_{KL}(\pi_i,\pi)$，那么有$\eta(\pi_i)=M_i(\pi_i), \ \eta(\pi_{i+1})\geq M_{i}(\pi_{i+1})$，因此$\eta(\pi_{i+1})-\eta(\pi_i)\geq M_i(\pi_{i+1})-M_i(\pi_i)\geq 0$。

PS:这一结论其实从直觉上是不容易想到的，因为大部分人第一反应是:两个序列$\{a_i\}$和$\{b_i\}$满足$a_i\geq b_i$且$b_i\leq b_{i+1}$，可是却不一定有$a_i\leq a_{i+1}$。其中的关键之处在于$M_i(\cdot)$与$M_{i+1}(\cdot)$是不一样的。实际上，当我们得到满足$\eta(\pi_i)\geq M_{i-1}(\pi_i)$的$\pi_i$后，就利用刚得到的$\pi_i$把$M_{i-1}$更换为$M_i$，使得$M_i(\pi_i)=\eta(\pi_i)$，即将小于$\eta(\pi_i)$的$M_{i-1}(\pi_i)$提升到等于$\eta(\pi_i)$的$M_{i}(\pi_i)$了，再在此基础上进一步更新，所以得到的新的$M_i(\pi_{i+1})\geq M_i(\pi_i)=\eta(\pi_i)$就是很正常的了。



















