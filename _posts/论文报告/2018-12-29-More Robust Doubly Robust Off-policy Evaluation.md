---
layout: post
title: More Robust Doubly Robust Off-policy Evaluation 阅读笔记
date: 2018-12-29 13:12:00
categories: 强化学习
tags: 强化学习理论 off-policy-evaluation robust 
mathjax: true

---

* content
{:toc}

论文题目: [More Robust Doubly Robust Off-policy Evaluation](https://arxiv.org/pdf/1802.03493.pdf)

## 简介

该文的主要工作是对于OPE(off-policy evaluation)问题，提出了一个更加稳定（robust）的评估器（estimator）。






## 背景

对于一条轨迹

$$
\xi=(x_0,a_0,r_0,\cdots,x_{T-1},r_{T-1},x_T)
$$

定义累计回报$R_{0:T-1}(\xi)=\sum_{t=0}^{T-1}\gamma^tr_t$，其中

$$
x_0 \sim P_0 \\
a_t \sim \pi(\cdot|x_t) \\
x_{t+1}\sim P(\cdot|x_t,a_t) \\
r_t \sim P_r(\cdot|x_t,a_t)
$$

将上面的分布一起定义为$P^\pi_\xi$，有

$\rho_T^\pi = \mathbb{E}_{\xi \sim P^\pi_\xi}[R_{0:T-1}(\xi)]$

OPE问题是说给定一组来自于策略$\pi_b$的采样轨迹$\mathcal{D}=\{\xi^{(i)}\}_{i=1}^{n}$，如何用这组轨迹去评估策略$\pi_e$。如果下式MSE不大，我们认为一个评估器$\hat{\rho}^{\pi_e}$是好的：

$$
\mathrm{MES}(\rho^{\pi_e},\hat{\rho}^{\pi_e}) \overset{\triangle}{=} \mathbb{E}_{P^{\pi_b}_\xi}[(\rho^{\pi_e}-\hat{\rho}^{\pi_e})^2]
$$

此处$\rho^{\pi_e}$是一个与期望无关的$\pi_e$的真实值。

## 算法

关于该文研究的稳定评估器课题，最初是采用DM(Direct Method)方法或IS(Importance Sampling)方法，后来有人结合了以上两者提出了DR(Doubly Robust)方法，该文则是在DR基础上提出了MRDR(More Robust Doubly Robust
)方法。

笔者将按算法发展顺序依次介绍这些算法。

### Direct Method

DM方法就是先学习一个模型：$\hat{Q}^{\pi_e}$，再用该模型来衡量策略。

$$
\hat{\rho}_{DM}^{\pi_e}=\frac{1}{n}\sum_{i=1}^n \sum_{a\in \mathcal{A}} \pi_e(a|x_0^{(i)})\hat{Q}^{\pi_e}(x_0^{(i)},a;\beta_n^*)
$$

求解$\hat{Q}$的参数$\beta$的方法就是直接最小化MES：

$$
\beta^* \in \arg \min_{\beta \in \mathbb{R}^{\kappa}} \mathbb{E}_{(x,a)\sim \mu_{\pi_e}}[(Q^{\pi_e}(x,a)-\hat{Q}^{\pi_e}(x,a;\beta))^2]
$$

DM的优点：方差小，偏差小。

但考虑到采样数据来自于$\pi_b$，所以我们不知道$\pi_e$下真实的Q值$Q^{\pi_e}$，因此引入重要性采样比$\omega_{t_1:t_2}=\Pi_{\tau=t_1}^{t_2}\frac{\pi_e(a_\tau|x_\tau)}{\pi_b(a_\tau|x_\tau)}$，不过由于我们不知道$\pi_b$，所以还需要用$\hat{\omega}_{t_1:t_2}=\Pi_{\tau=t_1}^{t_2}\frac{\pi_e(a_\tau|x_\tau)}{\hat{\pi}_b(a_\tau|x_\tau)}$来代替，其中$\hat{\pi}_b$也是基于采样数据计算期望得到。

引入重要性采样比后，我们有

$$
\rho^{\pi_e}=\mathbb{E}_{P_\xi^{\pi_e}}[\sum_{t=0}^{T-1}\gamma^t r_t]=\mathbb{E}_{P_\xi^{\pi_b}}[\sum_{t=0}^{T-1}\gamma^t \omega_{0:t}r_t] \\
V^{\pi_e}(x)=\mathbb{E}_{P_\xi^{\pi_e}}[\sum_{t=0}^{T-1}\gamma^t r_t|x_0=x]=\mathbb{E}_{P_\xi^{\pi_b}}[\sum_{t=0}^{T-1}\gamma^t \omega_{0:t}r_t|x_0=x] \\
Q^{\pi_e}(x,a)=\mathbb{E}_{P_\xi^{\pi_e}}[\sum_{t=0}^{T-1}\gamma^t r_t|x_0=x,a_0=a]=\mathbb{E}_{P_\xi^{\pi_b}}[\sum_{t=0}^{T-1}\gamma^t \omega_{0:t}r_t|x_0=x,a_0=a] 
$$

因此上面的优化目标可以改写成（证明略）：

$$
\sum_{t=0}^T\gamma^t\mathbb{E}_{P_\xi^{\pi_b}}[\omega_{0:t}(\bar{R}_{t:T-1}(\xi)-\hat{Q}^{\pi_e}(x_t,a_t;\beta))^2]
$$

其中$\bar{R}_{t:T-1}(\xi)=\sum_{\tau=t}^{T-1}\gamma^{\tau-t}\omega_{t+1:\tau}r(x_\tau,a_\tau)$就是$Q^{\pi_e}(x_t,a_t)$的蒙特卡洛估计。

考虑到$n$个轨迹样本，选择SAA(sample averaging approximation):

$$
\beta_n^* \in \arg \max_{\beta\in \mathbb{R}^\kappa} \sum_{t=0}^T\gamma^t\frac{1}{n}\sum_{i=1}^n\omega_{0:t}^{(i)}[\bar{R}_{t:T-1}(\xi^{(i)})-\hat{Q}^{\pi_e}(x_t^{(i)},a_t^{(i)};\beta)]^2
$$

由于SAA是无偏的，所以当$n$足够大时，$\beta_n^* \to \beta^* \mathrm{almost \ surely}$。

对于每个时刻都是独立事件的多臂赌博机（非常简单的环境）来说，可以写得更简洁（以weighted least square(WLS)的形式）：

$$
\beta_n^* \in \arg \max_{\beta\in \mathbb{R}^\kappa} \frac{1}{n}\sum_{i=1}^n\frac{\mathbf{1}\{\pi_e(x_i)=a_i\}}{\pi_b(x_i,a_I)}[r(x_i,a_i)-\hat{Q}(x_i,a_i;\beta)]^2
$$

### Importance Sampling

IS算法不估计模型，而是直接利用IS ratio调整权重来估计$\rho$:

$$
\hat{\rho}^{\pi_e}= \frac{1}{n}\sum_{i=1}^n\omega_{0:T-1}^{(i)}\sum_{t=0}^{T-1}\gamma^tr_t^{(i)} \\
= \frac{1}{n}\sum_{i=1}^n\omega_{0:T-1}^{(i)}R_{0:T-1}^{(i)}
$$

IS算法是无偏的。

我们把上式的$\omega$放入第二个$\sum$里面会发现$\omega_{t+1:T-1}$部分对于$r_t$影响而徒增了方差，所以可以去掉，从而得到同样无偏但方差更小的$\rho_{step-IS}$:

$$
\hat{\rho}^{\pi_e}_{step-IS}= \frac{1}{n}\sum_{i=1}^n\sum_{t=0}^{T-1}\gamma^t\omega_{0:t}^{(i)}r_t^{(i)}
$$

但值得一提的是，$IS$和$step-IS$的无偏性都依赖于$\pi_b$是已知的，如果不是已知的，就不再是无偏的了。不过此时偏差是可以估计的：

$$
\mathrm{Bias}(\hat{\rho}^{\pi_e}_{IS})=|\rho^{\pi_e}-\mathbb{E}_{P_{\xi}^{\pi_b}}[\hat{\rho}^{\pi_e}_{IS}]|=|\mathbb{E}_{P_{\xi}^{\pi_e}}[\delta_{0:T-1}(\xi)R_{0:T-1}(\xi)]|
$$

（只要把$\mathbb{E}_{P_{\xi}^{\pi_b}}$变换成$\mathbb{E}_{P_{\xi}^{\pi_e}}$就是显然的了）

$$
\mathrm{Bias}(\hat{\rho}^{\pi_e}_{step-IS})=|\rho^{\pi_e}-\mathbb{E}_{P_{\xi}^{\pi_b}}[\hat{\rho}^{\pi_e}_{step-IS}]|=|\sum_{t=0}^{T-1}\gamma^t\mathbb{E}_{P_{\xi}^{\pi_e}}[\delta_{0:t}(\xi)r_t]|
$$

（方法同上，也是显然的。）

其中$\delta_{0:t}(\xi)=1-\lambda_{0:t}(\xi)=1-\Pi_{\tau=0}^t\frac{\pi_b(a_\tau|x_\tau)}{\hat{\pi}_b(a_\tau|x_\tau)}$，$\hat{\pi}_b$是估计值。

上面两种方法虽然具有无偏性的好性质，但方差也是惊人的，同时方差还会随着$T$指数级增长。所以在实际应用中，可以暂时放松对无偏性的要求，下面是一种有偏、但方差远小于前面两者的估计方法——对权重进行归一化：

$$
\hat{\rho}^{\pi_e}_{WIS}= \frac{1}{n}\sum_{i=1}^n\frac{\omega_{0:T-1}^{(i)}}{\sum_{i=1}^{n}\omega_{0:T-1}^{(i)}}\sum_{t=0}^{T-1}\gamma^tr_t^{(i)} 
$$

当然上面两种思路可以结合在一起（同样也是有偏的，但随着$n$增大是一致收敛的）：

$$
\hat{\rho}^{\pi_e}_{step-WIS}= \frac{1}{n}\sum_{i=1}^n\sum_{t=0}^{T-1}\gamma^t\frac{\omega_{0:t}^{(i)}}{\sum_{i=1}^{n}\omega_{0:t}^{(i)}}r_t^{(i)} 
$$

### Doubly Robust

DR方法则是结合了DM和IS（step-IS）:

$$
\hat{\rho}^{\pi_e}_{DR}=\frac{1}{n}\sum_{i=1}^n\sum_{t=0}^{T-1}[\gamma^t\omega_{0:t}^{(i)}r_t^{(i)}-\gamma^t(\omega_{0:t}^{(i)}\hat{Q}^{\pi_e}(x_t^{(i)},a_t^{(i)};\beta)-\omega_{0:t-1}^{(i)}\hat{V}^{\pi_e}(x_t^{(i)};\beta))]
$$

定义估计器的偏差为$\Delta(x,a)=\hat{Q}^{\pi_e}(x,a;\beta)-Q^{\pi_e}(x,a)$，那么DR算法的偏差是（证明略，可参考原文附录）

$$
\mathrm{Bias}(\hat{\rho}^{\pi_e}_{DR})=|\mathbb{E}_{P_{\xi}^{\pi_e}}[\sum_{t=0}^{T-1}\gamma^t\lambda_{0:t-1}(\xi)\delta_{t}(\xi)\Delta(x_t,a_t)]|
$$

类似地，只要$\pi_b$已知，DR就是无偏的（因为$\lambda=0$）。

### More Robust Doubly Robust
MRDR的主要创新是提出通过最小化DR estimator的方差来优化参数$\beta$。
这里假设$\pi_b$是已知的，且IS和DR是无偏的，那么MRDR也是无偏的，同时由于MRDR是最小化方差，所以也是最小的MSE（在DR 的解空间中）。
以单步的Contextual Bandit为例。

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\1.png)

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\2.png)

当$\pi_b$已知时，方差还可以写成下式

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\4.png)

但这个式子里包含了$\Delta$，也就是包含了真实的$Q$，而其不可求，故该表达式不可行。

我们还可以把方差按下式表述，其优点在于消去了$\Delta$，那么就可以算出关于$\beta$的偏导了。

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\3.png)

同时从（11）可以看出这是关于$\pi_b$的期望，因此可以用SAA代替期望：

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\5.png)

其中$\Omega_{\pi_b}(x)=\mathrm{diag}[1/\pi_b(a|x)]_{a \in \mathcal{A}}-ee^T$是一个半正定矩阵，
$$e=[1,\cdots,1]^T \\
q_\beta(x,a,r)=D_{\pi_e}(x)\bar{Q}(x;\beta)-[\mathrm{I}(a'=a)]_{a'\in \mathcal{A}} \\
D_{\pi_e}=\mathrm{diag}[\pi_e(a|x)]_{a\in \mathcal{A}} \\
\bar{Q}(x;\beta)=[\hat{Q}(x,a;\beta)]_{a\in \mathcal{A}}
$$

在更为一般的强化学习环境中，有：

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\6.png)

去掉与$\beta$无关的项之后：

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\7.png)

同样可以写成SAA的形式：

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\8.png)

## 实验

实验设计如下：

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\expe_1.png)

中文版：令动作空间为$\{1,2,\cdots,l\}$，$\mu\in[-0.5,0.5]$，实验设置了三种类型的随机采样策略$\pi_b$：
1. 第一种策略是友好型策略，当选中的动作为$a$时，$\pi_{\alpha,\beta}(x)$会以概率$\alpha+\beta\times \mu$返回$a$，以概率$\frac{1-(\alpha+\beta\times \mu)}{l-1}$均匀随机地返回$\{1,2,\cdots,l\}/\{a\}$中的一个数。
2. 第二种策略是厌恶型策略，当选中的动作为$a$时，$\pi_{\alpha,\beta}(x)$会以概率$\alpha+\beta\times \mu$返回一个不等于$a$的动作，以概率$\frac{1-(\alpha+\beta\times \mu)}{l}$均匀随机地返回$\{1,2,\cdots,l\}$中的一个数。
3. 第三种策略是中立策略，无论选中什么动作，都均匀随机地返回$\{1,2,\cdots,l\}$中的一个数。

具体的策略设计如下，待估策略$\pi_e$也在其中（应该是友好型的策略）：

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\expe_2.png)



对DR和MRDR的方差采用95%显著性检验，加粗表示通过检验（DR0指代采用DM训练好的$Q(x,a)$，再用DR方法算$\beta$）：

![](images\2018-12-29-More Robust Doubly Robust Off-policy Evaluation\expe_3.png)

## 小结

该文章主要介绍了一类新的用于off-policy估计的估计器MRDR。主要工作是改进了DR方法，用优化方差代替原先的优化MSE。新方法能够获得无偏、相合、渐进正态的估计，并在实验中获得好的表现。
未来的工作方向：
       1）考虑复杂的behavior policy
       2）考虑动作空间更复杂的情况
       3）考虑behavior policy未知的情况

这篇是我这学期组会讲过的文章，现在贴出来也算是补交一下作业。当时就看得有些不太明白，现在回过头来重读一遍，感觉又有了些不同的理解。

其实我对OPE问题理解不深，不过目前觉得OPE也是RL中不可避免的一个重要问题。但我目前还没有发现有哪些真正用上了这里面的estimator的算法，大部分都还是在用古老的IS方法。我猜想这可能和MRDR较为复杂的计算过程有关，希望之后能看到有人（或者我未来有机会）用上这个方法并且显著降低方差。
