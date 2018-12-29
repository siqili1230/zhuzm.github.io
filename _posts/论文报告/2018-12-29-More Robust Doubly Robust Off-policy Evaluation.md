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

对于一条轨迹$\xi=(x_0,a_0,r_0,\cdots,x_{T-1},r_{T-1},x_T)$，定义累计回报$R_{0:T-1}(\xi)=\sum_{t=0}^{T-1}\gamma^tr_t$，其中$x_0 \sim P_0,\ a_t \sim \pi(\cdot|x_t),\ x_{t+1}\sim P(\cdot|x_t,a_t), \ r_t \sim P_r(\cdot|x_t,a_t)$，将上面的分布一起定义为$P^\pi_\xi$，有

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

DM方法就是先学习一个模型：$\hat{Q}^\pi$，再用该模型来衡量策略。

$$
\hat{\rho}_{DM}^{\pi_e}=\frac{1}{n}\sum_{i=1}^n \sum_{a\in \mathcal{A}} \pi_e(a|x_0^{(i)})\hat{Q}^{\pi_e}(x_0^{(i)},a;\beta_n^*)
$$

求解$\hat{Q}$的参数$\beta$的方法就是直接最小化MES：

$$
\beta^* \in \arg \min_{\beta \in \mathbb{R}^{\kappa}} \mathbb{E}_{(x,a)\sim \mu_{\pi_e}}[(Q^{\pi_e}(x,a)-\hat{Q}^{\pi_e}(x,a;\beta))^2]
$$

DM的优点：方差小，偏差小。

### Importance Sampling

### Doubly Robust

### More Robust Doubly Robust

## 实验


## 小结
