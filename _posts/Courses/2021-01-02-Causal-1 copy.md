---
layout: post
title:  "Notes of Causal Inference"
date:   2021-01-02 23:12:18 +0800
categories: Courses
tags: Notes, Causal Inference
author: Zheng-Mao Zhu
mathjax: true
---

* content
{:toc}

[Course Web](https://www.bradyneal.com/causal-inference-course)

# Notes of Causal Inference

## What Does Imply Causation?

![image-1](\images\2021-01-02-Causal-1\image-1.png)

Consider following example, we cannot take $Y_i|_{T=1}-Y_i|_{T=0}$ as the causal effect because $Y_i|_{T=1}$ cannot represent the potential outcome of "if we take $T=1$". 

The main difference between causation and correlation is $Y_i|_{T=1}\neq Y_i|_{do(T=1)}$  

Here we calculate the true causal effect, $Y_i|_{do(T=1)}-Y_i|_{do(T=0)}$, noted as $Y_i(1)-Y_i(0)$:

$$
\mathbb{E}[Y_i(1)-Y_i(0)]=\mathbb{E}[Y_i(1)]-\mathbb{E}[Y_i(0)]\\
\neq \mathbb{E}[Y_i|T=1]-\mathbb{E}[Y_i|T=0)]
$$


## Value Iteration
Note $\mathcal{T}$ as the bellman optimal operator and $Q^*$ as the optimal Q-function that:

$$
(\mathcal{T} Q) (s,a) = R(s,a)+\gamma \mathbb{E}P(s'|s,a)[\arg\max_{a'}Q(s',a')]
$$

$$
Q^*=\mathcal{T} Q^* (bellman  \ optimality \ equation)
$$

Similarly,
$$
(\mathcal{T}^\pi Q) (s,a) = R(s,a)+\gamma \mathbb{E}P(s'|s,a)[Q(s',\pi(s')] \\
Q^\pi=\mathcal{T}^\pi Q^\pi
$$

### Planning

Assume we know the DMP model, then we can do planning: compute $V^*,Q^*$ given $\mathcal{M}$ by **value iteration** and **policy iteration**.

We have the algorithm that,
$$
Q^{*,0}=Q_0\\
Q^{*,h}=\mathcal{T} Q^{*,h-1}
$$



Q: If we denote $\pi_h:=\pi_{Q^{*,h}}$ , and  $Q^{\pi_h}$ be stationary point of operator  $\mathcal{T}^{\pi_h}$ , then are  $Q^{\pi_h}$  and  $Q^{*,h}$  equivalent?

A: No. This problem equals that assume a policy $\pi$ is derived from $Q$, then whether $Q$ must be equal to the converged Q-function $Q^{\pi}$. It's obviously not.

Q: How good is the policy $\pi_{Q^{*,H}}$ ?

A: For arbitrary $f$, $\|V^*-V^{\pi_f}\|_\infty=\frac{2 \|f-Q^*\|_\infty}{1-\gamma}$

Q: Bound $\|Q^{*,H}-Q^*\|_\infty$ ?

A: Lemma : $\|\mathcal{T}f-\mathcal{T}f'\|_\infty\leq\gamma\|f-f'\|_\infty$

Prove:
$$
\|Q^{*,H}-Q^*\|_\infty=\|\mathcal{T}Q^{*,H-1}-\mathcal{T}Q^*\|_\infty \\
\leq\gamma\|Q^{*,H-1}-Q^*\|_\infty \\
\leq \gamma^H\|Q^{*,0}-Q^*\|_\infty\\
\leq \gamma^H\frac{R_{max}}{1-\gamma}
$$


Another prove:
$$
V^{\pi,H}(s):=\mathbb{E}[\sum_{t=1}^H\gamma^{t-1}r_t|\pi,s_1=s]
$$

$$
V^{*,H}=\max_\pi V^{\pi,H}(s)
$$

$$
Q^{*}-Q^{*,H}\leq Q^*-Q^{\pi^*,H} \\
(since \ \pi^* is \ optimal \ for \ infinity \ horizon, but \ not \ when \ evaluated \ in \ H \ horizon)\\
=Q^{\pi^*}-Q^{\pi^*,H}\\
$$

For $\forall s,a, $
$$
Q^{\pi^*}(s,a)-Q^{\pi^*,H}(s,a)=\mathbb{E}[\sum_{t=H+1}^\infty \gamma^{t-1} r_t|s_1=s,a_1=a,\pi^*]\\
\leq \gamma^H(\sum_{t=1}^\infty \gamma^{t-1}R_{max})\\
=\gamma^H\frac{R_{max}}{1-\gamma}
$$


![image-20201206165112421](/images/2020-11-26-CS598 notes 1/image-1.png)

If we use a non-stationary policy that we execute $\pi_i$ in the i-th step, then the bound can be reduced from $O(1/(1-\gamma)^2)$ down to $O(1/(1-\gamma))$.

## exercise



![image-20201206165344658](\images\2020-11-26-CS598 notes 1\image-2.png)