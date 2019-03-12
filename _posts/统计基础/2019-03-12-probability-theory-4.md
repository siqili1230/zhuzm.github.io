---
layout: post
title:  "概率论(4)"
date:   2019-03-12 14:00:00 +0800
categories: 统计基础
tags: 概率论
author: 朱正茂
mathjax: true
---
* content
{:toc}


# 第三章 数字特征与特征函数






---
## 数学期望

**定义**

对于离散型随机变量，当$\sum_{k=1}^\infty x_kp_k$绝对可和，

即$\sum_{k=1}^\infty |x_k|p_k<\infty$时，

数学期望 $\mathbb{E}X=\sum_{k=1}^\infty x_kp_k$。

对于连续型随机变量，当$\int_{-\infty}^\infty xp(x)dx$绝对可积，

即$\int_{-\infty}^\infty |x|p(x)dx<\infty$时，

数学期望 $\mathbb{E}X=\int_{-\infty}^\infty xp(x)dx$。

**性质**

$f:R\to R$ 为实值可测函数，则有

$\mathbb{E}(f(X))=\int_{-\infty}^\infty f(x)p(x)dx$ 

$E[(X,Y)]=(EX,EY)$

## 方差

**定义**

$Var(X)=E(X-EX)^2=EX^2-(EX)^2$

**性质**

$Var(a+bX)=b^2 Var(X)$

$Var(X+Y)=Var(X)+Var(Y)+2E(X-EX)(Y-EY)$

$Var(X)\leq E(X-c)^2$，当$c=EX$时取等号。

**协方差**

$$
\begin{aligned}
Cov(X,Y)&=E[(X-EX)(Y-EY)] \\
&=E(XY)-EX*EY
\end{aligned}
$$

**协方差阵**

$$
\Sigma=\Bigg( \begin{matrix}
Var(X) & Cov(X,Y) \\
Cov(X,Y) & Var(Y)  
\end{matrix}\Bigg)
$$

## Chebyschev不等式

$$
P(|X-EX|>\epsilon) \leq \frac{Var(X)}{\epsilon^2}
$$

推广：如果$f$单调不减，有$P(X>\epsilon) \leq \frac{Ef(X)}{f(\epsilon)}$

## Cauthy-Schwarz不等式

$$
E|X-EX||Y-EY| \leq (E(X-EX)^2 E(Y-EY)^2)^{1/2}
$$















