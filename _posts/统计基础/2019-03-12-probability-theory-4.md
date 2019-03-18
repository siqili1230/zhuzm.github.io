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

即$\sum_{k=1}^\infty \| x_k \| p_k<\infty$时，

数学期望 $\mathbb{E}X=\sum_{k=1}^\infty x_kp_k$。

对于连续型随机变量，当$\int_{-\infty}^\infty xp(x)dx$绝对可积，

即$\int_{-\infty}^\infty \| x \| p(x)dx<\infty$时，

数学期望 $\mathbb{E}X=\int_{-\infty}^\infty xp(x)dx$。

**性质**

$f:R\to R$ 为实值可测函数，则有

$\mathbb{E}(f(X))=\int_{-\infty}^\infty f(x)p(x)dx$ 

$E[(X,Y)]=(EX,EY)$

---
## 方差

**定义**

$Var(X)=E(X-EX)^2=EX^2-(EX)^2$

**性质**

$Var(a+bX)=b^2 Var(X)$

$Var(X+Y)=Var(X)+Var(Y)+2E(X-EX)(Y-EY)$

$Var(X)\leq E(X-c)^2$，当$c=EX$时取等号。

---
## Chebyschev不等式

$$
P(|X-EX|>\epsilon) \leq \frac{Var(X)}{\epsilon^2}
$$

推广：如果$f$单调不减，有$P(X>\epsilon) \leq \frac{Ef(X)}{f(\epsilon)}$

---
## Cauthy-Schwarz不等式

$$
E|X-EX||Y-EY| \leq (E(X-EX)^2 E(Y-EY)^2)^{1/2}
$$

---
## 随机向量

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

非负定性

$$
(x,y)\Sigma(x,y)^T\geq 0
$$

如果$Cov(X,Y)=0$，我们称$(X,Y)$不相关。

相关系数$\gamma = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}$

**条件期望**

$$
E(X|Y=y)=\int_{-\infty}^{\infty}xP(X=x|Y=y)dx
$$

**全期望公式**

记$g(y)=E(X|Y=y)$，则随机变量$g(Y)=E(X|Y)$，那么对$g(Y)$关于$Y$求期望得

$$
E_Y(E_X(X|Y))=EY
$$

---
## 特征函数

如果$E|X|^k<\infty$，则称$EX^k$为$k$阶矩，$E(X-EX)^k$为$k$阶中心矩。

但$k$阶矩也不能完全确定随机变量的分布，即当两个随机变量的任意阶矩都相同时，也不能确定两个随机变量的分布相同。下面我们将介绍一个能确保被比较的两个分布相同的变量/函数。

**特征函数**

$\phi(t)=Ee^{itX}=E\cos{tX}+iE\sin{tX}=\int$

其中$E\cos{tX}$和$E\sin{tX}$存在且有限。

常见分布的特征函数：

退化分布：$P(X=c)=1,\phi(t)=e^{ict}$
二项分布：$X\sim B(n,p),\phi(t)=(pe^{it}+q)^n$
泊松分布：$X\sim P(\lambda),\phi(t)=e^{\lambda(e^{it}-1)}$
均匀分布：$X\sim U(a,b),\phi(t)=\frac{e^{itb}-e^{ita}}{i(b-a)t}$
正态分布：$X\sim N(a,\sigma^2),\phi(t)=e^{iat-\frac{\sigma^2t^2}{2}}$
柯西分布：$f(t)=\int_{-\infty}^{\infty} e^{itx}\frac{1}{\pi (1+x^2)}dx=e^{- \vert t \vert }$

**性质**

1. 
$$
\vert f(t)\vert\leq f(0)=1 \\
f(-t)=\overline{f(t)}
$$

2. $f(t)$在$(-\infty,\infty)$上一致连续。

3. $f(t)$非负定，即对任意的正整数$n$，以及任意实数$t_1,t_2,\cdots,t_n$，复数$\lambda_1,\cdots,\lambda)n$，有

$$
\sum_{k=1}^n\sum_{j=1}^n f(t_k-t_j)\lambda_k\bar{\lambda_j}\geq 0
$$

4. 基本运算性质：若$X_1,X_2$独立且特征函数为$f_1(t),f_2(t)$，那么$X=X_1+X_2$的特征函数为$f(t)=f_1(t)f_2(t)$。

5. 若$E \xi^n$存在，则$f(t)$是$n$次可微的。进而，当$k\leq n$时，

$$
f^{(k)}(t)=i^k\int_{-\infty}^{\infty}x^ke^{itx}dF(x), \ \ f^{(k)}(0)=i^kE\xi^k
$$

因此，有$E\xi=-if'(0),E\xi^2=-f''(0),Var\xi=-f''(0)+[f'(0)]^2$。


### 逆转公式与唯一性定理

**逆转公式（了解）**

设分布函数$F(x)$的特征函数为$f(t)$，令$x_1,x_2$是$F$的连续点，那么有

$$
F(x_2)-F(x_1)=\lim_{T\to \infty} \frac{1}{2\pi}\int_{-T}^{T}\frac{e^{-itx_1}-e^{-itx_2}}{it}f(t)dt
$$

**唯一性**

分布函数可由特征函数唯一确定。

在逆转公式中，令$y=x_1\to -\infty$，则有

$$
F(x)=\lim_{y\to -\infty}\lim_{T\to \infty}\frac{1}{2\pi}\int_{-T}^{T}\frac{e^{-itx_1}-e^{-itx_2}}{it}f(t)
$$

一般地，若$f(t)$能写成$\sum a_n e^{ix_n t}$的形式，其中$a_n>0,\sum a_n=1$，则$f(t)$是特征函数，分布列为$P(\xi=x_n)=a_n$。

$\overline{f(t)}=f(-t)$是$-\xi$的特征函数，$\vert f(t)\vert ^2=f(t)\overline{f(t)}=f(t)f(-t)$是$\xi_1-\xi_2$的特征函数。

利用特征函数非常容易计算随机变量的和的分布（可加性/再生性）。






























