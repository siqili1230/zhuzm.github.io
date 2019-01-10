---
layout: post
title:  "概率论(2)"
date:   2019-01-10 21:00:00 +0800
categories: 统计基础
tags: 概率论
author: 朱正茂
mathjax: true

---
# 第二章 随机变量与分布函数






---
## 离散随机变量及分布

**退化分布：** $P(\xi=c)=1$

**两点分布：** $P(\xi=x_1)=p, \ P(\xi=x_2)=1-p$

**二项分布：** 

$P(\xi=k)=C_n^k p^k(1-p)^{n-k}$

$\xi$可以理解为$n$次两点分布试验中出现$\xi=x_1$的次数。记$\xi \sim B(n,p)$，$b(k;n,p)\overset{\Delta}=P(\xi=k|n,p)$。

性质：
1. $b(k;n,p)=b(n-k;n,1-p)$
2. $\frac{b(k;n,p)}{b(k-1;n,p)}=\frac{(n-k+1)p}{kq}=1+\frac{(n+1)p-k}{kq}$

    因此最可能出现的次数是$[(n+1)p]$
3. 渐进性质(泊松定理)：假设当$n\to \infty$时，有$n\cdot p_n\to \lambda$，则

$$
\lim_{n\to \infty}b(k;n,p)=\frac{\lambda^k}{k!}e^{-\lambda}
$$

**泊松分布：**

$$
P(\xi=k)=\frac{\lambda^k}{k!}e^{-\lambda}, \ \lambda>0, k\in \mathbb{N}
$$

记$\xi \sim P(\lambda)$。如果$n$个独立事件$A_1,\cdots,A_n$发生的概率很小，那么事件发生的次数近似服从泊松分布$P(np)$，在商场来到的顾客数目、货物出售的数目等情形中常用到。

**几何分布：** 

$$
P(\xi=k)=p(1-p)^{k-1}, \ k\in \mathbb{N}^+ 
$$

几何分布可以用于表示两点分布第一次成功$(\xi=x_1)$时累计的试验次数。

几何分布具有无记忆性：

$$
P(\xi>m+k|\xi>m)=P(\xi>k)
$$

**超几何分布：**

$$
P(\xi=k)=\frac{C_M^kC_{N-M}^{n-k}}{C_n^k} \\
k=0,1,\cdots,\min(n,M)
$$

可以用于表示质量抽查，$N$件产品中有$M$件次品，现在抽查$n$件，其中次品数目的分布。

当$N$很大时，可以用二项分布近似超几何分布，令$p=M/N$，就有

$$
\frac{C_M^kC_{N-M}^{n-k}}{C_n^k} \to C_n^k p^k(1-p)^{n-k}
$$


---
## 连续随机变量及分布

**分布函数：**$F(x)=P(\xi\leq x), \ -\infty < x < \infty$ 

性质：
1. 单调不减性：若$a < b$，则$F(a)\leq F(b)$
2. $\lim_{x\to -\infty}F(x)=0, \ \lim_{x\to \infty}F(x)=1$
3. 右连续性：$F(x+0)=F(x)$ （若定义$F(x)=P(\xi < x)$，则为左连续）

**连续随机变量：** 若$\xi$可取某个区间的一切值，且存在非负的可积函数$p(x)$，使分布函数满足：

$$
F(x)=\int_{-\infty}^x p(x)dx, \ \ -\infty < x <\infty
$$

则$\xi$为连续随机变量，$p(x)$称为概率密度函数。

注意，$P(\xi=c)=0$。

**均匀分布：** 

$$
p(x)=\begin{cases} 
		\frac{1}{b-a}, & a \leq x\leq b\\ 
		0, & \mathrm{otherwise} 
	\end{cases}
$$

**正态分布：**

$$
p(x)=\frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \ \ -\infty < x <\infty
$$

记$\xi \sim N(\mu,\sigma^2)$。对于$(\mu,\sigma^2)=(0,1)$的特殊情况，称为标准正态分布，密度函数记为$\phi(x)$，分布函数记为$\Phi(x)$。

$3\sigma$原则：正态分布99.73%的值落在$(\mu-3\sigma,\mu+3\sigma)$之间。

**指数分布：**

$$
p(x)=\begin{cases} 
		\lambda e^{-\lambda x}, & x\geq 0\\ 
		0, & \mathrm{otherwise} 
	\end{cases}, \ \lambda>0
$$

$$
F(x)=\begin{cases} 
		1-e^{-\lambda x}, & x\geq 0\\ 
		0, & \mathrm{otherwise} 
	\end{cases}, \ \lambda>0
$$

指数分布是唯一具有无记忆性的连续型分布：

$$P(\xi>s+t|\xi>t)=P(\xi>s)$$

**$\Gamma$分布：**

$$
p(x)=\begin{cases} 
		\frac{\lambda^r}{\Gamma(r)}x^{r-1}e^{-\lambda x}, & x\geq 0\\ 
		0, & \mathrm{otherwise} 
	\end{cases}, \ \lambda>0, \ \ r>0
$$

其中Gamma函数：$\Gamma(r)=\int_0^\infty x^{r-1}e^{-x}dx$

**$\beta$分布：**

$$
p(x)=\begin{cases} 
		\frac{1}{B(a,b)}x^{a-1}(1-x)^{b-1}, & 0\leq x\leq 1\\ 
		0, & \mathrm{otherwise} 
	\end{cases}
$$

其中$B(a,b)=\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}=\int_0^1 x^{a-1}(1-x)^{b-1}dx$

**Cauthy（柯西）分布：**

$$
p(x)=\frac{1}{\pi}\frac{1}{1+(x-\theta)^2}, \ \ \ -\infty < x <\infty, \\ 
-\infty < \theta <\infty
$$

---
### 随机向量

$n$维随机向量的分布函数与密度函数：

$$
F(x_1,\cdots,x_n)=\int_{-\infty}^{x_1}\cdots\int_{-\infty}^{x_n}p(y_1,\cdots,y_n)dy_1\cdots dy_n
$$

**边际分布:**

以2维为例，设$(\xi,\eta)$的分布函数为$F(x,y)$，则
$$
\begin{aligned}
F_\xi(x)&=F(x,\infty) \\
&=\int_{-\infty}^x\int_{-\infty}^\infty p(\mu,\nu)d\mu d\nu \\
&=\int_{-\infty}^x \Big[ \int_{-\infty}^\infty p(\mu,\nu)d\nu \Big] d\mu \\
&=\int_{-\infty}^x p_\xi(\mu) d\mu 
\end{aligned}
$$

**$n$维均匀分布：**

$$
p(x_1,\cdots,x_n)=\begin{cases}
        A,&(x_1,\cdots,x_n)\in G\\
        0,&\mathrm{otherwise}
\end{cases}
$$

**$n$维正态分布：**

设$\Sigma=(\sigma_{i,j})$为$n$维正定对称阵，$\mathbf{x}=(x_1,\cdots,x_n)^T$，$\mathbf{u}=(\mu_1,\cdots,\mu_n)^T$，则称

$$
p(\mathbf{x})=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}\exp\Big \{-\frac{1}{2}(\mathbf{x}-\mathbf{u})^T\Sigma^{-1} (\mathbf{x}-\mathbf{u}) \Big \}
$$

为$n$维正态分布密度函数。

考虑$n=2$的特殊情况，

$$
\Sigma=\Big( \begin{aligned}
\ & \sigma_1^2  \ &r\sigma_1\sigma_2 \ \\
\ & r\sigma_1\sigma_2  \ &\sigma_2^2 \ 
\end{aligned} \Big)
$$

$$
p(x,y)=\frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-r^2}}\exp\Bigg \{\ -\frac{1}{2(1-r^2)} \times \Big[ \frac{(x-a)^2}{\sigma_1^2} - \frac{2r(x-a)(y-b)}{\sigma_1\sigma_2} + \frac{(y-b)^2}{\sigma_2^2}\Big]\Bigg \}
$$

对$p(x,y)$求边际密度可以发现其边际分布依然是正态分布，但反过来，正态分布的联合分布却不一定是多元正态分布。反例如下：

$$
p(x,y)=\frac{1}{2\pi}e^{-\frac{x^2+y^2}{2}}(1+\sin x \sin y), \ \ -\infty < x < \infty
$$

---
### 随机向量的独立性

性质：
1. $\xi,\eta$独立 $\iff$ $p(x,y)=p_\xi(x)p_\eta(y)$
2. 若$(\xi,\eta)\sim N(a,b,\sigma_1^2,\sigma_2^2,r)$，则
    $\xi,\eta$独立 $\iff$  $r=0$
3. 相互独立的随机变量集$\{\xi_i\}$的子集也相互独立。
4. 若随机向量$\xi=(\xi_1,\cdots,\xi_n)$与$\eta=(\eta_1,\cdots,\eta_n)$独立，则它们各自的子向量也互相独立。

---
### 条件分布

**条件分布函数:**

$$
P(\eta \leq y|\xi=x)=\int_{-\infty}^y\frac{p(x,\nu)}{p_\xi(x)}d\nu
$$

**条件密度函数:**

$$
p_{\eta|\xi}(y|x)=\frac{p(x,y)}{p_\xi(x)}
$$

**贝叶斯公式**

$$
p_{\eta|\xi}(y|x)=\frac{p_{\xi|\eta}(x|y)p_\eta(y)}{\int_{-\infty}^\infty p_{\xi|\eta}(x|\nu)p_\eta(\nu)d\nu}
$$

上面的定理可以将随机变量$x,y$拓宽到$n$维随机向量的情况。


