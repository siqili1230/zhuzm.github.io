---
layout: post
title:  "概率论(3)"
date:   2019-03-12 14:00:00 +0800
categories: 统计基础
tags: 概率论
author: 朱正茂
mathjax: true
---
* content
{:toc}


# 第二章 随机变量与分布函数






---
## 独立随机变量

**独立性**

设$\mathbf{x}=(x_1,x_2,\cdots,x_m)$是随机向量，联合分布为$F_\mathbf{X}(\mathbf{x})$，边际分布为$F_{X_i}(x_i)$，那么如果对所有的$\mathbf{x}$，都有

$$
F_\mathbf{X}(\mathbf{x})=\prod_{i=1}^m F_{X_i}(x_i)
$$

那么称$X_1,X_2,\cdots,X_m$相互独立。

---
## 随机变量的运算

**加减**

考虑$Z=X+Y$的分布：

$$
\begin{aligned}
F_Z(z)&=P(X+Y\leq z) \\
&=\int_{(x,y):x+y\leq z}p(x,y)dxdy \\
&=\int_{-\infty}^\infty \int_{-\infty}^{z-x} p(x,y)dydx \\
&=\int_{-\infty}^z\int_{-\infty}^\infty p(x,z^*-x)dxdz^*  (令z^*=x+y) \\
p_Z(z)&=\int_{-\infty}^\infty p(x,z-x)dx
\end{aligned}
$$

同理，$Z=X-Y$的分布：

$$
\begin{aligned}
F_Z(z)&=P(Y-X\leq z) \\
&=\int_{(x,y):y-x\leq z}p(x,y)dxdy \\
&=\int_{-\infty}^\infty \int_{-\infty}^{x+z} p(x,y)dydx \\
&=\int_{-\infty}^z\int_{-\infty}^\infty p(x,z^*+x)dxdz^*  (令z^*=y-x) \\
p_Z(z)&=\int_{-\infty}^\infty p(x,z+x)dx
\end{aligned}
$$

**乘除**

$Z=X*Y$的分布：

$$
\begin{aligned}
p_Z(z)&=\int_{-\infty}^\infty \frac{1}{|x|}p(x,\frac{z}{x})dx
\end{aligned}
$$

$Z=\frac{Y}{X}$的分布：

$$
\begin{aligned}
p_Z(z)&=\int_{-\infty}^\infty |x|p(x,zx)dx
\end{aligned}
$$

**一般的变换**

假设基于$(X,Y)$的变换如下：

$$
\begin{cases}
&U=f_1(X,Y) \\
&V=f_2(X,Y)
\end{cases}
$$

假设存在逆变换：

$$
\begin{cases}
&X=g_1(U,V) \\
&Y=g_2(U,V)
\end{cases}
$$

且$g_1,g_2$可导，Jacobi变换存在，行列式为：

$$
J=det\Bigg(\begin{matrix}
\frac{\partial{g_1}}{\partial{u}} & \frac{\partial{g_1}}{\partial{v}} \\
\frac{\partial{g_2}}{\partial{u}} & \frac{\partial{g_2}}{\partial{v}} 
\end{matrix}
\Bigg)
$$

则有

$$
p_{(U,V)}(u,v)=p_{(X,Y)}(g_1(u,v),g_2(u,v))|J|
$$


**次序统计量**

设$X_1,X_2,\cdots,X_n$是独立随机变量，则按从小到大排序得到$X_{(1)},X_{(2)},\cdots,X_{(n)}$，称为次序统计量，则

$$
\begin{aligned}
F_{X_{(n)}}(x)&=[F(x)]^n \\
F_{X_{(1)}}(x)&=1-(1-F(x))^n \\
F_{X_{(k)}}(x)&=(n-k+1)C_{n}^{k-1}F^{k-1}(x)p(x)(1-F(x))^{n-k}
\end{aligned} 
$$