---
layout: post
title: 分位数回归简介
date: 2018-07-24 17:15:00
categories: 统计基础
tags: 分位数回归
mathjax: true
---
* content
{:toc}

## 分位数回归

一般的回归分析注重因变量的均值受自变量变化的影响，也称为均值回归。
而分位数回归是描述因变量的分位数受自变量变化的影响。






比如我们基于自变量$X$和因变量$Y$建立了一个回归模型：

$$
Y=f(X)+\epsilon
$$

均值回归求的是 $ E(Y \| X=x) $ ，而分位数回归求的是

$$
Q_\tau(Y|X=x)=arg \ inf\{F_{Y|X=x}(y)\geq\tau\}
$$

其中 $ \tau $ 表示分位数， $ F_{Y \| X=x}( \cdot ) $ 表示 $ Y $ 的条件分布函数。

对于分位数回归，我们先考虑一种简单的情况：假设 $ Y $ 是一个随机变量，且 $ F_Y(y)=P(Y<y) $ 是分布函数，我们该如何求解 $ Y $ 的 $ \tau $ 分位数？

定义损失函数

$$
\rho_\tau(x)=x(\tau I(x\geq0)-(1-\tau)I(x<0))
$$

那么可以通过求解下式来得到分位数：

$$
min_u \ \mathbb{E}_Y (\rho_\tau(Y-u))
$$

证明：

易知 $ \rho_\tau(x)\geq0 $ ，故当 $$ \mathbb{E}_Y (\rho_\tau (Y-u)) $$ 取到0时必为最小值，即

$$
0=\mathbb{E}_Y (\rho_\tau(Y-u))=\tau \int_u^\infty (y-u)dF_Y(y)+(\tau-1)\int_{-\infty}^u(y-u)dF_Y(y)=F_Y(u)-\tau
$$

证毕。

接下来考虑对一个回归模型 $ Y=f(X)+\epsilon $ ，求解其在 $ X=x $ 的条件下 $ Y $ 的 $ \tau $ 分位数估计。

为了便于理解，我们从均值回归角度介绍。对于一个随机变量 $ Y $ ，均值回归就是求解

$$
min_u \mathbb{E}_Y[(Y-u)^2]
$$

如果 $ Y $ 是由自变量 $ X $ 表示的因变量，比如 $ Y=X\beta+\epsilon $ ，那么均值回归就是求解

$$
min_\beta \mathbb{E}_{(Y,X)}[(Y-X\beta)^2]
$$

均值估计就是 $ E(Y \| X=x)=x\beta $ 

现在相应地，在分位数回归中，对于随机变量 $ Y $ ，求解公式为

$$
min_u \mathbb{E}_Y[\rho_\tau(Y-u)]
$$

如果 $ Y $ 是由自变量$X$表示的因变量， $ Y=X\beta+\epsilon $ ，那么求解公式为

$$
min_\beta \mathbb{E}_{(Y,X)}[\rho_\tau(Y-X\beta)]
$$

分位数估计就是 $ Q_\tau(Y \| X=x)=x\beta $ 。

---
参考资料：
https://en.wikipedia.org/wiki/Quantile_regression
https://blog.csdn.net/bbbeoy/article/details/79494788
https://wenku.baidu.com/view/4f4e26ba27d3240c8547ef1e.html
