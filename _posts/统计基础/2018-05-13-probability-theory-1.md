---
layout: post
title:  "概率论(1)"
date:   2018-05-13 23:12:18 +0800
categories: 概率论
tags: 概率论
author: 朱正茂
mathjax: true
---
# 概率论基础
## 随机现象
随机现象的基本属性：

1. 试验可重复
2. 会出现何种结果不可知
3. 所有可能的结果已知



记所有可能结果为$\Omega$，称为样本空间。记每一个基本结果为 \(\omega \in \Omega\)，称为样本点。基本结果可以构成事件，可以用\(A\)表示，\(A \subset \Omega\)。

\(P(A)\)表示事件$A$发生的概率，统计方法中用频率来估计概率，即重复实验$N$次，$A$发生$N_A$次，那么有$P(A)=\lim_{N \to \infty}\frac{N_A}{N}$

---
##事件的基本运算
De Morgan对偶运算原理：
$\bar{(\cap A_n)}=\cup \bar{A}_n$    $\bar{(\cup A_n)}=\cap \bar{A}_n$

如果$A_1,\cdots,A_n$互不相交，那么有
$$
P(\Sigma_{i=1}^nA_i)=\Sigma_{i=1}^nP(A_i)
$$


## 概率的公理化

概率空间有三个基本要素：

1. 样本空间$\Omega$，是样本点$\omega$的集合。
2. 事件域$\mathcal{F}$，是$\Omega$中某些满足下列条件的子集的全体组成的集类：
    1. $\Omega \in \mathcal{F}$
    2. 若$A \in \mathcal{F}$，则$\bar{A} \in \mathcal{F}$
    3. 若$A_1,\cdots,A_m,\cdots \in \mathcal{F}$，则$\cup_{n=1}^\infty A_n \in \mathcal{F}$
3. 概率P，是定义在$\mathcal{F}$上的实值集函数：$A \to P(A)$，且满足：
    1. 非负性：$P(A)\geq0$
    2. 规范性：$P(\Omega)=0$
    3. 可列可加性：$
P(\Sigma_{i=1}^\infty A_i) = \Sigma_{i=1}^\infty P(A_i)
$

定理：如果$A_1,A_2,\cdots$是一系列单调增加的事件序列，具有极限A，那么有
$$
P(A)=\lim_{n\to \infty}P(A_n)
$$
所以概率具有连续性。
另外还有

