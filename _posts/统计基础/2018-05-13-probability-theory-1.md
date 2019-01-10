---
layout: post
title:  "概率论(1)"
date:   2018-05-13 23:12:18 +0800
categories: 统计基础
tags: 概率论
author: 朱正茂
mathjax: true
---
# 第一章 概率论基础





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
$$ 
\overset{——}{(\cap A_n)}=\cup \bar{A}_n \\
\overset{——}{(\cup A_n)}=\cap \bar{A}_n
$$  


如果$A_1,\cdots,A_n$互不相交，那么有
$$
P(\Sigma_{i=1}^nA_i)=\Sigma_{i=1}^nP(A_i)
$$

---
## 概率的公理化

概率空间有三个基本要素：

1. 样本空间$\Omega$，是样本点$\omega$的集合。
2. 事件域$\mathcal{F}$，是$\Omega$中某些满足下列条件的子集的全体组成的集类：
    1. $\Omega \in \mathcal{F}$
    2. 若$A \in \mathcal{F}$，则$\bar{A} \in \mathcal{F}$
    3. 若$A_1,\cdots,A_m,\cdots \in \mathcal{F}$，则$\cup_{n=1}^\infty A_n \in \mathcal{F}$
3. 概率$P$，是定义在$\mathcal{F}$上的实值集函数：$A \to P(A)$，且满足：
    1. 非负性：$P(A)\geq0$
    2. 规范性：$P(\Omega)=1$
    3. 可列可加性：若$\{A_i\}$是两两互不相容的事件，则
    $
P(\Sigma_{i=1}^\infty A_i) = \Sigma_{i=1}^\infty P(A_i)
$

定理：如果$A_1,A_2,\cdots$是一系列单调增加的事件序列，具有极限A，那么有

$$
P(\lim_{n\to \infty}A_n)=P(A)=\lim_{n\to \infty}P(A_n)
$$

Proof：令$B_k=A_k-A_{k-1}$，易得。

所以概率具有连续性。对于单调递减的$A_n$，该定理也成立。

## 条件概率与独立性

**条件概率：**

$P(A|B)=\frac{P(AB)}{P(B)}$表示在事件$B$发生的情况下事件A发生的概率。

**全概率公式：**

$$
P(B)=\sum_{i=1}^\infty P(A_i)P(B|A_i)
$$

其中$\sum_{i=1}^\infty A_i=\Omega$且$P(A_i)>0$。

**贝叶斯公式：**

$$
P(A_i|B)=\frac{P(A_i)P(B|A_i)}{\sum_{i=1}^\infty P(A_i)P(B|A_i)}
$$

其中$P(A_i)$为先验概率，$P(A_i|B)$为后验概率。

**事件独立：**

$$
P(AB)=P(A)\cdot P(B)
$$

或

$$
P(A|B)=P(A)
$$

对于多个事件$A_1,\cdots,A_n$的独立性，则需要对所有可能的事件组合，都满足独立的性质：

$$
\begin{aligned}
\begin{cases}
&P(A_iA_j)=P(A_i)P(A_j) \\
&P(A_iA_jA_k)=P(A_i)P(A_j)P(A_k) \\
&\cdots \\
&P(A_iA_j\cdots A_n)=P(A_i)P(A_j)\cdots P(A_n) 
\end{cases}  \\
\forall 1\leq i < j \cdots \leq n 
\end{aligned}
$$