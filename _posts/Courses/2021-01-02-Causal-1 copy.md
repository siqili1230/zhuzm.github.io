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

Consider following example, we cannot take 
$$Y_i | _{T=1} - Y_i | _{T=0}$$
 as the causal effect because $Y_i|_{T=1}$ cannot represent the potential outcome of "if we take $T=1$". 

The main difference between causation and correlation is $Y_i|_{T=1}\neq Y_i|_{do(T=1)}$  

Here we calculate the true causal effect, $Y_i|_{do(T=1)}-Y_i|_{do(T=0)}$, noted as $Y_i(1)-Y_i(0)$:

$$
\mathbb{E}[Y_i(1)-Y_i(0)]=\mathbb{E}[Y_i(1)]-\mathbb{E}[Y_i(0)]\\
\neq \mathbb{E}[Y_i|T=1]-\mathbb{E}[Y_i|T=0)]
$$

which is because we have confounding association $C$:

![image-2](\images\2021-01-02-Causal-1\image-2.png)

## Identificability

![image-11](\images\2021-01-02-Causal-1\image-11.png)


## Assumption 1: Ignorability

The igorability can be showed as following:
$$
\mathbb{E}[Y(1)]-\mathbb{E}[Y(0)]
= \mathbb{E}[Y(1)|T=1]-\mathbb{E}[Y(0)|T=0)]
$$

which means that the potential outcome $Y(1)$ is regardless of what the $T$ value is in practice. In the following table, the blankets in $Y(1)$ and $Y(0)$ columns are the unobservable outcomes. Those values are independent of whether corresponding $T$ is taken.

![image-3](\images\2021-01-02-Causal-1\image-3.png)

The ignorability can be described as:

$$
(Y(1),Y(0)) \perp \perp T
$$

Well, in another perspective, the ingorability can be viewed as exchangeability:

![image-4](\images\2021-01-02-Causal-1\image-4.png)

which means that the outcomes is independent of what data you choose.

## Conditional exchangeability

However, we don't know whether the data set satisfies exchangeability. Looking at the following example, we find that the distribution of "drunk/sober" in $T=1$ is different from it in $T=0$, where exchangeability is not satisfied.

![image-5](\images\2021-01-02-Causal-1\image-5.png)

So the conditional exchangeability is proposed that $(Y(1),Y(0))\perp\perp T|X$, where $X$ represents "drunk/sober".
So we have:

![image-6](\images\2021-01-02-Causal-1\image-6.png)

## Unconfoundedness

Unconfoundedness is an untestable assumption that $X$ is the only confoundedness and there is no unobservable "W".

![image-7](\images\2021-01-02-Causal-1\image-7.png)

## Positivity

Positivity demands that the support set is consist of the set $X$.

![image-8](\images\2021-01-02-Causal-1\image-8.png)

And if we don't have positivity, we have to predict the potential outcomes:

![image-9](\images\2021-01-02-Causal-1\image-9.png)

## Consistency

The same $T$ must correspond to the same outcome.

![image-10](\images\2021-01-02-Causal-1\image-10.png)

## Summary

![image-11](\images\2021-01-02-Causal-1\image-11.png)
