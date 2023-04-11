# Neural-ODE
A repository to maintain assets related to "Neural Ordinary Differential Equations: Review of Literature", by Owen Capell, Evan Hyzer, and Malhar Vaishampayan (MATH 1275 - Honors ODE Course Project)

## Overview
Here we maintian a repository that will serve as a collection of all items related to our literature review on the contemporary state of Neural Ordinary Differential Equations

A brief development of Neural Ordinary Differential Equations can be conducted (as presented in [Neural Ordinary Differential Equations](https://proceedings.neurips.cc/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)) as follows:

Consider the typical transformation of hidden states within a residual neural network:
$$
\mathbf{h}_{n+1} = \mathbf{h}_n + f(n, \theta)
$$

Where $\mathbf{h}_i$ is the $i$th residual block and $f$ is the learned transformation of the block depending on parameters $\theta$. This can be recognized as an Euler discretization of a continuous function, prompting one to take a limit of infinite residual blocks with smaller step sizes to get continous dynamics of hidden units:
$$
\frac{d\mathbf{h}(t)}{d\mathbf{h}}=f(\mathbf{h}(t),t,\theta)
$$

We recognize that this is an initial value problem (IVP) with initial state (input layer) $\mathbf{h}(t_0)\in\mathbb{R}^D$. We define the output layer to be $\mathbf{h}(T)$ as the unique solution (imposing Lipschitz and continuously differentiable requirements on the dynamics) to the IVP.

Reverse-Mode Automatic Differention (Backpropagation) to compute the gradients of a Neural ODE solution are discussed in the lit review, as well as general ideas about replacing ResNET with ODE Solvers of Neural ODEs.

## Review of Literature

We present a review of the current state of Neural Ordinary Differential Equations (and, implicitly, continuous-depth Neural Networks) and their applications in machine learning. 

We begin by developing the necessary prerequisite knowledge to understand the motivation behind each successive advancement. First we develop an outline of Convolutional Neural Networks (CNNs), followed by developing the motivations and applications of Residual Neural Networks (ResNET). After sufficient construction of the ideas surrounding ResNet, we explore the development and motivations of Neural Ordinary Differential Equations.

Generally, existing literature has shown Neural ODEs to have clear-cut advantages and disadvantages.

### Advantages
By our definiton of the output layer of being the unique solution to the given (above) IVP, neural networks implementing ODE solvers for Neural ODEs are incredibly space efficient (something that is not taken for granted with deep learning). In terms of the number of layers $L$ (or, in the case of a continuous-depth network, we take $L$ to be the number of function evaluations of our ODE solver), we can achieve as good as $O(1)$ memory usage.

### Disadvantages
Being continous-depth networks, the hyperparamater of network depth is discarded. Instead, though, users must pick an error tolerance for ODE solver as well as choose the implementation of the ODE solver

Additionally, ODE solvers typically are not computationally negligible (employing substantial control flow), meaning that Neural ODEs do not enhance speed of training. If anything, it can be argued they hinder it.

## Test Results
We implement and test our own neural network that use an ODE Solver to learn the parameters $\theta$ of the dynamics function $f(\mathbf{h(t)},t,\theta)$.

Our code is based (heavily, nearly verbatim) off of the implementations presented in [Mikhail Surtsukov's Neural ODE Notebook](http://implicit-layers-tutorial.org/neural_odes/).

Trained on the MNIST Handwritten Digit Data Set, we were able to achieve similar results to the original paper ([Neural Ordinary Differential Equations](https://proceedings.neurips.cc/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)) and to Surtsukov's implementaiton (which ours is heavily modeled after).

We achieved accuracy as high as 99.1% before any other optimizations. This displays the power that continuous-depth networks posses, being just as capable as other deep network architectures.