# Lecture 24. Subject Review and Exam Info

This note is completed with the assistance of [ChatGPT](https://chat.openai.com/c/c8047535-2d6f-4e57-a6e8-e29fb8deebe2)


## 2023 S2 Exam Scope

---

### **Stats Background (Lectures 1 and 2)**
- **Basics of Statistics & Probability**: Descriptive stats, inferential stats, probability axioms, conditional probability.
- **Bayes' Theorem**:

$$ P(\theta|X) = \frac{P(\theta, X)}{P(X)} $$


### **Linear Regression (Lecture 3)**
- **Model**: $ y = \beta_0 + \beta_1 x $
- **MLE (Maximum Likelihood Estimation)**: Estimation method to find the parameters that maximize the likelihood of the observed data.

### **Regularising Linear Regression (Lecture 5)**
- **Ridge Regression**: Adds L2 penalty to linear regression.

$$ \text{Cost} = |y - X\beta|^2 + \lambda |\beta|^2 $$

- **Lasso Regression**: Adds L1 penalty to linear regression.

$$ \text{Cost} = |y - X\beta|^2 + \lambda |\beta| $$

- **Bayesian MAP (Maximum A Posteriori)**: Mode of the posterior distribution.

### **Non-linear Regression & Bias-Variance (Lecture 5)**
- **Bias-Variance Tradeoff**: Total Error = Bias^2 + Variance + Irreducible Error.

### **PAC Learning (Lecture 7)**
- **PAC (Probably Approximately Correct) Learning**: Framework for mathematical analysis of machine learning.
- **VC (Vapnik-Chervonenkis) Dimension**: Measure of the capacity of a hypothesis class.

### **SVMs (Support Vector Machines) (Lecture 9)**
- **Kernel Trick**: Efficiently compute dot products in high-dimensional spaces.
- **Popular Kernels**: Linear, Polynomial, RBF (Radial Basis Function), Sigmoid.
- **Mercer's Theorem**: Condition for a function to be a valid kernel.

### **Neural Networks (Lectures 11-14)**
- **Basics**: Neurons, activation functions (ReLU, Sigmoid, Tanh).
- **Backpropagation**: Algorithm to update network weights using gradients.
- **Autoencoders**: Neural networks trained to reproduce their input.
- **CNN (Convolutional Neural Networks)**: Specialized for grid-like data (e.g., images).
- **RNN (Recurrent Neural Networks)**: Suited for sequence data (e.g., time series).

### **PGMs (Probabilistic Graphical Models) (Lectures 20-22)**
- **Directed PGMs**: Represent conditional dependencies (e.g., Bayesian Networks).
- **Undirected PGMs**: Represent symmetric relationships (e.g., Markov Random Fields).
- **Inference in PGMs**: Compute conditional marginals from joint distributions.
- **Exact Inference**: Algorithms like variable elimination.
- **Approximate Inference**: Techniques like sampling.
- **Parameter Estimation**: MLE, EM (Expectation Maximization) for latent variables.

### **Application (Project 1)**
- **Review Key Steps**: Data preprocessing, model selection, evaluation metrics, results, and conclusions.

---

Given the extensive topics, it's essential to have a deeper understanding of each, especially the critical concepts. This cheatsheet provides a quick reference, but detailed notes, practice problems, and examples will help in mastering the material.