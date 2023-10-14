# SVM assignment
## Additional Resources
- [Stanford SVM Slides](https://see.stanford.edu/materials/aimlcs229/cs229-notes3.pdf)
- [MIT SVM Slides](https://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf)
- [StatQuest - Support Vector Machines Part 1 (of 3): Main Ideas!!!](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [Stochastic Gradient Descent, Clearly Explained!!!](https://www.youtube.com/watch?v=vMh0zPT0tLI)
- [Gradient Descent, Step-by-Step](https://youtu.be/sDv4f4s2SB8?t=653)

## SVM and Duality
- [Support Vector Machines for Beginners – Duality Problem](https://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-duality-problem/)
- [Method of Lagrange Multipliers: The Theory Behind Support Vector Machines (Part 1: The Separable Case)](https://machinelearningmastery.com/method-of-lagrange-multipliers-the-theory-behind-support-vector-machines-part-1-the-separable-case/)


## **Mathematical Foundations:**

1. **Hypothesis Representation**:
   - SVM aims to find a hyperplane that separates data points of different classes in a feature space:
   
   $$ w^T x + b = 0 $$
   
   where $w$ is the weight vector, $x$ is the feature vector, and $b$ is the bias term.

2. **Margin and Support Vectors**:
   - SVM maximizes the margin, which is the reciprocal of the magnitude of the weight vector:

   $$ \text{Margin} = \frac{1}{\|w\|} $$

3. **SVM Objective Function**:
   - The objective is to maximize $\frac{1}{2} \|w\|^2$ subject to the constraint that $\forall i$, $y^{(i)}(w^T x^{(i)} + b) \geq 1$. This leads to the optimization problem:

   $$ \text{Minimize } \frac{1}{2} \|w\|^2 \text{ subject to } y^{(i)}(w^T x^{(i)} + b) \geq 1, \forall i $$

**Modified SVM Objective Function with Slack Variables:**

4. **Introduction to Slack Variables**:
   - In real-world scenarios, it's common for data points to be challenging to separate perfectly, leading to misclassifications or data points within the margin.
   - Slack variables ($\epsilon_i$) are introduced to handle such situations.

5. **Trade-off with Margin and Misclassifications**:
   - The modified objective function combines the margin term, $\frac{1}{2}\|w\|^2$, with a term that penalizes misclassifications and points within the margin, $C \sum_{i=1}^{m} \epsilon_i$.
   - The hyperparameter $C$ allows you to control the trade-off between maximizing the margin and allowing classification errors (misclassifications and points within the margin).

## **Practical Concepts:**

6. **Constraints with Slack Variables**:
   - The constraints ensure that data points are correctly classified (or within the margin) based on the value of $\epsilon_i$.
   - The inequality $y^{(i)}(w^T x^{(i)} + b) \geq 1 - \epsilon_i$ enforces correct classification if $\epsilon_i = 0$ and allows for some tolerance ($\epsilon_i > 0$) for misclassifications.

7. **Hyperparameter $C$**:
   - The hyperparameter $C$ controls the balance between maximizing the margin and minimizing misclassifications.
   - Smaller $C$ values emphasize maximizing the margin, potentially allowing more misclassifications ($\epsilon_i > 0$).
   - Larger $C$ values emphasize minimizing misclassifications, possibly reducing the margin width and resulting in smaller $\epsilon_i$ values.

8. **Optimization**:
   - The optimization process aims to find optimal values for $w$, $b$, and $\epsilon_i$ by minimizing the modified objective function while satisfying the constraints.
   - The optimal solution balances margin width, misclassifications, and the penalty associated with $\epsilon_i$, based on the value of $C$.
---


## Functional & Geometric Margin

| Concept | Definition | Interpretation |
|---------|------------|----------------|
| Functional Margin ($f(x_i)$) | $f(x_i) = y_i(\mathbf{w} \cdot \mathbf{x}_i + b)$ | - Positive $f(x_i)$ indicates correct classification. - Negative $f(x_i)$ indicates misclassification. - Magnitude of $f(x_i)$ measures confidence in classification. |
| Geometric Margin | $ \text{Geometric Margin} = \frac{f(x_i)}{\|\mathbf{w}\|} $ | - Measures perpendicular distance from data point to hyperplane. - Normalized by magnitude of $\mathbf{w}$ to be scale-independent. - Represents the actual margin or gap between data point and hyperplane. |

## Parameter Update Rules


The update rules for the weights $w$ and bias $b$ in stochastic gradient descent with a learning rate $\eta$ (eta) and per-instance loss $L_i(w, b)$ are as follows:

$$
w_{t+1} = w_t - \eta \nabla_w L_i(w, b)
$$

$$
b_{t+1} = b_t - \eta \nabla_b L_i(w, b)
$$

Here, $L_i(w, b)$ is the loss function for the $i$-th example, and the total loss $L(w, b)$ is typically defined as:

$$
L(w, b) = \frac{1}{2}\|w\|^2 + \frac{1}{C} \sum_{i=1}^{n} L_i(w, b)
$$

In this context:
- [ok](/_build/html/week11.2.html)
- $\eta$ (eta) is the [learning rate](https://gracelovesyah.github.io/comp90051notes/basics.html#learning-rate)
- $C$ is a regularization parameter
- $\frac{1}{2\lambda}\|w\|^2$ will be used to simplify the computation of its derivative with respect to $w$
-  $\lambda$ (lambda) is related to the regularization parameter $C$.



## Derivative of Loss Function

$$ L(w,b) = \sum_{i} \max(0, 1 - y_i (w^T x_i + b)) + \frac{\lambda}{2} \| w \|_2^2 $$

The derivative with respect to $ w $:

$$
\frac{\partial L}{\partial w} = 
\left\{
	\begin{array}{ll}
		-y_i x_i + \lambda w & \mbox{if } 1 - y_i (w^T x_i + b) > 0 \\
		\lambda w & \mbox{otherwise}
	\end{array}
\right.
$$

And the derivative with respect to $ b $:

$$
\frac{\partial L}{\partial b} = 
\left\{
	\begin{array}{ll}
		-y_i & \mbox{if } 1 - y_i (w^T x_i + b) > 0 \\
		0 & \mbox{otherwise}
	\end{array}
\right.
$$



## Key Concepts
**Kernel Functions** : In order to make the mathematics possible, Support Vector Machines use something called Kernel Functions to systematically find Support Vector Classifiers in higher dimensions.


