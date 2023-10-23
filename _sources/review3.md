# Review 3
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/20c2aa50-1af7-489f-8cf6-3407f4264c37)

## ToC
- Iterative optimization
- gradient descent, Newton Raphson
- Regularization
- PAC, VC dimension

## normal equation
The normal equation is a method used to find the optimal parameters for a linear regression model without resorting to iterative optimization techniques like gradient descent. If you have a design matrix $ X $ (with each row being a training example and each column being a feature) and a vector $ y $ of target values, the normal equation provides a formula to directly compute the optimal weight vector $ \theta $ for the linear regression model.

The formula for the normal equation is:


$$ \theta = (X^T X)^{-1} X^T y $$

Where:
- $ \theta $ is the vector of model parameters.
- $ X $ is the design matrix.
- $ y $ is the vector of target values.
- $ X^T $ is the transpose of $ X $.
- $ (X^T X)^{-1} $ is the inverse of the matrix $ X^T X $.

This equation gives you the parameter values that minimize the cost function for linear regression. However, it's worth noting that the normal equation can be computationally expensive for very large datasets or when $ X^T X $ is non-invertible (singular). In such cases, iterative methods like gradient descent might be more appropriate.

## Newton Raphston

```{admonition} Question
"Recall that linear regression's objective can be written as $ L(\theta) = \frac{1}{2} ||X\theta - y||^2 $, where $ X $ is a $ n \times d $ matrix of training instances in rows, $ y \in \mathbb{R}^n $ are the training labels, and $ \theta \in \mathbb{R}^d $ are the parameters to learn by minimising $ L(\theta) $. Show that for any starting $ \theta \in \mathbb{R}^d $, a single step of Newton-Raphson optimisation of $ L(\theta) $ recovers the normal equation exact solution. (Hints for taking vector/matrix-valued derivatives with respect to $ \theta $, for constants matrix $ A $ and vector $ b $. Hint 1: $ \nabla (A\theta + b) = A $. Hint 2: $ \nabla^2(A\theta) = A $. Hint 3: $ \nabla ||f (\theta)||^2 $ = $ 2 \nabla f(\theta)^T f(\theta) $. Where prime ' denotes transpose, $ \nabla $ the gradient, and $ \nabla^2 $ the Hessian as in class, here always with respect to $ \theta $)."
```

```{admonition} Answer
:class: dropdown
Alright, let's correct that and go through the process again.

Given the objective function:

$$ L(\theta) = \frac{1}{2} ||X\theta - y||^2 $$

We want to show that a single step of the Newton-Raphson optimization recovers the normal equation solution.

The Newton-Raphson update rule for optimization is:

$$ \theta_{\text{new}} = \theta_{\text{old}} - \left( \nabla^2 L(\theta_{\text{old}}) \right)^{-1} \nabla L(\theta_{\text{old}}) $$

**Step 1:** Compute the gradient $ \nabla L(\theta) $.

Using the given hints:

$$ \nabla (A\theta + b) = A $$

$$ \nabla ||f(\theta)||^2 = 2 f(\theta)^T \nabla f(\theta) $$

The gradient is:

$$ \nabla L(\theta) = \frac{1}{2} \times 2 (X\theta - y)^T X $$

$$ \nabla L(\theta) = X^T (X\theta - y) $$

**Step 2:** Compute the Hessian $ \nabla^2 L(\theta) $.

Using the hint:

$$ \nabla^2 (A\theta) = A $$

The Hessian is:

$$ \nabla^2 L(\theta) = X^T X $$

**Step 3:** Plug into the Newton-Raphson update rule.


$$ \theta_{\text{new}} = \theta_{\text{old}} - (X^T X)^{-1} X^T (X\theta_{\text{old}} - y) $$

Notice that if $ \theta_{\text{old}} $ is any arbitrary starting point, then $ \theta_{\text{new}} $ will be the solution of the normal equation:

$$ \theta = (X^T X)^{-1} X^T y $$

This shows that a single step of Newton-Raphson optimization recovers the normal equation exact solution.
```

## PCA
The formula you provided breaks down the excess risk of a learned model $ f_m $ into two components: the estimation error and the approximation error. Let's dissect the formula and its components:

1. **Excess Risk**:

$$ R[f_m] - R^* $$

This represents the difference in risk (or error) between the model $ f_m $ that we've learned from our data and the optimal model with the lowest possible risk $ R^* $. This difference is what we refer to as the "excess risk" of $ f_m $.

2. **Estimation Error**:

$$ R[f_m] - R[f^*] $$

This is the difference in risk between the model $ f_m $ and the best possible model $ f^* $ from our hypothesis class. The estimation error arises because of the limitations of our learning algorithm and the finite amount of data we have. It captures the error due to the "estimation" process of learning $ f_m $ from data.

3. **Approximation Error**:

$$ R[f^*] - R^* $$
This represents the difference in risk between the best possible model $ f^* $ from our hypothesis class and the optimal model with the lowest possible risk $ R^* $. The approximation error arises because our hypothesis class might not contain the optimal model. It captures the error due to the "approximation" of the true underlying function by the best function in our hypothesis class.

To summarize:

- **Excess Risk**: The total additional risk our learned model $ f_m $ has compared to the optimal model.
- **Estimation Error**: The error due to the process of learning a model from data.
- **Approximation Error**: The error due to the limitations of our hypothesis class in approximating the true underlying function.

The formula essentially decomposes the excess risk into these two main sources of error, helping us understand where the shortcomings of our learned model come from.

### PCA Learnability

A concept class $ F $ is said to be PAC learnable if there exists a polynomial-time learning algorithm such that for any target concept $ f $ in $ F $, any distribution $ D $ over the input space, and for any $ \epsilon, \delta > 0 $ (where $ \epsilon $ is the approximation error and $ \delta $ is the probability bound), the algorithm will, with probability at least $ 1 - \delta $, output a hypothesis $ h $ whose error is at most $ \epsilon $, provided it is given a number of samples $ m $ that is polynomial in $ \frac{1}{\epsilon} $ and $ \frac{1}{\delta} $.

The notation $ m = \mathcal{O}(\text{poly}(\frac{1}{\epsilon}, \frac{1}{\delta})) $ means that the sample complexity $ m $ is bounded by a polynomial function of $ \frac{1}{\epsilon} $ and $ \frac{1}{\delta} $.

However, the statement "if $ m = 0 $" is confusing in this context. If $ m $ is 0, it means no samples are required, which doesn't make sense in the PAC learning framework. It's possible there's a misunderstanding or miscommunication about the notation or concept. If you have a specific question or context about PAC learnability, please provide more details.

## VC dimension

The VC (Vapnik-Chervonenkis) dimension is a fundamental concept in statistical learning theory. It provides a measure of the capacity or complexity of a hypothesis class. Specifically, the VC dimension gives an upper bound on the number of training samples that a hypothesis class can shatter, which in turn provides insights into the generalization ability of a learning algorithm.

Here's a more detailed breakdown:

1. **Shattering**: A hypothesis class $ H $ is said to "shatter" a set of points if, for every possible labeling of those points (binary labels, typically 0 or 1), there exists some hypothesis in $ H $ that perfectly separates the points according to that labeling.

2. **VC Dimension**: The VC dimension of a hypothesis class $ H $ is the largest number of points that $ H $ can shatter. If $ H $ can shatter an arbitrarily large number of points, then its VC dimension is said to be infinite.

3. **Implications for Learning**:
   - A higher VC dimension indicates that the hypothesis class is more complex and can represent a wider variety of functions.
   - However, a higher VC dimension also means that the hypothesis class can easily overfit to the training data. This is because a more complex hypothesis class can fit the training data in many ways, including fitting to the noise in the data.
   - The VC dimension plays a crucial role in deriving generalization bounds in machine learning. Specifically, it helps in determining how well a learning algorithm will perform on unseen data, given its performance on the training data.

4. **Examples**:
   - For a hypothesis class of linear classifiers in 2D (lines in the plane), the VC dimension is 3. This is because any 3 non-collinear points can be shattered by lines in the plane, but 4 general points cannot be.
   - For a hypothesis class of linear classifiers in $ d $-dimensions (hyperplanes), the VC dimension is $ d + 1 $.

5. **Relationship with PAC Learning**: In the PAC (Probably Approximately Correct) learning framework, the VC dimension helps determine the number of samples required to learn a concept within a certain error bound. Specifically, if the VC dimension is finite, then the sample complexity for PAC learning is polynomial in terms of the VC dimension, the inverse of the desired error $ \epsilon $, and the inverse of the confidence $ \delta $.

The VC dimension provides a bridge between the empirical performance of a learning algorithm (its performance on the training set) and its expected performance on unseen data, making it a foundational concept in understanding the generalization capabilities of learning algorithms.