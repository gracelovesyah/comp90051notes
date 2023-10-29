# Final Review Notes

## Overfitting and Underfitting
| Model                | Overfitting Techniques                                   | Underfitting Techniques                                 |
|----------------------|----------------------------------------------------------|---------------------------------------------------------|
| **Linear Regression**| - Ridge or Lasso regularization - Feature selection - Avoid high-degree polynomials   | - Polynomial features - Reduce regularization strength  |
| **Decision Trees**   | - Tree pruning - Random Forest or ensemble methods       | - Increase tree depth - Fewer constraints (e.g., min samples per split) |
| **SVM**              | - Increase regularization $ C $ - Simpler kernel (e.g., linear) | - Decrease regularization $ C $ - Complex kernel (e.g., RBF) |
| **Neural Networks**  | - Dropout - Weight regularization - Smaller network - Data augmentation - Early stopping | - Larger network - More epochs - Complex architectures  |
| **k-NN**             | - Increase $ k $ - Distance weighting                 | - Decrease $ k $                                      |
| **Logistic Regression**| - L1 or L2 regularization - Feature selection          | - Polynomial features - Reduce regularization strength  |

## Linearity


| Aspect                          | Linearly Separable Data                               | Non-Linearly Separable Data                                 |
|---------------------------------|-------------------------------------------------------|------------------------------------------------------------|
| **Choice of Model**             | - Linear models (e.g., Linear SVM, Logistic Regression) | - Non-linear models (e.g., Kernel SVM, Decision Trees, Neural Networks) |
| **Feature Engineering**         | - Minimal transformations required                    | - Polynomial features, interaction terms, or domain-specific transformations might be beneficial |
| **Regularization**              | - Might require stronger regularization to prevent overfitting due to perfect separation | - Regularization still important, but the balance might differ |
| **Model Complexity**            | - Simpler models often suffice                        | - More complex models may be needed to capture data patterns |
| **Training Time**               | - Typically faster due to simpler models              | - Potentially longer, especially with non-linear algorithms |
| **Interpretability**            | - Linear models are usually more interpretable         | - Complex models (e.g., deep neural networks) might be harder to interpret |
| **Validation Strategy**         | - Standard validation techniques apply                | - Ensuring diverse data in validation sets is crucial, given data's complexity |
| **Risk of Overfitting**         | - With perfect separation, there's a risk of overfitting | - Risk exists, especially with very flexible models. Techniques like pruning, dropout, or early stopping might be essential |
| **Kernel Methods (for SVM)**    | - Linear kernel is often suitable                     | - Non-linear kernels (e.g., RBF, polynomial) might be required |


## Optimization algorithms 
Optimization algorithms aim to find the best solution (or solutions) to a problem from a set of possible solutions. In machine learning and deep learning, the primary goal of an optimization algorithm is to minimize (or maximize) an objective function, typically known as the loss or cost function. By adjusting the model's parameters, optimization algorithms try to find the parameter values that result in the lowest possible loss for the given data.


| Optimization Algorithm     | Description                                                                                                     |
|----------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Gradient Descent (Batch)** | Updates the parameters in the direction of the negative gradient of the entire dataset at each iteration.         |
| **Stochastic Gradient Descent (SGD)** | Updates the parameters using only one training example at a time. It can be noisier but often faster than batch gradient descent. |
| **Mini-batch Gradient Descent** | A compromise between batch and stochastic gradient descent: updates parameters using a subset (or "mini-batch") of the training data. |
| **Momentum**                | Uses a moving average of past gradients to accelerate convergence and reduce oscillations.                         |
| **Nesterov Accelerated Gradient (NAG)** | A variant of momentum that computes the gradient after the momentum update, leading to more accurate parameter updates. |
| **AdaGrad**                 | Adjusts the learning rate for each parameter based on the historical squared gradients.                             |
| **RMSProp**                 | Modifies AdaGrad to use a moving average of squared gradients, preventing the learning rate from decreasing too rapidly. |
| **Adam**                    | Combines elements of Momentum and RMSProp. Maintains moving averages of both gradients and squared gradients.       |
| **Adadelta**                | An extension of AdaGrad that reduces its aggressive, monotonically decreasing learning rate.                         |
| **FTRL (Follow-the-Regularized-Leader)** | Especially suited for large-scale and online learning. Often used with L1 regularization for feature selection. |
| **L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)** | A quasi-Newton method that approximates the second-order derivative (Hessian) to guide the parameter updates. Suitable for smaller datasets. |
| **Conjugate Gradient**      | Uses conjugate directions (instead of just the gradient) to avoid re-visiting previously minimized directions. Used for non-linear optimizations. |

### **Gradient Descent Algorithm**:

1. Initialize the parameter vector $ \theta $ randomly or with zeros.
2. Repeat for a specified number of iterations or until convergence:
   1. Compute the gradient of the average loss over the training set:
   
$$
   \nabla_\theta J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta l(\theta, x_i, y_i)
   $$

   2. Update the parameter vector:
   
$$
   \theta = \theta - \alpha \nabla_\theta J(\theta)
   $$


Where:
- $ J(\theta) $ is the average loss over the training set.
- $ \alpha $ is the learning rate, a hyperparameter that determines the step size during each iteration of the gradient descent. A higher value makes the algorithm converge faster but risks overshooting the minimum, while a lower value ensures more accurate convergence but might be slower.

### **Stochastic Gradient Descent (SGD)**:

Stochastic Gradient Descent differs from the standard Gradient Descent in the following ways:

1. **Random Single Sample**: Instead of calculating the gradient based on the entire training set, SGD randomly selects one example ($ x_i, y_i $) at a time to compute the gradient.
   
2. **Frequent Updates**: Since it uses only one example at a time, the parameter vector $ \theta $ is updated more frequently, leading to much noisier steps.
   
3. **Convergence**: The frequent and noisy updates mean that SGD may never settle at the minimum. Instead, it will oscillate around the minimum. To mitigate this, the learning rate $ \alpha $ is often gradually decreased during training (e.g., learning rate annealing).

4. **Efficiency and Scalability**: SGD can be faster and more scalable than standard gradient descent, especially when the training set is large, because it starts making updates to $ \theta $ immediately.

In summary, while standard gradient descent computes the average gradient using the entire training set, stochastic gradient descent approximates the gradient using a single, randomly-selected training example. This makes SGD noisier but often faster and more suitable for large datasets.

## NN

### Backpropogation
#### RNN and BPTT

- **Sequential Data**:
  - RNNs process data one step at a time with a maintained hidden state.
  
- **Temporal Dependencies**:
  - RNNs capture relationships across sequence steps.

- **Backpropagation Through Time (BPTT)**:
  - Adjusts weights considering past steps.
  - Essential due to the influence of prior inputs on current outputs.

BPTT ensures RNNs account for the entire sequence when updating weights.


### Optimisation Algorithm (NN vs Other)

Optimization algorithms in neural networks (NN) and traditional machine learning models serve the same fundamental purpose: minimizing (or maximizing) an objective function, typically the loss or cost function. However, there are differences in the challenges posed by these models, leading to nuances in the optimization techniques used. Let's explore these differences:

1. **Scale of Parameters**:
   - **Neural Networks**: NNs, especially deep networks, can have millions to billions of parameters. Optimizing such a large parameter space introduces challenges not typically found in traditional models.
   - **Traditional ML Models**: These models often have fewer parameters. For instance, linear regression has one parameter for each feature (plus a bias).

2. **Non-Convexity**:
   - **Neural Networks**: The loss surfaces of deep NNs are non-convex, meaning they have many local minima, saddle points, and complex structures. This makes the optimization landscape challenging.
   - **Traditional ML Models**: Some traditional models (like linear regression with a squared error loss) have convex loss surfaces, ensuring a unique global minimum.

3. **Stochasticity**:
   - **Neural Networks**: Due to their size and complexity, NNs are often trained using stochastic methods (like mini-batch gradient descent) to speed up convergence.
   - **Traditional ML Models**: While stochastic methods can be used, many traditional algorithms can efficiently process the entire dataset in one iteration.

4. **Regularization Techniques**:
   - **Neural Networks**: NNs introduce unique regularization techniques like dropout, batch normalization, and weight normalization to combat overfitting and aid optimization.
   - **Traditional ML Models**: Regularization techniques for traditional models often revolve around adding penalty terms to the loss (e.g., L1 and L2 regularization).

5. **Learning Rate Scheduling**:
   - **Neural Networks**: Adaptive learning rate techniques and schedulers (like learning rate annealing or cyclical learning rates) are more commonly employed in NN training to ensure convergence in complex landscapes.
   - **Traditional ML Models**: While adaptive learning rates can be used, many traditional algorithms converge well with fixed or simpler learning rate strategies.

6. **Optimization Algorithms**:
   - **Neural Networks**: Advanced optimization algorithms like Adam, RMSProp, and Nadam, which combine momentum and adaptive learning rates, are popular in deep learning.
   - **Traditional ML Models**: Simpler algorithms like gradient descent, conjugate gradient, and L-BFGS are often sufficient for these models.

7. **Challenge of Vanishing/Exploding Gradients**:
   - **Neural Networks**: Deep networks face the issue of vanishing or exploding gradients, which can hinder training. Techniques like gradient clipping and careful weight initialization are used to mitigate this.
   - **Traditional ML Models**: These issues are less prevalent in traditional models.

8. **Parallelism and Hardware Acceleration**:
   - **Neural Networks**: Training large NNs benefits significantly from parallelism and hardware acceleration (e.g., GPUs). Optimization algorithms are sometimes adapted to better leverage these hardware capabilities.
   - **Traditional ML Models**: While some models can be parallelized or hardware-accelerated, the gains are often less pronounced than in deep learning.

In summary, while the core principles of optimization remain consistent across neural networks and traditional machine learning models, the scale, complexity, and challenges posed by deep learning have led to the development and adaptation of various optimization strategies specific to neural networks.

## SVM
### lamda and C

In the context of a soft-margin SVM, $ \lambda $ (often denoted as $ \alpha $ in many textbooks) and $ E $ (often referred to as $ \xi $ or slack variables) have specific interpretations:

1. **Lagrange Multipliers ($ \lambda $ or $ \alpha $)**:
    - $ \lambda = 0 $: The training example is correctly classified and lies outside the margin. It doesn't influence the decision boundary.
    - $ 0 < \lambda < C $: The training example lies on the margin's boundary and is correctly classified. It is a support vector.
    - $ \lambda = C $: The training example is wrongly classified, despite its position to the boundary.
    - $ \lambda > C $: impossible
2. **Slack Variables ($ E $ or $ \xi $)**:
    - $ E = 0 $: The training example is correctly classified and lies on the correct side of the margin.
    - $ 0 < E < 1 $: The training example is correctly classified but lies inside the margin.
    - $ E = 1 $: The training example lies exactly on the decision boundary.
    - $ 1 < E < 2 $: The training example is misclassified but lies within the margin.
    - $ E > 2 $: The training example is misclassified and lies outside the margin on the wrong side.

So, for the points A, B, C, and D, you'd use the criteria listed above to determine the values of $ \lambda $ and $ E $ based on their positions relative to the decision boundary and margin. The exact values would depend on the specific locations of these points in relation to the SVM's decision boundary and margin.



### Kernel Proof
#### sum
- [Proof of sum of kernels of concatenated vector](https://stats.stackexchange.com/questions/270967/proof-of-sum-of-kernels-of-concatenated-vector)
#### product
- [How do I formally proof the product of two kernels is a kernel? If K1 (x,x1) and K2 (x,x2) are both kernel function, then K1 (x,x1) K2 (x,x2) is also a kernel?](https://www.quora.com/How-do-I-formally-proof-the-product-of-two-kernels-is-a-kernel-If-K1-x-x1-and-K2-x-x2-are-both-kernel-function-then-K1-x-x1-K2-x-x2-is-also-a-kernel)
#### exp
- [Proof that exponential of a kernel is a kernel](https://stats.stackexchange.com/questions/320768/proof-that-exponential-of-a-kernel-is-a-kernel)

To prove that a function $ k $ is a valid kernel, one generally needs to demonstrate that the kernel matrix (or Gram matrix) constructed using $ k $ is positive semi-definite (PSD) for any set of input data points. Here are the general approaches to prove this:

1. **Direct Method**:
   - Construct the Gram matrix $ K $ using the kernel function $ k $ for any set of data points.
   - Show that for any vector $ \alpha $ of appropriate dimensions, the value of $ \alpha^T K \alpha $ is non-negative. If this is true for all such $ \alpha $, then $ K $ is PSD, and $ k $ is a valid kernel.

2. **Using Kernel Properties**:
   - **Closure Properties**: Kernels have several closure properties. For instance, if $ k_1 $ and $ k_2 $ are kernels, then the following are also kernels:
     - $ c \cdot k_1 $ for $ c > 0 $
     - $ k_1 + k_2 $
     - $ k_1 \times k_2 $
     - $ f(x) \cdot k_1(x, y) \cdot f(y) $ for any function $ f $
   - Using these properties, one can construct new kernels from known kernels.

3. **Mercer's Theorem**:
   - Mercer's theorem provides conditions under which a function can be expressed as an inner product in some (possibly infinite-dimensional) feature space, and therefore is a kernel. If a function satisfies Mercer's conditions (related to non-negative integrals over the product of the function with test functions), it is a valid kernel.

4. **Feature Map Representation**:
   - Demonstrate that there exists a feature map $ \phi $ such that $ k(x, y) = \langle \phi(x), \phi(y) \rangle $, where $ \langle \cdot, \cdot \rangle $ denotes the inner product. If you can explicitly find or describe such a feature map, then $ k $ is a kernel.

5. **Using Existing Kernels**:
   - Sometimes it's easier to derive new kernels based on known kernels. If you can express a function in terms of operations and compositions that preserve the kernel property, then the function is a valid kernel.

6. **Eigenvalues**:
   - Another way to prove a matrix is PSD is by showing all its eigenvalues are non-negative. However, computing eigenvalues might not always be feasible, especially for infinite-dimensional spaces.

## CNN 
```{image} ./images/cnn3.png
:alt: cnn
:class: bg-primary mb-1
:width: 800px
:align: center
```

### CNN and VI


- **Convolutional Layers**:
  - Use filters with shared weights.
  - Detect features anywhere in the image, ensuring consistent feature recognition regardless of position.

- **Pooling Layers**:
  - Down-sample feature maps (e.g., max-pooling).
  - Reduce sensitivity to exact feature locations, adding another layer of spatial invariance.

Together, these components enable CNNs to achieve translation-invariant classifications by recognizing and classifying image content regardless of its spatial position.

## Generative vs Discriminative

Of course, I can help clarify the difference between generative and discriminative models.

Generative models attempt to model how the data is generated. They capture the joint probability distribution $ P(X, Y) $, where $ X $ is the feature and $ Y $ is the label or class. Using this joint distribution, they can compute the conditional probability $ P(Y|X) $ for prediction.

Discriminative models focus on distinguishing between classes. They model the boundary between classes and learn the conditional probability $ P(Y|X) $ directly without worrying about how the data is generated.

| Criteria              | Generative Models                             | Discriminative Models                        |
|-----------------------|-----------------------------------------------|----------------------------------------------|
| **Description**       | Model how the data is generated.              | Model the boundary between classes.          |
| **What They Learn**   | Distribution of each class.                   | Boundary between classes.                    |
| **Examples**          | Gaussian Mixture Models, Naïve Bayes, Hidden Markov Models. | Logistic Regression, SVM, Random Forests, Neural Networks. |
| **Advantages**        | Can generate new data points. More robust with limited data. | Often more accurate in classification with ample data. |
| **Drawbacks**         | Might be less accurate in classification tasks. | Cannot generate new data points.             |
| **Analogy**           | Studying individual apples and oranges.       | Looking directly at differences between apples and oranges. |



## MAP
### **Difference between MAP and Full Bayesian Inference**:

- **Maximum a Posteriori (MAP)**:
  - MAP estimation provides a point estimate of a model parameter. Specifically, it estimates the mode of the posterior distribution.
  - It is formulated as:
    \[
    \theta_{\text{MAP}} = \arg\max_{\theta} P(\theta | \text{data}) = \arg\max_{\theta} P(\text{data} | \theta)P(\theta)
    \]
  - It combines the likelihood $ P(\text{data} | \theta) $ with the prior $ P(\theta) $ but doesn't fully account for the uncertainty in the parameter's distribution.

- **Full Bayesian Inference**:
  - Instead of a point estimate, full Bayesian inference considers the entire posterior distribution of the model parameter.
  - Predictions are made by integrating over all possible parameter values, weighted by their posterior probabilities.
  - This method captures the inherent uncertainty in parameter estimates.

**Situation where MAP and Full Bayesian Inference produce different predictions**:

Consider a scenario where you're estimating the bias of a coin based on a series of tosses. Let's assume you have a bimodal posterior distribution for the bias: one peak (mode) near $ \theta = 0.2 $ and another near $ \theta = 0.8 $, but the latter peak is broader.

- Using **MAP**, you'd pick the bias of the coin as the value corresponding to the highest peak, let's say $ \theta = 0.2 $. 
- With **Full Bayesian Inference**, when making predictions (like the probability of the next toss being heads), you'd average over both modes. Because the broader mode near $ \theta = 0.8 $ has a more substantial portion of the probability mass, the full Bayesian prediction might be closer to that of a coin with bias $ \theta = 0.8 $ than $ \theta = 0.2 $.

In this situation, despite the MAP estimate being $ \theta = 0.2 $, the full Bayesian approach might predict as if the bias is closer to $ \theta = 0.8 $, showcasing how the two methods can lead to different predictions.