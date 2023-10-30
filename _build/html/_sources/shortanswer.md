# Short Answer Questions

---

### **1. Neural Networks and Deep Learning**

**ProTip**: Deep learning questions often touch on architectures, optimization techniques, and challenges like vanishing gradients. Familiarize yourself with common architectures like CNNs, RNNs, and attention mechanisms.

- **Question**: In what respect is a recurrent neural network deep?
  - **Answer**: RNNs are "deep" in terms of **time or sequence length**. They process sequences and maintain a hidden state across time steps, allowing them to capture long-range dependencies in data.

- **Question**: How does the value of the learning rate, $ \eta $ in SGD affect the training progress?
  - **Answer**: A **high learning rate** may cause oscillations and overshooting, potentially missing the minimum. A **low learning rate** might converge slowly or get stuck in local minima. An optimal learning rate allows steady convergence to a global or good local minimum.

- **Question**: Can training a deep neural network with tanh activation using backpropagation lead to the vanishing gradient problem? Why?
  - **Answer**: Yes, because the derivatives of the tanh function are in the range (0,1). For deep networks, multiplying these small derivatives during backpropagation can cause gradients to **vanish** and hinder weight updates in early layers.

- **Question**: Why are vanishing gradients a more pressing concern for recurrent neural networks (RNNs) than for other neural architectures?
  - **Answer**: RNNs process sequences and backpropagate errors through time steps. When gradients are small, they can **vanish** over many time steps, leading to long-range dependencies not being captured effectively and making early layers hard to train.

- **Question**: Explain how convolutional neural networks (CNNs) can produce translation invariant representations when applied to vector or matrix inputs.
  - **Answer**: CNNs use shared weights in their convolutional filters. This allows them to detect features regardless of their position in the input, leading to **translation invariance**. Pooling layers further enhance this property by reducing spatial dimensions.

- **Question**: A stochastic gradient descent (SGD) training step for a convolutional neural network (CNN) involves loading a training image into memory: as we need prediction, loss, and gradients for this image. Graph convolutional networks (GCNs) can involve very large graphs — too large to load into memory all at once: how can we reduce the memory footprint of training over the entire graph, when training a GCN?
  - **Answer**: Use **mini-batch training** with **neighbor sampling** or **graph sampling** techniques. Instead of loading the entire graph, load and process smaller subgraphs or neighborhoods iteratively.

- **Question**: Both convolutional networks (CNNs) and recurrent networks (RNNs) can be applied to sequence inputs. Explain the key benefit that RNNs have over CNNs for this type of input.
  - **Answer**: RNNs are designed to handle sequences and capture temporal dependencies by maintaining a memory (hidden state) from previous steps, making them more suited for sequential data compared to CNNs.

- **Question**: Explain in words how Attention can be used to allow for neural models to process dynamic sized inputs.
  - **Answer**: Attention mechanisms weight input elements differently, allowing the model to "focus" on parts of the input that are more relevant for a particular task. This dynamic weighting enables models to handle variable-sized inputs efficiently.

---

### **2. SVMs and Kernel Methods**

**ProTip**: For SVMs, understanding the distinction between hard-margin and soft-margin, the role of the kernel trick, and the primal vs. dual formulations can be particularly beneficial.

- **Question**: For the hard-margin support vector machine, data points that are support vectors have a margin of 1 from the decision boundary. Explain how this can be the case even when support vectors are further than 1 unit away from the decision boundary in Euclidean space.
  - **Answer**: The margin of 1 refers to the **distance in the transformed feature space**, not the original space. Due to the transformation (e.g., by a kernel), points may appear closer/farther in the transformed space than in the original Euclidean space.

- **Question**: When is it better to use the primal program for training an SVM with a quadratic kernel?
  - **Answer**: When the **number of features (d) is less than the number of samples (n)**. The complexity of the primal is related to the number of features, while the dual's complexity is related to the number of samples.

- **Question**: Suppose you have trained a soft-margin support vector machine (SVM) with a RBF kernel, and the performance is very good on the training set while very poor on the validation set. How will you change the hyperparameters of the SVM to improve the performance of the validation set?
  - **Answer**: **1. Increase the regularization parameter $ C $**: A larger $ C $ will impose more regularization, preventing overfitting. **2. Adjust the RBF kernel parameter $ \sigma $ or $ \gamma $**: Increase the width of the Gaussian function, making the decision boundary smoother.

- **Question**: Weak duality guarantees that a primal optimum always upper bounds a dual optimum. How can we tell that the support vector machine’s primal and dual optima are always equal?
  - **Answer**: SVM's optimization problem satisfies the **Karush-Kuhn-Tucker (KKT) conditions**, which ensure strong duality. Thus, the primal and dual optima are equal when these conditions are met.

- **Question**: List two general strategies for proving that a learning algorithm can be kernelized.
  - **Answer**: **1. Representer Theorem**: Demonstrate that the solution to the optimization problem can be expressed as a linear combination of kernel evaluations with training data. **2. Dual Formulation**: Show that the problem's dual formulation only involves dot products between data points, which can then be replaced by kernel functions.

- **Question**: For a support vector machine with a quadratic kernel, in what situation, if any, would it be better to use the primal program instead of the dual program for training?
  - **Answer**: When the **number of features (d) is less than the number of samples (n)**. In such cases, the primal problem might be more efficient to solve than the dual.

---

### **3. Bayesian and Probabilistic Models**

**ProTip**: Understanding the distinction between the frequentist and Bayesian perspectives, as well as how evidence, likelihood, and priors interact, can be pivotal for this category.

- **Question**: How does Thompson sampling achieve exploration of infrequently played arms in multi-armed bandit learning?
  - **Answer**: Thompson sampling samples from the posterior distribution over each arm's reward. By occasionally sampling optimistic estimates, even for infrequently played arms, it ensures **exploration** of all arms over

 time.

- **Question**: Why compute evidence for a Bayesian posterior but ignore it for MAP estimation?
  - **Answer**: The evidence, which is the denominator in Bayes' rule, acts as a normalizing constant to ensure the posterior is a valid probability distribution. For MAP estimation, we're interested in the **mode** of the posterior, so we can ignore the evidence since it doesn't affect the location of the mode.

- **Question**: How does the expectation maximization algorithm relate to the maximum likelihood estimate?
  - **Answer**: EM is used to find the **maximum likelihood estimates** (MLE) of parameters when the data has missing or hidden variables. It iteratively estimates the hidden variables (E-step) and then optimizes the parameters (M-step).

- **Question**: Can the joint probability distribution described by any undirected probabilistic graphical model be expressed as a directed probabilistic graphical model? Explain why or why not.
  - **Answer**: Not always. Some factorizations and independence assumptions in undirected models (Markov random fields) may not have a straightforward equivalent in directed models (Bayesian networks) and vice versa.
**ProTip**: Understanding the distinction between the frequentist and Bayesian perspectives, as well as how evidence, likelihood, and priors interact, can be pivotal for this category.

**Questions**:
*(Including the previously listed ones and any additional related ones)*

- **Question**: For Bayesian linear regression (as presented in class) explain how the posterior variance can vary for different test points.
  - **Answer**: The posterior variance depends on the test input $ x_* $ and captures our uncertainty about the prediction. It will be higher for test points far from training data and lower for points close to training data, reflecting increased confidence.

---

### **4. Optimization and Learning Theory**

**ProTip**: Grasp the distinction between different optimization techniques like gradient descent, RMSProp, Adagrad, and coordinate descent. Familiarity with concepts like VC-dimension, PAC learning, and bounds can be beneficial.

- **Question**: How does the value of the learning rate, $ \eta $ in SGD affect the training progress?
  - **Answer**: A **high learning rate** may cause oscillations and overshooting, potentially missing the minimum. A **low learning rate** might converge slowly or get stuck in local minima. An optimal learning rate allows steady convergence to a global or good local minimum.

- **Question**: Probably approximately correct (PAC) learning theory aims to upper bound the true risk $ R[f] $ of a learned model $ f \in F $ by the empirical risk $ \hat{R}[f] $ of that model plus an error term that might involve a sample size $ m $, VC-dimension $ VC(F) $, confidence parameter $ \delta $, etc. Why do PAC bounds hold only “with high probability $ 1 - \delta $”, and not deterministically?
  - **Answer**: PAC bounds are probabilistic because they account for the **random nature** of drawing a finite sample from a distribution. The bound guarantees that the true risk is close to the empirical risk for most samples, but there's a small probability $ \delta $ where this might not hold.

- **Question**: Why might VC-dimension based PAC learning theory lead to very large or impractical risk bounds for deep neural networks with non-linear activations and multiple hidden layers?
  - **Answer**: Deep neural networks with non-linear activations have a very **high VC-dimension**, which reflects their capacity to fit a large variety of functions. Using VC-dimension in PAC bounds for such models can result in very large or overly pessimistic bounds, making them less informative or useful.

- **Question**: If a training problem allows a closed form solution, would there be any advantage to using an iterative gradient based optimization method?
  - **Answer**: While closed-form solutions are computationally efficient, gradient-based methods can be more **flexible**, allowing for regularization, easier integration with other components, or scalability to large datasets.


- **Question**: Gradient descent is typically preferred over coordinate descent. Given this, why is coordinate descent used in the Expectation Maximisation algorithm?
  - **Answer**: Coordinate descent can be more efficient in situations where optimizing one parameter at a time (holding others fixed) has a closed-form solution or is computationally simpler, as is often the case in the E-step and M-step of the Expectation Maximisation algorithm.

- **Question**: Conjugacy between the likelihood and prior is very important when using Bayesian models. However, conjugacy is not critical when using the maximum a posteriori (MAP) estimator. Explain why this is the case.
  - **Answer**: In MAP estimation, we're interested in finding the mode (peak) of the posterior distribution. Conjugacy ensures the posterior remains in the same family as the prior, which simplifies computations. However, for finding the mode, conjugacy isn't essential as we're not explicitly computing the full posterior distribution.

---


### **5. Miscellaneous**

**ProTip**: These questions touch on varied aspects of machine learning, from model types to optimization strategies. Ensure a broad understanding of fundamental ML concepts.

**Questions**:

- **Question**: Describe something that can go wrong specifically with gradient descent while training a learner. Describe one approach to mitigating the problem.
  - **Answer**: Gradient descent can get stuck in local minima or saddle points, especially in non-convex optimization landscapes. One approach to mitigate this is to use **momentum** or optimization techniques like RMSProp and Adam, which can help navigate such challenging terrains.

- **Question**: Explain why training examples are weighted within the AdaBoost algorithm.
  - **Answer**: In AdaBoost, examples are weighted to give more emphasis to the misclassified examples in previous rounds. This ensures that subsequent weak learners focus more on these harder examples, making the ensemble more robust.

- **Question**: How are generative models different from discriminative models?
  - **Answer**: Generative models learn the joint probability $ P(X, Y) $ and model how the data is generated. Discriminative models learn the conditional probability $ P(Y|X) $ and focus on separating the classes.

- **Question**: In words, define the tree width of a directed graph $ G $.
  - **Answer**: Tree width measures the "tree-likeness" of a graph. A smaller tree width indicates the graph is closer to being a tree. Specifically, it's the size of the largest set in a tree decomposition of the graph minus one.

---
