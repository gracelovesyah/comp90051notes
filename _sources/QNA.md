# Short Answers
**Question 1: Decision Theory and Kernel Methods**

(a) Is the loss function $ l(y,\hat{y}) = (\hat{y} - y)^3 $ a good idea or a bad idea for supervised regression? Why?
- **Answer:**
  - **Bad idea.** 
  - The cubic loss function does not penalize errors symmetrically. Positive and negative errors of the same magnitude will have different losses. This can lead to undesirable biases in regression estimates.

(b) Explain a benefit of using the kernel trick with a polynomial kernel of degree $ p = 12 $ in SVM vs explicit basis expansion.
- **Answer:**
  - **Computational efficiency.** 
  - Using the kernel trick, we can compute the inner product in the higher-dimensional space without explicitly computing the transformation. Explicit basis expansion for a degree 12 polynomial would involve computing a large number of polynomial features, which is computationally expensive.

(c) How are support vectors characterized by dual variables $ \lambda_i $ in the soft-margin SVM?
- **Answer:**
  - **Non-zero values.** 
  - In the soft-margin SVM, support vectors are data points for which the dual variable $ \lambda_i $ is greater than zero.

(d) Write an expression relating expected squared loss, bias, and variance.
- **Answer:**
  - **Expected squared loss = Bias^2 + Variance + Noise.**

(e) How is the objective function for soft-margin SVM a relaxation of the hard-margin SVM objective?
- **Answer:**
  - **Inclusion of slack variables.** 
  - Soft-margin SVM introduces slack variables to allow some misclassifications, whereas hard-margin SVM requires perfect separation of data. This makes the soft-margin SVM more flexible and applicable to non-linearly separable data.

**Question 2: General Machine Learning**

(a) Describe where within multi-armed bandit algorithms like ε-greedy, statistical estimation is performed and how these estimates are used.
- **Answer:**
  - **Estimation of action values.**
  - In ε-greedy, the value of each action (arm) is estimated based on the historical rewards. The algorithm exploits the action with the highest estimated value most of the time, and explores random actions with a probability of ε.

(b) Describe a problem with gradient descent during training and one approach to mitigate it.
- **Answer:**
  - **Vanishing or Exploding Gradients.**
  - Gradients can become too small (vanish) or too large (explode), making learning slow or unstable.
  - **Mitigation:** Use techniques like gradient clipping, batch normalization, or careful initialization.

(c) Why are training examples weighted within the AdaBoost algorithm?
- **Answer:**
  - **To emphasize misclassified examples.**
  - AdaBoost increases the weights of misclassified examples in each iteration to focus the next weak learner on those hard-to-classify instances.

(d) How are generative models different from discriminative models?
- **Answer:**
  - **Modeling approach.**
  - Generative models model the joint probability $ P(X,Y) $ and try to learn how the data is generated. Discriminative models model the conditional probability $ P(Y|X) $ and focus on the decision boundary between classes.

(e) Define the tree width of a directed graph $ G $ in words.
- **Answer:**
  - **Maximum clique size minus one.**
  - The tree width of a graph is the size of its largest clique (fully connected subgraph) minus one, when represented as a tree decomposition.

(f) Benefit of using the Gaussian mixture model (GMM) over k-means clustering.
- **Answer:**
  - **Flexibility in cluster shapes.**
  - GMM can model elliptical clusters, whereas k-means assumes spherical clusters. GMM also provides a probabilistic cluster assignment, as opposed to the hard assignment in k-means.

---

**Question 1:**

(a) Why are maximum likelihood estimation, max a posteriori, and empirical risk minimization all instances of extremum estimators?
- **Answer:**
  - All these methods involve **optimizing** (maximizing or minimizing) a particular objective function. MLE maximizes the likelihood, MAP maximizes the posterior, and ERM minimizes the empirical risk.

(b) Why was there a third irreducible error term in the bias-variance decomposition for supervised regression but not in parameter estimation?
- **Answer:**
  - In supervised regression, the third term represents the **noise** inherent in the data. This error is due to the randomness in the data and cannot be reduced regardless of the model. In parameter estimation, we're trying to estimate fixed parameters, so there's no inherent "noise" term.

(c) Key difference between learning with experts and multi-armed bandit settings?
- **Answer:**
  - In the **learning with experts** setting, the learner receives feedback on all expert predictions, regardless of which expert was chosen. In the **multi-armed bandit** setting, feedback is received only for the chosen action (arm).

(d) How do the frequentist and Bayesian approaches differ in their modeling of unknown parameters?
- **Answer:**
  - **Frequentists** view unknown parameters as fixed but unknown values. They estimate these parameters from data. **Bayesians** treat unknown parameters as random variables and use probability distributions to describe their uncertainty about these parameters.

(e) Why would a growth function $ S_F(m) $ of $ 2^m $ be bad for the PAC bound with growth function for $ F $?
- **Answer:**
  - A growth function of $ 2^m $ indicates that the hypothesis class is too complex, potentially leading to **overfitting**. The PAC bound would be loose or meaningless, making guarantees on generalization error weak or non-existent.

(f) Why doesn’t max a posteriori estimation require computation of the evidence through costly marginalization, while computing the posterior distributions does?
- **Answer:**
  - In MAP, we're finding the parameter value that **maximizes** the posterior. The evidence is a normalizing constant and does not affect the location of this maximum. For full posterior distributions, the evidence is needed to normalize and get a proper probability distribution.

(g) Is the loss function $ l(y; \hat{y}) = (\hat{y} - y)^5 $ a good idea or a bad idea for supervised regression?
- **Answer:**
  - **Bad idea.** 
  - Fifth power loss will greatly amplify the effect of outliers. It's also not symmetric for positive and negative errors. This can introduce biases and make regression unstable.

(h) Strategies to change the hyperparameters of the soft-margin SVM with a RBF kernel for better performance?
- **Answer:**
  - **1. Adjust the regularization parameter $ C $**: If overfitting, increase $ C $; if underfitting, decrease $ C $.
  - **2. Tune the RBF kernel parameter $ \sigma $**: If overfitting, increase $ \sigma $; if underfitting, decrease $ \sigma $.

(i) Main drawback of Adaptive Gradient (AdaGrad) compared to RMSProp?
- **Answer:**
  - AdaGrad's learning rate can **decrease too rapidly** and become extremely small, causing the algorithm to converge prematurely and stop learning.

(j) Strategy that allows VAE to apply gradient descent through the samples of latent representation $ z $ to the encoder?
- **Answer:**
  - VAE uses the **reparameterization trick**. Instead of sampling $ z $ directly, it samples from a standard distribution and then shifts and scales the sample using parameters from the encoder. This makes the process differentiable.

(k) For the graph neural network described, are the hidden states and input data located on the nodes, edges, or both?
- **Answer:**
  - The hidden states $ h $ and input data $ x $ are located on the **nodes**. The functions $ f $ and $ g $ describe the interactions between nodes, but the states themselves reside on nodes.

---

**Section A: Short Answer Questions**

1. In what respect is a recurrent neural network deep?
- **Answer:**
  - RNNs are "deep" in terms of **time or sequence length**. They process sequences and maintain a hidden state across time steps, allowing them to capture long-range dependencies in data.

2. Explain how support vectors can have a margin of 1 from the decision boundary but be further than 1 unit away in Euclidean space.
- **Answer:**
  - The margin of 1 refers to the **distance in the transformed feature space**, not the original space. Due to the transformation (e.g., by a kernel), points may appear closer/farther in the transformed space than in the original Euclidean space.

3. When is it better to use the primal program for training an SVM with a quadratic kernel?
- **Answer:**
  - When the **number of features (d) is less than the number of samples (n)**. The complexity of the primal is related to the number of features, while the dual's complexity is related to the number of samples.

4. How are artificial neural networks a form of non-linear basis function in learning a linear model?
- **Answer:**
  - ANNs transform the input using **activation functions** and multiple layers, effectively creating non-linear combinations of the input data. The final layer is a linear combination of these non-linear transformations, making it a linear model in a transformed, non-linear basis.

5. Effect of high uncertainty over model parameters on the maximum likelihood estimate?
- **Answer:**
  - High uncertainty can lead to **broad likelihood functions** with multiple peaks or wide regions of high likelihood, making the MLE less precise or less interpretable.

6. Why is using the training likelihood for model selection problematic when choosing between models from different families?
- **Answer:**
  - Using training likelihood can lead to **overfitting**. More complex models might fit the training data better (higher likelihood) but may not generalize well to new data.

7. How can momentum in an optimizer help avoid local optima or saddle points?
- **Answer:**
  - Momentum helps the optimizer **maintain direction** from previous steps. This can allow it to "roll over" small local optima or saddle points instead of getting stuck, by accumulating velocity in consistent directions.

8. When is it desirable to use an estimator with high bias?
- **Answer:**
  - When there's a risk of **overfitting** due to limited data or high variance. A high bias estimator can regularize the model, providing simpler and potentially more generalizable solutions.

9. A failure case from using complex non-linear encoder and decoder components in an autoencoder?
- **Answer:**
  - Complex components can **overfit** to the training data, allowing the autoencoder to memorize rather than learn meaningful representations. This reduces the generalization capability on new data.

10. Why is conjugacy not critical when using the MAP estimator?
- **Answer:**
  - For MAP, we're interested in finding the mode (peak) of the posterior, not the full distribution. Conjugacy simplifies computations for the full posterior but isn't strictly necessary for finding its mode.

11. How can the posterior variance vary for different test points in Bayesian linear regression?
- **Answer:**
  - The variance for a test point is influenced by its **location relative to the training data**. Points far from training data have higher uncertainty, leading to larger variances. The formula $ \sigma^2 + x_*^T V_N x_* $ shows that the variance depends on the test point $ x_* $.

12. Why is coordinate descent used in the Expectation Maximization algorithm despite the preference for gradient descent?
- **Answer:**
  - In EM, the objective function is often **decomposed** into components corresponding to different variables or parameters. Coordinate descent can optimize each component (E-step or M-step) separately, making the optimization more tractable or efficient for such decompositions.

---

**Question 1:**

(a) How will you change the hyperparameters of the SVM with an RBF kernel if it overfits the training set?
- **Answer:**
  - **1. Increase the regularization parameter $ C $**: A larger $ C $ will impose more regularization, preventing overfitting.
  - **2. Adjust the RBF kernel parameter $ \sigma $ or $ \gamma $**: Increase the width of the Gaussian function, making the decision boundary smoother.

(b) Main drawback of AdaGrad compared to RMSProp?
- **Answer:**
  - AdaGrad's learning rate can **decrease too rapidly** and become very small, causing the algorithm to converge prematurely and stop updating.

(c) Strategy that allows VAE to apply gradient descent through the samples of latent representation $ z $?
- **Answer:**
  - VAE uses the **reparameterization trick**. Instead of sampling $ z $ directly, it samples from a standard distribution and then shifts and scales the sample using parameters from the encoder, ensuring differentiability.

(d) Why use backpropagation through time for RNNs?
- **Answer:**
  - RNNs process sequences and maintain state across time steps. **Backpropagation through time (BPTT)** is used to compute gradients for weight updates by unrolling the RNN over the sequence length and applying the chain rule across time steps.

(e) Fill in the blanks: $ L $ ⊂ $ M $ ⊂ $ E $.
- **Answer:**
  - **L** ⊂ **M** ⊂ **E**.

(f) Why is there no irreducible error term in the square-loss risk for parameter estimate but it's present in supervised regression predictor's risk?
- **Answer:**
  - The irreducible error in the regression risk arises from the inherent noise in the data (unpredictable variability). When estimating parameters, we're trying to find the best fit to the data, but we aren't directly modeling the data's inherent variability. Thus, there's no term for irreducible error in parameter estimation risk.

(g) Objective function optimized during GMM training?
- **Answer:**
  - The **log-likelihood** of the data given the model parameters. Mathematically, for a GMM, the objective is:

$$
\mathcal{L}(\theta) = \sum_{i=1}^{N} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k) \right)
$$

Where:
- $ \pi_k $ is the mixture coefficient for component $ k $,
- $ \mathcal{N} $ denotes the Gaussian distribution with mean $ \mu_k $ and covariance $ \Sigma_k $,
- $ N $ is the number of data points, and $ K $ is the number of Gaussian components.

(h) Why only one iteration of Newton-Raphson is required for linear regression?
- **Answer:**
  - Linear regression has a **convex** cost function (squared error). Newton-Raphson directly finds the point where the gradient is zero, which, in the case of convex functions like the squared error, is the global minimum. Thus, one step is sufficient to reach the optimal solution.

---

**Section A: Short Answer Questions**

(a) How does the dynamic learning rate in Adagrad operate and its importance?
- **Answer:**
  - Adagrad adjusts the learning rate for each parameter based on the historical gradient information. Parameters with larger past gradients get smaller learning rates, and vice versa. This allows for **adaptive learning** and can help in situations where features have different scales or gradients.

(b) Key benefit of RNNs over CNNs for sequence inputs?
- **Answer:**
  - RNNs inherently maintain a **state across time steps**, allowing them to capture long-range dependencies in sequences. This makes them especially suitable for tasks where past information is crucial for future predictions.

(c) How can Attention be used to process dynamic sized inputs in neural models?
- **Answer:**
  - Attention mechanisms allow the model to **focus on different parts** of the input dynamically based on the context. By weighting parts of the input differently, the model can adapt to different input lengths and prioritize relevant parts of the input.

(d) Can the independence assumptions of all directed probabilistic graphical models be represented by undirected ones?
- **Answer:**
  - No, not all independence assumptions of directed models (Bayesian networks) can be directly represented by undirected models (Markov random fields). Some factorizations in directed models don't have a straightforward equivalent in undirected models.

(e) Why use coordinate ascent in the Expectation Maximization algorithm despite the preference for gradient descent?
- **Answer:**
  - In EM, the objective function is often **decomposed** into components corresponding to different variables or parameters. Coordinate ascent can optimize each component (E-step or M-step) separately, making the optimization more tractable or efficient for such decompositions.

(f) How can we tell the SVM's primal and dual optima are always equal?
- **Answer:**
  - SVM's optimization problem satisfies the **Karush-Kuhn-Tucker (KKT) conditions**, which ensure strong duality. Thus, the primal and dual optima are equal when these conditions are met.

(g) Why are both maximum-likelihood estimators and maximum a posteriori estimators asymptotically efficient?
- **Answer:**
  - Both MLE and MAP estimators achieve the **Cramér-Rao lower bound** asymptotically, meaning their variances approach the minimum possible variance for an unbiased estimator as the sample size grows.

(h) Why would a growth function $ S_F(m) $ of $ 2^m $ be problematic for the PAC bound with growth function for $ F $?
- **Answer:**
  - A growth function of $ 2^m $ indicates that the hypothesis class is too complex, potentially leading to **overfitting**. The PAC bound would be loose or very large, making guarantees on generalization error weak or non-existent.

---

**Section A: Short Answer Questions**

(a) How does the value of the learning rate, $ \eta $ in SGD affect the training progress?
- **Answer:**
  - A **high learning rate** may cause oscillations and overshooting, potentially missing the minimum. A **low learning rate** might converge slowly or get stuck in local minima. An optimal learning rate allows steady convergence to a global or good local minimum.

(b) Why are vanishing gradients more concerning for RNNs than other architectures?
- **Answer:**
  - RNNs process sequences and backpropagate errors through time steps. When gradients are small, they can **vanish** over many time steps, leading to long-range dependencies not being captured effectively and making early layers hard to train.

(c) How do CNNs produce translation invariant representations?
- **Answer:**
  - CNNs use shared weights in their convolutional filters. This allows them to detect features regardless of their position in the input, leading to **translation invariance**. Pooling layers further enhance this property by reducing spatial dimensions.

(d) Can any undirected graphical model's joint probability be expressed as a directed one?
- **Answer:**
  - Not always. Some factorizations and independence assumptions in undirected models (Markov random fields) may not have a straightforward equivalent in directed models (Bayesian networks) and vice versa.

(e) Advantage of using iterative gradient-based optimization when a closed form solution exists?
- **Answer:**
  - While closed-form solutions are computationally efficient, gradient-based methods can be more **flexible**, allowing for regularization, easier integration with other components, or scalability to large datasets.

(f) How does the expectation maximization (EM) algorithm relate to the MLE?
- **Answer:**
  - EM is used to find the **maximum likelihood estimates** (MLE) of parameters when the data has missing or hidden variables. It iteratively estimates the hidden variables (E-step) and then optimizes the parameters (M-step).

(g) How does the MAP estimate relate to the Bayesian posterior distribution over weights?
- **Answer:**
  - The **MAP estimate** is the mode of the Bayesian posterior distribution, i.e., the value of the weights that maximizes the posterior distribution, incorporating both the likelihood of the data and the prior on the weights.

(h) Why do PAC bounds hold "with high probability $ 1 - \delta $" and not deterministically?
- **Answer:**
  - PAC bounds are probabilistic because they account for the **random nature** of drawing a finite sample from a distribution. The bound guarantees that the true risk is close to the empirical risk for most samples, but there's a small probability $ \delta $ where this might not hold.

---

**Section A: Short Answer Questions**

(a) Can training a deep neural network with tanh activation using backpropagation lead to the vanishing gradient problem? Why?
- **Answer:**
  - Yes, because the derivatives of the tanh function are in the range (0,1). For deep networks, multiplying these small derivatives during backpropagation can cause gradients to **vanish** and hinder weight updates in early layers.

(b) Two benefits of max pooling.
- **Answer:**
  - **1. Dimensionality Reduction**: Reduces spatial dimensions, leading to fewer parameters and computational savings.
  - **2. Translation Invariance**: Provides a form of translational invariance, helping the model recognize features regardless of their exact position in the input.

(c) Another use of padding besides preserving spatial dimensions in convolutional layers?
- **Answer:**
  - **Edge Information Preservation**: Padding helps in preserving information at the edges of the input by ensuring that border pixels are adequately involved in convolutions.

(d) Why might VC-dimension based PAC learning theory give impractical risk bounds for deep neural networks?
- **Answer:**
  - Deep neural networks with non-linear activations have a very **high VC-dimension**, which reflects their capacity to fit a large variety of functions. Using VC-dimension in PAC bounds for such models can result in very large or overly pessimistic bounds, making them less informative or useful.

(e) How does Thompson sampling achieve exploration in multi-armed bandit learning?
- **Answer:**
  - Thompson sampling samples from the posterior distribution over each arm's reward. By occasionally sampling optimistic estimates, even for infrequently played arms, it ensures **exploration** of all arms over time.

(f) Why compute evidence for a Bayesian posterior but ignore it for MAP estimation?
- **Answer:**
  - The evidence, which is the denominator in Bayes' rule, acts as a normalizing constant to ensure the posterior is a valid probability distribution. For MAP estimation, we're interested in the **mode** of the posterior, so we can ignore the evidence since it doesn't affect the location of the mode.

(g) How to reduce memory footprint when training a GCN over a very large graph?
- **Answer:**
  - Use **mini-batch training** with **neighbor sampling** or **graph sampling** techniques. Instead of loading the entire graph, load and process smaller subgraphs or neighborhoods iteratively.

(h) Two general strategies for proving that a learning algorithm can be kernelized.
- **Answer:**
  - **1. Representer Theorem**: Demonstrate that the solution to the optimization problem can be expressed as a linear combination of kernel evaluations with training data.
  - **2. Dual Formulation**: Show that the problem's dual formulation only involves dot products between data points, which can then be replaced by kernel functions.

---

