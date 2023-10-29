# Basic Concepts

---

### **Probability Basics**
1. **Probability of an Event**:

$$ P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}} $$

2. **Conditional Probability**:

$$ P(A|B) = \frac{P(A \cap B)}{P(B)} $$

3. **Bayes' Theorem**:

$$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} $$

4. **Law of Total Probability**:

$$ P(A) = \sum_i P(A|B_i) \times P(B_i) $$


### **Random Variables**
1. **Expected Value**:

$$ \mathbb{E}[X] = \sum x_i P(X = x_i) $$
 (for discrete)

$$ \mathbb{E}[X] = \int x f(x) \, dx $$
 (for continuous)
2. **Variance**:

$$ \text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 $$

3. **Standard Deviation**:

$$ \sigma_X = \sqrt{\text{Var}(X)} $$


### **Common Distributions**
1. **Binomial** (n trials, p success probability):

$$ P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} $$

2. **Poisson** (λ rate):

$$ P(X=k) = \frac{e^{-\lambda} \lambda^k}{k!} $$

3. **Normal** (μ mean, σ standard deviation):

$$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} $$


### **Descriptive Statistics**
1. **Mean**:

$$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $$

2. **Sample Variance**:

$$ s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2 $$

3. **Correlation**:

$$ r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}} $$


### **Inferential Statistics**
1. **Z-Score**:

$$ z = \frac{x - \mu}{\sigma} $$

2. **t-Score** (for sample):

$$ t = \frac{\bar{x} - \mu}{s/\sqrt{n}} $$


### **Regression**
1. **Simple Linear Regression**:

$$ y = \beta_0 + \beta_1 x $$


$$ \beta_1 = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sum (x_i-\bar{x})^2} $$


$$ \beta_0 = \bar{y} - \beta_1 \bar{x} $$


---


## OWLS
06/10/2023
The equation $ \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} $ is the solution for the weights $ \mathbf{w}^* $ in linear regression using the method of least squares. Let's break down how we arrive at this solution:

---

**Derivation of the Optimal Weights in Linear Regression (OWLS第一万遍)**:
[Regression Modelling And Least-Squares](https://www.ma.imperial.ac.uk/~das01/GSACourse/Regression.pdf)
1. **Objective**:
   - In linear regression, we aim to minimize the sum of squared residuals (SSR), which is the difference between the observed values ($ \mathbf{y} $) and the predicted values ($ \mathbf{Xw} $).
   - The objective function (SSR) is given by:

     $$ SSR(\mathbf{w}) = (\mathbf{y} - \mathbf{Xw})^T (\mathbf{y} - \mathbf{Xw}) $$


2. **Differentiating with Respect to $ \mathbf{w} $**:
   - To find the weights that minimize the SSR, we differentiate the SSR with respect to $ \mathbf{w} $ and set the result to zero. This will give us the weights that minimize the SSR.
   - Differentiating, we get:

     $$ \frac{\partial SSR}{\partial \mathbf{w}} = -2\mathbf{X}^T (\mathbf{y} - \mathbf{Xw}) $$


3. **Setting the Gradient to Zero**:
   - For the minimum, we set the gradient to zero:

     $$ -2\mathbf{X}^T (\mathbf{y} - \mathbf{Xw}) = 0 $$

   - This simplifies to:

     $$ \mathbf{X}^T \mathbf{y} = \mathbf{X}^T \mathbf{Xw} $$

4. **Solving for $ \mathbf{w}^* $**:
   - To isolate $ \mathbf{w} $, we multiply both sides by the inverse of $ (\mathbf{X}^T \mathbf{X}) $:
   
     $$ \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} $$

---

This solution provides the optimal weights for a linear regression model in the least squares sense. It's worth noting that this solution assumes that $ (\mathbf{X}^T \mathbf{X}) $ is invertible. If it's not (e.g., in cases of multicollinearity), then additional techniques or regularization methods might be needed.

---
## Fisher Info

**Fisher Information Matrix (FIM)第一万次**:

1. **Definition**:
   - For a single parameter $ \theta $, the Fisher Information $ I(\theta) $ is defined as the variance of the score, which is the derivative of the log-likelihood:

     $$ I(\theta) = \text{Var}\left( \frac{\partial \log p(x|\theta)}{\partial \theta} \right) $$

   - For multiple parameters, the Fisher Information Matrix is defined as:

     $$ I_{ij}(\theta) = E\left[ \frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j} \right] $$

     where the expectation is taken over the distribution of the data $ x $.

2. **Interpretation**:
   - The FIM quantifies how much information the data provides about the parameters of the model.
   - Larger values in the FIM indicate that small changes in the parameter $ \theta $ would result in larger changes in the likelihood, meaning the data is more informative about that parameter.

3. **Properties**:
   - **Invertibility**: The inverse of the FIM (if it's invertible) provides a lower bound on the variance of any unbiased estimator of the parameter $ \theta $. This is known as the Cramér-Rao bound.
   - **Additivity**: For independent observations, the Fisher Information is additive. That is, the information from each observation can be summed up.

4. **Applications**:
   - **Parameter Estimation**: The FIM is used in the context of the Cramér-Rao bound to determine the minimum variance (best precision) that can be achieved by an unbiased estimator.
   - **Experimental Design**: In experimental settings, the FIM can be used to design experiments that maximize the information about parameters of interest.
   - **Model Comparison**: Models can be compared based on the amount of information they provide about parameters or underlying processes.

5. **Connection to Linear Regression**:
   - In the context of linear regression with Gaussian errors, the Fisher Information matrix is proportional to $ \mathbf{X}^T \mathbf{X} $, where $ \mathbf{X} $ is the design matrix. This relationship is particularly clear when considering the maximum likelihood estimation in linear regression.

---

In essence, the Fisher Information Matrix is a measure of how sensitive the likelihood function is to changes in the parameters. It plays a crucial role in understanding the precision and reliability of parameter estimates in statistical models.

Certainly! Let's use a simple example to illustrate the concepts of incorporating prior beliefs and uncertainty quantification in Bayesian linear regression.

---

07/10/2023
## Multivariate Gaussian
[(PP 6.1) Multivariate Gaussian - definition](https://www.youtube.com/watch?v=TC0ZAX3DA88)

---

14/10/2023
## Learning Rate
[Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8)
- In gradient descent, the learning rate represents the size of the steps taken during each parameter update and affects the trade-off between convergence speed and stability of the optimization process.
- Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient. 

```{image} ./images/learningrate1.png
:alt: learningrate1
:class: bg-primary mb-1
:width: 800px
:align: center
```

Absolutely, the concept of hidden layers can be abstract and sometimes confusing, especially when you're first getting into deep learning. Let's break it down:

17/10/2023
### Hidden Layers in Deep Learning:

1. **Basic Idea**: 
   - At the highest level, a neural network is designed to recognize patterns. It does this by taking input data and transforming it through multiple layers to produce an output.
   - Hidden layers are those layers in between the input and output layers. They are called "hidden" because we don't directly interact with them (like providing input or extracting output).

2. **Transformation of Data**: 
   - Each hidden layer transforms the data a little bit, extracting and refining features as data moves through the network.
   - The initial layers (closer to the input) might recognize basic features, like edges in the context of an image. As you progress deeper into the network, the layers recognize more complex patterns, like shapes, textures, and even entire objects.

3. **Neurons and Activations**: 
   - Each layer is made up of "neurons" that have associated weights and biases. These weights are adjusted during training.
   - The "activation" of a neuron is a value that results from passing an input (or multiple inputs) through a neuron, which processes it using a weighted sum and a specific activation function, like ReLU or Sigmoid.

4. **Why Not Directly From Input to Output?**:
   - Hidden layers allow the network to learn and represent hierarchical features. A direct input-to-output model wouldn't be able to capture complex relationships in the data.
   - Essentially, with each subsequent hidden layer, the model can represent more abstract and complex features, helping it generalize better to unseen data.

5. **Analogy**:
   - Think of learning a complex topic. You start with basics, and as you understand them, you move to more complex concepts built on those basics. Hidden layers in a neural network function similarly. The initial layers capture the basics, and as data progresses, each layer builds on the features extracted by the previous one.

6. **Utility in Practice**:
   - Hidden layers are essential for deep learning models to function effectively. Without them, the models wouldn't be "deep," and they'd lose their capacity to model complex data and relationships.
   - The intermediate representations in hidden layers are not typically interpretable by humans. However, they are valuable intermediate steps for the model to produce the final result.

### Conclusion:

In essence, hidden layers can be seen as a series of transformations that refine and process the input data to produce an output. They enable deep learning models to capture intricate patterns and relationships in the data, making these models powerful tools for a range of tasks.

---
18/10/2023
## Likelihood and Prob
[Probability is not Likelihood. Find out why!!!](https://www.youtube.com/watch?v=pYxNSUDSFH4)

```{image} ./images/likelihood.png
:alt: likelihood
:class: bg-primary mb-1
:width: 800px
:align: center
```
