# Bayesian regression
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/102a61a5-682c-499c-9004-1f43c7b798ed)

The equation $ \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} $ is the solution for the weights $ \mathbf{w}^* $ in linear regression using the method of least squares. Let's break down how we arrive at this solution:

---

**Derivation of the Optimal Weights in Linear Regression (OWLS第一万遍)**:

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

**Example: House Price Prediction**

Imagine you're trying to predict the price of houses based on their size (in square feet). You decide to use linear regression for this task.

**1. Prior Beliefs**:
Suppose you've been in the real estate business for a while, and you believe that, on average, houses are priced at about $100 per square foot, but you're not entirely sure. This belief can be encoded as a prior distribution on the slope of your regression line. 

You might choose a Normal distribution centered at 100 (reflecting your belief about the average price per square foot) with a certain standard deviation (reflecting your uncertainty about this belief). 

Mathematically, this might look like:

$$ w \sim \text{Normal}(100, 500) $$

Here, $ w $ is the slope of the regression line, and the large standard deviation (500) indicates you're quite uncertain about your belief.

**2. Observing Data**:
Now, you collect data on 50 houses – their sizes and their prices. Some houses are $80 per square foot, some are $120, and so on. You want to use this data to refine your beliefs about the average price per square foot.

**3. Posterior Distribution**:
Using Bayesian linear regression, you combine your prior beliefs with this new data to compute the posterior distribution of $ w $. 

After doing the math, you might find that the posterior distribution is:

$$ w|data \sim \text{Normal}(105, 50) $$

This tells you that, after seeing the data, your updated belief about the average price per square foot is $105, and you're more certain about this (as reflected by the smaller standard deviation of 50).

**4. Uncertainty Quantification**:
The posterior distribution not only gives you an updated point estimate (the mean, $105/sqft) but also a measure of uncertainty (the standard deviation, $50). This means that while you believe the average price is around $105/sqft, it could reasonably vary by about $50 in either direction due to the variability in the data and your prior uncertainty.

---

In this example, the Bayesian approach allowed you to incorporate your initial beliefs about house prices and then update those beliefs with observed data. The resulting posterior distribution provides both an updated estimate and a quantification of the uncertainty surrounding that estimate.


---
## 9.2 Bayesian Regression: Detailed Summary




**1. Introduction:**
- Traditional regression methods, like MLE and MAP, provide point estimates of parameters. These don't capture the inherent uncertainty about parameters.
- Bayesian regression offers a probabilistic approach, preserving and quantifying this uncertainty.

**2. Key Concepts & Questions Addressed:**

- **Point Estimates vs. Full Distribution**:
  - Point estimates, like $ \hat{w} $, represent single "best" values for parameters. They don't capture the inherent uncertainty about the parameters.
  - Bayesian regression focuses on the entire distribution of plausible parameter values, allowing for a richer representation of uncertainty.

- **Sequential Bayesian Updating**: 
  - This is the process of continuously refining the posterior distribution as new data is observed. The current posterior becomes the prior for the next update, allowing for a dynamic and adaptive model.

- **Conjugate Priors**: 
  - These are priors that, when combined with a specific likelihood, result in a posterior distribution of the same family as the prior. This property simplifies the computation of the posterior. For Bayesian linear regression, the Normal-Normal conjugacy is often used.

- **Why Gaussian for Prior?**
  - Gaussian distributions are mathematically tractable, especially when combined with linear models. Using a Gaussian prior centered at zero implies a belief that weights are likely to be small (or close to zero) unless the data provides strong evidence to the contrary.

**3. Mathematical Formulations:**

- **Prior Distribution**: 
  - $ w \sim Normal(0, \sigma^2 I_D) $
  - This represents our prior belief about the weights before observing any data.

- **Likelihood**: 
  - $ y \sim Normal(Xw, \sigma^2) $
  - This represents the probability of observing the data given the weights.

- **Posterior Distribution**:
  - Using Bayes' theorem: $ p(w|X, y, \sigma^2) \propto p(y|X, w, \sigma^2) p(w) $
  - Results in a Gaussian: $ Normal(w|w_N, V_N) $
  - Where:
    - $ w_N = \sigma^2 (X^T X + \sigma^2 I_D)^{-1} X^T y $
    - $ V_N = \sigma^2 (X^T X + \sigma^2 I_D)^{-1} $
  - The posterior mean $ w_N $ is analogous to the point estimate in ridge regression, with the regularization term emerging naturally from the prior.

- **Predictive Distribution**:
  - $ p(y_*|x_*, X, y, \sigma^2) = \int p(y_*|x_*, w, \sigma^2) p(w|X, y, \sigma^2) dw $
  - Results in a Gaussian: $ Normal(y_*|x^T_* w_N, \sigma^2_N(x_*)) $
  - Where:
    - $ \sigma^2_N(x_*) = \sigma^2 + x^T_* V_N x_* $
  - This distribution provides both a mean prediction and a measure of uncertainty (variance) for new data points.

**4. Advantages & Insights from Bayesian Regression**:

- **Uncertainty Quantification**: Bayesian regression captures uncertainty in parameter estimates, providing a full probabilistic description.
  
- **Robustness**: It's more robust to overfitting, especially with small datasets. As you noted, Bayesian methods shine with small data and high uncertainty, while frequentist methods are more suited for larger datasets with low uncertainty.

- **Predictions vs. Parameters**: In Bayesian thinking, the emphasis is more on making accurate predictions rather than precisely determining parameter values.

**5. Future Topics**:

- Bayesian Classification: Extending Bayesian principles to classification tasks.
- Probabilistic Graphical Models (PGMs): Representing complex probabilistic relationships using graphs.

---

This detailed summary incorporates the nuances and questions you raised, providing a comprehensive overview of Bayesian Regression.