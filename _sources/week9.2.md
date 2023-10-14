# Bayesian regression
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/102a61a5-682c-499c-9004-1f43c7b798ed)



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