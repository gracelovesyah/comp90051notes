# Review 0
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/9c7af942-d759-448c-a4ce-4e9bdeef5cbc)


## MLE
### Additional Resource
- [Maximum Likelihood For the Normal Distribution, step-by-step!!!](https://www.youtube.com/watch?v=Dn6b9fCIUpM)

## Bayesian
In the realm of statistics and machine learning, Bayesian inference is a method of statistical inference where Bayes' theorem is used to update the probability estimate for a hypothesis as more evidence becomes available. This is especially useful in parameter estimation.

Let's take a simple example: Estimating the bias of a coin.

### Setup

Imagine you have a coin, and you want to estimate the probability $ \theta $ of it landing heads. In a frequentist setting, you might flip it many times and compute the fraction of times it lands heads. In a Bayesian setting, you start with a prior belief about $ \theta $ and update this belief with observed data.

### Bayesian Approach

1. **Prior**: Start with a prior distribution over $ \theta $. A common choice for the bias of a coin is the Beta distribution, which is parameterized by two positive shape parameters, $ \alpha $ and $ \beta $. If you have no strong belief about the coin's bias, you might choose $ \alpha = \beta = 1 $, which is equivalent to a uniform distribution over [0, 1].

2. **Likelihood**: This is the probability of observing the data given a particular value of $ \theta $. If you flip the coin $ n $ times and observe $ h $ heads, the likelihood is given by the binomial distribution:

$$ P(data|\theta) = \binom{n}{h} \theta^h (1-\theta)^{n-h} $$

3. **Posterior**: Using Bayes' theorem, the posterior distribution for $ \theta $ after observing the data is:

$$ P(\theta|data) = \frac{P(data|\theta) \times P(\theta)}{P(data)} $$

$$
Posterior == \frac{Likelihood \cdot Prior}{Evidence}
$$

Given the conjugacy between the Beta prior and the Binomial likelihood, the posterior is also a Beta distribution but with updated parameters: $ \alpha' = \alpha + h $ and $ \beta' = \beta + n - h $.

4. **Estimation**: There are different ways to estimate $ \theta $ from the posterior distribution. A common choice is to use the mean of the posterior, which for a Beta distribution is:

$$ \hat{\theta} = \frac{\alpha'}{\alpha' + \beta'} $$

 
---
The posterior distribution $P(\theta | \text{data})$ represents the updated probability distribution of parameter(s) $\theta$ after incorporating observed data, providing estimates, quantifying uncertainty, and allowing the integration of prior knowledge in Bayesian analysis.


### Intuition

Let's say you start with a completely uniform prior (no knowledge about the coin's bias). After flipping the coin 10 times, you observe 7 heads. Your belief (posterior distribution) about the coin's bias will shift toward 0.7, but it will also consider your initial uncertainty. The more data (coin flips) you observe, the more your belief will be influenced by the observed data relative to the prior.

In Bayesian parameter estimation, rather than obtaining a single "best estimate" value as in frequentist statistics, you obtain a distribution over the possible parameter values that reflects both the data and the prior beliefs.

---
The posterior distribution $P(\theta | \text{data})$ represents the updated probability distribution of parameter(s) $\theta$ after incorporating observed data, providing estimates, quantifying uncertainty, and allowing the integration of prior knowledge in Bayesian analysis.


## Bayesian Linear Regression

Suppose you have data $ \mathbf{X} $ and $ \mathbf{y} $, and you're trying to model the relationship between them using linear regression:

$$ \mathbf{y} = \mathbf{X} \beta + \epsilon $$
where $ \epsilon \sim N(0, \sigma^2I) $ represents the errors.

### 1. Prior:

In the Bayesian approach, you start by defining prior distributions on the parameters you wish to estimate. For simplicity, let's consider priors on the regression coefficients $ \beta $:

$$ \beta \sim N(0, \lambda^2I) $$
This is a normal prior with mean 0 and variance $ \lambda^2 $. The choice of prior can be changed based on domain knowledge or other considerations.

### 2. Likelihood:

Given the linear regression model, the likelihood for the observed data is:

$$ P(\mathbf{y} | \mathbf{X}, \beta) \propto e^{-\frac{1}{2\sigma^2}(\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta)} $$

### 3. Posterior:

Using Bayes' theorem, the posterior distribution for $ \beta $ after observing the data is:

$$ P(\beta | \mathbf{y}, \mathbf{X}) \propto P(\mathbf{y} | \mathbf{X}, \beta) \times P(\beta) $$

Combining the likelihood and prior, the posterior becomes a multivariate normal distribution in the case of Gaussian priors and likelihoods.

### 4. Estimation:

The goal in Bayesian regression isn't just to find point estimates of the parameters (like the MLE in frequentist regression), but rather to describe the entire posterior distribution. This provides insight into the uncertainty associated with parameter estimates.

In practice, due to the high-dimensional nature of the posterior distribution, especially in regression with many predictors, sampling methods like Markov Chain Monte Carlo (MCMC) or Variational Inference might be used to approximate this distribution.

### 5. Prediction:

To make predictions for new data $ \mathbf{X_{new}} $ using the Bayesian model, you would consider the entire posterior distribution of $ \beta $. This results in a distribution over possible predicted values rather than a single predicted value.

### Advantages:

1. **Incorporate Prior Knowledge**: If you have prior knowledge or beliefs about the relationship being modeled, Bayesian methods allow you to incorporate that knowledge.

2. **Parameter Uncertainty**: Bayesian methods provide a full distribution over the parameters, offering a more comprehensive view of parameter uncertainty.

3. **Regularization Effect**: The inclusion of priors can have a regularization effect similar to Ridge or Lasso regression, preventing overfitting.

In summary, Bayesian regression provides a more comprehensive view of uncertainty, allowing for prior beliefs to be incorporated into the model and yielding full posterior distributions of model parameters instead of just point estimates.

## Bayes Theorem
Here's a step-by-step derivation of Bayes' theorem:

**Step 1: Define Conditional Probability**
- We start with the definition of conditional probability, denoted as $P(A | B)$, which represents the probability of event $A$ occurring given that event $B$ has occurred. Mathematically, this is defined as:
  $$P(A | B) = \frac{P(A \cap B)}{P(B)}$$
  Here, $P(A \cap B)$ represents the probability of both events $A$ and $B$ occurring together.

**Step 2: Rearrange the Conditional Probability Formula**
- We can rearrange the above formula to solve for $P(A \cap B)$:
  $$P(A \cap B) = P(A | B) \cdot P(B)$$

**Step 3: Apply the Symmetry of Conditional Probability**
- We know that $P(A \cap B) = P(B \cap A)$. This is because the order in which we consider events does not affect the probability of their intersection.

**Step 4: Express $P(B \cap A)$ in Terms of Conditional Probability**
- We can use the definition of conditional probability to express $P(B \cap A)$ as follows:
  $$P(B \cap A) = P(B | A) \cdot P(A)$$

**Step 5: Equate the Expressions for $P(A \cap B)$**
- Equate the expressions for $P(A \cap B)$ obtained in Step 2 and Step 4:
  $$P(A | B) \cdot P(B) = P(B | A) \cdot P(A)$$

**Step 6: Solve for Bayes' Theorem**
- Finally, we can rearrange the equation from Step 5 to obtain Bayes' theorem in its general form:
  $$P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$$

This is the general form of Bayes' theorem. It provides a way to update the probability of an event $A$ in light of new evidence or information provided by event $B$. The theorem is widely used in statistics, machine learning, and various fields for tasks such as probabilistic inference, Bayesian statistics, and Bayesian reasoning.

---

## Bayesian Regression 

### Equation Breakdown:

$$ p(y^*|x^*, x, y) = \int p(y^*|x^*, w) \times p(w|x, y) \, dw $$

Here:
- $ y^* $ is a new output (or set of outputs) that you're trying to predict.
- $ x^* $ is the new input (or set of inputs) corresponding to $ y^* $.
- $ x $ and $ y $ are the observed inputs and outputs (your data).
- $ w $ represents the parameters of the model.
- $ p(y^*|x^*, w) $ is the likelihood of observing $ y^* $ given the new input $ x^* $ and some fixed parameters $ w $.
- $ p(w|x, y) $ is the posterior distribution of the parameters $ w $ given the observed data $ x $ and $ y $.

### Interpretation:

1. **Posterior Predictive Distribution $ p(y^*|x^*, x, y) $**:
   - This is the distribution of the new outputs $ y^* $ given new inputs $ x^* $ and the observed data $ x, y $.
   - It integrates over all possible parameter values $ w $ weighted by their probability given the observed data.

2. **Likelihood $ p(y^*|x^*, w) $**:
   - This term describes how likely the new outputs $ y^* $ are for given inputs $ x^* $ if the model parameters were fixed at $ w $.

3. **Posterior $ p(w|x, y) $**:
   - This term represents the updated beliefs about the model parameters $ w $ after observing the data $ x $ and $ y $.

4. **Integration over $ w $**:
   - The integration accounts for the uncertainty in the model parameters. Instead of just making a prediction based on a single "best" estimate of $ w $, you're averaging predictions across all possible values of $ w $, weighted by their posterior probability.
   - This results in a more robust prediction that accounts for parameter uncertainty.

### Conclusion:

The equation is essentially saying: "To predict a new output $ y^* $ given new inputs $ x^* $ and the observed data $ x, y $, consider all possible parameter values $ w $, weigh the predictions for each parameter value by the probability of that parameter value given the data, and then sum (integrate) these weighted predictions."

This approach embodies a key tenet of Bayesian reasoning: consider all possibilities and weigh them by their probabilities. This leads to predictions that account for both the uncertainty in the model parameters and the inherent variability in the data-generating process.