# Bayesian classification


### **Mini Summary**

- **Bayesian Ideas in Discrete Settings**
  * Beta-Binomial conjugacy
  * Conjugate pairs; Uniqueness in proportionality

- **Bayesian Classification (Logistic Regression)**
  * Non-conjugacy necessitates approximation

- **Rejection Sampling**
  * Monte Carlo sampling: A classic method to approximate posterior

- **Next time: probabilistic graphical models**


#### **1. Bayesian View on Discrete Data:**
- **Generative vs. Discriminative Models**: 
  - Generative models capture the joint distribution $ p(x, y) $.
  - Discriminative models condition on the input, capturing $ p(y | x) $.
  - Example: Naïve Bayes (generative) vs. Logistic Regression (discriminative).

#### **2. Beta-Binomial Conjugacy:**
- **Likelihood**: $ p(k|n, q) = \binom{n}{k} q^k (1 - q)^{n-k} $.
- **Prior**: $ p(q) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} q^{\alpha-1} (1 - q)^{\beta-1} $.
- **Posterior**: $ p(q|k, n) = \text{Beta}(q; k + \alpha, n - k + \beta) $.

#### **3. Uniqueness up to Normalization:**
- If an unnormalized distribution is proportional to a recognized distribution, it's essentially that distribution once normalized.
- $ f(\theta) \propto g(\theta) $ implies $ f(\theta) = C \times g(\theta) $ with normalization constant $ C $.

#### **4. Laplace's Sunrise Problem:**
- Predict the probability of the sun rising based on historical observations.
- Bayesian approach with Beta-Binomial: $ p(q|k) = \text{Beta}(q; k + 1, 1) $.
- Expected probability: $ E_p[q|k] = \frac{k + 1}{k + 2} $.

#### **5. Bayesian Logistic Regression:**
- **Discriminative Classifier**: $ p(y=1|x, w) = \frac{1}{1 + e^{-w^T x}} $.
- **Gaussian Prior**: $ p(w) = \text{Normal}(0, \sigma^2I) $.
- **Inference**: Due to non-conjugacy, use approximation methods like MCMC, Laplace Approximation, or Variational Inference.

#### **6. Laplace Approximation:**
- Approximate a complex distribution with a Gaussian centered at its mode.
- **Procedure**:
  - Find the mode (MAP estimate).
  - Determine curvature using the Hessian matrix.
  - Form the Gaussian approximation.

---

### rejection sampling

- Additional resource
    - [ChatNote](https://chat.openai.com/c/a3c8924c-dea7-4051-8d5c-3aa0e251a2cb)
    - [Rejection Sampling - VISUALLY EXPLAINED with EXAMPLES!](https://youtu.be/si76S7QqxTU)
    - [What is Rejection Sampling?](https://towardsdatascience.com/what-is-rejection-sampling-1f6aff92330d)
    - [An introduction to rejection sampling](https://www.youtube.com/watch?v=kYWHfgkRc9s)


1. **Objective**:
   The goal of rejection sampling is to generate samples from a target distribution, in this case, the posterior $ p(\theta|y) $. However, directly sampling from this distribution might be difficult. So, we use an auxiliary or proposal distribution, denoted as $ g(\theta) $, from which we can easily sample.

2. **Un-normalised Density**:
   Often in Bayesian statistics, the exact posterior $ p(\theta|y) $ is hard to compute due to the denominator (the evidence or marginal likelihood). Instead, we work with an un-normalised version, $ q(\theta|y) $, which is proportional to the true posterior but doesn't necessarily integrate to 1.

3. **Condition**:
   The inequality $ q(\theta|y) \leq M' \times g(\theta) $ for all $ \theta $ means that when we scale the proposal distribution $ g(\theta) $ by a factor $ M' $, it should always be above or equal to the un-normalised posterior $ q(\theta|y) $. This ensures that the scaled proposal distribution "encases" or "envelopes" the target distribution.

4. **Proposal Distribution Details**:
   In the example you provided, $ g(\theta) = 0.5 $ is a constant function, and $ \theta $ is sampled from a uniform distribution between -1 and 1, denoted as $ U(-1,1) $. This means that for any $ \theta $ in the interval [-1, 1], the value of $ g(\theta) $ is 0.5.

Given this setup, the process of rejection sampling would be:

a. Sample a value $ \theta $ from $ U(-1,1) $.

b. Evaluate the un-normalised posterior $ q(\theta|y) $ at this $ \theta $.

c. Sample a value $ u $ from a uniform distribution between 0 and $ 0.5 \times M' $.

d. If $ u $ is less than $ q(\theta|y) $, accept $ \theta $ as a sample from the posterior. Otherwise, reject it.
