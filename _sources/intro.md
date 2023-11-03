# Welcome to COMP90051

This is a notebook to COMP90051 Statistical Machine Learning @ Unimelb

This note is composed by Jiahe (Grace) Liu (jiahe3@student.unimelb.edu.au) with assistance of ChatGPT.

Check out the content pages bundled with this sample book to see more.

```{tableofcontents}
```

---

### 1. **Machine Learning Theory**
*Pro Tip: Deepen your understanding of the fundamental concepts and theories to provide a solid foundation for your machine learning journey.*

**Questions:**
- Why are maximum likelihood estimation, max a posteriori, and empirical risk minimization all instances of extremum estimators?
- Why was there a third irreducible error term in the bias-variance decomposition for supervised regression but not in parameter estimation?
- How do the frequentist and Bayesian approaches differ in their modeling of unknown parameters?
- Why might VC-dimension based PAC learning theory give impractical risk bounds for deep neural networks?

---

### 2. **Support Vector Machines (SVM)**
*Pro Tip: Understanding the intricacies of SVM helps in harnessing its power for classification and regression problems.*

**Questions:**
- Explain a benefit of using the kernel trick with a polynomial kernel of degree $ p = 12 $ in SVM vs explicit basis expansion.
- How are support vectors characterized by dual variables $ \lambda_i $ in the soft-margin SVM?
- How is the objective function for soft-margin SVM a relaxation of the hard-margin SVM objective?
- Explain how support vectors can have a margin of 1 from the decision boundary but be further than 1 unit away in Euclidean space.
- Strategies to change the hyperparameters of the soft-margin SVM with a RBF kernel for better performance?

---

### 3. **Deep Learning**
*Pro Tip: Deep learning has revolutionized many domains. Staying updated with its nuances is essential for state-of-the-art results.*

**Questions:**
- In what respect is a recurrent neural network deep?
- How are artificial neural networks a form of non-linear basis function in learning a linear model?
- Can training a deep neural network with tanh activation using backpropagation lead to the vanishing gradient problem? Why?
- Another use of padding besides preserving spatial dimensions in convolutional layers?
- Key benefit of RNNs over CNNs for sequence inputs?
- How can Attention be used to process dynamic sized inputs in neural models?
- Why use backpropagation through time for RNNs?
- How are RNNs different from CNNs when handling sequential data?

---

### 4. **Optimization and Gradient Descent**
*Pro Tip: Effective optimization techniques are at the heart of training machine learning models.*

**Questions:**
- Describe a problem with gradient descent during training and one approach to mitigate it.
- How does the dynamic learning rate in Adagrad operate and its importance?
- How does the value of the learning rate, $ \eta $ in SGD affect the training progress?
- How is the dynamic learning rate in Adagrad different from that in RMSProp?
- Main drawback of AdaGrad compared to RMSProp?
- How can momentum in an optimizer help avoid local optima or saddle points?

---

### 5. **Probabilistic Models and Expectation Maximization**
*Pro Tip: Probabilistic models provide a principled way to handle uncertainty in data.*

**Questions:**
- How do the frequentist and Bayesian approaches differ in their modeling of unknown parameters?
- How can the posterior variance vary for different test points in Bayesian linear regression?
- Why use coordinate ascent in the Expectation Maximization algorithm despite the preference for gradient descent?
- Objective function optimized during GMM training?
- How does the expectation maximization (EM) algorithm relate to the MLE?
- Benefit of using the Gaussian mixture model (GMM) over k-means clustering.

---

### 6. **Other Machine Learning Techniques**
*Pro Tip: Diverse machine learning techniques cater to different types of data and problem settings.*

**Questions:**
- How are generative models different from discriminative models?
- Why are training examples weighted within the AdaBoost algorithm?
- Describe where within multi-armed bandit algorithms like ε-greedy, statistical estimation is performed and how these estimates are used.
- How does Thompson sampling achieve exploration in multi-armed bandit learning?
- How to reduce memory footprint when training a GCN over a very large graph?
- Strategy that allows VAE to apply gradient descent through the samples of latent representation $ z $?

---

### 7. **Definitions and Concepts**
*Pro Tip: Having clear definitions and understanding core concepts is foundational for any machine learning practitioner.*

**Questions:**
- Define the tree width of a directed graph $ G $ in words.
- Can any undirected graphical model's joint probability be expressed as a directed one?
- Two benefits of max pooling.
- Two general strategies for proving that a learning algorithm can be kernelized.

---
