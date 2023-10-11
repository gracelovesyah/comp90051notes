# PGM Representation
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/7887d9af-ce1c-4551-8f91-c576874448be)

**Lecture 20: PGM Representation (COMP90051 Statistical Machine Learning)**

- **Probabilistic Graphical Models**
  * Motivation: applications, unifies algorithms
  * Motivation: ideal tool for Bayesians
  * Independence lowers computational/model complexity
  * PGMs: compact representation of factorised joints
  * U-PGMs

- **Example PGMs and Applications**

- **Additional Resource**
- ((ML 13.1) Directed graphical models - introductory examples (part 1)
)[https://www.youtube.com/watch?v=3XysEf3IQN4]


- **Next time: elimination for probabilistic inference**


**Topics Covered:**

1. **(Directed) Probabilistic Graphical Models (D-PGMs)**
   - These are graphical representations of probability distributions. The directed nature means that the graph has arrows indicating a direction from one variable to another, representing causal relationships.

2. **Motivations for Using PGMs:**
   - **Applications & Unification of Algorithms:** PGMs are used in various applications and can unify different algorithms under a single framework.
   - **Ideal Tool for Bayesians:** Bayesians use probability distributions to represent uncertainty. PGMs provide a structured way to represent these distributions, making them an ideal tool for Bayesian analysis.

3. **Importance of Independence:**
   - **Lowers Computational/Model Complexity:** When variables are independent, it simplifies computations and the model itself. This is because you don't need to account for relationships between every pair of variables.
   - **Conditional Independence:** This is a specific type of independence where two variables are independent given the value of a third variable. For example, if A and B are conditionally independent given C, then knowing the value of C breaks any dependence between A and B.

4. **PGMs as a Compact Representation:**
   - **Factorized Joints:** PGMs allow for a compact representation of joint probability distributions by breaking them down into smaller factors. This makes computations more efficient.

5. **Undirected PGMs and Conversion from D-PGMs:**
   - While directed PGMs represent causal relationships, undirected PGMs represent non-causal relationships between variables. There are methods to convert between the two types.

6. **Example PGMs & Applications:**
   - This section will likely provide specific examples of PGMs and how they are applied in real-world scenarios.

**Notes:**
- PGMs provide a visual representation of complex probability distributions.
- They can represent both causal (directed) and non-causal (undirected) relationships.
- Independence and conditional independence are crucial concepts in PGMs, simplifying computations and the model.
- PGMs are widely used in machine learning and statistics for various applications.

---



**Lecture Summary: Probabilistic Graphical Models (PGMs) in Statistical Machine Learning**

1. **Introduction to PGMs:**
   - PGMs provide a compact representation of factorized joint distributions, making them ideal for Bayesian modeling.

2. **Joint Distributions:**
   - Represented as $Pr(X_1, X_2, ..., X_n)$. Directly working with these can be computationally intensive, especially as the number of variables increases.

3. **Probabilistic Inference:**
   - Uses Bayes Rule: 

     $$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} $$

   - And Marginalisation: 

     $$ P(A) = \sum_{B} P(A, B) $$

   These tools allow us to derive probabilities and update beliefs based on new evidence.

4. **Factoring Joint Distributions:**
   - Chain Rule:

     $$ Pr(X_1, X_2, ..., X_n) = \prod_{i=1}^{n} Pr(X_i | X_1, ..., X_{i-1}) $$

   This expresses joint distributions as products of conditional probabilities. Independence assumptions can simplify these products.

5. **Directed PGMs (Bayesian Networks):**
   - Represented with nodes (random variables) and acyclic edges (conditional dependencies).
   - Joint distribution factorization:

     $$ Pr(X_1, X_2, ..., X_n) = \prod_{i} Pr(X_i | parents(X_i)) $$

   This shows how the joint probability is a product of the probabilities of each variable given its parents in the graph.

6. **Naïve Bayes Classifier:**
   - Assumes features $X_1, ..., X_d$ are conditionally independent given class label $Y$.
   - Joint probability:

     $$ Pr(Y, X_1, ..., X_d) = Pr(Y) \times \prod_{i=1}^{d} Pr(X_i|Y) $$
     
   For prediction, it selects the $Y$ that maximizes $Pr(Y|X_1, ..., X_d)$.

7. **Benefits of PGMs:**
   - They allow for structured, efficient, and intuitive probabilistic modeling. The factorization and independence assumptions reduce the computational burden and the risk of overfitting.
