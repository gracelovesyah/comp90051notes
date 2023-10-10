# PGM Representation
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/7887d9af-ce1c-4551-8f91-c576874448be)


**Lecture 20: PGM Representation (COMP90051 Statistical Machine Learning)**

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

Feel free to provide more details or ask questions about specific points!