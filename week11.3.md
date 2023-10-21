# Lecture 22. Inference on PGMs
This note is completed with the assistance of [ChatGPT](https://chat.openai.com/c/c5be32aa-672c-4e59-8a2c-85704a846b8a)

**Lecture Notes: Probabilistic Inference in Probabilistic Graphical Models (PGMs)**

---
## Summary

**1. Exact Probabilistic Inference on PGMs**:
   - **Concept**: Calculate exact probabilities by marginalizing out certain variables and conditioning on others.
   - **Reasoning**: Useful for tasks like Bayesian posterior updates.
   - **Algorithm**: The elimination algorithm helps in sequentially removing nodes from the PGM to simplify the computation.
   - **Math**: Involves operations like matrix multiplication and summation over variables.

---

**2. Probabilistic Inference by Simulation**:
   - **Concept**: Approximate probabilistic distributions through sampling methods.
   - **Reasoning**: Exact inference can be computationally expensive or analytically impossible.
   - **Example**: Approximating a distribution using histograms of samples.

---

**3. Gibbs Sampling**:
   - **Concept**: An MCMC method that samples one variable at a time from its conditional distribution.
   - **Reasoning**: A divide and conquer approach to sampling; more computationally feasible than sampling all variables simultaneously.
   - **Math**: Relies on the conditional distributions of the PGM.
   - **Example**: Given a set of observed variables (evidence nodes), we can iteratively sample the other variables to get an approximate joint distribution.

---

**4. Markov Blanket**:
   - **Concept**: For any given node, its Markov blanket consists of its parents, children, and parents of its children.
   - **Reasoning**: Knowing the state of the nodes in the Markov blanket renders the node conditionally independent of all other nodes.
   - **Math**: The conditional distribution of the node given its Markov blanket is proportional to the product of its local conditional distribution and the conditional distributions of its children.

---

**5. Markov Chain Monte Carlo (MCMC)**:
   - **Concept**: A class of algorithms that sample from a probability distribution by constructing a Markov chain.
   - **Reasoning**: Useful when it's difficult to sample directly from a distribution.
   - **Key terms**: Burn-in (discard initial samples), Thinning (reduce correlation by spacing out samples), Limiting distribution.

---

**6. Initializing Gibbs via Forward Sampling**:
   - **Concept**: Start Gibbs sampling by setting evidence nodes and sampling the remaining nodes in a parent-first order.
   - **Reasoning**: Gives the Markov chain a starting point, although the chain's convergence isn't sensitive to initial values.

---

**7. Application of Gibbs Samples**:
   - **Concept**: Use the obtained samples to approximate different properties of the target distribution.
   - **Examples**: Creating histograms, estimating means, computing marginals, and posterior computations.
   - **Math**: Statistical estimates based on sample means, counts, and other summary statistics.

---

**Conclusion**: The lecture delved into both exact and approximate inference methods in PGMs. While exact methods provide precise answers, approximate methods, especially Gibbs sampling, offer practical solutions to complex inference problems. The Markov blanket concept further emphasized the localized dependencies in PGMs.

## Math 


---

**1. Elimination Algorithm in PGMs:**
- **Initialization**:
$$ 
\text{Append } Pr(X_i | \text{parents}(X_i)) \text{ to active} 
$$

- **Evidence Nodes**:

$$
 \text{Append } \delta(X_i, x_i) \text{ to active} 
$$

- **Elimination Steps**:
   - Potentials:

$$
 \text{Remove tables referencing } X_i \text{ from active} 
$$

   - New Nodes:

$$
 N_i = \text{nodes other than } X_i \text{ referenced by tables} 
$$

   - Table Computation:

$$
 \phi_i(X_i, X_{!i}) = \text{product of tables} 
$$


$$
 m_i X_{!i} = \sum_{!i} \phi_i(X_i, X_{!i}) 
$$

   - Final Probability:

$$
 Pr(X_Q|X_E = x_E) = \frac{\phi_Q(X_Q)}{\sum_{X_Q} \phi_Q(X_Q)} 
$$


---

**2. Gibbs Sampling:**
Given a D-PGM on 𝑑 random variables and evidence values:

$$
 X_t = \text{Sample } p(X_i | X_1, ..., X_{i-1}, X_{i+1}, ..., X_d) 
$$

The sample collection over iterations will approximate the desired posterior.

---

**3. Markov Blanket:**

$$
 p(X_i | X_{-i}) \propto p(X_i | X_{parents(i)}) \prod_{j: i \in parents(j)} p(X_j | X_{parents(j)}) 
$$

Where $X_{-i}$ refers to all variables except $X_i$ and $parents(i)$ and $parents(j)$ denote the parent nodes of $X_i$ and $X_j$ respectively.

---

**4. Markov Chain Monte Carlo (MCMC):**
Transition in Markov Chain:

$$
 p_t = T^t p_0 
$$

Where:
   - $p_t$ is the distribution at time $t$.
   - $T$ is the transition matrix of the Markov chain.
   - $p_0$ is the initial distribution.

---

**5. Using Gibbs Samples:**

$$
 \text{Expectation of a function } f: E[f(X)] \approx \frac{1}{N} \sum_{i=1}^{N} f(X^{(i)}) 
$$

Where $X^{(i)}$ are the samples from Gibbs sampling.

---

These equations form the mathematical backbone of the discussed concepts. They provide a means to execute the processes of inference in PGMs, whether that's through exact methods like elimination or approximate methods like Gibbs sampling.