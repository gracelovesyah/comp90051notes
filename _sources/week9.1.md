# Stochastic Multi-Armed Bandits (MABs)

This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/102a61a5-682c-499c-9004-1f43c7b798ed)

---

**1. Introduction to MABs**:
- **Definition**: A decision-maker faces $ k $ options (arms) and must choose one at a time to receive a reward. The goal is to maximize the cumulative reward over time.
- **Challenge**: Balancing exploration (trying out different options) with exploitation (leveraging known information).

**2. Bandit Setting vs. Learning with Experts**:
- **Expert Selection**: In the expert setting, one has to pick an expert (akin to an arm in MABs).
- **Observation Limitation**: Rewards are observed only for the chosen arm/expert.

**3. Exploration vs. Exploitation**:
- **MABs**: Represent the simplest setting for balancing exploration and exploitation.

**4. Stochastic MAB Setting**:
- **Actions**: Possible actions are represented by arms.
- **Reward Distribution**: Each arm $ i $ has a reward distribution $ P_i $ with mean $ \mu_i $.
- **Goal**: Minimize cumulative regret, defined as:

$$ \mu^*T - \sum_{t=1}^{T} E[R_{t,i_t}] $$

Where $ \mu^* $ is the maximum mean reward across all arms.

**5. Greedy Approach**:
- **Estimation**: For each arm $ i $, estimate its value $ Q_i $ as:

$$ Q_i = \frac{\sum_{s=1}^{t} R_s \mathbb{1}_{i_s = i}}{\sum_{s=1}^{t} \mathbb{1}_{i_s = i}} $$

Where $ \mathbb{1} $ is the indicator function.
- **Action Selection**: Always choose the arm with the highest estimated value.

**6. Upper Confidence Bound (UCB)**:
- **Estimation**: For each arm $ i $, compute its UCB as:

$$ Q_i = \mû_i + \sqrt{\frac{2 \log(t)}{N_i}} $$

Where $ \mû_i $ is the observed average reward for arm $ i $ and $ N_i $ is the number of times arm $ i $ has been pulled.
- **Action Selection**: Choose the arm with the highest UCB.

**7. Contextual Bandits**:
- **Reward Estimation**: The reward is estimated based on the context, turning the problem into a regression task.

$$ E[R_t | X_t] $$

Where $ X_t $ is the context vector.
- **Action Selection**: UCB is computed based on the regression prediction given the context vector.

**8. MABs vs. Reinforcement Learning (RL)**:
- **State Transitions**: In RL, the chosen action in a state determines or influences the next state.

**9. Summary & Looking Ahead**:
- **Key Takeaways**: The lecture covered the foundational concept of MABs, strategies like ε-greedy and UCB, and extensions like contextual bandits.

---

**Points of Clarification**:
1. **Difference between MABs and Learning with Experts**: In MABs, you only observe the reward for the chosen arm.
2. **UCB's Exploration Term**: The exploration term in UCB ensures that arms aren't prematurely discarded based on limited data.
3. **Contextual Bandits vs. RL**: The main distinction is the consideration of state transitions in RL, which are absent in contextual bandits.

---
