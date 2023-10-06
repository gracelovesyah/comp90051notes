# Workshop 10: Multi-armed bandits notes

## Summary of the Epsilon-Greedy Strategy:

**Epsilon-Greedy Strategy Overview:**
The epsilon-greedy strategy is a simple yet effective method for addressing the exploration-exploitation dilemma in multi-armed bandit problems.

**How It Works:**
1. **Exploitation:** With probability $1-\epsilon$, the strategy selects the current "best" arm. This is the arm with the highest estimated mean reward based on the rewards observed so far.
2. **Exploration:** With probability $\epsilon$, the strategy selects an arm uniformly at random, allowing the algorithm to explore other options and potentially discover a better arm.

**Estimating Mean Reward:**
The estimated mean reward for arm $k$ at round $t$ is denoted as $Q_{t-1,k}$. It is calculated as:
- If the arm $k$ has been played before (i.e., $N_{t-1,k} > 0$), then the estimated mean reward is the sample mean of the rewards observed for that arm so far, represented as $\hat{\mu}_{t-1,k}$.
- If the arm $k$ has never been played (i.e., $N_{t-1,k} = 0$), then the estimated mean reward is set to an initial value $Q_0$.

**Formulas:**
1. $N_{t-1,k} = \sum_{\tau=1}^{t-1} \mathbb{I}[a_\tau = k]$: This represents the number of times arm $k$ has been played up to round $t-1$.
2. $\hat{\mu}_{t-1,k} = \frac{1}{N_{t-1,k}} \sum_{\tau=1}^{t-1} r_{\tau} \mathbb{I}[a_\tau = k]$: This is the sample mean reward for arm $k$ up to round $t-1$.

**Hyperparameters:**
1. $\epsilon$: This is the probability with which the strategy will explore by selecting an arm at random. A higher $\epsilon$ means more exploration, while a lower $\epsilon$ means more exploitation.
2. $Q_0$: This is the initial value assigned to the estimated mean reward for any arm that hasn't been played yet.

**Implications:**
The epsilon-greedy strategy strikes a balance between exploration and exploitation. By occasionally exploring random arms, it ensures that it doesn't get stuck with a suboptimal choice. However, by mostly exploiting the best-known arm, it aims to maximize the cumulative reward. The choice of $\epsilon$ and $Q_0$ can influence the performance of the strategy, and they might need to be tuned based on the specific problem and environment.

## Summary of the Upper Confidence Bound (UCB) Strategy:

**UCB Strategy Overview:**
The Upper Confidence Bound (UCB) strategy is a sophisticated approach to the multi-armed bandit problem that balances exploration and exploitation by considering both the estimated mean reward and the uncertainty associated with each arm.

**How It Works:**
1. **Exploitation:** The strategy considers the estimated mean reward of each arm, represented as $\hat{\mu}_{t-1,k}$.
2. **Exploration:** The strategy adds an exploration term to the estimated mean reward. This term increases the value of arms that have been played less frequently or have higher uncertainty, encouraging the algorithm to explore them.

**Upper Confidence Bound Calculation:**
The UCB for arm $k$ at round $t$ is denoted as $Q_{t-1,k}$ and is calculated as:
- If the arm $k$ has been played before (i.e., $N_{t-1,k} > 0$), then the UCB is the sum of the sample mean reward and the exploration term: $\hat{\mu}_{t-1,k} + c \sqrt{\frac{\log t}{N_{t-1,k}}}$.
- If the arm $k$ has never been played (i.e., $N_{t-1,k} = 0$), then the UCB is set to an initial value $Q_0$.

**Formulas:**
1. $N_{t-1,k}$: Represents the number of times arm $k$ has been played up to round $t-1$.
2. $\hat{\mu}_{t-1,k}$: This is the sample mean reward for arm $k$ up to round $t-1$.
3. Exploration Term: $c \sqrt{\frac{\log t}{N_{t-1,k}}}$. This term increases as the current round $t$ progresses (encouraging exploration as time goes on) and decreases as the arm $k$ is played more frequently (reducing the need for exploration for well-known arms).

**Hyperparameters:**
1. $c$: This is the exploration parameter. A higher value of $c$ encourages more exploration, while a lower value emphasizes exploitation. The choice of $c$ can influence the balance between exploration and exploitation and might need to be tuned based on the specific problem and environment.

**Implications:**
The UCB strategy is particularly effective in scenarios where the reward distributions of the arms are uncertain or have high variance. By considering both the estimated mean reward and the uncertainty, UCB ensures that the algorithm doesn't prematurely converge to a suboptimal arm and continues to explore potentially better options.

## Summary of Offline Evaluation for Multi-Armed Bandits (MABs):

**Offline Evaluation Overview:**
Offline evaluation is a method to assess the performance of MAB algorithms using historical data, without the need to deploy the algorithm in a live environment.

**Online vs. Offline Evaluation:**
1. **Online Evaluation:** In a live setting, a MAB algorithm would compete with another algorithm, and their performances would be compared based on the cumulative rewards they achieve. However, this approach has drawbacks:
   - **Initial Poor Performance:** MABs start with little knowledge about the reward structure, leading to suboptimal decisions in the early rounds.
   - **User Experience:** In the context of the news website example, users might be exposed to less relevant or uninteresting articles during the exploration phase, affecting user satisfaction.

2. **Offline Evaluation:** This method bypasses the need for live deployment. It involves:
   - **Data Collection:** Gather a sequence of arm pulls and their corresponding rewards, ideally chosen uniformly at random.
   - **Simulation:** Evaluate the MAB algorithm's decisions and rewards using this historical dataset. Since the data is already collected, there's no need to expose users to potentially suboptimal choices.

**Benefits of Offline Evaluation:**
1. **Cost-Effective:** No need to deploy multiple algorithms in a live setting, saving resources.
2. **User Experience:** Users aren't exposed to potentially poor decisions made by the MAB during its exploration phase.
3. **Reusability:** The same historical dataset can be used to evaluate multiple MAB algorithms or configurations, ensuring a consistent benchmark.

**Implementation:**
The `offlineEvaluate` function provided simulates the performance of a MAB algorithm using a given historical dataset. This function allows for a comparison of different MAB strategies without the need for live testing.

**Reference:**
For those interested in a deeper dive into offline evaluation, especially in the context of personalized news article recommendations, the paper by Lihong Li et al. provides a comprehensive study. This paper introduces a contextual-bandit approach to news article recommendation and discusses the nuances of offline evaluation in detail.

**Implications:**
Offline evaluation is a powerful tool, especially in scenarios where online testing might be costly, risky, or detrimental to user experience. It allows for rapid testing and iteration of MAB strategies using consistent benchmarks. However, it's essential to ensure that the historical data used for offline evaluation is representative of the real-world scenario the MAB will be deployed in.