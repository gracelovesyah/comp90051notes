# Lecture 14. RNN

This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/e37f2c38-a8bd-49b6-ad32-65aadbb10dda)

## Summary
---

**Statistical Machine Learning & RNNs**:
- **RNNs**: Neural networks designed for sequential data. They maintain a hidden state to capture information from past inputs.
- **Importance**: Can handle inputs of varying lengths, unlike traditional feed-forward neural networks.
- **Math Involved**: 
  - Hidden state update: $ h_t = \text{activation}(W_{hh} h_{t-1} + W_{xh} x_t + b_h) $
  - Attention scores (in attention mechanisms): $ e_{t,i} = a(s_{t-1}, h_i) $
  - Attention weights: $ \alpha_{t,i} = \frac{exp(e_{t,i})}{\sum_j exp(e_{t,j})} $

**Vanilla Neural Networks**:
- Basic neural networks without recurrent or convolutional structures.

**Attention Mechanism**:
- Allows models to focus on specific parts of the input when producing an output.
- Used in Seq2Seq models for tasks like machine translation.
- **Math Involved**: 
  - Context vector: $ c_t = \sum_i \alpha_{t,i} h_i $
  - Attention scores: $ e_{t,i} = a(s_{t-1}, h_i) $
  - Attention weights: $ \alpha_{t,i} = \frac{exp(e_{t,i})}{\sum_j exp(e_{t,j})} $

**Activation Functions**:
- **Softmax**: Converts a vector into a probability distribution.
- **Sigmoid**: Maps any input into a value between 0 and 1.

**Gradient Vanishing Problem**:
- In deep networks, especially RNNs, gradients can become too small, causing weights to stop updating.
- Caused by repeated multiplications of values less than 1. (also the explosion if W > 1)
- LSTMs and GRUs were introduced to mitigate this issue.

**LSTM & GRU**:
- Variants of RNNs designed to capture long-term dependencies.
- **LSTM**: Uses three gates (input, forget, output) and maintains a cell state.
- **GRU**: Simplified version of LSTM with two gates (reset and update).
- **Reasoning**: LSTMs are more expressive but computationally heavier than GRUs. Choice depends on the specific task and computational resources.

**Transformers & Self-Attention**:
- **Transformers**: Use self-attention mechanisms to process sequences in parallel.
- **Self-Attention**: Allows each item in a sequence to consider all other items when computing its representation.
- **Math Involved**:
  - Attention scores: $ e_{t,i} = a(s_{t-1}, h_i) $
  - Attention weights: $ \alpha_{t,i} = \frac{exp(e_{t,i})}{\sum_j exp(e_{t,j})} $

**Attention in Vision**:
- Spatial attention allows models to focus on specific regions of an image.
- Useful for tasks like image captioning and visual question answering.

---

**Questions Discussed**:
1. Explanation of vanilla neural networks.
2. How RNNs handle varying input lengths.
3. Difference between softmax and sigmoid activations.
4. Understanding the gradient vanishing problem.
5. Initial weight distribution around 0 in neural networks.
6. Differences between LSTM and GRU.
7. Explanation of self-attention and its use in Transformers.
8. Understanding attention heatmaps in NLP tasks.

---


## Random Notes
- RNN: memory, issues when input data of different length
- Vanilla Neural Networks: A vanilla neural network, often simply called a feedforward neural network, is the most basic type of artificial neural network architecture. When dealing with sequential input, traditional fixed-size input neural networks like vanilla feedforward networks are not the most suitable. Sequential data has inherent temporal dependencies, meaning the order of the data points matters. Here's how we can handle sequential input in the context of deep neural networks (DNNs):




```{tip}
why is W distributed around 0 initially??
```

```{admonition} answer
:class: dropdown
Initializing the weights of a neural network, including the recurrent weight matrix $ W $ in RNNs, around 0 is a common practice. The reasons for this are:
1. **Symmetry Breaking**: If all weights are initialized to the same value, then all neurons in a given layer of the network would produce the same output and undergo the same weight updates during training. This means they would always remain identical, effectively making them useless as they wouldn't capture diverse features. Initializing weights with small random values breaks this symmetry, ensuring that neurons evolve differently during training.

2. **Activation Function Dynamics**: For many activation functions, such as the sigmoid or tanh, when the input is close to 0, the function operates in a region where its slope (or derivative) is the steepest. This means that, initially, the network will have a more substantial gradient and will learn faster. If weights are initialized with large values, the activations might end up in the saturated regions of these functions where gradients are near zero, slowing down learning significantly due to the vanishing gradient problem.

3. **Preventing Saturation**: For activation functions like sigmoid or tanh, large input values (either positive or negative) can saturate the neuron, meaning it's in a region where the function is almost flat. This can lead to vanishing gradients, making the network hard to train. Initializing weights around 0 ensures that neurons are in the non-saturated region of the activation function at the start of training.

4. **Variance Considerations**: For certain initialization techniques, like Xavier/Glorot or He initialization, the goal is to maintain the variance of activations and gradients across layers. These techniques often result in weights being initialized with values centered around 0, but the exact scale depends on the specific activation function and the number of input/output units.

5. **Regularization Perspective**: Initializing weights with smaller values can be seen as a form of implicit regularization, as it encourages the network to start with a simpler model, which can potentially prevent overfitting to some extent.

```

### Attention Score

The importance of a hidden state in the context of attention mechanisms is determined by its attention score, often denoted by $ e $. The attention score indicates how relevant or important a particular hidden state is for a given context, such as a query or another part of the sequence.

The computation of the attention score $ e $ can vary depending on the specific attention mechanism used, but here's a general overview:

---

**Computing Attention Scores**:

1. **Dot Product Attention**:
   - The simplest form of attention computes the score as the dot product between the hidden state and a query vector:
     $$ e = q^T \cdot h $$
     Here, $ q $ is the query vector, and $ h $ is the hidden state.

2. **Scaled Dot Product Attention**:
   - Similar to the dot product attention but scales the dot product by the inverse square root of the depth of the attention (used in the Transformer model):
     $$ e = \frac{q^T \cdot h}{\sqrt{d}} $$
     Where $ d $ is the dimension of the query and hidden state.

3. **Additive/Multiplicative Attention**:
   - This method uses a small feed-forward network to compute the attention score:
     $$ e = v^T \cdot \text{tanh}(W_1 \cdot q + W_2 \cdot h) $$
     Here, $ v $, $ W_1 $, and $ W_2 $ are learnable parameters. The idea is to project both the query and the hidden state into a shared space and then measure their compatibility.

4. **Content-based Attention**:
   - The score is computed based on the content of the hidden state itself, often using a neural network to produce the score.

---

**Determining Importance**:

- Once the raw attention scores $ e $ are computed for all hidden states, they are normalized using the softmax function to produce the attention weights $ \alpha $:
  $$ \alpha = \frac{exp(e)}{\sum exp(e)} $$
  
- These attention weights $ \alpha $ represent the importance or relevance of each hidden state to the current context or query. A higher weight means the corresponding hidden state is more important.

- The context vector, which is a weighted average of all hidden states based on the attention weights, is then computed. This context vector captures the most relevant information from the sequence for the current processing step.
