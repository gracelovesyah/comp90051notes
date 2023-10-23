# Review 2
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/88d4575b-4666-4b45-a495-683fb598bd35)

## ToC
- Kernel Trick
- common kernels
- prove valid kernel
- Discriminative Neural Networks Models
- Perceptron
- Feedforward Neural Networks
- Convolutional Neural Networks
- Recurrent Neural Networks
- Attention

## SVM vs NN

```{admonition} Question
What is the difference between Feedforward Neural Networks and SVM considering their approaches in dealing with non-linearly separable data?
```
```{admonition} Answer
:class: dropdown
Feedforward Neural Networks (FNNs) and Support Vector Machines (SVMs) both handle non-linearly separable data, but they approach the problem differently. FNNs, especially when equipped with non-linear activation functions, learn hierarchical representations of data by transforming input through multiple layers, allowing them to approximate any function, including non-linear boundaries. SVMs, on the other hand, utilize the kernel trick to map the input data into a higher-dimensional space where it becomes linearly separable. Instead of learning representations like FNNs, SVMs focus on finding the optimal hyperplane that maximally separates the classes in this transformed space.

```

```{admonition} Question
What are the advantage and disadvantage of manual feature engineering and automatic data transformation in Neural Network Models?
```

```{admonition} Answer
The advantages and disadvantages of manual feature engineering and automatic data transformation in neural network models:

| Aspect                                       | Manual Feature Engineering | Automatic Data Transformation in Neural Network Models |
|----------------------------------------------|----------------------------|-------------------------------------------------------|
| **Advantages**                               |                            |                                                       |
| Domain Knowledge Application                 | ✅ Uses domain expertise to create meaningful features. | ❌ Lacks domain-specific insights.                      |
| Model Interpretability                       | ✅ Leads to transparent and interpretable models.      | ❌ Often results in "black box" models.                |
| Efficiency                                   | ✅ Simpler models might suffice.                       | ❌ Requires complex models and more computational resources. |
| Adaptability                                 | ❌ Might not generalize across tasks.                  | ✅ Adaptable to various tasks without manual intervention.  |
| End-to-end Learning                          | ❌ Typically not end-to-end.                           | ✅ Allows optimization from raw data to output.         |
| Reduced Human Bias                           | ❌ Risk of introducing biases based on intuitions.     | ✅ Reduces human biases in feature creation.           |
| **Disadvantages**                            |                            |                                                       |
| Time Consumption                             | ✅ Requires significant time and effort.               | ❌ Faster adaptation to new data/tasks.                 |
| Risk of Overfitting                          | ✅ Possible if features don't generalize well.         | ❌ Can overfit if not regularized or if data is limited.  |
| Scalability                                  | ❌ Needs repeated efforts for new tasks/data.          | ✅ Scales better to different tasks without manual changes. |

Note: The "✅" and "❌" symbols indicate whether the mentioned aspect is an advantage or disadvantage, respectively, for each method.
```
## Partial Derivative of Loss Function 
```{admonition} Question
Given the following neural network for binary classification, if we use binary cross entropy as loss function, compute the partial derivative of loss function L with respect to vij (written as a formula that only contains xi, uj, wj, z and y, where y is the correct label of data. Assume we use sigmoid activation function everywhere, which are shown as f in the figure below.)
```
```{image} ./images/nn1.png
:alt: nn1
:class: bg-primary mb-1
:width: 500px
:align: center
```

```{admonition} Answer
:class: dropdown
Alright, let's compute the derivative of the binary cross entropy loss with respect to $ v_{ij} $.

Given:
Loss function $ L $ for binary classification with binary cross entropy:

$$ L = -y \log(z) - (1 - y) \log(1 - z) $$


Where:

$$ z = f(s) $$

and 

$$ s = \sum_j u_j w_j $$


Given that $ f $ is the sigmoid function:

$$ f(x) = \frac{1}{1 + e^{-x}} $$

The derivative $ f'(x) $ is:

$$ f'(x) = f(x)(1 - f(x)) $$


From the given neural network:

$$ u_j = f(\sum_i x_i v_{ij}) $$

or

$$ u_j = f(r_j) $$

where $ r_j $ is the net input to neuron $ j $ in the hidden layer.

Let's compute the chain of derivatives:
1. Derivative of $ L $ with respect to $ z $:

$$ \frac{\partial L}{\partial z} = \frac{z - y}{z(1 - z)} $$

2. Derivative of $ z $ with respect to $ s $:

$$ \frac{\partial z}{\partial s} = f'(s) = f(s)(1 - f(s)) = z(1 - z) $$

3. Derivative of $ s $ with respect to $ u_j $:

$$ \frac{\partial s}{\partial u_j} = w_j $$

4. Derivative of $ u_j $ with respect to $ r_j $:

$$ \frac{\partial u_j}{\partial r_j} = f'(r_j) = f(r_j)(1 - f(r_j)) $$

5. Derivative of $ r_j $ with respect to $ v_{ij} $:

$$ \frac{\partial r_j}{\partial v_{ij}} = x_i $$


Using the chain rule, the partial derivative of $ L $ with respect to $ v_{ij} $ is:

$$ \frac{\partial L}{\partial v_{ij}} = \frac{\partial L}{\partial z} \times \frac{\partial z}{\partial s} \times \frac{\partial s}{\partial u_j} \times \frac{\partial u_j}{\partial r_j} \times \frac{\partial r_j}{\partial v_{ij}} $$


Substitute in the computed derivatives:

$$ \frac{\partial L}{\partial v_{ij}} = \left(\frac{z - y}{z(1 - z)}\right) \times (z(1 - z)) \times w_j \times f(r_j)(1 - f(r_j)) \times x_i $$


After simplifying:

$$ \frac{\partial L}{\partial v_{ij}} = (z - y) \times w_j \times f(r_j)(1 - f(r_j)) \times x_i $$


And this is the partial derivative of the loss function $ L $ with respect to $ v_{ij} $.

```

## Derivative of Sigmoid Function


Given:

$$f(x) = \frac{1}{1 + e^{-x}} $$

To find $ f'(x) $, differentiate with respect to $ x $:

$$f'(x) = \frac{-1 \cdot (-e^{-x})}{(1 + e^{-x})^2} $$

$$= \frac{e^{-x}}{(1 + e^{-x})^2} $$

Notice:

$$f(x) = \frac{1}{1 + e^{-x}} $$

So:

$$e^{-x} = \frac{1}{f(x)} - 1 $$

Replacing $ e^{-x} $ in the derivative expression:

$$f'(x) = f(x)(1 - f(x)) $$

So, the derivative of the sigmoid function is:

$$f'(x) = f(x)(1 - f(x)) $$

```{Tip}
Use the quotient rule of differentiation:
If you have a function in the form $ \frac{u(x)}{v(x)} $, then its derivative is:

$$ \frac{u'v - uv'}{v^2} $$
```

## CNN

### Pooling
- [What is pooling? | CNN's #3](https://youtu.be/KKmCnwGzSv8?t=58)

### Calculations

**(a)Size of Output Map**

$$ O = \lceil \frac{W - K + 2P}{S} + 1 \rceil $$

Where:
- $ O $ is the output size.
- $ W $ is the width (or height) of the input.
- $ K $ is the kernel (or filter) size.
- $ P $ is the padding.
- $ S $ is the stride.


**(b) Number of parameters in each layer:**

- Parameters = $ K × K × C_{in} × C_{out} $

    - $ K $ is the kernel size.
    - $ C_{in} $ is the number of input channels.
    - $ C_{out} $ is the number of output channels (or filters).
    eg. input: 64 * 64 * 64 * 3 -> 3 is the number of input channels; 

    output channels = # of filters.

- Pooling: No learnable parameters in max-pooling layer.
- Fully Connected: 
    - n * m (no bias)
    - (n + 1) * m (with bias)





**(c) To keep the same size of the receptive field of conv2 with fewer parameters:**

1. **Use 1x1 Convolution (a.k.a Network in Network)**: Applying 1x1 convolutions can help reduce the depth (number of channels) before applying the 9x9 convolution. For example, using a 1x1 convolution to reduce the channels to 32, and then applying a 9x9 convolution.
 
2. **Use Dilated Convolution**: Instead of the regular convolution, dilated (or atrous) convolution can be used with a smaller kernel size but with increased dilation rate to achieve the same receptive field.

3. **Factorized Convolution**: Decompose the 9x9 convolution into two separate convolutions: one 9x1 convolution followed by a 1x9 convolution.

Note: Each of the above methods reduces the number of parameters while maintaining the same receptive field, but the exact effects on performance and the number of parameters saved will vary.