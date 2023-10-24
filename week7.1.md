# Lecture 13. Convolutional Neural Networks

This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/446f0a4a-334c-463e-8c08-b23536813867)

- Additional Resources
    - [Youtube: weight sharing](https://www.youtube.com/watch?v=ryJ6Bna-ZNU)
    - 
---
```{admonition} Quick Question
:class: tip
What are the choice of activation functions and loss functions or a binary classification deep learning model?
```

```{admonition} Answer
:class: dropdown

1. **Activation Function**:
   
   - **Output Layer**: 
     - **Sigmoid (Logistic) Activation**: It squashes the output between 0 and 1, which can be interpreted as the probability of belonging to the positive class.

       $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

2. **Loss Function**:

   - **Binary Cross-Entropy (Log Loss)**: It is the most commonly used loss function for binary classification problems. Given that $ p $ is the prediction from our model (output of sigmoid activation) and $ y $ is the actual label (0 or 1), the binary cross-entropy loss is:
   
     $$ L(p, y) = -y \log(p) - (1 - y) \log(1 - p) $$
   
     This loss function penalizes wrong predictions. The farther the prediction is from the actual label, the higher the loss.

**Notes**:

- Ensure that you have only one neuron in the output layer for binary classification with a sigmoid activation function.
  
- For binary classification, while the sigmoid activation is used in the output layer, you'd typically use other activation functions (like ReLU) in the hidden layers to introduce non-linearity and capture complex patterns.

- The optimizer used (like Adam, SGD, etc.) will try to minimize the Binary Cross-Entropy loss during training.

Using the sigmoid activation function in the output layer along with the binary cross-entropy loss is a well-established practice for binary classification problems in deep learning.

```

```{admonition} Quick Question
:class: tip
In image detection, why can CNN recognise a triangle image even if it not on the same spot?
```

```{admonition} Answer
:class: dropdown
The reason convolutional neural networks (CNNs) possess translational invariance, and can recognize patterns regardless of their position in the input, stems from their foundational operations and architecture. Let's break down the "why" behind this:

1. **Convolution Operation**: At its core, the convolution operation involves taking a small, fixed-size filter (or kernel) and sliding it over the entire input image to produce a feature map. This operation captures local spatial features from the input.

   - **Shared Weights**: Each position in the feature map is computed using the same weights from the filter. Thus, no matter where a particular feature (like an edge or a texture) appears in the image, the filter can detect it.

2. **Pooling**: Many CNN architectures also include pooling layers (often max pooling) which down-sample the feature maps. This operation provides a level of spatial invariance as it retains only the dominant feature in a local region, making the network less sensitive to the exact position of features.

3. **Hierarchical Feature Learning**: As you progress deeper into a CNN, the layers tend to recognize more complex, higher-level features. The initial layers might detect simple edges, while deeper layers might detect shapes or even more complex structures. Each subsequent layer builds upon the previous, allowing the network to recognize patterns regardless of their spatial arrangement.

4. **Benefit of Local Connectivity**: In CNNs, neurons in a given layer do not connect to every neuron in the previous layer (unlike fully connected networks). Instead, they only connect to a small local region of the input. This local connectivity ensures that the network learns local spatial hierarchies, which contribute to its ability to recognize patterns anywhere in the input.

In essence, CNNs are designed to automatically and adaptively learn spatial hierarchies of features from input images. The convolution operation's nature ensures that if a feature is learned once, it can be recognized anywhere in the image, granting the CNN its translational invariance property.
```


---
## Random notes by hand
- CNN are motivated by: efficiency and translation invariance.
- translation invariant: you can recognize an object as an object, even when its appearance varies in some way. 
- filters / kernel ($w$): for extracting features from image (a form of vector of weights)
- parameter sharing: learn translation invariant filters
- convolution operator: technique to match filter to image
---

```{tip}
• When input and kernel have the same pattern: high activation response
```

---

## **Convolutional Neural Networks (ConvNets or CNNs) Overview**

### **Basic Principles:**

1. **Convolution Operator:**
    - Originates from signal processing.
    - Defined by the integral/sum of the product of two functions, with one function being a flipped and shifted version of the other.
    - Measures how the shape of one function matches the other as it slides along.
    - In the context of CNNs, it is applied to discrete inputs like images or sequences.

2. **Convolution in 2D:**
    - Applied extensively for image processing tasks.
    - Helps in producing a "Feature Map", a 2D representation showing the presence of a specific pattern (defined by a kernel) at different locations in an input image.
    - Different kernels can detect different patterns (like edges, textures, etc.)

3. **Convolution in 1D:**
    - Primarily for sequential data such as time series or text.
    - Captures patterns or n-grams in sequences, like word combinations in text data.

### **Architectural Components:**

1. **Convolution Layers:**
    - Extract features from input data using the convolution operation.
    - Filter weights are learned during training.

2. **Downsampling via Max Pooling:**
    - Reduces the spatial dimension, retaining dominant features.
    - Helps prevent overfitting and reduces computational demand.
    - Not entirely differentiable, but gradients are defined based on the "sub-gradient" for backpropagation.

3. **Fully Connected Layers:**
    - Used towards the end of the network.
    - Combines the features learned and makes predictions or classifications.

4. **Residual Connections (specific to ResNet):**
    - Helps in training very deep networks by adding the input to the output of layers.
    - Alleviates the vanishing gradient problem in deep networks.

### **Applications:**

1. **Computer Vision:**
    - LeNet-5: An early and influential CNN model.
    - ResNet: Introduced residual connections, allowing for the training of much deeper networks.
    - Common practice involves using transfer learning, i.e., pretraining on a large dataset and fine-tuning on a smaller, task-specific dataset.

2. **Language Processing:**
    - 1D convolutions applied to word sequences.
    - CNNs can effectively classify sentences and other text-based tasks.
    - Research by Yoon Kim in 2014 highlighted the efficacy of CNNs for sentence classification.

### **Advantages of CNNs:**

1. **Translation Invariance:**
    - CNNs can detect patterns regardless of their position in the input.
    - Max-pooling further amplifies this characteristic.

2. **Parameter Sharing:**
    - Reduces the number of parameters to learn, as the same filter is used across the entire input.

3. **Hierarchical Pattern Learning:**
    - Stacking multiple convolution layers allows CNNs to learn complex patterns. Early layers might detect simple patterns like edges, while deeper layers can detect more abstract features.

