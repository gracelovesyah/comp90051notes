# Lecture 10. The Perceptron

This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/90f9d3bc-a155-4431-ac8e-ceef71477b2b)


**The Perceptron Model**

- **Introduction to the Perceptron**
  - The perceptron is a building block for artificial neural networks.
  - It is inspired by the biological structure and function of the human brain.
  - The human brain can be approximated as a network of interconnected processing nodes (neurons) that relay electrical signals.

- **Artificial Neural Network (ANN)**
  - ANNs consist of processing elements that convert inputs to outputs.
  - The output is a function (activation function) of a weighted sum of inputs.
  - To use an ANN, one needs to design the network topology and adjust weights according to the given data.

- **Perceptron Model Details**
  - Inputs: \(x_1, x_2\)
  - Synaptic weights: \(w_1, w_2\)
  - Bias weight: \(w_b\)
  - Activation function: \(f\)
  - The perceptron is a linear binary classifier.

- **Perceptron Training Rule**
  - The perceptron training rule serves as a gateway to stochastic gradient descent.
  - Training involves adjusting weights based on the perceptron loss function.
  - The perceptron uses a loss function that has no penalty for correctly classified examples, while misclassified examples have a penalty equal to \(s\).

- **Stochastic Gradient Descent**
  - All training examples are shuffled/split into batches.
  - The algorithm iteratively updates weights using data from each batch.
  - The advantage is computational feasibility for large datasets.

- **Kernel Perceptron**
  - The perceptron weights always take the form of a linear combination of data.
  - Predictions for a new point are based on the dot product with data.
  - This dot product can be replaced with a kernel, leading to the kernelized perceptron.

- **Deep Learning (to be covered in subsequent lectures)**
  - The lecture hints at a future topic on deep learning, which involves neural networks with multiple layers (deep architectures).

In essence, this lecture provides an introduction to the perceptron model, its training rule, and the concept of kernelization. The perceptron serves as a foundational building block for more complex neural network architectures.