# week7
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/446f0a4a-334c-463e-8c08-b23536813867)

[View Lecture 13. Convolutional Neural Networks (CNN)](./slide/13.pdf)

[View Lecture 14. Bayesian regression](./slide/14.pdf)

[View workshop8-slides](./slide/workshop8-slides.pdf)

[View workshop8](/workshop8-slides.pdf)


---

**Convolutional Neural Networks (CNN) - Summary**

1. **Overview**: 
   - CNNs are a type of deep learning model primarily used for image recognition and processing.
   - Inspired by the human visual system.

2. **Architecture**:
   - **Input Layer**: Takes raw pixel values of the image.
   - **Convolutional Layer**: Filters slide over the image to produce feature maps. Highlights important features.
   - **ReLU Layer**: Introduces non-linearity using a function like Rectified Linear Unit (ReLU). Makes the network powerful.
   - **Pooling/Subsampling Layer**: Reduces spatial size, reduces computation, and helps in achieving translational invariance.
   - **Fully Connected (FC) Layer**: Connects every neuron from the previous layer, making decisions based on learned features.

3. **Key Concepts**:
   - **Filter/Kernels**: Small, learnable weight matrices which slide over the input data (like a window) to produce a feature map.
   - **Stride**: Number of pixels the filter moves while sliding over the image. Affects the size of the output feature map.
   - **Padding**: Adding layers of zeros around the image border to ensure the spatial dimensions remain the same after convolution.
   - **Pooling**: Down-sampling technique. Max pooling (takes maximum value) and average pooling (takes average) are popular methods.

4. **Advantages**:
   - **Parameter Sharing**: Reduces the number of parameters, thus reducing computation and preventing overfitting.
   - **Translational Invariance**: Recognizes patterns irrespective of their position in the image.
   - **Hierarchical Learning**: Lower layers learn basic patterns; higher layers learn complex features.

5. **Applications**:
   - Image and Video Recognition
   - Image Classification
   - Medical Image Analysis
   - Self-driving Cars
   - Many others in vision-based tasks.

6. **Challenges**:
   - Requires a large amount of labeled data for training.
   - Computationally intensive, especially for larger images or deeper networks.

7. **Optimization and Training**:
   - Often trained using backpropagation and optimizers like SGD, Adam, or RMSprop.
   - Regularization techniques (e.g., dropout) are used to prevent overfitting.

---
