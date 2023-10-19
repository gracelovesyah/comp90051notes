# Lecture 16 Graph Convolution Networks (Deep Learning After You Drop The Camera)

This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/dcc0b72f-a768-4363-91b8-d8e382994a24)

Certainly! Here's a concise summary of the discussion and topics you've covered:

---

**Graph Networks & Graph Convolutional Networks (GCNs)**
- Graph networks exchange the rigid structure of tensors/matrices for more flexible discretizations.
- Information can be encoded on both graph nodes and edges. The properties can change over time due to local or global factors.
- Graphs are defined by adjacency matrices (A) and sometimes normalized using a modified Laplacian, $ L = I_N + D^{-1/2} A D^{1/2} $, aiding in feature propagation.

**Mathematical Formulations**
- The feature update rule: $ H^{(l+1)} = \sigma (H^{(l)} W^{(l)} + \tilde{A} H^{(l)} W^{(l)}) $, with $\tilde{A}$ often representing the normalized adjacency matrix.
- Aggregation methods in GNNs use functions like mean, max, or LSTMs to handle varying numbers of connections into a node.

**Queries & Answers**
- The modified Laplacian (L) is used for normalization and promotes smoother feature propagation across a graph.
- GPUs favor structured data due to their parallel processing nature.
- GCNs don't require global information as they can leverage local neighborhood data for training. This allows for efficient subgraph-based training and global behavior learning.

**Case Studies**
1. **Google Maps**: Demonstrated how GCNs can handle time-series data on graph structures and how they can be combined with LSTMs to analyze dynamic behaviors.
  
2. **Point Clouds**: Pointed out the challenges of handling point cloud data, such as irregularity and disorder. Highlighted a paper, "Towards Efficient Graph Convolutional Networks for Point Cloud Handling" by Li et. al, 2021, that offers an architectural solution. Stressed the flexibility of GNNs over CNNs and their potential in handling complex datasets and problems.

---