# Lecture 22. Inference on PGMs Cont. & Lecture 23. Gaussian Mixture Models

This note is completed with the assistance of [ChatGPT](https://chat.openai.com/c/c5be32aa-672c-4e59-8a2c-85704a846b8a)

### **Lecture Summary: Gaussian Mixture Model (GMM) in Statistical Machine Learning**

---

#### **1. Introduction to Unsupervised Learning**:

- **Definition**: Learning the structure of data without labels. It contrasts with supervised learning where data comes with predefined labels.
- **Main Paradigms**: Unsupervised learning is a shift from supervised learning, which is about predicting labels, and the contextual bandits setting, which deals with partial supervision.

---

#### **2. Kinds of Unsupervised Learning**:

- **Tasks**:
  - **Clustering**: Grouping data points based on their similarities.
  - **Dimensionality Reduction**: Reducing the number of random variables.
  - **Learning Probabilistic Models**: Estimating parameters for statistical models.
- **Applications**: Include market basket analysis, outlier detection, and various tasks within supervised machine learning pipelines.

---

#### **3. Refresher on K-means Clustering**:

- **Steps**:
  1. Initialize cluster centroids.
  2. Assign data points to the nearest centroid and recompute centroids.
  3. Terminate when no change in assignments or continue iterating.
- **Features**: Requires specifying the number of clusters in advance, uses Euclidean distance for dissimilarity, and typically finds spherical clusters.

---

#### **4. Gaussian Mixture Model (GMM)**:

- **Concept**: A probabilistic approach to clustering, where each data point can belong to multiple clusters with certain probabilities.
- **Benefits**: Allows modeling uncertainty about the origin of each data point. Each point originates from a particular cluster, but we aren't sure which one.
- **Application**: Clustering in a GMM becomes model fitting in a probabilistic sense.

---

#### **5. Clustering with a Probabilistic Model**:

- **Model**: Treats data points as i.i.d. samples from a mixture of distributions.
- **GMM**: When the components in the mixture are Gaussian distributions, we have a Gaussian Mixture Model.
- **Normal Distribution**: A key mathematical concept, with the 1D Gaussian given by the bell curve and the multi-dimensional Gaussian defined by a mean vector and a covariance matrix.

---

#### **6. Key Components of GMM**:

- **Cluster Assignment Probabilities**: Represents the likelihood a data point belongs to a particular cluster.
- **Location of Point**: Governed by the Gaussian distribution of the cluster it's assigned to.
- **Model Parameters**: Cluster weights, means, and covariance matrices.
- **Mixture Distribution**: Obtained by marginalizing latent variables and represents the likelihood of observed data points.

---

#### **Final Takeaway**:

GMM provides a nuanced probabilistic approach to clustering. Instead of hard assignments as in k-means, GMM gives soft assignments, making it more flexible and expressive for capturing uncertainties in real-world data structures.