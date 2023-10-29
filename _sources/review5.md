# Review 5
This notes is completed with assistance of [ChatGPT](https://chat.openai.com/c/d7b76882-24c2-4363-951c-47f33d0f2839)

## ToC
- Probabilistic Graphical Models
- representation, inference
- Hidden Markov Model
- Gaussian Mixture Model



## HMM
### Learning Resources
- [Lecture Notes for Stats325](https://stat.auckland.ac.nz/~fewster/325/notes.php)
- [Markov Chains - VISUALLY EXPLAINED + History!](https://youtu.be/CIe869Rce2k)

## KNN & Kmeans


| Feature/Aspect             | K-NN (K-Nearest Neighbors)        | K-Means              |
|----------------------------|----------------------------------|----------------------|
| **Type of Algorithm**      | Supervised Learning              | Unsupervised Learning|
| **Purpose**                | Classification or Regression     | Clustering           |
| **Input Data**             | Labeled data for training        | Unlabeled data       |
| **Output**                 | Predicted label/value            | Cluster centroids    |
| **Training Requirement**   | Yes, stores all training data    | Yes, finds centroids |
| **Prediction Mechanism**   | Votes from 'K' closest points    | Assigns to nearest centroid |
| **Parameter 'K'**          | Number of neighbors to consider  | Number of clusters   |
| **Distance Metric**        | Typically Euclidean, but can be others | Typically Euclidean |
| **Sensitivity to Outliers**| Sensitive (depends on 'K')       | Can be sensitive     |
| **Scalability**            | Can be computationally expensive for large datasets (unless optimized e.g., with KD-trees) | Can be more scalable with techniques like MiniBatch K-means |
| **Real-time Updates**      | Easy, just add data to dataset   | May require re-running the algorithm |
| **Interpretability**       | Direct relationship between input and output based on distance | Centroids represent cluster 'centers' but may not correspond to actual data points |


## Gaussian

```{admonition} List two similarities and two differences between K-means and Gaussian Mixture Model
:class: dropdown

**Similarities**:
1. **Objective**: Both K-Means and GMM are used for clustering, that is, partitioning a dataset into groups (clusters) based on similarities.
2. **Initialization**: Both can use similar initialization techniques. For example, the initial cluster centers in K-Means can be used as the initial means in GMM.

**Differences**:
1. **Model Assumption**:
   - **K-Means**: Assumes that clusters are spherical and equally sized. It achieves this by assigning points to the nearest centroid based on Euclidean distance.
   - **GMM**: Assumes that data is generated from a mixture of several Gaussian distributions. Each cluster can have a different elliptical shape, size, and orientation because GMM captures the variance and covariance of the data.
   
2. **Soft vs. Hard Assignment**:
   - **K-Means**: Performs hard assignments, meaning each point is assigned to one and only one cluster.
   - **GMM**: Performs soft assignments, meaning each point is assigned a probability for each cluster, indicating how likely it belongs to the cluster. This allows for a more nuanced understanding of cluster membership.

While K-Means can be thought of as a special case of GMM where each cluster's covariance along all dimensions approaches 0 (leading to spherical clusters), GMM provides a more general and flexible approach at the cost of being computationally more expensive.

```

```{admonition} How many free parameters do we need at least for a Gaussian Mixture Model in a D-dimensional space with k clusters?
:class: dropdown

For a Gaussian Mixture Model (GMM) in a $ D $-dimensional space with $ k $ clusters, the free parameters include:

1. **Means**: For each of the $ k $ clusters, we have a $ D $-dimensional mean vector. 
   Total parameters for means = $ k \times D $
   
2. **Covariances**: Each cluster has a $ D \times D $ covariance matrix.
   However, since the covariance matrix is symmetric, the number of unique values in the matrix is $ \frac{D(D + 1)}{2} $.
   Total parameters for covariances = $ k \times \frac{D(D + 1)}{2} $

3. **Mixture weights (priors)**: For $ k $ clusters, there are $ k $ mixture weights. But they sum to 1, so you only need to know $ k - 1 $ of them to determine the last one.
   Total parameters for mixture weights = $ k - 1 $

Now, summing up all the parameters:
Total parameters = $ k \times D + k \times \frac{D(D + 1)}{2} + k - 1 $

Let's simplify this expression.

The total number of free parameters for a Gaussian Mixture Model in a $ D $-dimensional space with $ k $ clusters is given by:

$$
Dk\left(\frac{D + 1}{2}\right) + Dk + k - 1
$$

This formula accounts for the means, covariance matrices, and mixture weights for each of the $ k $ clusters in the $ D $-dimensional space.

```
## EM
- soft assignment: computes the prob this point comes from this/that distribution

```{image} ./images/em1.png
:alt: em
:class: bg-primary mb-1
:width: 500px
:align: center
```


```{image} ./images/em2.png
:alt: em
:class: bg-primary mb-1
:width: 500px
:align: center
```
