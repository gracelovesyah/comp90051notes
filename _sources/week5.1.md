# Lecture 9. Kernel Methods

---

### **Conceptual Understanding**:
- **Notes**:
  - The primal problem represents the original formulation of a problem, while the dual provides an alternative perspective.
  - Dual formulation in SVMs focuses on maximizing the margin between support vectors, represented in terms of Lagrange multipliers.
  - The dual formulation only requires dot products between samples, making it kernel-friendly.

- **Example Question**:
  - Why is the dual formulation of SVMs often preferred over the primal form?

---

### **Mathematical Derivation**:
- **Notes**:
  - The dual formulation can be derived from the primal using Lagrange multipliers.
  - KKT conditions ensure optimality in constrained optimization problems.

- **Example Question**:
  - Derive the dual problem of SVMs from its primal using Lagrange multipliers.

---

### **Implementation**:
- **Notes**:
  - Solving the dual problem typically involves quadratic programming.
  - The optimization focuses on finding the Lagrange multipliers.

- **Example Problem**:
  - Implement a simple SVM solver using the dual formulation for a given small dataset.

---

### **Kernels and the Dual Formulation**:
- **Notes**:
  - Kernels implicitly map data to a higher-dimensional space, allowing non-linear separation.
  - The "kernel trick" utilizes the fact that we only need dot products in the transformed space.

- **Example Question**:
  - Given the polynomial kernel of degree 2, compute the dual objective for a dataset with points \( (1, 2) \) and \( (3, 4) \).

---

### **Optimization and Solution**:
- **Notes**:
  - Dual formulation results in a convex quadratic programming problem.
  - Support vectors are data points with non-zero Lagrange multipliers.

- **Example Problem**:
  - Given a trained SVM in its dual form, identify the support vectors from a set of data points.

---

### **Application and Interpretation**:
- **Notes**:
  - Dual coefficients (Lagrange multipliers) indicate the importance of corresponding data points.
  - Predictions involve computing dot products (or kernel values) with support vectors.

- **Example Problem**:
  - Using a trained SVM with known dual coefficients and support vectors, classify a new data point \( (2, 3) \).

---

### **Regularization and the Dual Formulation**:
- **Notes**:
  - Regularization in SVMs (parameter \( C \)) controls the trade-off between maximizing the margin and minimizing classification error.
  - Appears as bounds on the Lagrange multipliers in the dual problem.

- **Example Question**:
  - How does increasing the value of \( C \) affect the dual solution of an SVM?

---

### **Comparative Questions**:
- **Notes**:
  - Primal formulation directly optimizes the weights, while the dual optimizes Lagrange multipliers.
  - Dual formulation is typically preferred when the number of features is large compared to the number of samples.

- **Example Question**:
  - In what scenarios might the primal formulation of SVMs be computationally more efficient than the dual?

---

### **Problem Solving**:
- **Notes**:
  - Setting up the dual problem involves creating the matrix of dot products or kernel values.
  - Solving it gives the Lagrange multipliers.

- **Example Problem**:
  - For a dataset with points \( (1, 1) \) and \( (2, 3) \), set up the dual problem using the linear kernel.

---

### **Advanced Topics**:
- **Notes**:
  - The dual formulation concept extends to other areas, like neural networks, where dual methods can be applied to training processes.

- **Example Question**:
  - Discuss the relationship between the dual formulation in SVMs and kernel PCA.

---

This cheat sheet provides an overview of key concepts, accompanied by typical questions or problems one might encounter regarding the dual formulation in SVMs.