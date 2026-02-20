# MNIST-digits-classifier-using-various-estimators-
Gemini said
MNIST Handwritten Digit Classification (MLE, LDA, QDA)
A machine learning project focused on handwritten digit classification using statistical methods. The system implements Maximum Likelihood Estimation (MLE) to model the distribution of MNIST digits (0, 1, and 2) and evaluates performance through Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA).

🚀 Key Features
Statistical Modeling: Implements manual Maximum Likelihood Estimation (MLE) for mean vectors and covariance matrices of high-dimensional image data.

Discriminant Analysis: Features custom implementations of Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) to classify 784-dimensional feature vectors.

High-Dimensional Preprocessing: Processes raw MNIST data, including flattening 28x28 images and performing range normalization [0, 1].

Visual Analytics: Utilizes t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualize the clustering and separability of digits in a 2D space.

🏗️ Technical Architecture
Data Sampling: Specifically filtered to classes 0, 1, and 2, with 300 samples used for training and 300 for testing.

Mathematical Precision: Employs manual log-gaussian discriminant calculations, handling potential matrix singularity with identity matrix regularizers.

Performance Comparison: Validated that QDA (98.00% accuracy) outperformed LDA (91.67%) by capturing unique class-specific pixel distribution patterns.

🛠️ Tech Stack
Language: Python

Libraries: NumPy (Matrix operations), Scikit-Learn (t-SNE), Matplotlib (Visualization), idx2numpy (Dataset parsing)

Methods: MLE, LDA, QDA, Log-Likelihood Discriminants
