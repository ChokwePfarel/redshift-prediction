Redshift Prediction from Spectral Data

Author: Pfarelo Chokwe
Date: 5 August 2025

This repository contains the complete implementation of a machine-learning pipeline for automated redshift (z) prediction from astronomical flux spectra. The project includes Python notebooks, datasets, and the full methodology.

Project Objective

Redshift ($z$) measures how much light from distant objects is stretched due to the expansion of the universe. Accurate estimation is essential for determining distances, velocities, and large-scale structure
(e.g., Hogg, 1999).

Modern surveys such as the Sloan Digital Sky Survey (SDSS; York et al., 2000) produce millions of spectra, making automated redshift prediction a critical task.

This project develops a full pipeline that:

Processes high-dimensional spectral data (≈4001 points per spectrum)

Extracts meaningful features via Principal Component Analysis (PCA)

Predicts redshift using two regressors: Random Forest (RF) and Support Vector Regression (SVR)

Compares model performance using MSE, R², and Pearson correlation

 Methodology
1. Dataset

The dataset consists of:

spectra.npy: Flux intensities across 4000–8000 Å

metadata.npy: True redshift values

Both arrays were truncated to equal lengths.

2. Feature Extraction — PCA

PCA reduces the ~4001-dimensional flux vector into two principal components, following standard dimensionality-reduction techniques (Jolliffe & Cadima, 2016).

PC1: Global spectral shape / continuum

PC2: Emission and absorption line variations

The two components retained ≈97.8% of total variance.

3. Regression Models

Two regressors were implemented using scikit-learn:

 Random Forest Regressor (RF)

100 decision trees

Captures nonlinear patterns

Based on Breiman’s (2001) ensemble method

 Support Vector Regression (SVR)

RBF kernel

$C = 10$, gamma = scale

Good for smooth, continuous functions (Smola & Schölkopf, 2004)

 Results
Model	Mean Squared Error (MSE)	$R^{2}$	Pearson $r$
SVR	≈ 0.008	≈ 0.93	≈ 0.96
Random Forest	≈ 0.014	≈ 0.84	≈ 0.91

SVR outperformed RF across all metrics, showing:

Lower error

Stronger linear relationship

Smoother prediction behavior

The SVR RMSE of ≈0.09 indicates highly accurate redshift estimation for many astronomical use cases.

 Discussion

PCA successfully compressed the spectral structure into a compact representation while preserving essential variance.

RF performs well but can overfit with low-dimensional PCA features.

SVR generalizes better and captures continuous spectral–redshift relationships.

The clear color gradients in PCA scatter plots indicate that redshift information is well-encoded in PC1 and PC2.

 Conclusion

Combining:

PCA for feature extraction

SVR for regression modeling

produces a robust, efficient, and accurate pipeline for redshift estimation from spectral data.

This framework can be extended with deep learning methods or additional spectral features in future work.
