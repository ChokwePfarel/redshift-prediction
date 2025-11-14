# 🌌 Redshift Prediction from Spectral Data

**Author: Pfarelo Chokwe** | **Date: 5 August 2025**


This repository contains the complete implementation of a **machine-learning pipeline** for **automated redshift ($z$) prediction** from astronomical flux spectra. The project includes Python notebooks, datasets, and the full methodology.

---

## Project Objective

Redshift ($z$) is a fundamental measure of how much light from distant objects is stretched due to the expansion of the universe (e.g., Hogg, 1999). Accurate estimation is essential for determining **distances, velocities, and the large-scale structure** of the cosmos.

With modern surveys (e.g., SDSS; York et al., 2000) generating millions of spectra, automated redshift prediction is a critical task. This project develops a full pipeline that:

* Processes **high-dimensional spectral data** ($\approx 4001$ points per spectrum).
* Extracts meaningful features via **Principal Component Analysis (PCA)**.
* Predicts redshift using two regressors: **Random Forest (RF)** and **Support Vector Regression (SVR)**.
* Compares model performance using **MSE, R², and Pearson correlation ($r$)**.

---

## Methodology

### 📊 Dataset and Preprocessing

The dataset consists of:
* `spectra.npy`: Flux intensities measured across the **4000–8000 Å** wavelength range.
* `metadata.npy`: The true (ground-truth) redshift values.

### 🔬 Feature Extraction — Principal Component Analysis (PCA)

PCA was employed to reduce the $\approx 4001$-dimensional flux vector into just **two principal components** (Jolliffe & Cadima, 2016).

* **PC1** captures the **global spectral shape and continuum**.
* **PC2** captures **emission and absorption line variations** linked to redshift.

This two-component projection successfully retained $\approx \mathbf{97.8\%}$ of the total variance, demonstrating effective feature compression.

### ⚙️ Regression Models

Two regressors were implemented using scikit-learn for comparative analysis:

| Model | Key Parameters & Description |
| :--- | :--- |
| **Random Forest Regressor (RF)** | **100 decision trees**. An ensemble method (Breiman, 2001) well-suited for capturing nonlinear patterns. |
| **Support Vector Regression (SVR)** | **RBF kernel**, with $C=10$ and $\text{gamma} = \text{'scale'}$. Excellent for generalizing and producing smooth, continuous functions (Smola & Schölkopf, 2004). |

---

## Results and Conclusion

### Model Performance Comparison

| Model | Mean Squared Error (MSE) | $R^{2}$ (Coefficient of Determination) | Pearson Correlation ($r$) |
| :--- | :--- | :--- | :--- |
| **Support Vector Regression (SVR)** | $\approx 0.008$ | $\approx 0.93$ | $\approx 0.96$ |
| **Random Forest (RF)** | $\approx 0.014$ | $\approx 0.84$ | $\approx 0.91$ |

### Discussion

* **SVR outperformed RF** across all metrics, exhibiting lower error and a stronger linear relationship between predicted and actual redshift.
* The SVR Root Mean Squared Error (RMSE) of $\approx 0.09$ indicates **highly accurate redshift estimation** for many astronomical use cases.
* The clear results confirm that while RF is robust, SVR **generalizes better** and is less prone to overfitting when using low-dimensional PCA features.

### Final Conclusion

The framework of combining **PCA for feature extraction** with **SVR for regression** produces a robust, efficient, and highly accurate pipeline for automated redshift estimation from spectral data. This framework can be extended with deep learning or additional spectral features in future work.

---

## Repository Contents

* `Redshift_Prediction_Report.pdf`: The full research paper.
* `Python notebooks/scripts`: Contains the code for PCA and both regression models.
* `datasets/`: Includes `spectra.npy` and `metadata.npy`.
