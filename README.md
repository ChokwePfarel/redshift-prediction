# Redshift Prediction from Spectral Data

[cite_start]This repository contains the complete research project, including the final report, Python code, and data files, which implements a machine learning pipeline for **automated redshift (z) prediction** from observed astronomical flux spectra[cite: 5, 17].

## Project Objective and Challenge

[cite_start]Redshift ($z$) is a fundamental quantity in astrophysics, describing how much light from distant celestial objects is stretched due to cosmic expansion[cite: 13]. [cite_start]Accurate estimation is crucial for inferring distances, velocities, and the large-scale structure of the universe[cite: 14].

[cite_start]This project addresses the challenge of high dimensionality in spectral data ($\approx 4001$ flux points per spectrum) [cite: 6, 18] [cite_start]by using **Principal Component Analysis (PCA)** for efficient feature extraction[cite: 11, 21]. [cite_start]The pipeline then compares two robust regression models to predict redshift directly from the compressed features[cite: 22].

## Methodology & Pipeline

1.  [cite_start]**Data Preprocessing**: The dataset comprises spectra (`spectra.npy`) and ground-truth redshift values (`metadata.npy`)[cite: 26]. [cite_start]The raw flux intensity values cover a wavelength range of 4000-8000 Å[cite: 18].
2.  [cite_start]**Feature Extraction**: **Principal Component Analysis (PCA)** was applied to reduce the dimensionality to just **two components** (PC1 and PC2)[cite: 29].
    * [cite_start]This compression successfully retained $\approx \mathbf{97.79\%}$ of the explained variance[cite: 151].
    * [cite_start]**PC1** captures the **global spectral shape and continuum**, while **PC2** captures **emission or absorption line variations** linked to redshift[cite: 30].
3.  [cite_start]**Regression Modeling**: Two machine learning models were implemented using `scikit-learn` and trained on the PCA-transformed features[cite: 33, 38]:
    * [cite_start]**Random Forest Regressor (RF)**: An ensemble of decision trees ($n\_estimators=100$)[cite: 35].
    * [cite_start]**Support Vector Regressor (SVR)**: A kernel-based method using an **RBF kernel** ($C=10$, $\text{gamma}=\text{'scale'}$)[cite: 37].

## Key Results

[cite_start]The **Support Vector Regression (SVR)** model demonstrated superior performance, achieving a stronger linear correlation and lower error magnitude compared to the Random Forest model[cite: 10, 160].

| Model | Mean Squared Error (MSE) | Coefficient of Determination ($R^{2}$) | Pearson Correlation ($r$) |
| :--- | :--- | :--- | :--- |
| **Support Vector Regression (SVR)** | [cite_start]$\approx 0.008$ [cite: 43] | [cite_start]$\approx 0.93$ [cite: 43] | [cite_start]$\approx 0.96$ [cite: 43] |
| **Random Forest (RF)** | [cite_start]$\approx 0.014$ [cite: 43] | [cite_start]$\approx 0.84$ [cite: 43] | [cite_start]$\approx 0.91$ [cite: 43] |

[cite_start]**Conclusion:** The combination of PCA-based feature compression with Support Vector Regression provides an efficient and highly accurate approach for redshift estimation from spectral data[cite: 11, 171]. [cite_start]The resulting Root Mean Squared Error (RMSE) of $\approx 0.09$ for SVR is highly accurate for many astronomical applications[cite: 168].
