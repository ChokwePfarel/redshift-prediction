# 🌌 Redshift Prediction from Spectral Data

[cite_start]This repository contains the complete research project, including the final report, Python code, and data files, which implements a machine learning pipeline for **automated redshift (z) prediction** from observed astronomical flux spectra[cite: 5].

## Project Overview

[cite_start]Redshift ($z$) is a fundamental quantity in astrophysics, describing how much light from distant celestial objects is stretched due to cosmic expansion[cite: 13]. [cite_start]Accurate estimation is crucial for inferring distances, velocities, and the large-scale structure of the universe[cite: 14].

[cite_start]This project addresses the challenge of high dimensionality in spectral data ($\approx 4001$ flux points per spectrum [cite: 18][cite_start]) by applying **Principal Component Analysis (PCA)** for efficient feature extraction[cite: 6, 21]. The pipeline then compares two robust regression models to predict redshift directly from the compressed features.

## Methodology & Pipeline

1.  [cite_start]**Data Preprocessing**: Spectra data (`spectra.npy`) and ground-truth redshift values (`metadata.npy`) were prepared[cite: 26].
2.  [cite_start]**Feature Extraction**: **Principal Component Analysis (PCA)** was applied to reduce the $\approx 4001$-dimensional input to just **two components**[cite: 29].
    * [cite_start]This compression successfully retained **$97.79\%$** of the explained variance[cite: 151].
    * [cite_start]**PC1** captures the global spectral shape, while **PC2** captures emission/absorption line variations[cite: 30].
3.  [cite_start]**Regression Modeling**: Two machine learning models were trained and evaluated on the PCA-transformed features using a $97\% - 3\%$ train-test split[cite: 7, 38]:
    * [cite_start]**Random Forest Regressor (RF)**: An ensemble method for robust non-linear modeling[cite: 35].
    * [cite_start]**Support Vector Regressor (SVR)**: A kernel-based method using an **RBF kernel**[cite: 37].
4.  [cite_start]**Evaluation**: Model performance was assessed using Mean Squared Error (MSE), Coefficient of Determination ($R^2$), and Pearson Correlation Coefficient ($r$)[cite: 9].

## 🚀 Key Results

[cite_start]The **Support Vector Regression (SVR)** model demonstrated superior performance, achieving a stronger linear correlation and lower error magnitude compared to the Random Forest model[cite: 10, 160].

| Model | Mean Squared Error (MSE) | Coefficient of Determination ($R^{2}$) | Pearson Correlation ($r$) |
| :--- | :--- | :--- | :--- |
| **Support Vector Regression (SVR)** | [cite_start]$\approx 0.008$ [cite: 43] | [cite_start]$\approx 0.93$ [cite: 43] | [cite_start]$\approx 0.96$ [cite: 43] |
| **Random Forest (RF)** | [cite_start]$\approx 0.014$ [cite: 43] | [cite_start]$\approx 0.84$ [cite: 43] | [cite_start]$\approx 0.91$ [cite: 43] |

[cite_start]**Conclusion:** The combination of PCA-based feature compression with Support Vector Regression provides an efficient and highly accurate approach for redshift estimation from spectral data[cite: 11, 170]. [cite_start]The Root Mean Squared Error (RMSE) of $\approx 0.09$ for SVR is highly accurate for astronomical applications[cite: 168].

## Files in this Repository

* `Redshift_Prediction_Report.pdf`: The complete research paper detailing the methods, results, and discussion.
* `spectra.npy`: The 2D array of flux intensities.
* `metadata.npy`: Contains the corresponding ground-truth redshift values.
* *\[Your Python Code File Name(s) Here]*: The scripts implementing the PCA and regression models.

## **Contact**

* [cite_start]**Name:** Pfarelo Chokwe [cite: 2]
* [cite_start]**Student ID:** 4022839 [cite: 3]
* [cite_start]**GitHub Link (Code):** [https://github.com/ChokwePfarel/redshift-prediction/tree/main](https://github.com/ChokwePfarel/redshift-prediction/tree/main) [cite: 25]
