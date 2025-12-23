# Naive-Bayes---Titanic-survival-Probabilistic-Machine-Learning
##Overview

This project implements a Gaussian Naive Bayes classifier to predict passenger survival on the Titanic dataset. It demonstrates a full probabilistic machine learning pipeline, including data preprocessing, exploratory analysis, correlation checks, model training, incremental learning, evaluation, and an optional interactive prediction dashboard using ipywidgets.
The repository is suitable for academic submission, learning probabilistic ML, and portfolio/GitHub showcasing.

## Project Objectives

* Apply Bayes' Theorem to a real-world classification problem
* Understand assumptions behind Naive Bayes (feature independence & Gaussian distribution)
* Perform data preprocessing and feature engineering
* Train and evaluate a Gaussian Naive Bayes model
* Compare performance with partial vs full training data
* Visualize correlations and feature distributions
* Build an interactive front-end for live predictions

## Dataset

* **Dataset:** Titanic Passenger Dataset
* **Target Variable:** Survived (0 = No, 1 = Yes)
* **Features Used:**
  * Age (binned)
  * Sex
  * Embarked
  * Fare
  * Parch
  * SibSp

The dataset is preprocessed to handle missing values, categorical encoding, and feature selection.

## Technologies Used

* **Python 3**
* Libraries:
* numpy, pandas
* scikit-learn
* seaborn, matplotlib
* scipy
* ipywidgets (for interactive dashboard)

## Methodology

### 1. Data Preprocessing
* Merged train and test datasets for uniform preprocessing
* Encoded categorical features (Sex, Embarked)
* Discretized Age into quantile-based bins
* Removed highly correlated feature (Pclass) using Pearson correlation
* Handled missing values via dropping (train) and mean imputation (test)

### 2. Correlation Analysis
* Used Pearson Correlation Coefficient (PCC)
* Visualized correlations with a heatmap
* Ensured reduced dependency between predictors to align with Naive Bayes assumptions

### 3. Distribution Analysis
* Checked Gaussian assumptions for numeric features
* Observed mild skew but acceptable approximation for Gaussian Naive Bayes

## Model Training

### Algorithm
* Gaussian Naive Bayes (sklearn.naive_bayes.GaussianNB)

### Training Strategy
* Split data into:
  * Training (80%)
  * Validation (20%)
* Further split training set into:
  * 30% initial training
  * 70% incremental update using partial_fit
    
This demonstrates Naive Bayes' ability to support incremental learning.

## Evaluation Metrics

The following metrics were used:
* Accuracy
* Recall
*Precision

### Results Summary
* Model performance improved after updating from 30% to 100% training data
* Validation accuracy and recall showed consistent gains
* Confirms effectiveness of probabilistic learning with more evidence

Bar charts are included to visually compare performance across training stages.

## Model Interpretability

The trained model exposes:
* Class Prior Probabilities (P(Survived))
* Feature-wise Mean (θ) per class
* Feature-wise Variance (σ²) per class

This makes Naive Bayes highly transparent and interpretable compared to black-box models.

## Interactive Dashboard
An optional interactive dashboard is implemented using ipywidgets.

### Features
* Input widgets for:
  * Age (binned)
  * Sex
  * Embarked
  * Fare
  * Parch
  * SibSp
* Predict button
* Outputs:
  * Survival prediction (0 / 1)
  * Probability of survival and non-survival
    
**Usage**
1. Run the notebook/script in Jupyter or Google Colab
2. Adjust feature values using sliders and dropdowns
3. Click Predict Survival to view results instantly

## Pros and Cons of Naive Bayes

### Advantages
* Fast training and inference
* Works well with small datasets
* Highly interpretable
* Supports incremental updates

### Limitations
* Strong independence assumption
* Sensitive to correlated features
* Probability estimates may be poorly calibrated
* Zero-frequency problem (can be mitigated with smoothing)

## Conclusion
This project showcases how probabilistic machine learning can be applied effectively to real-world data. Despite its simplicity, Naive Bayes delivers competitive results, strong interpretability, and fast training — making it an excellent baseline and educational model.

## References
* Analytics Vidhya – Naive Bayes Explained
* Machine Learning Mastery – Bayesian Networks
* Scikit-learn Documentation
