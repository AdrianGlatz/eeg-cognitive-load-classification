# EEG-Based Cognitive Load Classification

This project classifies different levels of cognitive load from EEG using bandpower features and machine learning.

## Dataset

The project uses the public COG-BCI dataset with EEG recordings from 29 participants performing N-back tasks:

- 0-back → low cognitive load
- 1-back → medium cognitive load
- 2-back → high cognitive load

Each participant completed three sessions. Only normal trials were used.

## Method

EEG data is filtered from 1–40 Hz and split into 2-second epochs.

Bandpower features are extracted using Welch's method from five scalp regions and five frequency bands:

- Theta
- Low Alpha
- High Alpha
- Low Beta
- High Beta

A Logistic Regression classifier is evaluated using 5-fold subject-wise cross-validation.

## Results

Accuracy: **0,383**  
Macro F1: **0.376**

Recall per class:

- Low: **40.89%**
- Medium: **24.78%**
- High: **49.32%**

High and low cognitive load were easier to distinguish than the medium condition.

Using features from every EEG channel gave only a small improvement over the original coarse ROI features, while increasing training time substantially. Using finer frequency bands with ROI features gave the best result while keeping the feature set small and interpretable.