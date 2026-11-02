# MLE-Based Multi-Class Pattern Recognition System

**Pattern Matching Lab (BCS-DS-738), Project 6**
**Team:** Krishna Bhatia, Shivani Kuntal | **COs:** CO2, CO3 | **Bloom's Level:** BTL-5

Implementation of **Maximum Likelihood Estimation (MLE)** for multi-class pattern recognition, with an evaluation of both the parameter estimates and the classification performance.

## What's inside
All work is in [`MLE_Multiclass_Pattern_Recognition.ipynb`](MLE_Multiclass_Pattern_Recognition.ipynb) (already executed, so outputs and plots are visible on GitHub).

| Part | Content |
|---|---|
| 1 | Theory: MLE for Gaussian class-conditionals, Bayes decision rule, covariance variants |
| 2 | Parameter estimation on synthetic data: likelihood surface, consistency (error ∝ 1/√n), bias of MLE variance |
| 3 | `MLEGaussianClassifier` written from scratch in NumPy (full / tied / diagonal covariance + ridge λ) |
| 4 | Decision boundaries on a synthetic 4-class problem |
| 5 | Wine & Iris: accuracy, precision, recall, F1, confusion matrices, posterior confidence, repeated 5-fold CV |
| 6 | Validation against scikit-learn QDA / LDA / GaussianNB (100% prediction agreement) |
| 7 | Small-sample behaviour, singular covariance when n_k ≤ d, choosing λ, AIC/BIC |
| 8 | Evaluation, limitations, conclusion |

## Key results (repeated stratified 5-fold CV, 50 folds)
| Model | Wine (3 classes, 13 features) | Iris (3 classes, 4 features) |
|---|---|---|
| MLE-Full (QDA) | 98.93% ± 1.67 | 97.00% ± 2.77 |
| MLE-Full + λ=0.1 | **99.16% ± 1.40** | 96.07% ± 3.57 |
| MLE-Tied (LDA) | 98.93% ± 1.57 | **97.87% ± 2.29** |
| MLE-Diag (Naive Bayes) | 97.47% ± 2.26 | 95.60% ± 3.74 |

## Run it
```bash
pip install -r requirements.txt
jupyter notebook MLE_Multiclass_Pattern_Recognition.ipynb
```
Or open it in Google Colab. Uses only scikit-learn's built-in datasets, so there's nothing to download.
