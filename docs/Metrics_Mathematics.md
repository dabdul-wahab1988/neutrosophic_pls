# Performance Metrics for Regression and Classification

## Mathematical Foundation for Model Evaluation

**Author:** Neutrosophic PLS Development Team  
**Date:** December 2024  
**Version:** 1.0

---

## 1. Introduction

This document provides the mathematical definitions and properties of all performance metrics used in the Neutrosophic PLS framework for model evaluation.

---

## 2. Regression Metrics

### 2.1 Root Mean Square Error of Prediction (RMSEP)

**Definition:**

$$\text{RMSEP} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Properties:**

- Same units as the response variable
- Range: $[0, \infty)$
- Lower is better
- Sensitive to outliers

**Alternative notation (RMSE):**

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\mathbb{E}[(Y - \hat{Y})^2]}$$

### 2.2 Mean Squared Error (MSE)

**Definition:**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Bias-Variance Decomposition:**

$$\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Where:

- $\text{Bias} = \mathbb{E}[\hat{Y}] - Y_{\text{true}}$
- $\text{Variance} = \mathbb{E}[(\hat{Y} - \mathbb{E}[\hat{Y}])^2]$

### 2.3 Mean Absolute Error (MAE)

**Definition:**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Properties:**

- Same units as response
- Less sensitive to outliers than RMSE
- Median-optimal (minimized by median)
- Not differentiable at zero

**Relationship to RMSE:**

$$\text{MAE} \leq \text{RMSE} \leq \sqrt{n} \cdot \text{MAE}$$

### 2.4 Coefficient of Determination (R²)

**Definition:**

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

Where:

- $\text{SS}_{\text{res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ (residual sum of squares)
- $\text{SS}_{\text{tot}} = \sum_{i=1}^{n} (y_i - \bar{y})^2$ (total sum of squares)

**Alternative formulation:**

$$R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)}$$

**Properties:**

- Range: $(-\infty, 1]$
- $R^2 = 1$: Perfect prediction
- $R^2 = 0$: Model predicts the mean
- $R^2 < 0$: Model is worse than predicting the mean

**Interpretation thresholds:**

- $R^2 > 0.9$: Excellent
- $0.7 < R^2 \leq 0.9$: Good
- $0.5 < R^2 \leq 0.7$: Moderate
- $R^2 \leq 0.5$: Poor

### 2.5 Adjusted R²

**Definition:**

$$R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

Where:

- $n$ = number of samples
- $p$ = number of predictors

**Properties:**

- Penalizes additional predictors
- $R^2_{\text{adj}} \leq R^2$
- Can be negative

### 2.6 Ratio of Performance to Deviation (RPD)

**Definition:**

$$\text{RPD} = \frac{\text{SD}(y)}{\text{RMSEP}} = \frac{\sigma_y}{\text{RMSEP}}$$

**Properties:**

- Dimensionless ratio
- Higher is better

**Interpretation (Williams, 2014):**

- $\text{RPD} > 3.0$: Excellent (screening)
- $2.5 < \text{RPD} \leq 3.0$: Very good
- $2.0 < \text{RPD} \leq 2.5$: Good (quality control)
- $1.5 < \text{RPD} \leq 2.0$: Poor (rough screening only)
- $\text{RPD} \leq 1.5$: Unreliable

**Relationship to R²:**

$$\text{RPD} = \frac{1}{\sqrt{1 - R^2}}$$

### 2.7 Ratio of Performance to Inter-Quartile (RPIQ)

**Definition:**

$$\text{RPIQ} = \frac{Q_3(y) - Q_1(y)}{\text{RMSEP}} = \frac{\text{IQR}(y)}{\text{RMSEP}}$$

Where:

- $Q_1$ = 25th percentile
- $Q_3$ = 75th percentile
- IQR = Interquartile Range

**Properties:**

- More robust than RPD (uses IQR instead of SD)
- Better for non-normal distributions
- Higher is better

### 2.8 Standard Error of Prediction (SEP)

**Definition:**

$$\text{SEP} = \sqrt{\frac{\sum_{i=1}^{n}(e_i - \bar{e})^2}{n - 1}}$$

Where $e_i = y_i - \hat{y}_i$ and $\bar{e}$ is the mean error.

**Relationship to RMSEP:**

$$\text{RMSEP}^2 = \text{SEP}^2 + \text{Bias}^2$$

### 2.9 Bias

**Definition:**

$$\text{Bias} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) = \bar{e}$$

**Properties:**

- Can be positive or negative
- Indicates systematic over/under prediction
- $\text{Bias} = 0$ for unbiased predictor

---

## 3. Classification Metrics

### 3.1 Confusion Matrix

For binary classification with classes $\{0, 1\}$:

|                | Predicted 0 | Predicted 1 |
|----------------|-------------|-------------|
| **Actual 0**   | TN          | FP          |
| **Actual 1**   | FN          | TP          |

Where:

- TP = True Positives
- TN = True Negatives
- FP = False Positives (Type I error)
- FN = False Negatives (Type II error)

### 3.2 Accuracy

**Definition:**

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{\text{Correct}}{\text{Total}}$$

**Properties:**

- Range: $[0, 1]$
- Can be misleading for imbalanced classes

### 3.3 Error Rate

**Definition:**

$$\text{Error Rate} = 1 - \text{Accuracy} = \frac{\text{FP} + \text{FN}}{\text{Total}}$$

### 3.4 Precision (Positive Predictive Value)

**Definition:**

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Interpretation:** Of all positive predictions, what fraction are correct?

### 3.5 Recall (Sensitivity, True Positive Rate)

**Definition:**

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Interpretation:** Of all actual positives, what fraction were correctly identified?

### 3.6 Specificity (True Negative Rate)

**Definition:**

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

**Interpretation:** Of all actual negatives, what fraction were correctly identified?

### 3.7 F1 Score

**Definition (harmonic mean of precision and recall):**

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \text{TP}}{2 \text{TP} + \text{FP} + \text{FN}}$$

**Properties:**

- Range: $[0, 1]$
- Balances precision and recall
- Equals 0 if either precision or recall is 0

### 3.8 F-beta Score

**Definition (general form):**

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

**Special cases:**

- $F_1$: Equal weight to precision and recall
- $F_2$: More weight to recall
- $F_{0.5}$: More weight to precision

### 3.9 Matthews Correlation Coefficient (MCC)

**Definition:**

$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$

**Properties:**

- Range: $[-1, 1]$
- $\text{MCC} = 1$: Perfect prediction
- $\text{MCC} = 0$: Random prediction
- $\text{MCC} = -1$: Total disagreement
- Works well for imbalanced datasets

### 3.10 Cohen's Kappa

**Definition:**

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

Where:

- $p_o$ = observed agreement (accuracy)
- $p_e$ = expected agreement by chance

$$p_e = \frac{(\text{TP} + \text{FN})(\text{TP} + \text{FP}) + (\text{FP} + \text{TN})(\text{FN} + \text{TN})}{n^2}$$

**Interpretation (Landis & Koch, 1977):**

- $\kappa > 0.81$: Almost perfect
- $0.61 < \kappa \leq 0.80$: Substantial
- $0.41 < \kappa \leq 0.60$: Moderate
- $0.21 < \kappa \leq 0.40$: Fair
- $\kappa \leq 0.20$: Slight/Poor

---

## 4. Multi-Class Extensions

### 4.1 Macro-Averaging

Compute metric for each class, then take unweighted mean:

$$\text{Metric}_{\text{macro}} = \frac{1}{K} \sum_{k=1}^{K} \text{Metric}_k$$

Where $K$ is the number of classes.

### 4.2 Micro-Averaging

Aggregate TP, FP, FN across all classes, then compute:

$$\text{Precision}_{\text{micro}} = \frac{\sum_k \text{TP}_k}{\sum_k (\text{TP}_k + \text{FP}_k)}$$

$$\text{Recall}_{\text{micro}} = \frac{\sum_k \text{TP}_k}{\sum_k (\text{TP}_k + \text{FN}_k)}$$

### 4.3 Weighted-Averaging

Weight by class frequency:

$$\text{Metric}_{\text{weighted}} = \sum_{k=1}^{K} \frac{n_k}{n} \cdot \text{Metric}_k$$

Where $n_k$ is the number of samples in class $k$.

---

## 5. Cross-Validation Metrics

### 5.1 Mean CV Score

$$\bar{M} = \frac{1}{K} \sum_{k=1}^{K} M_k$$

Where $M_k$ is the metric for fold $k$.

### 5.2 Standard Deviation of CV Score

$$\text{SD}(M) = \sqrt{\frac{1}{K-1} \sum_{k=1}^{K} (M_k - \bar{M})^2}$$

### 5.3 Standard Error of Mean

$$\text{SE}(\bar{M}) = \frac{\text{SD}(M)}{\sqrt{K}}$$

### 5.4 Confidence Interval

Approximate 95% CI:

$$\bar{M} \pm 1.96 \cdot \text{SE}(\bar{M})$$

---

## 6. Uncertainty-Aware Metrics (Neutrosophic)

### 6.1 Falsity-Weighted RMSE

Weight errors by falsity of predictions:

$$\text{RMSE}_F = \sqrt{\frac{\sum_i w_i (y_i - \hat{y}_i)^2}{\sum_i w_i}}$$

Where $w_i = 1 - F_i$ (downweight high-falsity predictions).

### 6.2 Reliability-Adjusted R²

$$R^2_{\text{rel}} = R^2 \cdot \bar{R}$$

Where $\bar{R}$ is the mean reliability score.

### 6.3 Prediction Confidence

$$\text{Confidence}_i = 1 - \max(I_i, F_i)$$

Mean confidence:

$$\bar{C} = \frac{1}{n} \sum_{i=1}^{n} \text{Confidence}_i$$

---

## 7. Metric Selection Guidelines

### 7.1 Regression

| Use Case | Recommended Metric |
|----------|-------------------|
| General prediction quality | RMSEP, R² |
| Robust to outliers | MAE, RPIQ |
| Calibration performance | RPD |
| Bias detection | Bias, SEP |

### 7.2 Classification

| Use Case | Recommended Metric |
|----------|-------------------|
| Balanced classes | Accuracy, F1 |
| Imbalanced classes | MCC, F1-weighted |
| Cost-sensitive | Precision or Recall |
| Agreement | Cohen's Kappa |

---

## 8. Implementation

### 8.1 Core Function (Regression)

```python
def compute_metrics(y_true, y_pred, include_extended=True):
    y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
    
    # Basic metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmsep = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    metrics = {"RMSEP": rmsep, "R2": r2, "MAE": mae}
    
    if include_extended:
        sd_y = np.std(y_true, ddof=1)
        rpd = sd_y / rmsep if rmsep > 0 else np.inf
        iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
        rpiq = iqr / rmsep if rmsep > 0 else np.inf
        bias = np.mean(y_true - y_pred)
        
        metrics.update({"RPD": rpd, "RPIQ": rpiq, "Bias": bias})
    
    return metrics
```

---

## 9. Summary Table

### Regression Metrics

| Metric | Formula | Range | Better |
|--------|---------|-------|--------|
| RMSEP | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | $[0, \infty)$ | Lower |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | $[0, \infty)$ | Lower |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | $(-\infty, 1]$ | Higher |
| RPD | $\frac{\sigma_y}{\text{RMSEP}}$ | $[0, \infty)$ | Higher |
| RPIQ | $\frac{IQR}{\text{RMSEP}}$ | $[0, \infty)$ | Higher |

### Classification Metrics

| Metric | Formula | Range | Better |
|--------|---------|-------|--------|
| Accuracy | $\frac{TP+TN}{Total}$ | $[0, 1]$ | Higher |
| Precision | $\frac{TP}{TP+FP}$ | $[0, 1]$ | Higher |
| Recall | $\frac{TP}{TP+FN}$ | $[0, 1]$ | Higher |
| F1 | $\frac{2 \cdot Prec \cdot Rec}{Prec + Rec}$ | $[0, 1]$ | Higher |
| MCC | See formula | $[-1, 1]$ | Higher |

---

## 10. References

1. Williams, P. (2014). The RPD statistic: a tutorial note. *NIR News*, 25(1), 22-26.

2. Bellon-Maurel, V., et al. (2010). Critical review of chemometric indicators commonly used for assessing the quality of the prediction of soil attributes by NIR spectroscopy. *TrAC*, 29(9), 1073-1081.

3. Matthews, B. W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta*, 405(2), 442-451.

4. Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37-46.

5. Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174.

---

*Document generated for the Neutrosophic PLS package.*
