# Neutrosophic PLS - Interactive Mode Guide

## A Step-by-Step Guide for Non-Programmers

**For researchers who want to use N-PLS without writing code**

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Running the Interactive Mode](#2-running-the-interactive-mode)
3. [Step-by-Step Walkthrough](#3-step-by-step-walkthrough)
4. [Understanding the Results](#4-understanding-the-results)
5. [Interpreting VIP Analysis](#5-interpreting-vip-analysis)
6. [Saving Your Results](#6-saving-your-results)
7. [Troubleshooting](#7-troubleshooting)
8. [Glossary](#8-glossary)

---

## 1. Getting Started

### What is N-PLS?

Neutrosophic Partial Least Squares (N-PLS) is an advanced machine learning technique for:

- **Predicting** continuous values (like protein content) from spectral data
- **Classifying** samples into categories
- **Handling noisy data** better than traditional methods

### Prerequisites

Before using the interactive mode, you need:

1. **Python installed** (version 3.8 or higher)
2. **The N-PLS package installed**
3. **Your data file** (CSV or ARFF format)

### Installing Python (if needed)

1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, check ✅ "Add Python to PATH"
3. Click "Install Now"

### Installing N-PLS Package

Open a terminal (Command Prompt on Windows, Terminal on Mac) and type:

```bash
pip install -e .
```

Run this command from the N-PLS folder.

---

## 2. Running the Interactive Mode

### Step 1: Open a Terminal

**Windows:**

- Press `Win + R`, type `cmd`, press Enter
- OR search for "Command Prompt" in the Start menu

**Mac:**

- Press `Cmd + Space`, type "Terminal", press Enter

### Step 2: Navigate to the N-PLS Folder

```bash
cd path/to/neutrosophic_pls
```

For example:

```bash
cd C:\Users\YourName\Desktop\neutrosophic_pls
```

### Step 3: Start Interactive Mode

Type this command and press Enter:

```bash
python -m neutrosophic_pls --interactive
```

You should see:

```
===============================================================
         Neutrosophic Partial Least Squares (N-PLS)

   Uncertainty-aware PLS with Truth/Indeterminacy/Falsity
===============================================================


╔═══════════════════════════════════════════════════════════════╗
║            Neutrosophic PLS - Interactive Analysis            ║
╚═══════════════════════════════════════════════════════════════╝
           For researchers without coding experience
```

---

## 3. Step-by-Step Walkthrough

The interactive mode guides you through 7 steps:

### STEP 1/7: Load Your Data

```
STEP 1/7: Load Your Data
──────────────────────────────────────────────────
Available datasets in 'data/':
  1. A3 (CSV, 1.84 MB)
  2. B1 (CSV, 2.24 MB)
  3. MA_A2 (CSV, 1.84 MB)
  4. micro-mass (ARFF, 1.4 MB)

Enter selection (1-5) or file path:
```

**What to do:**

- Type a number (e.g., `3`) to select a dataset from the list
- OR type the full path to your own data file

**Your data should have:**

- One row per sample
- One column per feature (e.g., wavelengths)
- One target column (what you want to predict)

---

### STEP 2/7: Data Summary

```
STEP 2/7: Data Summary
──────────────────────────────────────────────────
Columns (742): Protein, 730, 730.5, 731, ...

Enter target column name [Protein]:
Task type (r=regression, c=classification) [r]:
Columns to exclude (comma-separated, or empty) []:
```

**What to do:**

| Prompt | What it means | Your action |
|--------|---------------|-------------|
| Target column | The column you want to predict | Press Enter to accept default, or type column name |
| Task type | Regression (numbers) or classification (categories) | Type `r` or `c` |
| Exclude columns | Columns to ignore (like sample ID) | Leave empty or type names |

**Note:** Normalization (like SNV) is now handled internally by the encoders. The NDG encoder, for example, has configurable normalization options.

---

### STEP 3/7: Encoder Selection

```
STEP 3/7: Encoder Selection
──────────────────────────────────────────────────
How would you like to encode your data?

  [A] Automatic - Let the system find the best encoder
  [M] Manual    - Choose a specific encoder

Selection [A]:
```

**Recommendation:** Choose `A` (Automatic)

The system will test multiple encoders and select the best one.

**What are encoders?**

Encoders convert your data into a special format with three components:

- **Truth (T)**: The clean signal
- **Indeterminacy (I)**: Measurement uncertainty
- **Falsity (F)**: Noise and outliers

---

### STEP 4/7: Model Selection

```
STEP 4/7: NPLS Variant Selection
──────────────────────────────────────────────────
Which NPLS variant would you like to use?

  [A] Automatic - Let the system select based on your data
  [M] Manual    - Choose from available variants:
        1. NPLS   - Standard (sample weighting)
        2. NPLSW  - Reliability-weighted (best for noisy samples)
        3. PNPLS  - Probabilistic (best for localized noise)

Selection [A]:
```

**Recommendation:** Choose `A` (Automatic)

**If choosing manually:**

| Model | Best for |
|-------|----------|
| NPLS | General purpose, clean data |
| NPLSW | When some samples are unreliable |
| PNPLS | When noise affects specific wavelengths |

**Training configuration:**

```
Training configuration:
  Number of components [5]:
  CV folds [5]:
  CV repeats [3]:
```

- **Components**: How many latent variables (5-15 is typical)
- **CV folds**: Cross-validation splits (5 is standard)
- **CV repeats**: How many times to repeat CV (3 is good)

---

### STEP 5/7: Run Analysis

```
STEP 5/7: Run Analysis
──────────────────────────────────────────────────
Compare with Classical PLS? [Y]:

Running 5-fold × 3-repeat cross-validation...
  Evaluating: NPLS vs Classical PLS

  [████████████████████████████████████████] 100%
```

**What's happening:**

- The system trains and tests your model
- It compares N-PLS with traditional PLS
- Shows a progress bar

**Results appear:**

```
══════════════════════════════════════════════════════════════════════
              RESULTS COMPARISON: NPLS vs Classical PLS
══════════════════════════════════════════════════════════════════════

┌───────────┬──────────────────────┬──────────────────────┬────────────┐
│ Metric    │      Classical PLS   │   NPLS               │ Improve    │
├───────────┼──────────────────────┼──────────────────────┼────────────┤
│ RMSEP     │  0.8500 ± 0.0650    │  0.7385 ± 0.0563    │ +13.1% ↓   │
│ R²        │  0.8200 ± 0.0400    │  0.8711 ± 0.0280    │ +28.4% ↑   │
│ MAE       │  0.6500              │  0.5729              │ +11.9% ↓   │
└───────────┴──────────────────────┴──────────────────────┴────────────┘

  ✓ NPLS OUTPERFORMS Classical PLS!
```

---

### STEP 6/7: VIP Analysis

```
STEP 6/7: Feature Importance (VIP) Analysis
──────────────────────────────────────────────────
Would you like to analyze feature importance?

Run VIP analysis? [Y]:
```

**What is VIP?**

VIP (Variable Importance in Projection) tells you which features (e.g., wavelengths) are most important for predictions.

**Results show:**

```
══════════════════════════════════════════════════════════════════════
              TOP 10 MOST IMPORTANT FEATURES
══════════════════════════════════════════════════════════════════════
┌──────┬─────────────────┬─────────┬─────────┬─────────┬─────────┐
│ Rank │ Feature         │   VIP   │  VIP^T  │  VIP^I  │  VIP^F  │
├──────┼─────────────────┼─────────┼─────────┼─────────┼─────────┤
│    1 │ 1100            │   1.796★│   1.650 │   0.095 │   0.051 │
│    2 │ 1099.5          │   1.783★│   1.620 │   0.105 │   0.058 │
│    3 │ 730             │   1.753★│   1.700 │   0.033 │   0.020 │
└──────┴─────────────────┴─────────┴─────────┴─────────┴─────────┘
  ★ = VIP > 1 (important feature)
```

**Understanding the columns:**

| Column | Meaning |
|--------|---------|
| VIP | Total importance score (higher = more important) |
| VIP^T | Importance from the signal (Truth) |
| VIP^I | Importance from uncertainty |
| VIP^F | Importance from noise/outliers |

---

### STEP 7/7: Export Figures

```
STEP 7/7: Export Figures
──────────────────────────────────────────────────
Would you like to export analysis report figures?

Export figures? [N]:
```

Type `Y` to save figures, or press Enter to skip.

---

## 4. Understanding the Results

### Regression Metrics

| Metric | What it means | Good value |
|--------|---------------|------------|
| **RMSEP** | Average prediction error | Lower is better |
| **R²** | How well model explains data | 0.8+ is good, 0.9+ is excellent |
| **MAE** | Average absolute error | Lower is better |
| **RPD** | Prediction quality ratio | >2.5 is good, >3.0 is excellent |

### Classification Metrics

| Metric | What it means | Good value |
|--------|---------------|------------|
| **Accuracy** | % of correct predictions | Higher is better |
| **F1 Score** | Balance of precision/recall | 0.8+ is good |
| **Precision** | % of positive predictions that are correct | Higher is better |
| **Recall** | % of actual positives found | Higher is better |

### Comparing N-PLS to Classical PLS

The comparison table shows:

- **↓** for RMSEP/MAE = N-PLS has lower (better) error
- **↑** for R² = N-PLS has higher (better) explanation
- **Improve %** = How much better N-PLS is

---

## 5. Interpreting VIP Analysis

### What VIP Scores Mean

| VIP Value | Interpretation |
|-----------|----------------|
| **VIP > 1** ★ | Important feature - keep it! |
| **0.8 < VIP ≤ 1** | Moderately important |
| **VIP < 0.8** | Less important - can consider removing |
| **VIP < 0.5** ○ | Very low importance - removal candidate |

### Channel Breakdown (NVIP)

The N-PLS advantage is seeing WHERE importance comes from:

| If high in... | It means... |
|---------------|-------------|
| **VIP^T** | Feature has strong, reliable signal |
| **VIP^I** | Uncertainty pattern is informative |
| **VIP^F** | Noise pattern matters (check data quality!) |

### Signal Quality Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                   SIGNAL QUALITY ANALYSIS                       │
├─────────────────────────────────────────────────────────────────┤
│ High quality (SNR > 2)    :  166 features  ✓ Good       │
│ Low quality (SNR < 1)     :  392 features  ⚠ Check data │
└─────────────────────────────────────────────────────────────────┘
```

**SNR (Signal-to-Noise Ratio):**

- SNR > 2: Good quality, trust these features
- SNR < 1: Noisy feature, may need attention

---

## 6. Saving Your Results

### Saving VIP Analysis

When prompted:

```
Save VIP analysis to CSV? (path or ENTER to skip): vip_results.csv
```

Type a filename to save, or press Enter to skip.

**The CSV will contain:**

- Feature names
- VIP scores (aggregate, T, I, F)
- Dominant channel
- SNR values

### Saving Cross-Validation Results

When prompted:

```
Save results to directory? (path or ENTER to skip): results/
```

Type a folder name to save results.

---

## 7. Troubleshooting

### "Command not found" error

**Problem:** `python` command not recognized

**Solution:**

1. Make sure Python is installed
2. Try `python3` instead of `python`
3. Reinstall Python and check "Add to PATH"

### "Module not found" error

**Problem:** N-PLS package not installed

**Solution:**

```bash
pip install -e .
```

### "File not found" error

**Problem:** Data file doesn't exist

**Solution:**

1. Check the file path is correct
2. Make sure the file is in the `data/` folder
3. Use the full path: `C:\Users\...\file.csv`

### Very high RMSE values

**Problem:** PNPLS showing unrealistic numbers

**Solution:**

- Try a different model (NPLS or NPLSW)
- Use a different encoder (try "probabilistic")
- Check your data for extreme outliers

### Low R² values

**Problem:** Model isn't predicting well

**Solutions:**

- Increase number of components (try 10-15)
- Try SNV normalization for spectral data
- Check if target column is correct
- Verify data quality

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **PLS** | Partial Least Squares - a regression method for high-dimensional data |
| **N-PLS** | Neutrosophic PLS - adds uncertainty handling to PLS |
| **Component** | A latent variable that captures patterns in data |
| **VIP** | Variable Importance in Projection - measures feature importance |
| **Cross-validation** | Testing method that splits data into training/test sets |
| **RMSEP** | Root Mean Square Error of Prediction |
| **R²** | Coefficient of Determination (0-1, higher is better) |
| **Encoder** | Converts data into Truth/Indeterminacy/Falsity format |
| **Truth (T)** | The estimated clean signal in your data |
| **Indeterminacy (I)** | Measurement uncertainty |
| **Falsity (F)** | Noise or outlier component |

---

## Quick Start Checklist

☐ Python installed  
☐ N-PLS package installed  
☐ Data file ready (CSV with target column)  
☐ Open terminal in N-PLS folder  
☐ Run: `python -m neutrosophic_pls --interactive`  
☐ Follow the 7-step wizard  
☐ Review results  
☐ Save VIP analysis for later  

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#7-troubleshooting) section
2. Review error messages carefully
3. Try with the example datasets first
4. Contact the development team

---

## Example Session (Complete)

Here's what a typical session looks like:

```text
$ python -m neutrosophic_pls --interactive

═══════════════════════════════════════════════════════════════
         Neutrosophic Partial Least Squares (N-PLS)
═══════════════════════════════════════════════════════════════

STEP 1/7: Load Your Data
Enter selection (1-5) or file path: 3
✓ Loaded: MA_A2.csv

STEP 2/7: Data Summary
Enter target column name [Protein]: [Enter]
Task type [r]: [Enter]
Continue? [Y/n]: [Enter]

STEP 3/7: Encoder Selection
Selection [A]: [Enter]
✓ Selected: ndg encoder

STEP 4/7: NPLS Variant Selection
Selection [A]: [Enter]
Number of components [5]: 10
✓ Selected: NPLS

STEP 5/7: Run Analysis
Compare with Classical PLS? [Y]: [Enter]
[████████████████████████████████████████] 100%

  ✓ NPLS matches Classical PLS (clean data bypass)

STEP 6/7: VIP Analysis
Run VIP analysis? [Y]: [Enter]
  ★ Top feature: 1100 (VIP=1.796)

Save VIP to CSV? vip_results.csv
✓ Saved

STEP 7/7: Export Figures
Export figures? [N]: [Enter]

Interactive session completed successfully!
```

---

*Document created for the Neutrosophic PLS package - December 2024*
