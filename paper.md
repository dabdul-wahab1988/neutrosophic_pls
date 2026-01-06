---
title: 'Neutrosophic Partial Least Squares (N-PLS): Uncertainty-aware PLS regression with Truth–Indeterminacy–Falsity encoding for chemometrics'
tags:
  - Python
  - chemometrics
  - spectroscopy
  - partial least squares
  - uncertainty quantification
  - robust regression
authors:
  - name: Ebenezer Aquisman Asare
    orcid: 0000-0003-1185-1479
    affiliation: 1
  - name: Dickson Abdul-Wahab
    orcid: 0000-0001-7446-5909
    corresponding: true
    affiliation: 2
affiliations:
  - index: 1
    name: Ghana Atomic Energy Commission, Ghana
  - index: 2
    name: University of Ghana, Ghana
date: 06 January 2026
bibliography: paper.bib
---

# Summary

Partial least squares (PLS) regression is a core method in chemometrics and spectroscopy for building predictive models from many (often highly correlated) measurements such as spectral wavelengths [@wold2001pls; @geladi1986pls]. In practice, spectroscopic measurements are affected by heterogeneous noise, outliers, and varying measurement quality across samples and wavelengths. `neutrosophic-pls` is a Python package that extends classical PLS with an explicit representation of measurement reliability using three aligned channels—Truth (signal), Indeterminacy (uncertainty), and Falsity (corruption/noise)—motivated by neutrosophic set theory [@smarandache1999neutro; @smarandache2005neutro]. The package targets researchers who need robust multivariate calibration with interpretable diagnostics, and it also includes an interactive command-line wizard intended for non-programmers.

# Statement of need

Standard PLS implementations assume that each measurement contributes equally to the latent-variable model. When noise is heteroscedastic or corruption is localized (e.g., detector artifacts at specific wavelengths), this assumption can reduce predictive accuracy and obscure which variables are informative versus unreliable. Common practices—such as global preprocessing, manual outlier removal, or robust estimators—can help, but they typically do not provide a unified mechanism to (1) encode uncertainty and corruption alongside the signal and (2) propagate that information into both fitting and interpretation.

N-PLS addresses this need by encoding each observation into three aligned channels: a Truth channel (signal), an Indeterminacy channel (uncertainty), and a Falsity channel (noise/outlier evidence). These channels allow models to downweight unreliable samples or cells during training while preserving the original data for prediction and interpretation. The package provides three model variants: NPLS (sample-weighted), NPLSW (reliability-weighted), and PNPLS (element-wise probabilistic weighting via an EM-NIPALS procedure). For interpretability, the package implements Neutrosophic Variable Importance in Projection (NVIP), decomposing variable importance into T/I/F contributions to distinguish informative signal from uncertainty- or corruption-driven effects [@wold1993vip; @mehmood2012plsvarsel].

The target audience includes chemometricians and spectroscopy practitioners performing multivariate calibration and feature interpretation under variable measurement quality, including use cases where analysts prefer guided, interactive workflows instead of programming.

# State of the field

The Python ecosystem already includes mature implementations of classical PLS (e.g., `PLSRegression` in scikit-learn) [@pedregosa2011sklearn]. Related PLS tooling also exists in other ecosystems commonly used in chemometrics (e.g., the R packages `pls` and `mixOmics`) [@mevik2007pls; @rohart2017mixomics]. There is also extensive literature on robust and weighted approaches to latent-variable regression [@serneels2005prm] and on separating structured signal from sparse corruption (e.g., robust PCA) [@candes2011rpca]. However, these toolkits typically operate on a single data matrix and treat reliability indirectly (through preprocessing or global weighting), which makes it difficult to attribute predictive performance and variable importance to signal versus uncertainty.

N-PLS contributes a practical bridge between neutrosophic uncertainty representation and PLS-style multivariate calibration: it standardizes multiple encoding strategies (probabilistic residual-based, RPCA-based, wavelet-based, and domain-oriented methods), integrates them into fitting procedures that can apply sample-wise or element-wise reliability weights, and provides channel-decomposed variable-importance outputs (NVIP) for diagnosis. This combination is designed to support research workflows where data quality varies across samples and wavelengths, and where interpretability about reliability is as important as prediction.

# Software design

The package is organized around three layers: (1) encoding methods that map raw data to T/I/F tensors, (2) model variants that consume these tensors and learn latent structures with reliability-aware weighting, and (3) user-facing workflows that run reproducible studies via configuration files or an interactive wizard. Encoding can be selected explicitly or via cross-validation-based automatic selection. For usability and stability, the implementation includes a “clean-data bypass” that detects when indeterminacy and falsity are low and, in that regime, dispatches to scikit-learn’s `PLSRegression` to match classical PLS behavior and numerical properties [@pedregosa2011sklearn].

![High-level workflow of `neutrosophic-pls`: raw data are encoded into Truth/Indeterminacy/Falsity (T/I/F) channels, then used by N-PLS model variants to produce predictions and channel-decomposed variable importance (NVIP).](paper_figures/npls_workflow.png){#fig:npls-workflow width="100%"}

# Mathematics

N-PLS uses neutrosophic channels to modulate the contribution of unreliable observations. In the sample-weighted setting, the per-sample reliability weight is computed from mean indeterminacy and falsity, $w_i \propto \exp\left(-\lambda_I \bar{I}_i - \lambda_F \bar{F}_i\right)$, so samples with larger uncertainty/corruption contribute less during fitting.

For interpretability, Neutrosophic Variable Importance in Projection (NVIP) extends the classical VIP concept [@wold1993vip] by decomposing importance into channel contributions and aggregating them via an L2 relationship,
$\mathrm{VIP}_{\mathrm{agg}}(j) = \sqrt{\mathrm{VIP}_T^2(j) + \mathrm{VIP}_I^2(j) + \mathrm{VIP}_F^2(j)}$

# Research impact statement

`neutrosophic-pls` is archived on Zenodo [@asare2026npls]. The repository includes unit tests, multiple real spectroscopic datasets, and scripts for generating publication-style figures and tables to support reproducibility. On the included IDRC 2016 Chemometrics ShootOut (MA_A2.csv) near-infrared (NIR) spectroscopy dataset for protein prediction (248 samples, 741 features; 5-fold cross-validation repeated 3 times), the NPLS model reduces RMSEP by approximately 10% compared to classical PLS (1.6540 ± 0.97 to 1.4867 ± 1.06), demonstrating a measurable improvement on noisy spectroscopy data under a transparent evaluation protocol.

# AI usage disclosure

Generative AI (OpenAI GPT models via the Codex CLI) was used to assist with copy-editing the manuscript text and preparing bibliographic entries. All technical claims, results statements, and references were reviewed and verified by the authors.

# Acknowledgements

The authors acknowledge the open-source scientific Python ecosystem that this package builds upon, including NumPy, SciPy, and scikit-learn [@pedregosa2011sklearn].

# References
