# Breast Lesion Classification Using Radiomics and Logistic Regression

This repository contains a baseline machine learning pipeline for the classification of breast lesions as **benign** or **malignant** using radiomic features extracted from medical images.

## Overview

The proposed approach uses:

- **PyRadiomics** for quantitative feature extraction
- **Binary segmentation masks** to define the lesion region of interest (ROI)
- A **Logistic Regression** model for lesion classification



## Project structure

- `main_code.py` : Main training and evaluation script
- `functions/` : Model and feature extraction functions


## Technologies

- NumPy
- Pandas
- Scikit-learn
- PyRadiomics
- SimpleITK
- OpenCV
- Statsmodels

## Author

Natalia Cabeza