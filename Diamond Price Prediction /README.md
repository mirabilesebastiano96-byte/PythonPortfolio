# Diamond Price Prediction

Machine learning project to predict diamond prices based on their physical and quality characteristics, using the classic **diamonds dataset** (53,940 observations, 10 variables).

## Overview

The goal is to build and evaluate supervised regression models that predict a diamond's price (`price`) from its characteristics:

- **carat** – weight of the diamond
- **cut** – quality of the cut (Fair → Ideal)
- **color** – diamond color grade
- **clarity** – measure of internal purity
- **depth**, **table** – proportions
- **x, y, z** – physical dimensions (mm)

Since `price` is continuous, the problem is framed as a **regression task**.

## Project structure

1. **Data loading & exploration (EDA)** – summary statistics, missing values, data types
2. **Preprocessing** – ordinal/label encoding of categorical features, feature scaling (`StandardScaler`)
3. **Modeling** – two approaches compared:
   - **XGBoost Regressor** (with `GridSearchCV` for hyperparameter tuning)
   - **Neural Network** (Keras `Sequential`, with `Dropout`, `BatchNormalization`, `EarlyStopping`)
4. **Evaluation** – MSE, RMSE, R² comparison between models
5. **Conclusions** – key price drivers and model comparison

## Results

| Model          | R²     |
|----------------|--------|
| XGBoost        | ~0.98  |
| Neural Network | ~0.96  |

XGBoost outperforms the neural network, confirming that tree-based models are highly effective on tabular data. Diamond prices are mainly driven by size-related variables (`y`, `carat`), with clarity and color playing a secondary role.

## Requirements

```bash
pandas
numpy
seaborn
matplotlib
scikit-learn
xgboost
keras
```

## Usage

1. Clone the repository
2. Place `diamonds.csv` in the project root (or update the path in the notebook)
3. Run `Script_Statistical_Machine_Learning.ipynb`

## Dataset

The dataset used is the well-known [diamonds dataset](https://ggplot2.tidyverse.org/reference/diamonds.html), commonly used for regression benchmarking.
