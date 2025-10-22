# 🚘 Car Price Prediction Using Linear & Lasso Regression
## 📘 Overview

This project builds a machine learning model to predict car selling prices using regression-based algorithms.
The notebook implements Linear Regression and Lasso Regression to estimate prices based on key vehicle attributes such as fuel type, transmission, and year of manufacture.

The workflow demonstrates a complete supervised learning pipeline: data cleaning → feature encoding → model training → evaluation → comparison.
It provides valuable insights into how linear models can capture market trends in car pricing.

## 📊 Dataset

The dataset car data.csv contains real-world data of cars listed for resale.
Each record includes vehicle specifications and their corresponding selling prices.

### Key Features:

Car_Name – Name of the car model

Year – Year of manufacture

Selling_Price – Price at which the car was sold (Target)

Present_Price – Current ex-showroom price

Driven_kms – Total kilometers driven

Fuel_Type – Type of fuel (Petrol, Diesel, CNG)

Seller_Type – Type of seller (Dealer, Individual)

Transmission – Gear type (Manual, Automatic)

Owner – Number of previous owners

## 🧹 Data Preprocessing

Handled Missing Values: Checked for null entries (none found).

Categorical Encoding:

Fuel_Type: Petrol → 0, Diesel → 1, CNG → 2

Seller_Type: Dealer → 0, Individual → 1

Transmission: Manual → 0, Automatic → 1

Feature Selection:

Dropped Car_Name (non-numeric, non-predictive).

Split dataset into features (X) and target (Y = Selling_Price).

Train-Test Split: 90% training, 10% testing with random state = 2.

## 🤖 Model Implementation
### 🔹 Linear Regression

Objective: Capture the linear relationship between features and car price.

Model: sklearn.linear_model.LinearRegression

Training: Fit on X_train and Y_train

Evaluation Metrics:

R² Score on training and test data

Visualization of predicted vs actual prices

### 🔹 Lasso Regression

Objective: Regularized version of Linear Regression to prevent overfitting by applying L1 penalty.

Model: sklearn.linear_model.Lasso

Comparison: Both models’ performances were evaluated to identify which provides better generalization.

## 📈 Model Evaluation
Model	              R² (Train)	    R² (Test)	     Remarks
Linear Regression	    ~0.96	          ~0.84	       High accuracy, slight overfitting
Lasso Regression	    ~0.84	          ~0.82	       Slightly lower accuracy but better generalization

Both models accurately predict car prices, with Linear Regression performing slightly better in-sample, and Lasso offering better control of overfitting.

Visual Analysis:

Predicted vs Actual price plots show close alignment, validating the models’ performance.

Distribution of residuals indicates that errors are approximately normally distributed.

## 🚀 Future Enhancements

Include additional features like car brand reputation, location, or condition score.

Compare with advanced models (Ridge Regression, Random Forest, XGBoost).

Build an interactive Streamlit dashboard for user-input-based predictions.

Apply hyperparameter tuning for Lasso to optimize the alpha value.
