# House Price Prediction and Property Segmentation

## Description

This project aims to predict house prices using Random Forest Regression and perform property segmentation using K-Means Clustering based on the CRISP-DM methodology.

## Dataset

House Prices: Advanced Regression Techniques

Source:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Algorithms

### Regression

* Linear Regression
* Random Forest Regression

### Clustering

* K-Means Clustering

## Evaluation Metrics

### Regression

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Clustering

* Silhouette Score

## Features Used

* OverallQual
* GrLivArea
* GarageCars
* TotalBsmtSF
* YearBuilt

## Framework

CRISP-DM

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

## Web Application

The web application is developed using Streamlit and consists of:

* Home
* Dataset Overview
* Prediction
* Visualization
* About

## Project Structure

```plaintext
HousePriceProject
│
├── dataset
│
├── notebook
│   └── analysis.ipynb
│
├── model
│   ├── rf_model.pkl
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│
├── app
│   ├── app.py
│   ├── pages
│   └── assets
│
├── laporan
│   └── laporan.pdf
│
├── requirements.txt
│
└── README.md
```

## Authors

* Steven Krisna Putra
* Muhammad Nashrul Aziz

## Repository

GitHub:
https://github.com/ENRIST/house-price-prediction-and-property-segmentation.git

## Deployment

Streamlit:
belom
