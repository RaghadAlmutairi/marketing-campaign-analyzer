# Superstore Customer Response Prediction

A machine learning project that predicts customer responses to marketing campaigns using the Superstore dataset. This project implements and compares multiple classification models including K-Nearest Neighbors (KNN) and AdaBoost.

## Overview

This project analyzes customer behavior data and builds predictive models to determine whether customers will respond positively to marketing campaigns. The analysis includes comprehensive exploratory data analysis (EDA), data preprocessing, and model evaluation using multiple machine learning algorithms.

## Project Objectives

- Analyze customer demographics and purchasing behavior
- Identify key factors influencing customer response to marketing campaigns
- Build and compare machine learning classification models
- Optimize model hyperparameters using GridSearchCV
- Provide actionable insights for targeted marketing strategies

## Dataset

**Key Features:**
- `Year_Birth`: Customer's birth year
- `Income`: Customer's annual income
- `Education`: Customer's education level
- `Marital_Status`: Customer's marital status
- `Dt_Customer`: Date when customer registered
- `Recency`: Days since last purchase
- `NumDealsPurchases`: Number of purchases made with a discount
- `NumWebPurchases`: Number of web purchases
- `NumStorePurchases`: Number of in-store purchases
- `Complain`: Whether customer has complained (1/0)
- `Response`: Target variable (1 = Responded to campaign, 0 = Did not respond)

## Project Structure

### Data Exploration & Preprocessing
1. **Data Loading & Inspection**
   - Load data from CSV file
   - Display basic information (shape, info, describe)

2. **Exploratory Data Analysis (EDA)**
   - Correlation analysis between variables
   - Visualizations: correlation heatmaps, pair plots, histograms, pie charts, category plots
   - Distribution analysis of features

3. **Data Cleaning**
   - Check for duplicate values
   - Handle missing values (NULL replacement)
   - Data type conversions (datetime, string, encoded categories)

4. **Feature Engineering & Preprocessing**
   - Label encoding for categorical variables (Education, Marital_Status)
   - Class balancing using RandomOverSampler to handle imbalanced data
   - Feature scaling using StandardScaler
   - Train-test split (70-30 split)

### Model Development

#### 1. K-Nearest Neighbors (KNN)
- **Hyperparameter Optimization:** GridSearchCV with k values 1-9
- **Best Parameters:** n_neighbors = 1
- **Metric:** Minkowski (p=2)
- **Accuracy:** 83.15%

#### 2. AdaBoost Classifier
- **Hyperparameter Optimization:** GridSearchCV for n_estimators and learning_rate
- **Best Parameters:** n_estimators = 500, learning_rate = 1.7
- **Accuracy:** 85.29%

### Model Evaluation
- Classification reports with precision, recall, and F1-scores
- Confusion matrices
- Accuracy scores
- GridSearchCV cross-validation (k-fold)

## Installation

### Requirements
- Python 3.7+
- Jupyter Notebook

### Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scikit-learn>=0.24.0
imbalanced-learn>=0.8.0
```

## Usage

Open the Jupyter notebook and run the cells sequentially:

1. **Import Libraries:** Execute the first cell to load all required packages
2. **Load Data:** Run data loading cell and verify dataset shape
3. **EDA:** Run exploration cells to generate visualizations and statistics
4. **Preprocessing:** Execute preprocessing cells to clean and prepare data
5. **Model Training:** Run KNN and AdaBoost model cells
6. **Evaluation:** Review classification reports and accuracy scores

## Results Summary

| Model | Accuracy | Best Parameters |
|-------|----------|-----------------|
| KNN | 83.15% | n_neighbors=1, metric=Minkowski (p=2) |
| AdaBoost | 85.29% | n_estimators=500, learning_rate=1.7 |

**Key Findings:**
- AdaBoost outperforms KNN with ~2% higher accuracy
- Data balancing significantly improves model performance
- Feature scaling is crucial for distance-based algorithms like KNN
- Cross-validation helps prevent overfitting

## Model Comparison & Insights

### KNN Advantages
- Simple and intuitive algorithm
- No training phase
- Effective for this balanced dataset

### AdaBoost Advantages
- Better generalization capability
- Handles complex decision boundaries
- Ensemble approach reduces overfitting

## Future Improvements

- [ ] Implement additional models (Random Forest, SVM, Neural Networks)
- [ ] Perform feature selection and dimensionality reduction (PCA, RFE)
- [ ] Create ROC curves and AUC scores for better evaluation
- [ ] Implement cross-validation strategies (k-fold, stratified)
- [ ] Deploy model as REST API
- [ ] Create interactive visualizations with Plotly
- [ ] Hyperparameter tuning with Bayesian optimization
- [ ] Analyze feature importance and model explainability (SHAP)


## Key Visualizations

- **Correlation Heatmap:** Shows relationships between all numeric variables
- **Pair Plot:** Visualizes distributions and relationships between features
- **Histograms:** Display feature distributions across the dataset
- **Pie Charts:** Show categorical variable distributions (Education)
- **Category Plots:** Count plots for categorical variables (Marital Status)


## Acknowledgments

- Dataset source: [Kaggle - Superstore Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/superstore-dataset)
- Built with scikit-learn, pandas, and other open-source Python libraries
