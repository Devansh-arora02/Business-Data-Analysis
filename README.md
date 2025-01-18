# Expanding Customer Base for an Electronics Retail Store

[![PDF Report](https://img.shields.io/badge/View-PDF_Report-blue)](https://github.com/Devansh-arora02/Business-Data-Analysis/blob/main/Analysis%20Result.pdf)

This repository contains the code, data, and analysis performed for the Business Data Management (BDM) capstone project aimed at expanding the customer base of **Infinity Technologies**, an HP-authorized electronics retail chain. The project involves analyzing sales data, identifying trends, segmenting customers, and proposing actionable strategies to improve customer engagement and sales.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Details](#data-details)
3. [Analysis and Models](#analysis-and-models)
4. [Key Findings](#key-findings)
5. [Recommendations](#recommendations)

---

## Project Overview

Infinity Technologies, operating in both B2B and B2C channels in Delhi, faced challenges due to shifting customer preferences toward online platforms and large format retailers. This project aimed to:
- Analyze sales data.
- Identify customer segments based on loyalty.
- Develop optimized discount strategies.
- Provide actionable recommendations for sales growth.

---

## Data Details

### Dataset
- **Source**: Infinity Technologies' sales transactions over one year.
- **Format**: CSV
- **Size**: 4,474 records and 139 parameters
- **Key Features**:
  - Transaction details (date, time, quantity, etc.)
  - Product-specific information
  - Customer-specific metrics (e.g., loyalty, longevity)

### Preprocessing
- **Handling Missing Values**: Imputation with zeros, means, or modes.
- **Outlier Detection**: Interquartile Range (IQR) method.
- **Transformations**: Normalization and one-hot encoding for modeling.

---

## Analysis and Models

### 1. Data Analysis
- **Pareto Analysis**: Identified high-performing product groups.
- **Sales Trend Analysis**: Peak sales during evening hours (6–8 PM).

### 2. Customer Segmentation
- **Method**: K-Means Clustering (k=3).
- **Features**: Visits, profit contribution, longevity, and customer type.
- **Output**: Loyalty segments (Platinum, Gold, Silver).

### 3. Predictive Discounting
- **Model**: Random Forest Regression.
- **Optimization**: GridSearchCV for hyperparameters.
- **Evaluation**: RMSE and R² metrics.
- **Insights**: Optimal discounts to maximize profit.

---

## Key Findings

- **Sales Trends**: High foot traffic in the evening.
- **Product Insights**:
  - Retail Notebooks lead in sales and profit.
  - Consumer Volume Desktop products show potential for growth.
- **Discount Impacts**: High-MRP products are less sensitive to large discounts.

---

## Recommendations

1. **Human Resource Management**: Optimize workforce deployment during peak hours.
2. **Customer Loyalty**:
   - Assign points based on frequency, profit contribution, and longevity.
   - Offer tiered discounts for Platinum, Gold, and Silver groups.
3. **Discount Strategy**:
   - Free giveaways for high-value purchases.
   - Avoid discounting high-performing products unnecessarily.
