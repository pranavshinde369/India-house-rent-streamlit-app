# 🏠 House Rent Prediction (India)

## 📌 Project Overview
This project focuses on predicting house rent prices in various cities across India using Machine Learning. By analyzing features such as location, area, furnishing status, and property size, the model estimates the monthly rent, helping tenants and landlords make informed decisions.

The project explores various regression algorithms to determine the best model for accurate rent prediction.

## 📂 Dataset Details
The dataset contains **7691 records** with **10 columns**, capturing various aspects of residential properties.

| Column Name | Description | Data Type |
| :--- | :--- | :--- |
| `house_type` | Description of the property (e.g., "2 BHK Flat...") | Object |
| `locality` | Specific neighborhood or area within the city | Object |
| `city` | The city where the property is located (e.g., Mumbai, Pune) | Object |
| `area` | Built-up area of the property in square feet | Float |
| `beds` | Number of bedrooms | Integer |
| `bathrooms` | Number of bathrooms | Integer |
| `balconies` | Number of balconies | Integer |
| `furnishing` | Furnishing status (Furnished, Semi-Furnished, Unfurnished) | Object |
| `area_rate` | Rate per square foot | Float |
| `rent` | **(Target Variable)** Monthly rent in INR | Float |

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:**
    * **Data Manipulation:** `pandas`, `numpy`
    * **Visualization:** `matplotlib`, `seaborn`
    * **Machine Learning:** `scikit-learn`, `xgboost`

## ⚙️ Project Workflow
1.  **Data Ingestion:** Loading the dataset and performing initial inspection.
2.  **Data Cleaning:** Handling missing values and removing outliers to ensure data quality.
3.  **Exploratory Data Analysis (EDA):** Visualizing relationships between features (e.g., Rent vs. City, Rent vs. Furnishing) to understand trends.
4.  **Feature Engineering:**
    * Encoding categorical variables (OneHotEncoding for City, Furnishing, etc.).
    * Scaling numerical features for better model performance.
5.  **Model Training:** Training multiple regression models to find the best fit.
6.  **Model Evaluation:** Comparing models using metrics like RMSE, MAE, and R2 Score.

## 📊 Model Performance
Multiple algorithms were tested. The **Random Forest Regressor** emerged as the best-performing model.

| Model | Test R² Score | Test RMSE | Test MAE |
| :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **0.8606** | **41,523** | **17,497** |
| Gradient Boosting | 0.8005 | 43,226 | 18,830 |
| XGBoost Regressor | 0.7996 | 44,126 | 19,070 |
| Linear Regression | 0.5284 | 58,278 | 28,155 |
| Decision Tree | 0.5549 | 56,619 | 22,526 |

*Note: The Random Forest model explains approximately 76% of the variance in rent prices.*

## 📈 Future Improvements
* Hyperparameter tuning for the Random Forest and XGBoost models to further improve accuracy.
* Deployment of the model using Flask or Streamlit as a web application.
* Incorporating more features like "Distance to City Center" or "Proximity to Metro" for better predictions.

## 🤝 Contribution
Contributions are welcome! Feel free to fork this repository and submit a Pull Request.

---
**Author:** 
Pranav Shinde 

"""
