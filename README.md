# ⚡ Smart Grid Electricity Theft Detection


### Men tavsiya qiladigan kichik yaxshilash

README'ning eng yuqorisiga **GitHub badges** ham qo‘shsak, repository ancha professional ko‘rinadi. Masalan:

```markdown
![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning project for detecting potential electricity theft from electricity and gas consumption patterns using an **XGBoost Classifier**.

The project includes data preprocessing, exploratory analysis, feature engineering, model training, evaluation, model serialization, and deployment as an interactive **Streamlit web application**.

---

## 🚀 Live Demo

🔗 **Try the deployed application:**

https://electricity-theft-detection0.streamlit.app

The web application allows users to enter hourly electricity/gas consumption values and select a building class. The trained model then predicts whether the consumption pattern is:

- ✅ **Normal Electricity Consumption**
- 🚨 **Electricity Theft**

---

## 📌 Project Overview

Electricity theft is a significant challenge for modern power grids. Detecting suspicious consumption patterns automatically can help utilities identify potentially fraudulent or abnormal electricity usage.

In this project, a supervised Machine Learning model is trained to classify electricity consumption records into two classes:

| Target | Meaning |
|---|---|
| `0` | Normal electricity consumption |
| `1` | Electricity theft |

The final model is deployed as an interactive Streamlit application so that predictions can be made through a simple web interface.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze electricity and gas consumption data
- Explore relationships between consumption features and electricity theft
- Prepare categorical and numerical features for Machine Learning
- Train an XGBoost classification model
- Evaluate model performance using multiple classification metrics
- Save the trained model for deployment
- Build an interactive Streamlit application
- Deploy the application using Streamlit Community Cloud

---

## 🧠 Machine Learning Model

The project uses an:

**XGBoost Classifier**

The trained model contains **26 input features**:

- 10 numerical consumption features
- 16 one-hot encoded building-class features

## The model was trained with the following main parameters:

n_estimators = 500
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8

## Building Class Features

Class_FullServiceRestaurant
Class_Hospital
Class_LargeHotel
Class_LargeOffice
Class_MediumOffice
Class_MidriseApartment
Class_OutPatient
Class_PrimarySchool
Class_QuickServiceRestaurant
Class_SecondarySchool
Class_SmallHotel
Class_SmallOffice
Class_Stand_aloneRetail
Class_StripMall
Class_SuperMarket
Class_Warehouse


## Model Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **95.72%** |
| Precision | **99.89%** |
| Recall    | **89.37%** |
| F1 Score  | **94.34%** |
| ROC-AUC   | **95.50%** |


## Project Structure

electricity-theft-detection/
│
├── app.py
├── electricity_theft_model.pkl
├── requirements.txt
└── README.md

## 🛠️ Technologies Used

Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
Streamlit
Jupyter Notebook
GitHub
Streamlit Community Cloud

## Installation

git clone https://github.com/Suhrobjonibodullayev/electricity-theft-detection.git

## 👨‍💻 Author

Suhrobjon Ibodullayev
Machine Learning / Data Science Project
