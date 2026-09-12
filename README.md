# ⚡ Smart Grid Electricity Theft Detection

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

The model was trained with the following main parameters:

```text
n_estimators = 500
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
