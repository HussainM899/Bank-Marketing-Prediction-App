# Bank Marketing Campaign Analysis and Prediction

## Overview

This project aims to predict whether a bank customer will subscribe to a term deposit based on their profile and historical data. It leverages machine learning and a user-friendly interface to assist banks in optimizing their marketing strategies and making data-driven decisions.

---

## Features

- **Prediction Model**: Uses machine learning (XGBoost) to classify customer subscription likelihood.
- **User-Friendly UI**: Built with Streamlit to allow non-technical users to input data and get predictions.
- **Data Visualization**: Insights into trends such as conversion rates, age-group analysis, and feature importance.
- **Offline Capability**: Batch processing of customer data in CSV files.

---

## Tech Stack

- **Programming Language**: Python
- **Machine Learning Libraries**: Scikit-learn, XGBoost
- **Data Visualization**: Matplotlib, Seaborn
- **Frontend Framework**: Streamlit
- **Model Serialization**: Joblib

---

## Steps to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/bank-marketing-prediction.git
cd bank-marketing-prediction
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Application
```bash
streamlit run app.py
```
### 4. Run the Application
- Upload a CSV file containing customer data for bulk predictions.
- Use the provided batch processing script to generate predictions in another CSV file.

  
