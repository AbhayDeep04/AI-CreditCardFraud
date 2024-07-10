# Credit Card Fraud Detection Web App

This project is a web application for detecting fraudulent credit card transactions using a Logistic Regression model. The application is built with Python and uses libraries like NumPy, Pandas, Scikit-Learn, and Streamlit.

## Overview

The dataset used is from Kaggle's credit card fraud detection dataset, which contains transactions made by European cardholders in September 2013. The dataset includes 492 fraudulent transactions and a large number of legitimate transactions.

## Features

- Data preprocessing and exploratory data analysis (EDA)
- Training a Logistic Regression model
- Evaluating model performance on training and testing data
- A web application built with Streamlit for predicting fraud in credit card transactions

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. Install the required libraries

3. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project directory.

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run creditFraud.py
    ```

2. Enter the credit card transaction details in the input box and click "Predict" to see if the transaction is legitimate or fraudulent. (Without the 0 or 1 at the end that symolizes legit or fraud respectively)


