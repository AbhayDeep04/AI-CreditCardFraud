#Import Libraries
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

#Load the dataset
credit_card_data = pd.read_csv('creditcard.csv')

#Exploratory Data Analysis below
#print(credit_card_df.head())
#print(credit_card_data.sample())
#print(credit_card_data.info())  
#credit_card_data.isnull().sum()
#print(credit_card_data['Class'].value_counts())

#Data Preprocessing (splitting the data into legit and fraud)
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

#Randomly selecting 492 samples from legit to match the number of fraud samples
legit_sample = legit.sample(n=492, random_state=2)
#Concatenating the two dataframes
credit_card_data = pd.concat([legit_sample, fraud], axis=0)

#X is independent variable and Y is dependent variable ie 0 or 1 where 0 is legit and 1 is fraud
X = credit_card_data.drop('Class', axis=1)
Y = credit_card_data['Class']

#Splitting the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

#Model Evaluation for both training and testing data
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

#Creating the Web App

st.title("Credit Card Fraud Detection") 
input_df = st.text_input("Enter the credit card details")
input_df_splitted = input_df.split(",")

submit = st.button("Predict")

if submit:
    features = np.asarray(input_df_splitted,dtype=np.float64)
    prediction= model.predict(features.reshape(1,-1))

    if prediction[0] == 0:
        st.write("The transaction is Legitimate")
    else:    
        st.write("The transaction is Fraudulent")