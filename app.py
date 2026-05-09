import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ohe = pickle.load(open("ohe.pkl", "rb"))

st.title("CreditWise Loan Approval System")

st.write("Enter Applicant Details")

# Numerical inputs
income = st.number_input("Applicant Income")
education = st.selectbox("Education",["Graduate","Not Graduate"])
dependents = st.number_input("Dependents")
existing_loans = st.number_input("Existing Loans")
savings = st.number_input("Savings")
collateral_value = st.number_input("Collateral Value")
co_income = st.number_input("Coapplicant Income")
age = st.number_input("Age")
credit_score = st.number_input("Credit Score")
dti = st.number_input("DTI Ratio")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")

# Categorical inputs (ALL columns you used)
employment = st.selectbox("Employment Status", ["Salaried", "Self-employed"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Other"])
property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
gender = st.selectbox("Gender", ["Male", "Female"])
employer_category = st.selectbox("Employer Category", ["Private", "Government", "Self-employed"])

# Convert to dataframe
cat_df = pd.DataFrame({
    "Employment_Status": [employment],
    "Marital_Status": [marital],
    "Loan_Purpose": [loan_purpose],
    "Property_Area": [property_area],
    "Gender": [gender],
    "Employer_Category": [employer_category]
})



# Feature engineering
credit_score_sq = credit_score ** 2
dti_sq = dti ** 2

# Combine features (order matters!)
num_features = np.array([[income, co_income, age, credit_score, dti, loan_amount]])



# ----------- FEATURE CREATION -----------

education_level = 1 if education == "Graduate" else 0

emp_salaried = 1 if employment == "Salaried" else 0
emp_self = 1 if employment == "Self-employed" else 0
emp_unemployed = 1 if employment == "Unemployed" else 0

marital_single = 1 if marital == "Single" else 0

loan_car = 1 if loan_purpose == "Car" else 0
loan_edu = 1 if loan_purpose == "Education" else 0
loan_home = 1 if loan_purpose == "Home" else 0
loan_personal = 1 if loan_purpose == "Personal" else 0

prop_semiurban = 1 if property_area == "Semiurban" else 0
prop_urban = 1 if property_area == "Urban" else 0

gender_male = 1 if gender == "Male" else 0

emp_gov = 1 if employer_category == "Government" else 0
emp_mnc = 1 if employer_category == "MNC" else 0
emp_private = 1 if employer_category == "Private" else 0
emp_unemp = 1 if employer_category == "Unemployed" else 0

dti_sq = dti ** 2
credit_sq = credit_score ** 2

features = np.array([[
    income, co_income, age, dependents, existing_loans,
    savings, collateral_value, loan_amount, loan_term,
    education_level,
    emp_salaried, emp_self, emp_unemployed,
    marital_single,
    loan_car, loan_edu, loan_home, loan_personal,
    prop_semiurban, prop_urban,
    gender_male,
    emp_gov, emp_mnc, emp_private, emp_unemp,
    dti_sq, credit_sq
]])

# Scale
features = scaler.transform(features)

# Predict
if st.button("Predict Loan Approval"):
    result = model.predict(features)

    if result[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")
