import streamlit as st
import pandas as pd
import joblib

# Load models
scaler = joblib.load("scaler.pkl")
lr = joblib.load("logisic_reg.pkl")
xgb = joblib.load("xgboost_model.pkl")
rf = joblib.load("random_forest.pkl")

# Set page config
st.set_page_config(page_title="Loan Default Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Loan Default Prediction App")

st.markdown("### Enter your information below:")

# --- INPUT FORM ---
with st.container():
    st.markdown("#### 📋 Personal & Financial Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income ($)", min_value=0, value=50000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=20000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        months_employed = st.number_input("Months Employed", min_value=0, value=24)
    with col2:
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=5)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=4.5)
        loan_term = st.number_input("Loan Term (months)", min_value=6, value=36)
        dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=0.25)

with st.container():
    st.markdown("#### 🧑‍💼 Other Information")
    col3, col4 = st.columns(2)
    with col3:
        education = st.selectbox("Education Level 🎓", ['High School', 'Bachelor\'s', 'Master\'s', 'PhD'])
        employment_type = st.selectbox("Employment Type 💼", ['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
    with col4:
        marital_status = st.selectbox("Marital Status ❤️", ['Divorced', 'Married', 'Single'])
        loan_purpose = st.selectbox("Loan Purpose 🏡", ['Auto', 'Business', 'Education', 'Home', 'Other'])

has_mortgage = st.radio("Do you have a mortgage?", ['Yes', 'No'], horizontal=True)
has_dependents = st.radio("Do you have dependents?", ['Yes', 'No'], horizontal=True)
has_cosigner = st.radio("Do you have a co-signer?", ['Yes', 'No'], horizontal=True)

# --- PROCESS INPUT ---
education_dict = {"High School": [0, 1, 0, 0], "Bachelor's": [1, 0, 0, 0], "Master's": [0, 0, 1, 0], "PhD": [0, 0, 0, 1]}
employment_dict = {"Full-time": [1, 0, 0, 0], "Part-time": [0, 1, 0, 0], "Self-employed": [0, 0, 1, 0], "Unemployed": [0, 0, 0, 1]}
marital_dict = {"Divorced": [1, 0, 0], "Married": [0, 1, 0], "Single": [0, 0, 1]}
loan_purpose_dict = {"Auto": [1, 0, 0, 0, 0], "Business": [0, 1, 0, 0, 0], "Education": [0, 0, 1, 0, 0], "Home": [0, 0, 0, 1, 0], "Other": [0, 0, 0, 0, 1]}

education_encoded = education_dict[education]
employment_encoded = employment_dict[employment_type]
marital_encoded = marital_dict[marital_status]
loan_purpose_encoded = loan_purpose_dict[loan_purpose]

input_data = {
    'Age': [age],
    'Income': [income],
    'LoanAmount': [loan_amount],
    'CreditScore': [credit_score],
    'MonthsEmployed': [months_employed],
    'NumCreditLines': [num_credit_lines],
    'InterestRate': [interest_rate],
    'LoanTerm': [loan_term],
    'DTIRatio': [dti_ratio],
    'HasMortgage': [1 if has_mortgage == 'Yes' else 0],
    'HasDependents': [1 if has_dependents == 'Yes' else 0],
    'HasCoSigner': [1 if has_cosigner == 'Yes' else 0],
    'Education_Bachelor\'s': education_encoded[0],
    'Education_High School': education_encoded[1],
    'Education_Master\'s': education_encoded[2],
    'Education_PhD': education_encoded[3],
    'EmploymentType_Full-time': employment_encoded[0],
    'EmploymentType_Part-time': employment_encoded[1],
    'EmploymentType_Self-employed': employment_encoded[2],
    'EmploymentType_Unemployed': employment_encoded[3],
    'MaritalStatus_Divorced': marital_encoded[0],
    'MaritalStatus_Married': marital_encoded[1],
    'MaritalStatus_Single': marital_encoded[2],
    'LoanPurpose_Auto': loan_purpose_encoded[0],
    'LoanPurpose_Business': loan_purpose_encoded[1],
    'LoanPurpose_Education': loan_purpose_encoded[2],
    'LoanPurpose_Home': loan_purpose_encoded[3],
    'LoanPurpose_Other': loan_purpose_encoded[4],
}

new_data = pd.DataFrame(input_data)

# Scale numeric columns
numeric_col = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
new_data[numeric_col] = scaler.transform(new_data[numeric_col])

# --- MODEL SELECTION ---
st.sidebar.title("🔍 Model Selection")
model_choice = st.sidebar.selectbox("Choose a Model:", ["Logistic Regression", "XGBoost", "Random Forest"])

# --- PREDICT ---
if st.button("🚀 Predict"):
    if model_choice == "Logistic Regression":
        prediction = lr.predict(new_data)
    elif model_choice == "XGBoost":
        prediction = xgb.predict(new_data)
    else:
        prediction = rf.predict(new_data)

    if prediction[0] == 0:
        st.success("✅ Prediction: The loan will **NOT** default.")
    else:
        st.error("⚠️ Prediction: The loan is **likely to default**.")
