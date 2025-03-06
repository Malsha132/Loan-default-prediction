import streamlit as st
import joblib
import pandas as pd

# Load models and scalers
personal_model = joblib.load("classification_model_personal.pkl")
personal_scaler = joblib.load("c_scaler.pkl")
personal_columns = joblib.load("c_X_train.pkl")

housing_model = joblib.load("classification_model_housing.pkl")
housing_scaler = joblib.load("h_scaler.pkl")
housing_columns = joblib.load("h_X_train.pkl")

# Function to predict loan default
def predict_loan_default(input_data, model, scaler, columns):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Main Interface
st.title("Loan Default Risk Prediction")
st.markdown("---")

# Section 1: Loan Type Selection
st.header("Select Loan Type")
loan_type = st.radio("Choose a loan type:", ["Personal Loan", "Housing Loan"], index=0)

st.markdown("---")

# Section 2: Input Fields
st.header("Enter Loan Details")
if loan_type == "Personal Loan":
    st.subheader("Personal Loan Risk Prediction")
    with st.form(key="personal_loan_form"):
        col1, col2 = st.columns(2)
        with col1:
            qspurposedes = st.selectbox("Loan Purpose", ["CONSTRUCTION", "EDUCATION", "INVESTMENT", "PERSONAL NEEDS", "PURCHASE OF PROPERTY", "PURCHASE OF VEHICLE", "WORKING CAPITAL REQUIREMENT"])
            qsector = st.selectbox("Sector", ["OTHER SERVICES", "CONSUMPTION", "MANUFACTURING & LOGISTIC", "FINANCIAL", "CONSTRUCTION & INFRASTRUCTURE", "EDUCATION", "TECHNOLOGY & INNOVATION", "TOURISM", "HEALTHCARE", "TRADERS", "AGRICULTURE & FISHING", "PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV"])
            lnbase = st.selectbox("Base", ["FINANCIAL INSTITUTIONS", "INDIVIDUALS", "MICRO FINANCE", "MIDDLE MARKET CORPORATES", "SME", "UNCLASSIFIED"])
            sex = st.selectbox("Gender", ["M", "F"])
            lnpayfreq = st.selectbox("Payment Frequency", ["2", "5", "12"])
            credit_card_used = st.radio("Used Credit Card", ["No", "Yes"])
            debit_card_used = st.radio("Used Debit Card", ["No", "Yes"])
            lnamount = st.slider("Loan Amount", min_value=1000, max_value=1000000, step=1000)
            lninstamt = st.slider("Installment Amount", min_value=100, max_value=100000, step=100)
            average_sagbal = st.slider("Average Savings Account Balance", min_value=0, max_value=1000000, step=1000)
            age = st.slider("Age", min_value=18, max_value=80)
            lnintrate = st.slider("Interest Rate", min_value=0.1, max_value=20.0, step=0.1)
        
        submit_button = st.form_submit_button(label="Predict Default Risk")
    
    if submit_button:
        user_input = pd.DataFrame({
            "LNAMOUNT": [lnamount],
            "LNINTRATE": [lnintrate],
            "LNINSTAMT": [lninstamt],
            "AGE": [age],
            "AVERAGE_SAGBAL": [average_sagbal],
            "QSPURPOSEDES": [qspurposedes],
            "QS_SECTOR": [qsector],
            "LNBASELDESC": [lnbase],
            "SEX": [sex],
            "LNPAYFREQ": [lnpayfreq],
            "CREDIT_CARD_USED": [credit_card_used],
            "DEBIT_CARD_USED": [debit_card_used]
        })
        user_input = pd.get_dummies(user_input, columns=["QSPURPOSEDES", "QS_SECTOR", "LNBASELDESC", "SEX", "LNPAYFREQ", "CREDIT_CARD_USED", "DEBIT_CARD_USED"], drop_first=True)
        missing_cols = set(personal_columns) - set(user_input.columns)
        for col in missing_cols:
            user_input[col] = 0
        user_input = user_input[personal_columns]
        prediction = predict_loan_default(user_input, personal_model, personal_scaler, personal_columns)
        
        st.markdown("---")
        st.header("Prediction Result")
        if prediction == 1:
            st.error("Prediction: The loan is at risk of default.")
        else:
            st.success("Prediction: The loan is not at risk of default.")
