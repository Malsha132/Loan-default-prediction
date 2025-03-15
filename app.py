# import streamlit as st
# import joblib
# import pandas as pd
# import base64

# # Function to get base64 encoding for image
# def get_base64_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode()

# # Function to set background image for the whole page
# def set_background(image_path):
#     base64_str = get_base64_image(image_path)
#     page_bg_img = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/jpeg;base64,{base64_str}");
#         background-size: cover; /* Ensures the image covers the entire screen */
#         background-position: center;
#         background-repeat: no-repeat;
#     }}
#     </style>
#     """
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# # Function to set background image for Section 1 only with improved scaling
# def set_section_one_background(image_path):
#     base64_str = get_base64_image(image_path)
#     section_bg_img = f"""
#     <style>
#     .section-one {{
#         background-image: url("data:image/jpeg;base64,{base64_str}");
#         background-size: 100% auto; /* Adjust width while maintaining aspect ratio */
#         background-position: center;
#         background-repeat: no-repeat;
#         padding: 50px;
#         text-align: center;
#         color: white;
#         width: 100%;
#         min-height: 250px; /* Ensures it has a proper height */
#     }}
#     </style>
#     """
#     st.markdown(section_bg_img, unsafe_allow_html=True)


# # Function to set background color for Section 2 only
# def set_section_two_background():
#     section_bg_color = """
#     <style>
#     .section-two {{
#         background-color: #f0f8ff;  /* Light blue background */
#         padding: 20px;
#         border-radius: 10px;
#     }}
#     </style>
#     """
#     st.markdown(section_bg_color, unsafe_allow_html=True)

# # Set the background image for the first section (before loan type selection)
# set_section_one_background("image.jpg")

# # Load models and scalers
# personal_model = joblib.load("classification_model_personal.pkl")
# personal_scaler = joblib.load("c_scaler.pkl")
# personal_columns = joblib.load("c_X_train.pkl")

# housing_model = joblib.load("classification_model_housing.pkl")
# housing_scaler = joblib.load("h_scaler.pkl")
# housing_columns = joblib.load("h_X_train.pkl")

# # Function to predict loan default
# def predict_loan_default(input_data, model, scaler, columns):
#     input_data_scaled = scaler.transform(input_data)
#     prediction = model.predict(input_data_scaled)
#     return prediction[0]

# # Main Interface
# st.title("Loan Default Risk Prediction")

# # Section 1 - Background image before loan type selection
# st.markdown('<div class="section-one">', unsafe_allow_html=True)
# st.header("Select Loan Type")
# loan_type = st.radio("", ["Personal Loan", "Housing Loan"])
# st.markdown('</div>', unsafe_allow_html=True)

# st.divider()

# # Section 2 - Background color after loan type selection
# set_section_two_background()  # Set background color for this section

# st.markdown('<div class="section-two">', unsafe_allow_html=True)

# # Personal Loan Input Fields
# if loan_type == "Personal Loan":
#     st.header("Personal Loan Risk Prediction")
    
#     with st.form(key="personal_loan_form"):
#         qspurposedes = st.selectbox("Loan Purpose", ["CONSTRUCTION", "PERSONAL NEEDS","PURCHASE OF VEHICLE","WORKING CAPITAL REQUIREMENT","EDUCATION","INVESTMENT","PURCHASE OF PROPERTY"])
#         qsector = st.selectbox("Sector", ["OTHER SERVICES", "MANUFACTURING & LOGISTICS", "MANUFACTURING & LOGISTICS", "FINANCIAL", "TOURISM", "TECHNOLOGY & INNOVATION","HEALTHCARE","CONSUMPTION","EDUCATION","CONSTRUCTION & INFRASTRUCTURE","TRADERS","AGRICULTURE & FISHINIG","PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV"])
#         lnbase = st.selectbox("Base", ["INDIVIDUALS", "SME", "MICRO FINANCE", "MIDDLE MARKET CORPORATES", "FINANCIAL INSTITUTIONS", "UNCLASSIFIED"])
#         lnperiod = st.selectbox("Loan Period", ["MEDIUM-TERM", "SHORT-TERM", "LONG-TERM"])
#         region=st.selectbox("Region",["Colombo Region","Other","Greater Colombo Region","North Central Region","Uva / Sabaragamuwa Region","Southern Region","Central Region","Nothern Region","South Western Region","North Western Region","Eastern Region"])
#         sex =  st.radio("Gender", ["F", "M"])
#         lnpayfreq = st.selectbox("Payment Frequency", ["5", "12", "2"])
#         credit_card_used = st.radio("Used Credit Card", ["No", "Yes"])
#         debit_card_used = st.radio("Used Debit Card", ["No", "Yes"])
        
#         # Replaced sliders with text input
#         lnamount = st.text_input("Loan Amount", value="1000")
#         lninstamt = st.text_input("Installment Amount", value="100")
#         average_sagbal = st.text_input("Average Savings Account Balance", value="0")
#         age = st.text_input("Age", value="18")
#         lnintrate = st.text_input("Interest Rate", value="0.1")
        
#         submit_button = st.form_submit_button(label="Predict Default Risk")

#     if submit_button:
#         # Convert inputs to appropriate types
#         user_input = pd.DataFrame({
#             "LNAMOUNT": [float(lnamount)],
            
#             "LNINSTAMT": [float(lninstamt)],
            
#             "AVERAGE_SAGBAL": [float(average_sagbal)],
#             "AGE": [int(age)],
#             "LNINTRATE": [float(lnintrate)],
#             "QSPURPOSEDES": [qspurposedes],
#             "QS_SECTOR": [qsector],
#             "LNBASELDESC": [lnbase],
#             "LNPERIOD_CATEGORY": [lnperiod],
#             "SEX": [sex],
#             "LNPAYFREQ": [lnpayfreq],
#             "CREDIT_CARD_USED": [credit_card_used],
#             "DEBIT_CARD_USED": [debit_card_used],
#             "LNRGNNAME" : [region]
#         })

#         user_input = pd.get_dummies(user_input, columns=["QSPURPOSEDES", "QS_SECTOR", "LNBASELDESC", "SEX", "LNPAYFREQ", "CREDIT_CARD_USED", "DEBIT_CARD_USED",'LNPERIOD_CATEGORY','LNRGNNAME'], drop_first=True)
#         missing_cols = set(personal_columns) - set(user_input.columns)
#         for col in missing_cols:
#             user_input[col] = 0
#         user_input = user_input[personal_columns]
#         prediction = predict_loan_default(user_input, personal_model, personal_scaler, personal_columns)

#         st.subheader("Prediction Result")
#         if prediction == 1:
#             st.error("The loan is at risk of default.")
#         else:
#             st.success("The loan is not at risk of default.")
            
# # Housing Loan Input Fields (similar to Personal Loan fields)
# elif loan_type == "Housing Loan":
#     st.header("Housing Loan Risk Prediction")
    
#     with st.form(key="housing_loan_form"):
#         qspurposedes = st.selectbox('Loan Purpose', ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
#         qsector = st.selectbox('Sector', ['OTHER SERVICES', 'CONSUMPTION', 'MANUFACTURING & LOGISTIC', 'FINANCIAL', 'CONSTRUCTION & INFRASTRUCTURE', 'EDUCATION', 'TECHNOLOGY & INNOVATION', 'TOURISM', 'HEALTHCARE', 'TRADERS', 'AGRICULTURE & FISHING', 'PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV'])
#         lnbase = st.selectbox('Base', ['FINANCIAL INSTITUTIONS', 'INDIVIDUALS', 'MICRO FINANCE', 'MIDDLE MARKET CORPORATES', 'SME', 'UNCLASSIFIED'])
#         sex = st.radio('Gender', ['M', 'F'])
#         lnpayfreq = st.selectbox('Payment Frequency', ['2', '5', '12'])
#         credit_card_used = st.radio('Used Credit Card', ['No','Yes'])
#         debit_card_used = st.radio('Used Debit Card', ['No','Yes'])
#         lnperiod_category = st.selectbox('Loan Period Category', ['Short-Term', 'Medium-Term', 'Long-Term'])
        
#         # Replaced sliders with text input
#         lnamount = st.text_input('Loan Amount', value="1000")
#         lninstamt = st.text_input('Installment Amount', value="100")
#         average_sagbal = st.text_input('Average Savings Account Balance', value="0")
#         age = st.text_input('Age', value="18")
#         lnintrate = st.text_input('Interest Rate', value="0.1")
        
#         submit_button = st.form_submit_button(label="Predict Default Risk")

#     if submit_button:
#         # Convert inputs to appropriate types
#         user_input = pd.DataFrame({
#             "LNAMOUNT": [float(lnamount)],
#             "LNINTRATE": [float(lnintrate)],
#             "LNINSTAMT": [float(lninstamt)],
#             "AGE": [int(age)],
#             "AVERAGE_SAGBAL": [float(average_sagbal)],
#             "QSPURPOSEDES": [qspurposedes],
#             "QS_SECTOR": [qsector],
#             "LNBASELDESC": [lnbase],
#             "LNPERIOD_CATEGORY" :[lnperiod_category],
#             "SEX": [sex],
#             "LNPAYFREQ": [lnpayfreq],
#             "CREDIT_CARD_USED": [credit_card_used],
#             "DEBIT_CARD_USED": [debit_card_used]
#         })

#         user_input = pd.get_dummies(user_input, columns=["QSPURPOSEDES", "QS_SECTOR", "LNBASELDESC", "SEX", "LNPAYFREQ", "CREDIT_CARD_USED", "DEBIT_CARD_USED",'LNPERIOD_CATEGORY'], drop_first=True)
#         missing_cols = set(housing_columns) - set(user_input.columns)
#         for col in missing_cols:
#             user_input[col] = 0
#         user_input = user_input[housing_columns]
#         prediction = predict_loan_default(user_input, housing_model, housing_scaler, housing_columns)

#         st.subheader("Prediction Result")
#         if prediction == 1:
#             st.error("The loan is at risk of default.")
#         else:
#             st.success("The loan is not at risk of default.")


            

# st.markdown('</div>', unsafe_allow_html=True)



import streamlit as st
import joblib
import pandas as pd
import base64

# Function to get base64 encoding for image
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to set background image for the whole page
def set_background(image_path):
    base64_str = get_base64_image(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_str}");
        background-size: cover; /* Ensures the image covers the entire screen */
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to set background image for Section 1 only with improved scaling
def set_section_one_background(image_path):
    base64_str = get_base64_image(image_path)
    section_bg_img = f"""
    <style>
    .section-one {{
        background-image: url("data:image/jpeg;base64,{base64_str}");
        background-size: 100% auto; /* Adjust width while maintaining aspect ratio */
        background-position: center;
        background-repeat: no-repeat;
        padding: 50px;
        text-align: center;
        color: white;
        width: 100%;
        min-height: 30px; /* Ensures it has a proper height */
    }}
    </style>
    """
    st.markdown(section_bg_img, unsafe_allow_html=True)


# Function to set background color for Section 2 only
def set_section_two_background():
    section_bg_color = """
    <style>
    .section-two {{
        background-color: #f0f8ff;  /* Light blue background */
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(section_bg_color, unsafe_allow_html=True)

# Set the background image for the first section (before loan type selection)
set_section_one_background("background.jpg")

# Load models and scalers
personal_model = joblib.load("dilan.pkl")
personal_scaler = joblib.load("inuka_scaler.pkl")
personal_columns = joblib.load("dilan_X_train.pkl")

housing_model = joblib.load("classification_model_housing (2).pkl")
housing_scaler = joblib.load("h_scaler (2).pkl")
housing_columns = joblib.load("h_X_train (2).pkl")

# Function to predict loan default
def predict_loan_default(input_data, model, scaler, columns):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Main Interface
st.title("Loan Default Risk Prediction")

# Section 1 - Background image before loan type selection
st.markdown('<div class="section-one">', unsafe_allow_html=True)
st.header("Select Loan Type")
loan_type = st.radio("", ["Personal Loan", "Housing Loan"])
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# Section 2 - Background color after loan type selection
set_section_two_background()  # Set background color for this section

st.markdown('<div class="section-two">', unsafe_allow_html=True)

# Personal Loan Input Fields
if loan_type == "Personal Loan":
    st.header("Personal Loan Risk Prediction")
    
    with st.form(key="personal_loan_form"):
        qspurposedes = st.selectbox("Loan Purpose", ["CONSTRUCTION", "PERSONAL NEEDS","PURCHASE OF VEHICLE","WORKING CAPITAL REQUIREMENT","EDUCATION","INVESTMENT","PURCHASE OF PROPERTY"])
        qsector = st.selectbox("Sector", ["OTHER SERVICES",  "MANUFACTURING & LOGISTICS", "FINANCIAL", "TOURISM", "TECHNOLOGY & INNOVATION","HEALTHCARE","CONSUMPTION","EDUCATION","CONSTRUCTION & INFRASTRUCTURE","TRADERS","AGRICULTURE & FISHINIG","PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV"])
        lnbase = st.selectbox("Base", ["INDIVIDUALS", "SME", "MICRO FINANCE", "MIDDLE MARKET CORPORATES", "FINANCIAL INSTITUTIONS", "UNCLASSIFIED"])
        lnperiod = st.selectbox("Loan Period", ["MEDIUM-TERM", "SHORT-TERM", "LONG-TERM"])
        region=st.selectbox("Region",["Colombo Region","Other","Greater Colombo Region","North Central Region","Uva / Sabaragamuwa Region","Southern Region","Central Region","Nothern Region","South Western Region","North Western Region","Eastern Region"])
        sex =  st.radio("Gender", ["F", "M"])
        lnpayfreq = st.selectbox("Payment Frequency", ["5", "12", "2"])
        credit_card_used = st.radio("Used Credit Card", ["No", "Yes"])
        debit_card_used = st.radio("Used Debit Card", ["No", "Yes"])
        
        # Replaced sliders with text input
        lnamount = st.text_input("Loan Amount", value="1000")
        lninstamt = st.text_input("Installment Amount", value="100")
        average_sagbal = st.text_input("Average Savings Account Balance", value="10000")
        age = st.text_input("Age", value="18")
        lnintrate = st.text_input("Interest Rate", value="0.1")
        
        submit_button = st.form_submit_button(label="Predict Default Risk")

    if submit_button:
        # Convert inputs to appropriate types
        user_input = pd.DataFrame({
            "LNAMOUNT": [float(lnamount)],
            
            "LNINSTAMT": [float(lninstamt)],
            
            "AVERAGE_SAGBAL": [float(average_sagbal)],
            "AGE": [int(age)],
            "LNINTRATE": [float(lnintrate)],
            "QSPURPOSEDES": [qspurposedes],
            "QS_SECTOR": [qsector],
            "LNBASELDESC": [lnbase],
            "LNPERIOD_CATEGORY": [lnperiod],
            "SEX": [sex],
            "LNPAYFREQ": [lnpayfreq],
            "CREDIT_CARD_USED": [credit_card_used],
            "DEBIT_CARD_USED": [debit_card_used],
            "LNRGNNAME" : [region]
        })

        # One-hot encoding to match training data
        user_input_dummies = pd.get_dummies(user_input, 
            columns=["QSPURPOSEDES", "QS_SECTOR", "LNBASELDESC", "SEX", "LNPAYFREQ", 
                    "CREDIT_CARD_USED", "DEBIT_CARD_USED", "LNPERIOD_CATEGORY", "LNRGNNAME"], 
                    drop_first=False
            )

        # Ensure all columns from training exist (Fill missing ones with 0)
        for col in personal_columns:
            if col not in user_input_dummies:
                user_input_dummies[col] = 0
        
        # Ensure order of columns matches training data
        user_input_dummies = user_input_dummies[personal_columns]
        
        # Scale numerical features
        num_features = ['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']

        user_input_scaled = personal_scaler.transform(user_input_dummies[num_features])
        
        user_input_scaled_df = pd.DataFrame(user_input_scaled, columns=num_features, index=user_input_dummies.index)
        
        user_input_final = user_input_dummies.drop(columns=num_features).reset_index(drop=True)

        # Add scaled numerical features
        user_input_final = pd.concat([user_input_scaled_df.reset_index(drop=True), user_input_final], axis=1)
        
        user_input_final = user_input_final[personal_columns]
        
        # Make prediction
        prediction = personal_model.predict(user_input_final)

        # Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("The loan is at risk of default.")
        else:
            st.success("The loan is not at risk of default.")
            
# Housing Loan Input Fields (similar to Personal Loan fields)
elif loan_type == "Housing Loan":
    st.header("Housing Loan Risk Prediction")
    
    with st.form(key="housing_loan_form"):
        qspurposedes = st.selectbox('Loan Purpose', ['CONSTRUCTION', 'EDUCATION', 'INVESTMENT', 'PERSONAL NEEDS', 'PURCHASE OF PROPERTY', 'PURCHASE OF VEHICLE', 'WORKING CAPITAL REQUIREMENT'])
        qsector = st.selectbox('Sector', ['AGRICULTURE & FISHING','CONSTRUCTION & INFRASTRUCTURE','FINANCIAL','MANUFACTURING & LOGISTIC','OTHER SERVICES','PROFESSIONAL, SCIENTIFIC & TECHNICAL ACTIV','TECHNOLOGY & INNOVATION', 'TOURISM','TRADERS'])
        lnbase = st.selectbox('Base', ['INDIVIDUALS','LARGE CORPORATES',  'MICRO FINANCE', 'MIDDLE MARKET CORPORATES', 'SME'])
        sex = st.radio('Gender', ['M', 'F'])
        lnpayfreq = st.selectbox('Payment Frequency', ['2', '5', '6','12'])
        credit_card_used = st.radio('Used Credit Card', ['No','Yes'])
        debit_card_used = st.radio('Used Debit Card', ['No','Yes'])
        lnperiod_category = st.selectbox('Loan Period Category', ['Short-Term', 'Medium-Term', 'Long-Term'])
        region = st.selectbox("Select a Region", ["Colombo Region","Other","Greater Colombo Region","North Central Region","Uva / Sabaragamuwa Region","Southern Region","Central Region","Nothern Region","South Western Region","North Western Region","Eastern Region"])
        # Replaced sliders with text input
        lnamount = st.text_input('Loan Amount', value="1000")
        lninstamt = st.text_input('Installment Amount', value="100")
        average_sagbal = st.text_input('Average Savings Account Balance', value="0")
        age = st.text_input('Age', value="18")
        lnintrate = st.text_input('Interest Rate', value="0.1")
        
        submit_button1 = st.form_submit_button(label="Predict Default Risk")

    if  submit_button1:
        # Convert inputs to appropriate types
        user_input = pd.DataFrame({
        "LNAMOUNT": [float(lnamount)],
        
        "LNINSTAMT": [float(lninstamt)],
        
        "AVERAGE_SAGBAL": [float(average_sagbal)],
        "AGE": [int(age)],
        "LNINTRATE": [float(lnintrate)],
        "QSPURPOSEDES": [qspurposedes],
        "QS_SECTOR": [qsector],
        "LNBASELDESC": [lnbase],
        "LNPERIOD_CATEGORY": [lnperiod_category],
        "SEX": [sex],
        "LNPAYFREQ": [lnpayfreq],
        "CREDIT_CARD_USED": [credit_card_used],
        "DEBIT_CARD_USED": [debit_card_used],
        "LNRGNNAME" : [region]
    })
    
        # One-hot encoding to match training data
        user_input_dummies = pd.get_dummies(user_input, 
            columns=["QSPURPOSEDES", "QS_SECTOR", "LNBASELDESC", "SEX", "LNPAYFREQ", 
                "CREDIT_CARD_USED", "DEBIT_CARD_USED", "LNPERIOD_CATEGORY", "LNRGNNAME"], 
                drop_first=False
        )

        # Ensure all columns from training exist (Fill missing ones with 0)
        for col in housing_columns:
            if col not in user_input_dummies:
                user_input_dummies[col] = 0
    
        # Ensure order of columns matches training data
        user_input_dummies = user_input_dummies[housing_columns]
    
        # Scale numerical features
        num_features = ['LNAMOUNT', 'LNINSTAMT', 'AVERAGE_SAGBAL', 'AGE', 'LNINTRATE']

        user_input_scaled = housing_scaler.transform(user_input_dummies[num_features])
    
        user_input_scaled_df = pd.DataFrame(user_input_scaled, columns=num_features, index=user_input_dummies.index)
    
        user_input_final = user_input_dummies.drop(columns=num_features).reset_index(drop=True)

        # Add scaled numerical features
        user_input_final = pd.concat([user_input_scaled_df.reset_index(drop=True), user_input_final], axis=1)
    
        user_input_final = user_input_final[housing_columns]
    
        # Make prediction
        prediction = housing_model.predict(user_input_final)

        # Display result
        st.subheader("Prediction Result")
        if  prediction[0] == 1:
           st.error("The loan is at risk of default.")
        else:
           st.success("The loan is not at risk of default.")
        

