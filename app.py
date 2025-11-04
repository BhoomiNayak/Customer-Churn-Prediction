import streamlit as st
import pandas as pd
import joblib

# === Load Model ===
MODEL_PATH = "cat_churn_pipeline.joblib"
pipeline = joblib.load(MODEL_PATH)

# === Page Config ===
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📞", layout="centered")

# === Custom Styling ===
st.markdown("""
    <style>
    .main {
        background-color: #f9fafc;
        padding: 20px;
        border-radius: 12px;
    }
    h1 {
        color: #0077B6;
        text-align: center;
        font-weight: 700;
    }
    h3 {
        color: #023E8A;
        margin-top: 20px;
    }
    .stButton button {
        background-color: #0077B6;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 1em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #023E8A;
        color: #fff;
    }
    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# === Title Section ===
st.title("📞 Customer Churn Prediction ")
st.markdown("<p style='text-align:center; color:gray;'>Predict customer churn probability using an AI-powered CatBoost model.</p>", unsafe_allow_html=True)

# === Sidebar Info ===
st.sidebar.title("⚙️ About This App")
st.sidebar.info("""
This app predicts whether a telecom customer is likely to churn 
based on their profile, contract, and billing details.
""")
st.sidebar.markdown("**Model:** CatBoost Classifier")
st.sidebar.markdown("**Version:** 1.0.0")

# === Input Fields ===
st.header("🧍‍♂️ Customer Information")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, value=12)

with col2:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])

st.divider()

st.header("💻 Services & Billing Details")

col3, col4 = st.columns(2)
with col3:
    device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

with col4:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

st.divider()

# === Prediction Button ===
if st.button("🔍 Predict Churn"):
    # Prepare input
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

    # Predict
    proba = pipeline.predict_proba(input_data)[0, 1]
    pred = "Churn" if proba >= 0.5 else "Not Churn"

    # === Display Results ===
    st.markdown("### 📊 Prediction Result")
    st.metric(label="Churn Probability", value=f"{proba*100:.2f}%")

    if pred == "Not Churn":
        st.success("✅ Customer is likely to stay.")
        st.balloons()
    else:
        st.error("⚠️ Customer is likely to churn.")

    if proba >= 0.6:
        st.warning("💡 High churn risk — consider retention offers or feedback calls.")

    

# === Footer ===
st.markdown("<br><hr><p style='text-align:center; color:gray;'>Mini Project done with Love</p>", unsafe_allow_html=True)
