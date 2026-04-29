import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Label encoding maps (matching sklearn LabelEncoder alphabetical order)
ENCODINGS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "default": ["no", "yes"],
    "housing": ["no", "yes"],
    "loan": ["no", "yes"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["apr", "aug", "dec", "feb", "jan", "jul", "jun",
              "mar", "may", "nov", "oct", "sep"],
    "poutcome": ["failure", "other", "success", "unknown"],
}

def encode(val, key):
    return ENCODINGS[key].index(val)

st.set_page_config(page_title="Bank Term Deposit Predictor", page_icon="🏦", layout="wide")
st.title("🏦 Bank Term Deposit Subscription Predictor")
st.markdown("Predict whether a customer will subscribe to a term deposit based on campaign and profile data.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Profile")
    age = st.slider("Age", 18, 95, 35)
    job = st.selectbox("Job", ENCODINGS["job"])
    marital = st.selectbox("Marital Status", ENCODINGS["marital"])
    education = st.selectbox("Education", ENCODINGS["education"])
    default = st.selectbox("Has Credit Default?", ENCODINGS["default"])
    balance = st.number_input("Average Annual Balance (€)", -10000, 100000, 1000)

with col2:
    st.subheader("Loan & Contact")
    housing = st.selectbox("Has Housing Loan?", ENCODINGS["housing"])
    loan = st.selectbox("Has Personal Loan?", ENCODINGS["loan"])
    contact = st.selectbox("Contact Type", ENCODINGS["contact"])
    month = st.selectbox("Last Contact Month", ENCODINGS["month"])
    day = st.slider("Last Contact Day of Month", 1, 31, 15)
    duration = st.slider("Last Contact Duration (seconds)", 0, 5000, 300)

with col3:
    st.subheader("Campaign Data")
    campaign = st.slider("Contacts This Campaign", 1, 50, 2)
    pdays = st.slider("Days Since Last Contact (-1 = never)", -1, 999, -1)
    previous = st.slider("Previous Campaign Contacts", 0, 50, 0)
    poutcome = st.selectbox("Previous Campaign Outcome", ENCODINGS["poutcome"])

# Build feature vector
features = pd.DataFrame([{
    "age": age,
    "job": encode(job, "job"),
    "marital": encode(marital, "marital"),
    "education": encode(education, "education"),
    "default": encode(default, "default"),
    "balance": balance,
    "housing": encode(housing, "housing"),
    "loan": encode(loan, "loan"),
    "contact": encode(contact, "contact"),
    "day": day,
    "month": encode(month, "month"),
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": encode(poutcome, "poutcome"),
}])

st.divider()

if st.button("🔍 Predict", type="primary", use_container_width=True):
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if pred == 1:
            st.success("✅ **LIKELY TO SUBSCRIBE** to term deposit")
        else:
            st.error("❌ **UNLIKELY TO SUBSCRIBE** to term deposit")
    with col_r2:
        st.metric("Subscription Probability", f"{prob[1]*100:.1f}%")
        st.metric("Non-Subscription Probability", f"{prob[0]*100:.1f}%")

st.divider()
st.caption("Built with XGBoost · Hussain Murtaza Ali · Bank Marketing Analytics Project")
