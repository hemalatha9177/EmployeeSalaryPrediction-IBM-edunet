
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# Load the trained model pipeline (with preprocessing)
model = joblib.load("best_model.pkl")

# Set page config
st.set_page_config(page_title="Employee Salary Prediction", page_icon="💵", layout="centered")

# Title
st.title("💵 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# 1. Age
age = st.sidebar.slider("Age", 18, 90, 30)

# 2. Workclass
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
])

# 3. fnlwgt
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=100000)

# 4. Educational-num
educational_num = st.sidebar.slider("Education Level (numeric)", 1, 16, 10)

# 5. Marital Status
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])

# 6. Occupation
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])

# 7. Relationship
relationship = st.sidebar.selectbox("Relationship", [
    "Husband", "Wife", "Not-in-family", "Own-child", "Unmarried", "Other-relative"
])

# 8. Race
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])

# 9. Gender
gender = st.sidebar.radio("Gender", ["Male", "Female"])

# 10. Capital Gain
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=99999, value=0)

# 11. Capital Loss
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=99999, value=0)

# 12. Hours per Week
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# 13. Native Country
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany", "Canada", "India", "England",
    "Cuba", "China", "South", "Iran", "Italy", "Poland", "Jamaica", "Vietnam",
    "Japan", "France", "Columbia", "Cambodia", "Thailand", "Laos", "Taiwan", "Haiti",
    "Portugal", "Dominican-Republic", "El-Salvador", "Guatemala", "Greece", "Yugoslavia",
    "Peru", "Hong", "Ireland", "Trinadad&Tobago", "Honduras", "Outlying-US(Guam-USVI-etc)", "Scotland", "Ecuador", "Nicaragua", "Hungary", "Holand-Netherlands"
])

# Collect input into a DataFrame
input_data = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}


input_df = pd.DataFrame([input_data])
st.write("###  Input Data")
st.write(input_df)


# Prediction
if st.button("Predict Salary class"):
    prediction = model.predict(input_df)[0]
    result = ">50K" if prediction == 1 else "≤50K"
    st.success(f" Predicted Salary Category: **{result}**")


# Batch prediction section
st.markdown("---")
st.markdown("### Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write(" Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data["PredictedClass"] = batch_preds
    st.write("Predictions:")
    st.write(batch_data.head())

    # Downloadable CSV
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name="predicted_classes.csv", mime="text/csv")
