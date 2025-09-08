import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# -------------------------------
# Load and prepare the dataset
# -------------------------------


@st.cache_data
def load_data():
    diabetesDataset = pd.read_csv("diabetes.csv")
    X = diabetesDataset.drop(columns="Outcome", axis=1)
    Y = diabetesDataset["Outcome"]

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return diabetesDataset, X, Y, scaler


diabetesDataset, X, Y, scaler = load_data()

# -------------------------------
# Train the model
# -------------------------------


@st.cache_resource
def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    classifier = svm.SVC(kernel="linear")
    classifier.fit(X_train, Y_train)

    train_acc = accuracy_score(classifier.predict(X_train), Y_train)
    test_acc = accuracy_score(classifier.predict(X_test), Y_test)

    return classifier, train_acc, test_acc


classifier, train_acc, test_acc = train_model(X, Y)

# -------------------------------
# Sidebar - dataset insights
# -------------------------------
st.sidebar.header("📊 Dataset Insights")

# Show dataset preview
if st.sidebar.checkbox("Show raw data"):
    st.sidebar.write(diabetesDataset.head())

# Outcome distribution
st.sidebar.subheader("Outcome Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Outcome", data=diabetesDataset, ax=ax)
st.sidebar.pyplot(fig)

# Feature histogram
feature = st.sidebar.selectbox(
    "Feature distribution", diabetesDataset.columns[:-1])
fig2, ax2 = plt.subplots()
sns.histplot(diabetesDataset[feature], bins=20, kde=True, ax=ax2)
st.sidebar.pyplot(fig2)

# -------------------------------
# Main app UI
# -------------------------------
st.title("🩺 Diabetes Prediction App")
st.write("This app predicts whether a person is diabetic based on medical input data.")

# Show accuracy
st.info(f"📈 Model Training Accuracy: **{train_acc:.2f}**")
st.info(f"🧪 Model Testing Accuracy: **{test_acc:.2f}**")

# Collect user input
st.subheader("Enter Patient Data")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, step=1)
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0, max_value=100, step=1)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function",
                          min_value=0.0, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Prediction button
if st.button("Predict"):
    input_data = (pregnancies, glucose, blood_pressure,
                  skin_thickness, insulin, bmi, dpf, age)
    input_data_as_array = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_as_array)
    prediction = classifier.predict(std_data)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"✅ Prediction: **{result}**")
