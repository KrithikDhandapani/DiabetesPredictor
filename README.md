# Diabetes Prediction App

A **machine learning-powered web app** built with **Python, Scikit-learn, and Streamlit** to predict whether a patient is diabetic based on medical input data.  
This project demonstrates end-to-end development — from **data preprocessing and model training** to **interactive deployment with a clean UI**.

---

## Project Overview

Diabetes is one of the most common chronic conditions worldwide, and early prediction can help with timely lifestyle changes or treatment.  
This project applies **supervised machine learning (Support Vector Machine)** to predict diabetes using the **Pima Indians Diabetes Dataset**.

The project was designed as a self-learning exercise to practice:

- **Data preprocessing** (scaling, splitting, cleaning)  
- **Model selection & training** (SVM classifier with linear kernel)  
- **Performance evaluation** (train/test accuracy tracking)  
- **Deployment** using Streamlit for an intuitive web-based interface  

---

## Features

- **Interactive Web App** – Enter patient medical data and get instant predictions  
- **Model Accuracy Tracking** – Displays both training and test accuracy  
- **Data Insights Dashboard** – Explore dataset distribution and feature histograms  
- **Streamlit-powered UI** – Lightweight, fast, and easy to use  

---

## Dataset

**Source:** Pima Indians Diabetes Database (UCI Repository / Kaggle)  

**Features:**  
- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function (genetic risk factor)  
- Age  

**Target:**  
- **Outcome** (0 = Not Diabetic, 1 = Diabetic)  

---

## Machine Learning Pipeline

### Preprocessing
- StandardScaler applied for feature normalization  
- Train-test split (80/20, stratified by outcome)  

### Model
- Support Vector Machine (SVM) with linear kernel  
- Chosen for high performance on binary classification tasks  

### Performance
- Training Accuracy: ~78–80%  
- Testing Accuracy: ~75–78%  

---

## Demo Preview

**Web App UI**  
- **Sidebar**: Dataset preview, outcome distribution, feature histograms  
- **Main Panel**:  
  - Model accuracy metrics  
  - Input form for patient data  
  - Prediction result with *"Diabetic"* or *"Not Diabetic"*  
 

---

## Why This Project is Useful

- **Practical ML Application** – Shows how machine learning can support healthcare predictions  
- **End-to-End Development** – From raw data to live, interactive app  
- **Clarity and simplicity** – Demonstrates data science, machine learning, and deployment skills in one project  

---

## Tech Stack

- **Python** (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)  
- **Streamlit** (for deployment and interactivity)  
- **Machine Learning**: Support Vector Machine (SVM)  

---

## Future Improvements

- Experiment with additional models (Random Forest, Logistic Regression, XGBoost)  
- Add cross-validation for more robust accuracy estimates  
- Deploy online via Streamlit Cloud, Heroku, or AWS for public access  
- Implement feature importance visualization for better interpretability  

---

**This project highlights my ability to take a dataset, apply machine learning techniques, evaluate performance, and deploy a working app that makes predictions — a full-stack data science workflow.**
