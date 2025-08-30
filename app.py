import streamlit as st
import pickle
import numpy as np

# -----------------
# Load trained models
# -----------------
diabetes_model = pickle.load(open("models/diabetes_model.sav", "rb"))
heart_model = pickle.load(open("models/heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("models/parkinsons_model.sav", "rb"))

# -----------------
# Streamlit App
# -----------------
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

st.title("🧑‍⚕️ Multiple Disease Prediction System")

# Sidebar navigation
st.sidebar.title("Navigation")
options = ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"]
choice = st.sidebar.radio("Go to", options)

# -----------------
# Home Page
# -----------------
if choice == "Home":
    st.write("""
    ## Welcome to the Health Assistant!  
    This app predicts the likelihood of:  
    - **Diabetes**  
    - **Heart Disease**  
    - **Parkinson's Disease**  

    👉 Choose a prediction tool from the sidebar to get started.
    """)

# -----------------
# Diabetes Prediction
# -----------------
elif choice == "Diabetes Prediction":
    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.2f")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

    if st.button("Predict Diabetes"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ The model predicts that this person is **diabetic**.")
        else:
            st.success("✅ The model predicts that this person is **not diabetic**.")

# -----------------
# Heart Disease Prediction
# -----------------
elif choice == "Heart Disease Prediction":
    st.header("Heart Disease Prediction")

    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    sex = st.number_input("Sex (1 = Male, 0 = Female)", min_value=0, max_value=1, step=1)
    cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3, step=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", min_value=0, max_value=1, step=1)
    restecg = st.number_input("Resting ECG (0–2)", min_value=0, max_value=2, step=1)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0)
    exang = st.number_input("Exercise Induced Angina (1 = Yes, 0 = No)", min_value=0, max_value=1, step=1)
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, format="%.1f")
    slope = st.number_input("Slope of ST Segment (0–2)", min_value=0, max_value=2, step=1)
    ca = st.number_input("Number of Major Vessels (0–4)", min_value=0, max_value=4, step=1)
    thal = st.number_input("Thalassemia (0 = Normal, 1 = Fixed defect, 2 = Reversible defect)", min_value=0, max_value=2, step=1)

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                                exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ The model predicts that this person **has heart disease**.")
        else:
            st.success("✅ The model predicts that this person **does not have heart disease**.")

# -----------------
# Parkinson’s Prediction
# -----------------
elif choice == "Parkinson's Prediction":
    st.header("Parkinson's Disease Prediction")

    st.write("Enter the required voice measurement values:")

    # Split into multiple columns for better UI
    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0)
        flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0)
        rap = st.number_input("MDVP:RAP", min_value=0.0)

    with col2:
        ppq = st.number_input("MDVP:PPQ", min_value=0.0)
        ddp = st.number_input("Jitter:DDP", min_value=0.0)
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0)
        apq3 = st.number_input("Shimmer:APQ3", min_value=0.0)
        apq5 = st.number_input("Shimmer:APQ5", min_value=0.0)

    with col3:
        apq = st.number_input("MDVP:APQ", min_value=0.0)
        dda = st.number_input("Shimmer:DDA", min_value=0.0)
        nhr = st.number_input("NHR", min_value=0.0)
        hnr = st.number_input("HNR", min_value=0.0)
        rpde = st.number_input("RPDE", min_value=0.0)
        dfa = st.number_input("DFA", min_value=0.0)
        spread1 = st.number_input("spread1", min_value=0.0)
        spread2 = st.number_input("spread2", min_value=0.0)
        d2 = st.number_input("D2", min_value=0.0)
        ppe = st.number_input("PPE", min_value=0.0)

    if st.button("Predict Parkinson's"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, jitter_abs,
                                rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5,
                                apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        prediction = parkinsons_model.predict(input_data)
        if prediction[0] == 1:
            st.error("⚠️ The model predicts that this person **has Parkinson's disease**.")
        else:
            st.success("✅ The model predicts that this person **does not have Parkinson's disease**.")
