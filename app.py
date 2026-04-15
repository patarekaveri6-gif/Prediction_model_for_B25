import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 20px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        background-color: #45a049;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie_student = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_kd54vced.json")

def load_model():
    # Loading the SVM model from your uploaded file
    with open('model (2).pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# --- HEADER ---
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Academic Success Predictor")
        st.write("Input student metrics below to predict the final grade category using our trained SVM model.")
    with col2:
        st_lottie(lottie_student, height=150)

st.divider()

# --- INPUT FORM ---
st.header("📊 Student Data Entry")

with st.expander("Enter Metrics", expanded=True):
    c1, c2 = st.columns(2)
    
    with c1:
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        attendance_rate = st.slider("Attendance Rate (%)", 0, 100, 85)
        study_hours = st.number_input("Study Hours Per Week", 0, 100, 15)
        prev_grade = st.slider("Previous Grade", 0, 100, 75)
        extra_curr = st.selectbox("Extracurricular Activities", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
    with c2:
        parental_support = st.selectbox("Parental Support Level", options=[0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])
        final_grade_ref = st.number_input("Last Midterm Grade", 0, 100, 70)
        study_hours_alt = st.number_input("Daily Study Hours", 0.0, 24.0, 3.5)
        attendance_alt = st.number_input("Attendance (Alternative Metric)", 0, 100, 90)

# --- PREDICTION ---
if st.button("🚀 Predict Final Outcome"):
    # The model expects 9 features based on your .pkl metadata
    features = np.array([[gender, attendance_rate, study_hours, prev_grade, 
                          extra_curr, parental_support, final_grade_ref, 
                          study_hours_alt, attendance_alt]])
    
    with st.spinner('Calculating...'):
        prediction = model.predict(features)
        
    st.markdown("---")
    st.balloons()
    
    # Display Result in an attractive way
    st.markdown(f"""
        <div class="result-card">
            <h3>Prediction Result</h3>
            <h1 style="color: #4CAF50;">Class {prediction[0]}</h1>
        </div>
    """, unsafe_allow_html=True)

st.sidebar.info("This app uses a Support Vector Machine (SVM) model to classify student performance.")
