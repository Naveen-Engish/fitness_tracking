import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('calorie_model.pkl')  # Ensure correct model file name
scaler = joblib.load('scaler.pkl')

# Set Streamlit page config
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🔥", layout="wide")

# Custom styling with a clean and massive title
st.markdown("""
    <style>
        .main-title {
            font-size: 50px !important;  /* Massive font size */
            font-weight: bold;
            text-align: center;
            color: #006ba6;  /* Dark blue */
            margin-bottom: 20px;
        }
        .sub-title {
            font-size: 50px;  /* Larger subtitle */
            color: #0496ff;  /* Bright blue */
            text-align: center;
            margin-bottom: 40px;
        }
        .stButton>button {
            background-color: #0496ff;  /* Bright blue */
            color: white;
            font-size: 16px;
            padding: 10px;
            width: 100%;
            border-radius: 10px;
            border: none;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #006ba6;  /* Dark blue on hover */
        }
        .result-box {
            border: 2px solid #0496ff;  /* Bright blue */
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            background-color: #f0f8ff;  /* Light blue background */
            color: #006ba6;  /* Dark blue text */
            margin: 20px auto;
            width: 50%;
        }
        .about-section {
            font-size: 18px;
            color: #006ba6;  /* Dark blue */
            line-height: 1.6;
            margin-top: 40px;
        }
        .image-container {
            display: flex;
            justify-content: center;
            margin-top: 40px;
        }
        .image-container img {
            max-width: 100%;
            border-radius: 10px;
            border: 2px solid #0496ff;  /* Bright blue border */
        }
        .sidebar .sidebar-content {
            background-color: #f0f8ff;  /* Light blue background for sidebar */
        }
        .sidebar .stRadio>div>div {
            color: #006ba6;  /* Dark blue text for radio buttons */
        }
        .sidebar .stSlider>div>div>div {
            background-color: #0496ff;  /* Bright blue slider */
        }
        .about-me {
            font-size: 20px;
            color: #006ba6;  /* Dark blue */
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background-color: #f0f8ff;  /* Light blue background */
            border-radius: 10px;
            border: 2px solid #0496ff;  /* Bright blue border */
        }
    </style>
""", unsafe_allow_html=True)

# App title with a massive and clean design
st.markdown('<p class="main-title">🔥 Personal Fitness Tracker 🔥</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Estimate the calories burned based on your workout session</p>', unsafe_allow_html=True)

# Layout - Inputs in the main area
st.sidebar.header("🔹 User Input Parameters")
gender = st.sidebar.radio("Select Gender", ["Female", "Male"], index=1)
age = st.sidebar.slider("Age", 10, 100, 25)
height = st.sidebar.slider("Height (cm)", 100, 250, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
duration = st.sidebar.slider("Exercise Duration (mins)", 1, 120, 30)
heart_rate = st.sidebar.slider("Average Heart Rate", 60, 200, 90)
body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)

# Convert gender to numerical (0 for Female, 1 for Male)
gender_num = 0 if gender == "Female" else 1

# Prepare input array with correct shape (7 features)
input_data = np.array([[gender_num, age, height, weight, duration, heart_rate, body_temp]])

# Ensure correct feature shape
expected_features = scaler.n_features_in_
if input_data.shape[1] != expected_features:
    st.error(f"Feature mismatch: Expected {expected_features} features, but got {input_data.shape[1]}. Check feature selection.")
else:
    # Scale the input data using the saved scaler
    scaled_input = scaler.transform(input_data)

    # Make prediction using the loaded model
    predicted_calories = model.predict(scaled_input)

    # Show results with a styled box (centered)
    st.markdown('<div class="result-box">📊 Estimated Calories Burned: {:.2f} calories</div>'.format(predicted_calories[0]), unsafe_allow_html=True)

# About section
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
    <div class="about-section">
        <p>
            This <strong>Personal Fitness Tracker</strong> app helps you estimate the calories burned during your workout sessions. 
            Simply input your details, such as age, height, weight, exercise duration, heart rate, and body temperature, 
            and the app will calculate the estimated calories burned using a machine learning model.
        </p>
        <p>
            <strong>How to Use:</strong>
            <ul>
                <li>Fill in your details in the sidebar.</li>
                <li>Click anywhere outside the sidebar to see the estimated calories burned.</li>
                <li>The result will be displayed in the center of the screen.</li>
            </ul>
        </p>
        <p>
            <strong>Note:</strong> This app is for educational purposes and provides estimates based on a pre-trained model. 
            For accurate fitness tracking, consult a professional.
        </p>
    </div>
""", unsafe_allow_html=True)

# Images at the bottom
st.markdown("---")
st.markdown("### Workout Illustrations")
col1, col2 = st.columns(2)
with col1:
    st.image("https://img.freepik.com/free-vector/man-running-concept-illustration_114360-1836.jpg", use_column_width=True, caption="Male Runner")
with col2:
    st.image("https://img.freepik.com/free-vector/fitness-woman-running-illustration_23-2148998519.jpg", use_column_width=True, caption="Female Runner")

# About Me section
st.markdown("---")
st.markdown("### About Me")
st.markdown("""
    <div class="about-me">
        <p><strong>Name:</strong> Naveen Jayaraj</p>
        <p><strong>Institution:</strong> SRM Institute of Science and Technology</p>
        <p><strong>Year/Semester:</strong> 2nd Year, 4th Semester</p>
    </div>
""", unsafe_allow_html=True)