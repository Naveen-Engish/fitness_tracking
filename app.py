import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('calorie_model.pkl')  
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🔥", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        /* Main Title */
        .main-title {
            font-size: 60px !important;
            font-weight: bold;
            text-align: center;
            color: #333;  
            padding-bottom: 10px;
        }
        /* Subtitle */
        .sub-title {
            font-size: 30px;
            text-align: center;
            color: #666; 
            margin-bottom: 40px;
        }
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: #f4f4f4; 
            padding: 20px;
            border-radius: 10px;
        }
        /* Button */
        .stButton>button {
            background-color: #008CBA; 
            color: white;
            font-size: 18px;
            padding: 12px;
            width: 100%;
            border-radius: 8px;
            border: none;
            transition: background 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #005f73;
        }
        /* Result Box */
        .result-box {
            border: 3px solid #008CBA;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            background-color: #f0f8ff;
            color: #005f73;
            margin: 30px auto;
            width: 60%;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        /* Image Container */
        .image-container {
            display: flex;
            justify-content: center;
            margin-top: 30px;
        }
        .image-container img {
            max-width: 100%;
            border-radius: 10px;
            border: 2px solid #008CBA;
        }
        /* About Section */
        .about-section {
            font-size: 20px;
            color: #333;
            line-height: 1.8;
            text-align: justify;
            background-color: #f9f9f9;
            padding: 25px;
            border-radius: 10px;
        }
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 18px;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-title">🔥 Personal Fitness Tracker 🔥</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Track your calories burned with AI predictions!</p>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🔹 Enter Your Details")
gender = st.sidebar.radio("Select Gender", ["Female", "Male"], index=1)
age = st.sidebar.slider("Age", 10, 100, 25)
height = st.sidebar.slider("Height (cm)", 100, 250, 170)
weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
duration = st.sidebar.slider("Exercise Duration (mins)", 1, 120, 30)
heart_rate = st.sidebar.slider("Average Heart Rate", 60, 200, 90)
body_temp = st.sidebar.slider("Body Temperature (°C)", 35.0, 42.0, 37.0)

# Convert gender to numerical
gender_num = 0 if gender == "Female" else 1

# Prepare input data
input_data = np.array([[gender_num, age, height, weight, duration, heart_rate, body_temp]])

# Feature Check
expected_features = scaler.n_features_in_
if input_data.shape[1] != expected_features:
    st.error(f"Feature mismatch: Expected {expected_features} features, got {input_data.shape[1]}.")
else:
    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Prediction
    predicted_calories = model.predict(scaled_input)

    # Display Result
    st.markdown('<div class="result-box">🔥 Estimated Calories Burned: {:.2f} kcal</div>'.format(predicted_calories[0]), unsafe_allow_html=True)

# About Section
st.markdown("---")
st.markdown("## ℹ️ About This App")
st.markdown("""
    <div class="about-section">
        <p><strong>Personal Fitness Tracker</strong> helps you estimate the calories burned based on workout parameters. 
        The AI model predicts calorie expenditure based on age, gender, heart rate, and other factors.</p>
        <p><strong>How to Use:</strong></p>
        <ul>
            <li>Enter your details in the sidebar.</li>
            <li>Click outside the sidebar to process the data.</li>
            <li>Your estimated calorie burn will be displayed instantly.</li>
        </ul>
        <p><strong>Disclaimer:</strong> This tool is for informational purposes and not a medical substitute.</p>
    </div>
""", unsafe_allow_html=True)

# Workout Illustrations
st.markdown("---")
st.markdown("## 🏃 Workout Illustrations")

col1, col2 = st.columns(2)

with col1:
    st.image("https://unblast.com/wp-content/uploads/2022/04/Running-Illustration.jpg", 
             use_container_width=True, caption="Male Runner")
with col2:
    st.image("https://t4.ftcdn.net/jpg/02/28/85/81/360_F_228858108_bK3t2Dpw09mShxcPaaalRQrNnA2SHeEj.jpg", 
             use_container_width=True, caption="Male and Female Runner")

# Importance of Running and Exercise
st.markdown("""
    ### 🏋️‍♂️ Why is Running and Exercise Important?
    - Running and exercise help improve cardiovascular health, strengthen muscles, and boost overall stamina.
    - Regular physical activity reduces the risk of chronic diseases such as diabetes, obesity, and heart conditions.
    - Exercise enhances mental well-being by reducing stress, anxiety, and depression while improving sleep quality.
    - It helps maintain a healthy weight, increases endurance, and keeps the body agile and active.
    - A daily workout routine, even for 30 minutes, can significantly improve longevity and overall quality of life.
""")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        Developed by <strong>Naveen Jayaraj</strong> | B.Tech CSE (AIML) | SRM Institute of Science and Technology
    </div>
""", unsafe_allow_html=True)
