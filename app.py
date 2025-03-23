import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load('calorie_model.pkl')  
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🔥", layout="wide")

# Custom Styling with modern design elements
st.markdown("""
    <style>
        /* Global Page Background */
        body {
            background: linear-gradient(135deg, #f0f4ff, #ffffff);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Main Title */
        .main-title {
            font-size: 60px !important;
            font-weight: 800;
            text-align: center;
            color: #1d3557;  
            padding-bottom: 10px;
            margin-top: 20px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        }
        /* Subtitle */
        .sub-title {
            font-size: 30px;
            text-align: center;
            color: #457b9d; 
            margin-bottom: 40px;
        }
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e0f7fa, #ffffff) !important;
            padding: 20px;
            border-right: 2px solid #b2ebf2;
        }
        /* Buttons */
        .stButton>button {
            background-color: #1d3557; 
            color: white;
            font-size: 18px;
            padding: 12px 0;
            width: 100%;
            border-radius: 10px;
            border: none;
            transition: background 0.3s ease-in-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #457b9d;
        }
        /* Result Box */
        .result-box {
            border: 3px solid #1d3557;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            background: #f1faee;
            color: #1d3557;
            margin: 30px auto;
            width: 70%;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .result-box:hover {
            transform: scale(1.02);
        }
        /* Image Container */
        .image-container img {
            max-width: 100%;
            border-radius: 10px;
            border: 2px solid #1d3557;
            transition: transform 0.3s ease;
        }
        .image-container img:hover {
            transform: scale(1.05);
        }
        /* About Section */
        .about-section {
            font-size: 20px;
            color: #333;
            line-height: 1.8;
            text-align: justify;
            background: #ffffff;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        /* Section Headers */
        h2 {
            color: #1d3557;
            text-align: center;
            margin-top: 40px;
        }
        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 18px;
            color: #666;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown('<p class="main-title">🔥 Personal Fitness Tracker 🔥</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Track your calories burned with AI predictions!</p>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🔹 Enter Your Details")
gender = st.sidebar.radio("🚻 Select Gender", ["Female", "Male"], index=1)
age = st.sidebar.slider("🔵 Age", 10, 100, 25)
height = st.sidebar.slider("📏 Height (cm)", 100, 250, 170)
weight = st.sidebar.slider("⚖️ Weight (kg)", 30, 150, 70)
duration = st.sidebar.slider("⏳ Exercise Duration (mins)", 1, 120, 30)
heart_rate = st.sidebar.slider("❤️ Average Heart Rate", 60, 200, 90)
body_temp = st.sidebar.slider("🌡️ Body Temperature (°C)", 35.0, 42.0, 37.0)

# Convert gender to numerical
gender_num = 0 if gender == "Female" else 1

# Prepare input data
input_data = np.array([[gender_num, age, height, weight, duration, heart_rate, body_temp]])

# Ensure correct feature size before scaling
if input_data.shape[1] != scaler.n_features_in_:
    st.error(f"Feature mismatch: Expected {scaler.n_features_in_} features, got {input_data.shape[1]}.")
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
        <p><strong>Personal Fitness Tracker</strong> helps you estimate the calories burned based on your workout parameters. 
        Using an AI model trained on various exercise data, this tool predicts calorie expenditure from inputs such as age, gender, heart rate, and more.</p>
        <p><strong>How to Use:</strong></p>
        <ul>
            <li>Fill in your personal details in the sidebar.</li>
            <li>Adjust the exercise parameters.</li>
            <li>Your estimated calorie burn will be automatically updated.</li>
        </ul>
        <p><strong>Disclaimer:</strong> This tool is for informational purposes only and should not be used as a substitute for professional medical advice.</p>
    </div>
""", unsafe_allow_html=True)

# Workout Illustrations Section
st.markdown("---")
st.markdown("## 🏃 Workout Illustrations")

col1, col2 = st.columns(2)

with col1:
    st.image("https://unblast.com/wp-content/uploads/2022/04/Running-Illustration.jpg", 
             use_container_width=True, caption="Male Runner")
with col2:
    st.image("https://t4.ftcdn.net/jpg/02/28/85/81/360_F_228858108_bK3t2Dpw09mShxcPaaalRQrNnA2SHeEj.jpg", 
             use_container_width=True, caption="Male and Female Runner")

# Exercise Importance Section
st.markdown("""
    ### 🏋️‍♂️ Why Exercise is Essential
    - **Boosts Cardiovascular Health:** Regular exercise improves heart function and blood circulation.
    - **Strengthens Muscles & Bones:** Weight-bearing exercises increase muscle strength and bone density.
    - **Enhances Mental Health:** Physical activity helps reduce stress, anxiety, and depression.
    - **Maintains a Healthy Weight:** Consistent exercise supports a balanced metabolism and weight management.
    - **Increases Longevity:** Even moderate activity, like a 30-minute walk, can enhance overall quality of life.
""")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        Developed by <strong>Naveen Jayaraj</strong> | B.Tech CSE (AIML) | SRM Institute of Science and Technology
    </div>
""", unsafe_allow_html=True)
