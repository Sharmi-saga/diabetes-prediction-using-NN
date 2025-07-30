import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Removed custom CSS styling function as requested.
# The app will now rely on Streamlit's default styling.

# Load model and scaler
# @st.cache_resource is used to cache heavy resources like ML models
# This prevents the model from reloading on every rerun of the app.
@st.cache_resource
def load_model():
    try:
        # Load the Keras deep learning model
        # model = tf.keras.models.load_model('model/diabetes_model.h5')
        model = tf.keras.models.load_model('diabetes_model.h5')
        # Load the scikit-learn scaler object for data preprocessing
        # scaler = joblib.load('model/scaler.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except Exception as e:
        # Display an error message if model or scaler files cannot be loaded
        st.error(f"Error loading model or scaler: {e}")
        return None, None, False

def main():
    # Removed the call to set_custom_style()
    
    # Load the pre-trained model and scaler
    model, scaler, model_loaded = load_model()
    
    # Removed the custom main-container div. Streamlit's default layout handles this.
    
    # Display the main title and subtitle of the application using Streamlit components
    st.title("🩺 Diabetes Prediction System")
    st.subheader("Enter your health parameters to get a diabetes risk prediction")
    
    # If the model failed to load, display an error and exit
    if not model_loaded:
        st.error("❌ Model could not be loaded. Please check if the model files exist in the 'model/' directory.")
        return # Removed closing div as main-container is no longer used
    
    # Create two columns for organizing input fields
    col1, col2 = st.columns(2)
    
    # Input fields for basic information (left column)
    with col1:
        # Removed custom input-container div
        st.subheader("📊 Basic Information")
        pregnancies = st.number_input(
            "Number of Pregnancies", 
            min_value=0, 
            max_value=20, 
            value=0,
            help="Number of times pregnant"
        )
        
        glucose = st.number_input(
            "Glucose Level (mg/dL)", 
            min_value=0.0, 
            max_value=300.0, 
            value=120.0,
            help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test"
        )
        
        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)", 
            min_value=0.0, 
            max_value=200.0, 
            value=80.0,
            help="Diastolic blood pressure (mm Hg)"
        )
        
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", 
            min_value=0.0, 
            max_value=100.0, 
            value=20.0,
            help="Triceps skin fold thickness (mm)"
        )
        # Removed closing input-container div
    
    # Input fields for medical parameters (right column)
    with col2:
        # Removed custom input-container div
        st.subheader("🔬 Medical Parameters")
        insulin = st.number_input(
            "Insulin Level (μU/mL)", 
            min_value=0.0, 
            max_value=1000.0, 
            value=80.0,
            help="2-Hour serum insulin (mu U/ml)"
        )
        
        bmi = st.number_input(
            "BMI (Body Mass Index)", 
            min_value=0.0, 
            max_value=70.0, 
            value=25.0,
            help="Body mass index (weight in kg/(height in m)^2)"
        )
        
        diabetes_pedigree_function = st.number_input(
            "Diabetes Pedigree Function", 
            min_value=0.0, 
            max_value=3.0, 
            value=0.5,
            step=0.001, # Allows for more precise input
            format="%.3f", # Formats the display to 3 decimal places
            help="Diabetes pedigree function (a genetic likelihood of diabetes based on family history)"
        )
        
        age = st.number_input(
            "Age (years)", 
            min_value=0, 
            max_value=120, 
            value=25,
            help="Age in years"
        )
        # Removed closing input-container div
    
    # Separator before the prediction button
    st.markdown("---")
    
    # Center the prediction button using columns
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Diabetes Risk", key="predict")
    
    # Logic to perform prediction when the button is clicked
    if predict_button:
        try:
            # Prepare the input data as a NumPy array for the model
            input_data = np.array([[
                pregnancies, glucose, blood_pressure, skin_thickness, 
                insulin, bmi, diabetes_pedigree_function, age
            ]])
            
            # Scale the input data using the loaded scaler
            input_data_scaled = scaler.transform(input_data)
            
            # Make a prediction using the loaded Keras model
            prediction = model.predict(input_data_scaled)
            # Extract the probability from the prediction output
            prediction_proba = prediction[0][0]
            
            # Display the prediction result
            st.markdown("---")
            st.subheader("📋 Prediction Result") # Changed to subheader for consistency
            
            # Determine if the prediction indicates diabetes or not, using Streamlit's built-in message components
            if prediction_proba > 0.5:
                result_text = "Oops..! You have diabetes!"
                confidence = prediction_proba * 100
                st.error(f"⚠️ {result_text}\n\nConfidence: {confidence:.1f}%") # Using st.error for positive result
                
                st.markdown("""
                #### 🏥 Recommendations:
                - Consult with a healthcare provider immediately
                - Monitor blood glucose levels regularly
                - Follow a healthy diet and exercise routine
                - Consider medication if prescribed by your doctor
                """)
                
            else:
                result_text = "Don't worry! You do not have diabetes."
                confidence = (1 - prediction_proba) * 100 # Confidence for negative prediction
                st.success(f"✅ {result_text}\n\nConfidence: {confidence:.1f}%") # Using st.success for negative result
                
                st.markdown("""
                #### 🌟 Keep it up:
                - Maintain a healthy lifestyle
                - Regular exercise and balanced diet
                - Periodic health check-ups
                - Monitor your health parameters
                """)
            
            # Allow users to view a summary of their input data
            with st.expander("📊 View Input Summary"):
                col_summary1, col_summary2 = st.columns(2)
                with col_summary1:
                    st.write(f"**Pregnancies:** {pregnancies}")
                    st.write(f"**Glucose:** {glucose} mg/dL")
                    st.write(f"**Blood Pressure:** {blood_pressure} mmHg")
                    st.write(f"**Skin Thickness:** {skin_thickness} mm")
                
                with col_summary2:
                    st.write(f"**Insulin:** {insulin} μU/mL")
                    st.write(f"**BMI:** {bmi}")
                    st.write(f"**Diabetes Pedigree:** {diabetes_pedigree_function}")
                    st.write(f"**Age:** {age} years")
                    
                st.write(f"**Raw Prediction Score:** {prediction_proba:.4f}")
            
        except Exception as e:
            # Catch any errors during the prediction process
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("Please check your input values and try again.")
    
    # Footer with a disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <small>
        ⚠️ <strong>Disclaimer:</strong> This is a machine learning prediction tool for educational purposes only. 
        Always consult with healthcare professionals for medical advice.
        </small>
    </div>
    """, unsafe_allow_html=True)
    
    # Removed closing main-container div.

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
