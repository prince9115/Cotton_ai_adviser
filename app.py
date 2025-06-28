import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import cohere
import base64
import io
import json
import os

# Page configuration
st.set_page_config(
    page_title="Cotton Disease AI Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disease information database
DISEASE_INFO = {
    "Bacterial Blight": {
        "description": "Bacterial infection causing water-soaked lesions",
        "severity": "High",
        "symptoms": "Angular lesions, yellow halos, leaf drop",
        "causes": "Xanthomonas citri bacteria, high humidity, wounded plants"
    },
    "Curl Virus": {
        "description": "Viral disease causing leaf curling and stunting",
        "severity": "Very High",
        "symptoms": "Upward leaf curling, yellowing, reduced plant size",
        "causes": "Whitefly transmission, infected plant material"
    },
    "Healthy Leaf": {
        "description": "Normal, healthy cotton leaf",
        "severity": "None",
        "symptoms": "Green color, normal shape and size",
        "causes": "Good agricultural practices"
    },
    "Herbicide Growth Damage": {
        "description": "Chemical damage from herbicide exposure",
        "severity": "Medium",
        "symptoms": "Distorted growth, chlorosis, stunted development",
        "causes": "Herbicide drift, over-application, wrong timing"
    },
    "Leaf Hopper Jassids": {
        "description": "Insect pest damage causing leaf damage",
        "severity": "Medium",
        "symptoms": "Yellowing, stippling, reduced vigor",
        "causes": "Leaf hopper insects, dry conditions"
    },
    "Leaf Redding": {
        "description": "Physiological disorder causing reddish discoloration",
        "severity": "Low",
        "symptoms": "Red/purple leaf margins, premature aging",
        "causes": "Potassium deficiency, water stress, temperature fluctuations"
    },
    "Leaf Variegation": {
        "description": "Genetic or viral-induced color variation",
        "severity": "Low to Medium",
        "symptoms": "Irregular color patterns, mosaic appearance",
        "causes": "Genetic factors, viral infections, nutrient imbalances"
    }
}

class CottonDiseaseAI:
    def __init__(self):
        self.model = None
        self.groq_client = None
        self.cohere_client = None
        self.class_names = [
            "Bacterial Blight", "Curl Virus", "Healthy Leaf",
            "Herbicide Growth Damage", "Leaf Hopper Jassids",
            "Leaf Redding", "Leaf Variegation"
        ]
        # Load pre-installed model
        self.model = self.load_model()
    
    @st.cache_resource
    def load_model(_self):
        """Load the pre-installed cotton disease model"""
        model_path = "cotton_leaf_cnn_best_model.keras"
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    def setup_apis(self, groq_api_key, cohere_api_key):
        """Initialize API clients"""
        try:
            if groq_api_key:
                self.groq_client = Groq(api_key=groq_api_key)
                st.session_state['groq_configured'] = True
            if cohere_api_key:
                # Fixed: Updated Cohere client initialization
                self.cohere_client = cohere.Client(api_key=cohere_api_key)
                st.session_state['cohere_configured'] = True
            return True
        except Exception as e:
            st.error(f"Error setting up APIs: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        try:
            # Resize to model input size
            image = image.resize((144, 144))
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to array and normalize
            img_array = np.array(image) / 255.0
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def predict_disease(self, image):
        """Predict disease from image"""
        if self.model is None:
            return None, None
        
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return None, None
        
        try:
            predictions = self.model.predict(processed_image)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_disease = self.class_names[predicted_class_idx]
            
            # Get all predictions for confidence analysis
            all_predictions = {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
            
            return predicted_disease, confidence, all_predictions
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None, None, None
    
    def generate_comprehensive_analysis(self, disease, confidence, user_query):
        """Generate comprehensive analysis including user query response"""
        if not self.groq_client:
            return "Groq API not configured"
        
        disease_data = DISEASE_INFO.get(disease, {})
        
        prompt = f"""
        As an expert agricultural AI advisor, provide a comprehensive analysis for this cotton plant diagnosis:

        DIAGNOSIS RESULTS:
        - Detected Disease: {disease}
        - Model Confidence: {confidence:.2%}
        - Severity Level: {disease_data.get('severity', 'Unknown')}
        - Symptoms: {disease_data.get('symptoms', 'Not available')}

        FARMER'S QUESTION/CONCERN:
        "{user_query if user_query else 'General advice needed'}"

        DISEASE CONTEXT:
        {disease_data.get('description', 'No description available')}
        Typical Causes: {disease_data.get('causes', 'Unknown')}

        Provide a detailed response covering:

        1. CONFIDENCE ASSESSMENT: 
           - Reliability of this diagnosis
           - What this means for the farmer

        2. DIRECT RESPONSE TO FARMER'S QUESTION:
           - Address their specific concern
           - Provide relevant context

        3. IMMEDIATE RECOMMENDATIONS:
           - Actions for next 24-48 hours
           - Emergency steps if needed

        4. TREATMENT STRATEGY:
           - Primary treatment approach
           - Alternative options
           - Timeline expectations

        5. MONITORING AND PREVENTION:
           - What to watch for
           - Long-term prevention
           - When to seek additional help

        6. ECONOMIC CONSIDERATIONS:
           - Potential yield impact
           - Cost-effective solutions
           - ROI of treatment options

        Use clear, practical language suitable for farmers. Focus on actionable advice.
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating analysis: {e}"
    
    def generate_cohere_embeddings(self, text_content):
        """Generate embeddings for similarity analysis"""
        if not self.cohere_client:
            return None
        
        try:
            response = self.cohere_client.embed(
                texts=[text_content],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            return response.embeddings[0]
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return None
    
    def get_treatment_recommendations(self, disease):
        """Get specific treatment recommendations"""
        treatments = {
            "Bacterial Blight": {
                "immediate": ["Remove affected leaves", "Improve air circulation", "Reduce overhead watering"],
                "chemical": ["Copper-based bactericides", "Streptomycin treatments"],
                "organic": ["Neem oil spray", "Baking soda solution", "Compost tea"],
                "prevention": ["Resistant varieties", "Crop rotation", "Proper spacing"]
            },
            "Curl Virus": {
                "immediate": ["Isolate infected plants", "Control whitefly vectors", "Remove severely affected plants"],
                "chemical": ["Insecticides for whitefly control", "Systemic treatments"],
                "organic": ["Yellow sticky traps", "Reflective mulches", "Beneficial insects"],
                "prevention": ["Virus-free seeds", "Vector control", "Quarantine measures"]
            },
            "Healthy Leaf": {
                "maintenance": ["Continue current practices", "Regular monitoring", "Preventive care"],
                "optimization": ["Nutrient management", "Water optimization", "Integrated pest management"]
            },
            "Herbicide Growth Damage": {
                "immediate": ["Stop herbicide applications", "Increase watering", "Apply growth regulators"],
                "recovery": ["Foliar nutrition", "Root stimulants", "Stress reduction"],
                "prevention": ["Proper application timing", "Correct dosage", "Weather considerations"]
            },
            "Leaf Hopper Jassids": {
                "immediate": ["Insecticide spray", "Remove weeds", "Monitor populations"],
                "chemical": ["Systemic insecticides", "Contact sprays"],
                "organic": ["Predatory insects", "Soap sprays", "Diatomaceous earth"],
                "prevention": ["Early detection", "Beneficial habitat", "Resistant varieties"]
            },
            "Leaf Redding": {
                "immediate": ["Soil test for potassium", "Adjust irrigation", "Check drainage"],
                "nutritional": ["Potassium fertilization", "Balanced nutrition", "Micronutrient correction"],
                "management": ["Stress reduction", "Temperature control", "Water management"]
            },
            "Leaf Variegation": {
                "assessment": ["Determine cause", "Check for virus", "Evaluate genetics"],
                "management": ["Symptom monitoring", "Supportive care", "Vector control"],
                "prevention": ["Quality seeds", "Sanitation", "Resistant varieties"]
            }
        }
        
        return treatments.get(disease, {"general": ["Consult agricultural expert", "Regular monitoring", "Integrated management"]})

def main():
    st.title("Cotton Disease AI Advisor")
    
    # Initialize the AI system
    ai_system = CottonDiseaseAI()
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Check model status
        st.subheader("Model Status")
        if ai_system.model:
            st.success("Model loaded successfully!")
        else:
            st.error("Model not available")
        
        # API Keys
        st.subheader("API Configuration")
        groq_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key for AI analysis")
        cohere_key = st.text_input("Cohere API Key", type="password", help="Enter your Cohere API key for embeddings")
        
        if st.button("Setup APIs"):
            if ai_system.setup_apis(groq_key, cohere_key):
                st.success("APIs configured!")
            else:
                st.error("API setup failed")
        
        # Information
        st.subheader("About")
        st.info("""
        This AI system analyzes cotton leaf diseases using:
        - CNN model for image classification
        - Groq for intelligent analysis
        - Cohere for semantic understanding
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Cotton Leaf Image")
        
        uploaded_image = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear image of cotton leaf for disease detection"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # User query input
        st.header("Your Questions About the Crop")
        user_query = st.text_area(
            "Describe your concerns or ask questions about your cotton crop:",
            placeholder="e.g., 'The leaves are turning yellow', 'What preventive measures should I take?', 'How severe is this condition?'",
            height=100
        )
        
        # Analysis button
        if st.button("Analyze Disease and Answer Query", type="primary"):
            if ai_system.model is None:
                st.error("Model not available!")
            elif not uploaded_image:
                st.error("Please upload an image first!")
            else:
                with st.spinner("Analyzing image and processing your query..."):
                    disease, confidence, all_predictions = ai_system.predict_disease(image)
                    
                    if disease:
                        st.session_state['prediction_results'] = {
                            'disease': disease,
                            'confidence': confidence,
                            'all_predictions': all_predictions,
                            'user_query': user_query
                        }
                        st.success("Analysis complete!")
                    else:
                        st.error("Prediction failed!")
    
    with col2:
        st.header("Analysis Results")
        
        if 'prediction_results' in st.session_state:
            results = st.session_state['prediction_results']
            disease = results['disease']
            confidence = results['confidence']
            all_predictions = results['all_predictions']
            
            # Main prediction display
            st.subheader(f"Detected Disease: {disease}")
            st.write(f"**Confidence:** {confidence:.1%}")
            st.write(f"**Severity:** {DISEASE_INFO.get(disease, {}).get('severity', 'Unknown')}")
            
            # Confidence visualization
            fig = px.bar(
                x=list(all_predictions.keys()),
                y=list(all_predictions.values()),
                title="Prediction Confidence for All Classes",
                labels={'x': 'Disease Class', 'y': 'Confidence Score'}
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Disease information
            disease_info = DISEASE_INFO.get(disease, {})
            if disease_info:
                st.subheader("Disease Information")
                st.write(f"**Description:** {disease_info.get('description', 'N/A')}")
                st.write(f"**Common Symptoms:** {disease_info.get('symptoms', 'N/A')}")
                st.write(f"**Typical Causes:** {disease_info.get('causes', 'N/A')}")
    
    # AI Analysis Section
    if 'prediction_results' in st.session_state:
        st.header("AI-Powered Analysis and Recommendations")
        
        results = st.session_state['prediction_results']
        disease = results['disease']
        confidence = results['confidence']
        user_query = results.get('user_query', '')
        
        # Generate AI analysis
        if st.button("Generate Detailed AI Analysis"):
            if groq_key and groq_key.strip():
                ai_system.setup_apis(groq_key, cohere_key)
                with st.spinner("Generating AI analysis..."):
                    analysis = ai_system.generate_comprehensive_analysis(disease, confidence, user_query)
                    
                    st.subheader("Expert AI Analysis")
                    st.write(analysis)
            else:
                st.warning("Groq API not configured. Please add your API key in the sidebar.")
        
        # Treatment recommendations
        st.subheader("Treatment Recommendations")
        treatments = ai_system.get_treatment_recommendations(disease)
        
        # Create tabs for different treatment types
        if disease != "Healthy Leaf":
            tab1, tab2, tab3, tab4 = st.tabs(["Immediate Actions", "Chemical Treatments", "Organic Options", "Prevention"])
            
            with tab1:
                if 'immediate' in treatments:
                    for action in treatments['immediate']:
                        st.write(f"• {action}")
                else:
                    st.info("No immediate actions required for this condition.")
            
            with tab2:
                if 'chemical' in treatments:
                    for treatment in treatments['chemical']:
                        st.write(f"• {treatment}")
                else:
                    st.info("No specific chemical treatments recommended.")
            
            with tab3:
                if 'organic' in treatments:
                    for treatment in treatments['organic']:
                        st.write(f"• {treatment}")
                else:
                    st.info("No specific organic treatments available.")
            
            with tab4:
                if 'prevention' in treatments:
                    for measure in treatments['prevention']:
                        st.write(f"• {measure}")
                else:
                    st.info("General good agricultural practices recommended.")
        else:
            st.success("Your cotton plant appears healthy! Continue with current care practices.")
            if 'maintenance' in treatments:
                st.write("**Maintenance Recommendations:**")
                for rec in treatments['maintenance']:
                    st.write(f"• {rec}")
    
    # Footer
    st.markdown("---")
    st.markdown("Cotton Disease AI Advisor - Powered by Deep Learning and GenAI")
    st.markdown("For best results, ensure clear, well-lit images of cotton leaves")
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Cotton Disease AI Advisor - Created by Prince Patel and Krupal Varma</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
