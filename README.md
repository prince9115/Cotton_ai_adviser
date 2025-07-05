# Cotton Disease AI Adviser

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cottondiseaseaiadviser.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An AI-powered solution for early detection and management of cotton leaf diseases using deep learning and generative AI.**

## Overview

The Cotton Disease AI Adviser is a comprehensive web application that combines computer vision and generative AI to help farmers and agricultural professionals identify cotton leaf diseases, understand their severity, and receive personalized treatment recommendations.

### **Key Features**

- **Disease Detection**: CNN-based image classification for 7 cotton diseases
- **Confidence Analysis**: Detailed prediction confidence scores
- **AI-Powered Insights**: Intelligent analysis using Groq's LLM
- **Treatment Recommendations**: Categorized treatment options (Chemical, Organic, Preventive)
- **User-Friendly Interface**: Clean, responsive Streamlit interface

## Supported Disease Classes

| Disease | Severity | Common Symptoms |
|---------|----------|----------------|
| **Bacterial Blight of Cotton** | High | Angular lesions, yellow halos, leaf drop |
| **Curl Virus** | Very High | Upward leaf curling, yellowing, stunted growth |
| **Healthy Leaf** | None | Normal green coloration, healthy appearance |
| **Herbicide Growth Damage** | Medium | Distorted growth, chlorosis, stunted development |
| **Leaf Hopper Jassids** | Medium | Yellowing, stippling, reduced vigor |
| **Leaf Redding** | Low | Red/purple margins, premature aging |
| **Leaf Variegation** | Low-Medium | Irregular color patterns, mosaic appearance |

## Live Demo

**Try the app now:** [Cotton Disease AI Adviser](https://cottondiseaseaiadviser.streamlit.app/)

## Technical Architecture

### **Machine Learning Pipeline**
```
Input Image → Preprocessing → CNN Model → Disease Classification → Confidence Analysis
```

### **AI Enhancement Pipeline**
```
Classification Results → Groq LLM → Comprehensive Analysis → Treatment Recommendations
```

### **Technology Stack**
- **Frontend**: Streamlit
- **Deep Learning**: TensorFlow/Keras
- **AI Analysis**: Groq API (Llama 4)
- **Embeddings**: Cohere API
- **Data Visualization**: Plotly, Matplotlib
- **Deployment**: Streamlit Cloud

## Prerequisites

- Python 3.8 or higher
- API Keys:
  - [Groq API Key](https://console.groq.com/) (for AI analysis)
  - [Cohere API Key](https://dashboard.cohere.ai/) (for embeddings)

## Installation & Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/prince9115/Cotton_ai_adviser.git
cd Cotton_ai_adviser
```

### **2. Create Virtual Environment**
```bash
python -m venv cotton_ai_env
source cotton_ai_env/bin/activate  # On Windows: cotton_ai_env\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Download Model File**
Ensure `cotton_leaf_cnn_best_model.keras` is in the root directory.

### **5. Run the Application**
```bash
streamlit run app.py
```

## Requirements

```
streamlit>=1.28.0
tensorflow>=2.13.0
numpy>=1.21.0
pillow>=9.0.0
pandas>=1.5.0
plotly>=5.0.0
groq>=0.4.0
cohere>=4.0.0
```

## Usage Guide

### **Step 1: Configure APIs**
1. Go to the sidebar "Configuration" section
2. Enter your Groq API key for AI analysis
3. Enter your Cohere API key for embeddings *(optional)*
4. Click "Setup APIs"

### **Step 2: Upload Image**
1. Navigate to the web application
2. Upload a clear image of cotton leaf (JPG, PNG, BMP supported)
3. Ensure good lighting and minimal background

### **Step 3: Ask Questions** *(Optional)*
- Describe your observations: *"The leaves are turning yellow"*
- Ask for advice: *"What preventive measures should I take?"*
- Inquire about severity: *"How serious is this condition?"*

### **Step 4: Analyze**
1. Click "Analyze Disease and Answer Query"
2. View prediction results and confidence scores
3. Generate detailed AI analysis (requires API key)

### **Step 5: Review Recommendations**
- **Immediate Actions**: Urgent steps to take
- **Chemical Treatments**: Recommended pesticides/fungicides
- **Organic Options**: Natural treatment alternatives
- **Prevention**: Long-term management strategies

## Model Details

### **Architecture**
- **Input**: 224x224x3 RGB images
- **Base**: Custom CNN with 3 convolutional blocks
- **Regularization**: L2 regularization, Dropout, BatchNormalization
- **Output**: 7-class softmax classification

### **Training Details**
- **Dataset**: 7,000+ augmented cotton leaf images
- **Augmentation**: Rotation, shifting, zoom, brightness adjustment
- **Optimizer**: Adam (lr=0.0005)
- **Loss**: Categorical crossentropy with class weights
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### **Performance Metrics**
- **Validation Accuracy**: ~70%
- **Validation Precision**: ~70%
- **Validation Recall**: ~70%

## API Configuration

### **Groq API Setup**
1. Visit [Groq Console](https://console.groq.com/)
2. Create an account and generate API key
3. Used for: Comprehensive disease analysis and treatment recommendations

### **Cohere API Setup**
1. Visit [Cohere Dashboard](https://dashboard.cohere.ai/)
2. Create an account and generate API key
3. Used for: Text embeddings and semantic analysis


## Model Performance

### **Confusion Matrix**
```
                         Predicted
Actual              BB   CV   HL  HGD  LHJ   LR   LV
Bacterial Blight   112    0   53    0    8   26    1
Curl Virus           0   69   54    0   22   54    1
Healthy Leaf         2    2  139    0   52    4    1
Herbicide Growth    23    5    9  163    0    0    0
Leaf Hopper          0    1    3    0  131   44   21
Leaf Redding         0    0    0    0    9  191    0
Leaf Variegation     0    0    0    0    5    7  188
```

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### **Contribution Guidelines**
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Prince Patel** - *Developer* - [GitHub](https://github.com/prince9115)

## Acknowledgments

- **Dataset**: Cotton leaf disease dataset from agricultural research
- **APIs**: Groq for LLM capabilities, Cohere for embeddings
- **Framework**: Streamlit for rapid web app development
- **ML Libraries**: TensorFlow, scikit-learn, OpenCV

## Support

- **Issues**: [GitHub Issues](https://github.com/prince9115/Cotton_ai_adviser/issues)
- **Email**: [Contact Us](mailto:prince9115@example.com)

## Future Enhancements

- [ ] Multi-language support
- [ ] Mobile app development
- [ ] Integration with IoT sensors
- [ ] Historical disease tracking
- [ ] Weather-based recommendations
- [ ] Batch image processing
- [ ] PDF report generation

---

<div align="center">

**Made with ❤️ for farmers and agricultural professionals**

[Live Demo](https://cottondiseaseaiadviser.streamlit.app/) | | [Report Bug](https://github.com/prince9115/Cotton_ai_adviser/issues)

</div>
