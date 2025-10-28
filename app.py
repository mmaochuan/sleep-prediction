import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os
import shap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Sleep Quality Prediction System",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styles
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    .risk-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-color: #4caf50;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-color: #ff9800;
    }
    .risk-high {
        background-color: #ffebee;
        border-color: #f44336;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 3em;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .section-header {
        color: #667eea;
        font-size: 1.5em;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# Model path configuration
def get_model_dir():
    """Auto-detect model directory path (optimized deployment compatibility)"""
    possible_paths = [
        os.path.join(".", "saved_models_selected_features"),
        os.path.join(os.path.dirname(__file__), "saved_models_selected_features")
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path

    default_path = os.path.join(".", "saved_models_selected_features")
    st.warning(f"Model directory not found, will try default path: {default_path}")
    return default_path


MODEL_DIR = get_model_dir()

# Feature label mapping (updated version)
FEATURE_LABELS = {
    'age': {
        'label': 'Age',
        'type': 'number',
        'min': 45,
        'max': 120,
        'step': 1,
        'is_integer': True
    },
    'gender': {
        'label': 'Gender',
        'options': {'0': 'Female', '1': 'Male'}
    },
    'education': {
        'label': 'Education Level',
        'options': {'1': 'Below Middle School', '2': 'High School/Vocational', '3': 'Higher Education'}
    },
    'smoke': {
        'label': 'Smoking',
        'options': {'0': 'No', '1': 'Yes'}
    },
    'digeste': {
        'label': 'Digestive Disease',
        'options': {'0': 'No', '1': 'Yes'}
    },
    'lunge': {
        'label': 'Lung Disease',
        'options': {'0': 'No', '1': 'Yes'}
    },
    'arthre': {
        'label': 'Arthritis',
        'options': {'0': 'No', '1': 'Yes'}
    },
    'chronum': {
        'label': 'Number of Comorbid Conditions',
        'type': 'number',
        'min': 0,
        'max': 14,
        'step': 1,
        'is_integer': True,
        'desc': '''**Chronic conditions include the following 14 diseases:**

1. Hypertension
2. Dyslipidemia
3. Diabetes
4. Cancer
5. Lung disease
6. Liver disease
7. Heart disease
8. Stroke
9. Kidney disease
10. Digestive disease
11. Psychiatric disorder
12. Memory disorder
13. Arthritis
14. Asthma

**Please enter the total number of diseases present (0-14)**'''
    },
    'adl': {
        'label': 'ADL Score',
        'type': 'number',
        'min': 0,
        'max': 6,
        'step': 1,
        'is_integer': True,
        'desc': '''**ADL (Activities of Daily Living) Score Description:**

ADL assesses difficulty in 6 basic daily activities:

1. **Dressing**: Difficulty dressing yourself
2. **Bathing**: Difficulty bathing yourself
3. **Eating**: Difficulty eating yourself
4. **Transferring**: Difficulty getting in/out of bed or chair
5. **Toileting**: Difficulty using the toilet
6. **Continence**: Difficulty controlling bowel/bladder

**Scoring method:**
- Each activity with difficulty scores 1 point
- Total score range: 0-6 points
- Higher score indicates worse daily living ability

**Please enter the number of activities with difficulty (0-6)**'''
    },
    'iadl': {
        'label': 'IADL Score',
        'type': 'number',
        'min': 0,
        'max': 5,
        'step': 1,
        'is_integer': True,
        'desc': '''**IADL (Instrumental Activities of Daily Living) Score Description:**

IADL assesses difficulty in 5 instrumental daily activities:

1. **Housework**: Difficulty doing housework
2. **Cooking**: Difficulty cooking meals
3. **Shopping**: Difficulty shopping
4. **Money management**: Difficulty managing money
5. **Medication**: Difficulty taking medication on time

**Scoring method:**
- Each activity with difficulty scores 1 point
- Total score range: 0-5 points
- Higher score indicates worse instrumental daily activity ability

**Please enter the number of activities with difficulty (0-5)**'''
    },
    'cog': {
        'label': 'Cognitive Function Score',
        'type': 'number',
        'min': 0,
        'max': 21,
        'step': 0.5,
        'is_integer': False,
        'desc': '''**CHARLS Cognitive Function Score Description:**

Total cognitive score consists of two parts, maximum 21 points:

**I. Mental Status (0-11 points)**

1. **Time orientation** (3 points total)
   - What is today's date? (year, month, day each 1 point)

2. **Time orientation** (1 point)
   - What day of the week is it?

3. **Time orientation** (1 point)
   - What season is it? (Spring/Summer/Fall/Winter)

4. **Calculation ability** (5 points total)
   - Starting from 100, subtract 7 five times
   - i.e.: 100-7=93, 93-7=86, 86-7=79, 79-7=72, 72-7=65
   - 1 point for each correct answer, maximum 5 points

5. **Visuospatial ability** (1 point)
   - Copy two overlapping pentagons
   - 1 point for correct drawing

**II. Episodic Memory (0-10 points)**

1. **Immediate Recall** (0-10 points)
   - Interviewer reads 10 words (e.g., apple, table, book, etc.)
   - Respondent recalls immediately
   - 1 point per correct word, maximum 10 points

2. **Delayed Recall** (0-10 points)
   - Several minutes later, recall the same set of words again
   - 1 point per correct word, maximum 10 points

3. **Final score calculation**
   - Episodic memory score = (Immediate recall score + Delayed recall score) ÷ 2
   - Range: 0-10 points
   - ⚠️ **Note: Decimal values possible (e.g., 5.5 points) due to averaging**

**Total score calculation:**
- Total score = Mental status score (0-11) + Episodic memory score (0-10)
- Total range: 0-21 points
- **Decimal values allowed, e.g., 10.5, 15.5, etc.**
- **Higher score indicates better cognitive function**

**Please enter total score (0-21, decimals allowed)**'''
    },
    'cesd': {
        'label': 'CESD Depression Score',
        'type': 'number',
        'min': 0,
        'max': 30,
        'step': 1,
        'is_integer': True,
        'desc': '''**CESD-10 Depression Scale Score Description:**

Includes 10 questions assessing feelings over the past week:

**Scoring criteria (1-4 points per question):**
- **1 point** = Rarely or none of the time (<1 day)
- **2 points** = Some or a little of the time (1-2 days)
- **3 points** = Occasionally or a moderate amount of time (3-4 days)
- **4 points** = Most or all of the time (5-7 days)

**10 Questions:**

1. **DC009** I was bothered by things that don't usually bother me
2. **DC010** I had trouble keeping my mind on what I was doing
3. **DC011** I felt depressed
4. **DC012** I felt that everything I did was an effort
5. **DC013** I felt hopeful about the future ⭐ (Reverse scored)
6. **DC014** I felt fearful
7. **DC015** My sleep was restless
8. **DC016** I was happy ⭐ (Reverse scored)
9. **DC017** I felt lonely
10. **DC018** I could not get going

**Reverse scoring (Questions 5 & 8):**
- Original 0 point → Scored as 3 points
- Original 1 points → Scored as 2 points
- Original 2 points → Scored as 1 point
- Original 3 points → Scored as 0 points

**Depression risk level determination:**
- **0-9 points**: No significant depressive symptoms
- **10-12 points**: Mild depressive tendency
- **≥13 points**: Significant depressive symptoms (possible depressive disorder)

**Total score range: 0-30 points**
**Higher score indicates more severe depression**

**Please enter total score (0-30)**'''
    },
    'selfhealth': {
        'label': 'Self-rated Health',
        'options': {'1': 'Very Poor', '2': 'Poor', '3': 'Fair', '4': 'Good', '5': 'Very Good'}
    },
    'lonely': {
        'label': 'Loneliness Frequency',
        'options': {'1': 'Rarely', '2': 'Sometimes', '3': 'Often', '4': 'Always'}
    },
    'lifesat': {
        'label': 'Life Satisfaction',
        'options': {'5': 'Extremely Satisfied', '4': 'Very Satisfied', '3': 'Somewhat Satisfied', '2': 'Not Very Satisfied', '1': 'Not at All Satisfied'}
    },
    'hchild': {
        'label': 'Number of Living Children',
        'type': 'number',
        'min': 0,
        'max': 20,
        'step': 1,
        'is_integer': True
    }
}


@st.cache_resource
def load_models():
    """Load all required models and preprocessors"""
    try:
        if not os.path.exists(MODEL_DIR):
            st.error(f"❌ Model directory does not exist: {MODEL_DIR}")
            return None, None, None, None, None

        # Load feature information
        selected_features_pkl = os.path.join(MODEL_DIR, 'selected_features.pkl')
        if os.path.exists(selected_features_pkl):
            selected_features_data = joblib.load(selected_features_pkl)
            features_info = {
                'selected_features': selected_features_data['selected_features'],
                'selected_categorical': selected_features_data.get('selected_categorical', []),
                'selected_continuous': selected_features_data.get('selected_continuous', [])
            }
        else:
            features_path = os.path.join(MODEL_DIR, 'model_features_info.json')
            with open(features_path, 'r', encoding='utf-8') as f:
                features_info = json.load(f)

        # Load model
        model_name = features_info.get('best_model_name')
        if not model_name:
            model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('best_model_') and f.endswith('.pkl')]
            if model_files:
                model_name = model_files[0].replace('best_model_', '').replace('.pkl', '')
            else:
                st.error("❌ Unable to determine model name")
                return None, None, None, None, None

        model_path = os.path.join(MODEL_DIR, f'best_model_{model_name}.pkl')
        model = joblib.load(model_path)
        features_info['best_model_name'] = model_name

        # Load encoder
        encoder_path = os.path.join(MODEL_DIR, 'ordinal_encoder.pkl')
        ordinal_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None

        # Load scaler
        scaler_path = os.path.join(MODEL_DIR, 'scaler_continuous.pkl')
        scaler_cont = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)

        return model, ordinal_encoder, scaler_cont, features_info, explainer

    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None, None, None, None, None


def preprocess_input(data, features_info, ordinal_encoder, scaler_cont):
    """Preprocess input data"""
    try:
        selected_features = features_info['selected_features']
        selected_categorical = features_info.get('selected_categorical', [])
        selected_continuous = features_info.get('selected_continuous', [])

        missing_features = [f for f in selected_features if f not in data]
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")

        important_data = {k: v for k, v in data.items() if k in selected_features}
        df = pd.DataFrame([important_data])

        # Encode categorical features
        if selected_categorical and ordinal_encoder is not None:
            cat_encoded = pd.DataFrame(
                ordinal_encoder.transform(df[selected_categorical]),
                columns=selected_categorical
            )
        else:
            cat_encoded = pd.DataFrame()

        # Standardize continuous features
        if selected_continuous and scaler_cont is not None:
            cont_scaled = pd.DataFrame(
                scaler_cont.transform(df[selected_continuous]),
                columns=selected_continuous
            )
        else:
            cont_scaled = pd.DataFrame()

        # Merge features
        if not cat_encoded.empty and not cont_scaled.empty:
            X_processed = pd.concat([cat_encoded, cont_scaled], axis=1)
        elif not cat_encoded.empty:
            X_processed = cat_encoded
        else:
            X_processed = cont_scaled

        X_processed = X_processed[selected_features]

        return X_processed

    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        raise


def configure_chinese_fonts():
    """Configure Chinese font display"""
    import platform
    import matplotlib.font_manager as fm

    system = platform.system()

    # Get available system fonts
    available_fonts = set([f.name for f in fm.fontManager.ttflist])

    # Define preferred fonts for different platforms
    if system == 'Windows':
        preferred_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        preferred_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti']
    else:  # Linux / Cloud
        preferred_fonts = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
            'Droid Sans Fallback', 'AR PL UMing CN', 'Noto Sans SC'
        ]

    # Add common fallback fonts
    preferred_fonts.extend(['DejaVu Sans', 'sans-serif'])

    # Find the first available font
    for font in preferred_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        # If none available, use all fallback fonts
        plt.rcParams['font.sans-serif'] = preferred_fonts

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'


def generate_shap_plot(shap_values, feature_values, base_value, features_info):
    """Generate SHAP waterfall plot (improved Chinese font support)"""
    try:
        # Configure Chinese fonts
        configure_chinese_fonts()

        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        feature_names = features_info['selected_features']
        sorted_idx = np.argsort(np.abs(shap_values))[::-1]

        cumsum = base_value
        positions = []
        values = []
        colors = []
        labels = []

        feature_name_map = {
            'gender': 'Gender', 'age': 'Age', 'education': 'Education',
            'cog': 'Cognitive Function', 'cesd': 'Depression Score', 'lonely': 'Loneliness',
            'selfhealth': 'Self-rated Health', 'depre': 'Depression Level', 'lifesat': 'Life Satisfaction',
            'chronum': 'Chronic Conditions', 'smoke': 'Smoking', 'digeste': 'Digestive Disease',
            'lunge': 'Lung Disease', 'arthre': 'Arthritis', 'hchild': 'Number of Children',
            'iadl': 'IADL Score', 'adl': 'ADL Score'
        }

        for idx in sorted_idx:
            positions.append(cumsum)
            values.append(shap_values[idx])
            colors.append('#FF6B6B' if shap_values[idx] > 0 else '#4ECDC4')

            feat_name = feature_name_map.get(feature_names[idx], feature_names[idx])
            feat_val = feature_values[idx]
            labels.append(f'{feat_name} = {feat_val:.2f}')

            cumsum += shap_values[idx]

        y_pos = np.arange(len(values))

        for i, (pos, val, color, label) in enumerate(zip(positions, values, colors, labels)):
            ax.barh(i, val, left=pos, color=color, alpha=0.8, height=0.6)
            text_x = pos + val / 2
            ax.text(text_x, i, f'{val:+.3f}',
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        ax.axvline(base_value, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'Baseline: {base_value:.3f}')
        ax.axvline(cumsum, color='red', linestyle='-', linewidth=2, alpha=0.7,
                   label=f'Prediction: {cumsum:.3f}')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('SHAP Value Impact on Prediction', fontsize=12, fontweight='bold')
        ax.set_title('Feature Impact Analysis on Sleep Quality Risk', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        return fig

    except Exception as e:
        st.error(f"SHAP plot generation failed: {str(e)}")
        return None





def main():
    """Main function"""

    # Load models
    model, ordinal_encoder, scaler_cont, features_info, explainer = load_models()

    if model is None:
        st.error("❌ Model loading failed, please check model path and files")
        st.stop()

    # Title
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 12px; color: white; margin-bottom: 2rem;'>
        <h1>🌙 Sleep Quality Prediction System</h1>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Information")
        st.info(f"""
        **Model Type**: {features_info['best_model_name']}  
        **Number of Features**: {len(features_info['selected_features'])}
        """)

        st.markdown("### 📋 Instructions")
        st.write("""
        1. Fill in all required health information
        2. Click "Start Prediction" button
        3. View risk assessment results
        4. Take preventive measures based on recommendations

        💡 **Tip**: Click ❓ next to input boxes for detailed descriptions
        """)

    # Main content area
    selected_features = features_info['selected_features']

    # Categorize features
    categories = {
        'Basic Information': ['gender', 'age', 'education'],
        'Health Status': ['smoke', 'digeste', 'lunge', 'arthre', 'chronum'],
        'Functional Assessment': ['adl', 'iadl', 'cog', 'cesd'],
        'Subjective Evaluation': ['selfhealth', 'lonely', 'lifesat'],
        'Family Information': ['hchild']
    }

    # Create form
    with st.form("prediction_form"):
        input_data = {}

        for category, features in categories.items():
            important_features = [f for f in features if f in selected_features]

            if not important_features:
                continue

            st.markdown(f"<div class='section-header'>📋 {category}</div>", unsafe_allow_html=True)

            cols = st.columns(2)

            for idx, feature in enumerate(important_features):
                if feature not in FEATURE_LABELS:
                    continue

                label_info = FEATURE_LABELS[feature]
                label = label_info['label']

                with cols[idx % 2]:
                    if 'options' in label_info:
                        options_dict = label_info['options']
                        options_list = list(options_dict.keys())
                        options_display = [f"{options_dict[k]}" for k in options_list]

                        selected = st.selectbox(
                            f"{label}",
                            options=options_display,
                            key=feature
                        )

                        selected_idx = options_display.index(selected)
                        input_data[feature] = float(options_list[selected_idx])
                    else:
                        min_val = label_info.get('min', 0)
                        max_val = label_info.get('max', 100)
                        step = label_info.get('step', 1)
                        desc = label_info.get('desc', '')
                        is_integer = label_info.get('is_integer', False)

                        help_text = desc if desc else None

                        # Use integer as default value and step
                        value = st.number_input(
                            f"{label}",
                            min_value=int(min_val) if is_integer else float(min_val),
                            max_value=int(max_val) if is_integer else float(max_val),
                            value=int(min_val) if is_integer else float(min_val),
                            step=int(step) if is_integer else float(step),
                            help=help_text,
                            key=feature
                        )

                        # Store as float for consistency
                        input_data[feature] = float(value)

        # Submit button
        submitted = st.form_submit_button("🔮 Start Prediction", use_container_width=True)

    # Handle prediction
    if submitted:
        with st.spinner('🔄 Calculating...'):
            try:
                # Preprocessing
                X = preprocess_input(input_data, features_info, ordinal_encoder, scaler_cont)

                # Prediction
                probability = model.predict_proba(X)[0, 1]
                risk_score = probability * 100

                # Risk classification
                if risk_score < 25:
                    risk_class = "Low Risk"
                    risk_color = "risk-low"
                    description = "The patient has a low risk of developing sleep quality problems in the next two years. Current sleep status is good, and it is recommended to continue maintaining a healthy lifestyle."
                    recommendations = """
                    - Maintain regular sleep schedule with fixed bedtime and wake time
                    - Continue moderate exercise such as walking, tai chi, etc.
                    - Maintain balanced diet, avoid caffeine before bedtime
                    - Maintain good mental state, actively participate in social activities
                    - Regular health checkups to monitor health status
                    """
                elif risk_score < 35:
                    risk_class = "Medium Risk"
                    risk_color = "risk-medium"
                    description = "The patient has a moderate risk of developing sleep quality problems in the next two years. Attention is needed and preventive measures should be taken to avoid further risk elevation."
                    recommendations = """
                    - **Establish good sleep hygiene habits**: Keep bedroom comfortable, quiet, and dark
                    - **Control chronic diseases**: Regular medical visits, follow medication instructions
                    - **Increase social activities**: Participate in community activities, reduce loneliness
                    - **Mental health attention**: If experiencing depression or anxiety symptoms, consult a mental health professional promptly
                    - **Avoid bad habits**: Quit smoking and limit alcohol, maintain regular schedule
                    - **Regular follow-up**: Check-up every 3-6 months
                    """
                else:
                    risk_class = "High Risk"
                    risk_color = "risk-high"
                    description = "The patient has a high risk of developing sleep quality problems in the next two years. Immediate intervention measures are strongly recommended with close monitoring of sleep status."
                    recommendations = """
                    - **Seek medical attention promptly**: Recommend professional evaluation at hospital sleep clinic
                    - **Actively treat underlying conditions**: Control hypertension, diabetes and other chronic diseases
                    - **Psychological intervention**: Receive psychological counseling or cognitive behavioral therapy if necessary
                    - **Medication treatment**: Use sleep aids under doctor's guidance
                    - **Lifestyle adjustment**: Strictly maintain sleep schedule, avoid long daytime naps
                    - **Social support**: Seek emotional support from family and friends
                    - **Close follow-up**: Monthly check-ups, adjust treatment plan promptly
                    """

                # Display results
                st.markdown("---")
                st.markdown("## 📊 Prediction Results")

                # Risk score display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class='metric-container'>
                        <div>Sleep Quality Risk Score</div>
                        <div class='metric-value'>{risk_score:.1f}</div>
                        <div style='font-size: 1.2em;'>{risk_class}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Risk explanation
                st.markdown(f"""
                <div class='risk-box {risk_color}'>
                    <h3>🎯 Risk Level: {risk_class}</h3>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)

                # Recommendations
                st.markdown("### 💡 Health Recommendations")
                st.info(recommendations)

                # SHAP explanation
                st.markdown("### 📈 Feature Impact Analysis")

                try:
                    shap_values = explainer.shap_values(X)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]

                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

                    fig = generate_shap_plot(shap_values[0], X.values[0], base_value, features_info)

                    if fig:
                        st.pyplot(fig)
                        st.caption("📌 SHAP values show the contribution of each feature to the sleep quality risk prediction. Red indicates increased risk, blue indicates decreased risk.")

                except Exception as e:
                    st.warning(f"⚠️ Feature impact analysis generation failed: {str(e)}")

                # Prediction details
                with st.expander("📋 View Prediction Details"):
                    st.write("**Input Data:**")
                    st.json(input_data)
                    st.write(f"**Prediction Probability:** {probability:.4f}")


            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


if __name__ == '__main__':
    main()
