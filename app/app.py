# app.py - Medical Insurance Cost Predictor (Cloud Compatible)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# ============================================
# PATH CONFIGURATION - WORKS ON LOCAL AND CLOUD
# ============================================

def get_models_directory():
    """Find models directory - works locally and on Streamlit Cloud"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Possible locations to check
    possible_paths = [
        # Same directory as app.py
        os.path.join(script_dir, "models"),
        # Parent directory (app/models)
        os.path.join(script_dir, "..", "models"),
        # Grandparent directory (project root/models)
        os.path.join(script_dir, "..", "..", "models"),
        # Streamlit Cloud specific path
        "/mount/src/ins_app/models",
        # Alternative cloud path
        os.path.join(os.getcwd(), "models"),
    ]
    
    # Check each path
    for path in possible_paths:
        path = os.path.abspath(path)
        if os.path.exists(path):
            # Verify it has .pkl files
            files = os.listdir(path)
            if any(f.endswith('.pkl') for f in files):
                return path
    
    # If not found, return the most likely path and show error later
    return os.path.join(script_dir, "..", "models")

# Get the models directory
MODELS_DIR = get_models_directory()

# Debug info in sidebar
st.sidebar.markdown("---")
st.sidebar.caption("🔧 Debug Info")

# Check if directory exists
if not os.path.exists(MODELS_DIR):
    st.sidebar.error(f"❌ Directory not found: {MODELS_DIR}")
    st.error("❌ Models directory not found!")
    
    # Show debugging info
    st.write("### Debugging Information")
    st.write(f"Script location: `{os.path.abspath(__file__)}`")
    st.write(f"Current working directory: `{os.getcwd()}`")
    st.write(f"Looking for models in: `{MODELS_DIR}`")
    
    # List what's actually available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    st.write(f"Contents of `{script_dir}`:")
    if os.path.exists(script_dir):
        st.write(os.listdir(script_dir))
    
    parent_dir = os.path.dirname(script_dir)
    if os.path.exists(parent_dir):
        st.write(f"Contents of parent `{parent_dir}`:")
        st.write(os.listdir(parent_dir))
    
    st.stop()
else:
    st.sidebar.success(f"✅ Found: {MODELS_DIR}")
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    st.sidebar.caption(f"📁 {len(files)} model files found")


# ============================================
# LOAD ALL MODELS
# ============================================

@st.cache_resource
def load_all_models():
    """Load all trained models and scalers"""
    models = {}
    
    # Define model files and their display names
    model_files = {
        'Linear Regression': 'linear_model.pkl',
        'Ridge Regression': 'ridge_model.pkl',
        'Lasso Regression': 'lasso_model.pkl',
        'Random Forest': 'best_model.pkl'  # Your champion model
    }
    
    # Alternative names if you saved them differently
    alt_names = {
        'Linear Regression': ['linear_model.pkl', 'linear_regression.pkl', 'model_linear.pkl'],
        'Ridge Regression': ['ridge_model.pkl', 'ridge_regression.pkl', 'model_ridge.pkl'],
        'Lasso Regression': ['lasso_model.pkl', 'lasso_regression.pkl', 'model_lasso.pkl'],
        'Random Forest': ['best_model.pkl', 'random_forest.pkl', 'model_rf.pkl']
    }
    
    loaded_models = {}
    errors = []
    
    for model_name, possible_files in alt_names.items():
        loaded = False
        for filename in possible_files:
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.exists(filepath):
                try:
                    loaded_models[model_name] = joblib.load(filepath)
                    loaded = True
                    break
                except Exception as e:
                    errors.append(f"{model_name}: {str(e)}")
                    continue
        
        if not loaded:
            errors.append(f"{model_name}: File not found (tried: {possible_files})")
    
    # Load scalers
    try:
        scaler_age_bmi = joblib.load(os.path.join(MODELS_DIR, 'scaler_age_bmi.pkl'))
        scaler_children = joblib.load(os.path.join(MODELS_DIR, 'scale_children.pkl'))
    except Exception as e:
        return None, None, f"Scaler error: {str(e)}"
    
    return loaded_models, (scaler_age_bmi, scaler_children), None

# Load everything
all_models, scalers, error = load_all_models()

if error:
    st.sidebar.error(f"❌ {error}")
    st.error("Failed to load models/scalers.")
    st.stop()

if not all_models:
    st.error("❌ No models found!")
    st.stop()

scaler_age_bmi, scaler_children = scalers
st.sidebar.success(f"✅ Loaded {len(all_models)} models")

# ============================================
# SIDEBAR - MODEL SELECTION & INPUTS
# ============================================

st.sidebar.markdown("---")
st.sidebar.header("🤖 Model Selection")

# Model selector
available_models = list(all_models.keys())
selected_model = st.sidebar.selectbox(
    "Choose Prediction Model:",
    available_models,
    index=available_models.index('Random Forest') if 'Random Forest' in available_models else 0,
    help="Compare different algorithms' predictions"
)

# Show model info
model_info = {
    'Linear Regression': {
        'type': 'Linear',
        'pros': 'Simple, fast, interpretable',
        'cons': 'Assumes linear relationships only',
        'expected_accuracy': 'Low-Medium (~72% R²)',
        'color': '#FF6B6B'
    },
    'Ridge Regression': {
        'type': 'Linear (Regularized)',
        'pros': 'Handles multicollinearity, stable',
        'cons': 'Still assumes linearity',
        'expected_accuracy': 'Medium (~72% R²)',
        'color': '#4ECDC4'
    },
    'Lasso Regression': {
        'type': 'Linear (Feature Selection)',
        'pros': 'Automatic feature selection',
        'cons': 'Can be too aggressive, lower accuracy',
        'expected_accuracy': 'Lower (~49% R²)',
        'color': '#45B7D1'
    },
    'Random Forest': {
        'type': 'Ensemble (Decision Trees)',
        'pros': 'Handles non-linearity, most accurate',
        'cons': 'Black box, slower',
        'expected_accuracy': 'High (~90% R²)',
        'color': '#96CEB4'
    }
}

if selected_model in model_info:
    info = model_info[selected_model]
    st.sidebar.markdown(f"""
    <div style='background-color: {info['color']}20; padding: 10px; border-radius: 5px; margin-top: 10px;'>
        <b>Type:</b> {info['type']}<br>
        <b>Expected:</b> {info['expected_accuracy']}<br>
        <small>✅ {info['pros']}</small><br>
        <small>⚠️ {info['cons']}</small>
    </div>
    """, unsafe_allow_html=True)

# Compare all models checkbox
compare_all = st.sidebar.checkbox("📊 Compare All Models", value=False)

st.sidebar.markdown("---")
st.sidebar.header("👤 Your Details")

# Collect all inputs in sidebar
age = st.sidebar.number_input("Age", 18, 100, 30, step=1)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
children = st.sidebar.number_input("Children", 0, 5, 0, step=1)

bmi = st.sidebar.number_input("BMI", 15.0, 55.0, 25.0, 0.1)

# BMI indicator
if bmi < 18.5:
    st.sidebar.caption("🟦 Underweight")
elif bmi < 25:
    st.sidebar.caption("🟩 Normal weight")
elif bmi < 30:
    st.sidebar.caption("🟨 Overweight")
else:
    st.sidebar.caption("🟧 Obese")

smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
region = st.sidebar.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

# ============================================
# MAIN CONTENT
# ============================================

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Compare ML models to estimate your insurance premium")

# Prediction function
def predict_with_model(model, age, bmi, children, smoker, sex, region):
    """Make prediction with any model"""
    # Encode
    smoker_encoded = 1 if smoker == "Yes" else 0
    sex_encoded = 1 if sex == "Male" else 0
    
    # Scale
    age_bmi_scaled = scaler_age_bmi.transform([[age, bmi]])
    children_scaled = scaler_children.transform([[children]])[0][0]
    
    # Feature vector
    features = np.array([
        age_bmi_scaled[0][0],  # age scaled
        age_bmi_scaled[0][1],  # bmi scaled
        children_scaled,
        smoker_encoded,
        sex_encoded,
        1 if region == "Northeast" else 0,
        1 if region == "Northwest" else 0,
        1 if region == "Southeast" else 0,
        1 if region == "Southwest" else 0
    ]).reshape(1, -1)
    
    # Predict and convert
    log_pred = model.predict(features)[0]
    real_pred = np.exp(log_pred)
    
    return real_pred, log_pred

# Calculate button
if st.button("🚀 Calculate Premium", type="primary", use_container_width=True):
    
    if compare_all:
        # ==========================================
        # COMPARE ALL MODELS VIEW
        # ==========================================
        
        st.header("📊 Model Comparison Results")
        
        results = []
        for model_name, model in all_models.items():
            try:
                pred, log_pred = predict_with_model(model, age, bmi, children, smoker, sex, region)
                results.append({
                    'Model': model_name,
                    'Prediction': pred,
                    'Log_Scale': log_pred,
                    'Color': model_info.get(model_name, {}).get('color', '#808080')
                })
            except Exception as e:
                st.error(f"Error with {model_name}: {str(e)}")
        
        if not results:
            st.error("No predictions could be made.")
        else:
            # Convert to DataFrame for display
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values('Prediction')
            
            # Display as table
            st.subheader("Predicted Annual Premiums")
            
            # Create comparison table
            cols = st.columns(len(results))
            for idx, (_, row) in enumerate(df_results.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div style='background-color: {row['Color']}30; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {row['Color']};'>
                        <h3 style='margin: 0; color: {row['Color']};'>{row['Model']}</h3>
                        <h2 style='margin: 10px 0; font-size: 2rem;'>${row['Prediction']:,.0f}</h2>
                        <small>Log: {row['Log_Scale']:.2f}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Highlight best (Random Forest usually)
            best_model = df_results.loc[df_results['Prediction'].idxmax() if smoker == "Yes" else df_results['Prediction'].idxmin(), 'Model']
            st.info(f"💡 **Recommendation:** Random Forest is most reliable (90% accuracy). Linear models may underestimate costs for high-risk profiles.")
            
            # Visualization
            st.subheader("Prediction Comparison Chart")
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.barh(df_results['Model'], df_results['Prediction'], 
                          color=[row['Color'] for _, row in df_results.iterrows()],
                          alpha=0.7, edgecolor='black')
            
            # Add value labels
            for i, (bar, pred) in enumerate(zip(bars, df_results['Prediction'])):
                ax.text(pred + 500, bar.get_y() + bar.get_height()/2, 
                       f'${pred:,.0f}', va='center', fontweight='bold')
            
            ax.set_xlabel('Predicted Annual Premium ($)')
            ax.set_title('Model Predictions Comparison', fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)
            
            # Add average line
            avg_cost = 13270
            ax.axvline(avg_cost, color='red', linestyle='--', alpha=0.7, label=f'National Avg: ${avg_cost:,.0f}')
            ax.legend()
            
            st.pyplot(fig)
            
            # Analysis
            st.subheader("📈 Analysis")
            min_pred = df_results['Prediction'].min()
            max_pred = df_results['Prediction'].max()
            range_pred = max_pred - min_pred
            
            st.write(f"**Prediction Range:** ${range_pred:,.0f} (from ${min_pred:,.0f} to ${max_pred:,.0f})")
            st.write(f"**Model Agreement:** {'High' if range_pred < 5000 else 'Medium' if range_pred < 15000 else 'Low'} (range: ${range_pred:,.0f})")
            
            if smoker == "Yes":
                st.warning("🔴 **High Variance:** Linear models struggle with smoker premiums (non-linear relationship). Trust Random Forest.")
            else:
                st.success("🟢 **Low Variance:** Models agree fairly well for low-risk profiles.")
    
    else:
        # ==========================================
        # SINGLE MODEL VIEW
        # ==========================================
        
        model = all_models[selected_model]
        prediction, log_pred = predict_with_model(model, age, bmi, children, smoker, sex, region)
        
        # Display prediction prominently
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"### Estimated Annual Premium")
            st.markdown(f"<h1 style='font-size: 3.5rem; color: #1f77b4; margin: 0;'>${prediction:,.2f}</h1>", 
                       unsafe_allow_html=True)
            st.caption(f"Using: **{selected_model}** | Log scale: {log_pred:.4f}")
            
            # Comparison
            avg_cost = 13270
            diff = float(prediction - avg_cost)
            pct = (diff / avg_cost) * 100
            
            if diff > 0:
                st.info(f"📊 ${abs(diff):,.0f} **({pct:+.1f}%)** above national average (${avg_cost:,.0f})")
            else:
                st.info(f"📊 ${abs(diff):,.0f} **({pct:+.1f}%)** below national average (${avg_cost:,.0f})")
        
        with col2:
            # Risk factors summary
            st.markdown("### Risk Factors")
            
            if smoker == "Yes":
                st.error("🚬 Smoker\n+280% impact")
            else:
                st.success("✅ Non-smoker")
            
            if bmi > 30:
                st.warning(f"⚖️ BMI {bmi:.1f}\nObese")
            elif bmi > 25:
                st.info(f"⚖️ BMI {bmi:.1f}\nOverweight")
            else:
                st.success(f"⚖️ BMI {bmi:.1f}\nHealthy")
            
            if age > 50:
                st.warning(f"🎂 Age {age}\nSenior")
            else:
                st.success(f"🎂 Age {age}\nAdult")
        
        # Model-specific insights
        st.divider()
        
        if selected_model == 'Random Forest':
            st.success("✅ **Most Accurate Model**: This prediction has ~90% confidence based on historical data patterns.")
        elif selected_model in ['Linear Regression', 'Ridge Regression']:
            st.warning("⚠️ **Linear Model Limitation**: May underestimate costs if you have multiple risk factors (smoking + high BMI + age).")
            st.info("💡 **Tip:** Switch to Random Forest for more accurate estimation.")
        elif selected_model == 'Lasso Regression':
            st.warning("⚠️ **Lower Accuracy**: This model has ~49% accuracy. Use for reference only.")
        
        # Recommendation based on prediction
        st.divider()
        if prediction > 30000:
            st.error("🚨 **Critical Risk Profile**: Premium is in top 10%. Lifestyle changes (quit smoking, weight loss) could save $20,000+/year.")
        elif prediction > 15000:
            st.warning("⚠️ **Above Average**: Consider health improvements to reduce long-term costs.")
        else:
            st.success("✅ **Favorable Rate**: Your profile qualifies for competitive premiums.")

else:
    # Initial state
    st.info("👈 Enter your details in the sidebar and click **Calculate Premium** to get started!")
    
    # Show feature importance from Random Forest
    if 'Random Forest' in all_models:
        st.divider()
        st.subheader("🔍 What Drives Insurance Costs? (Feature Importance)")
        
        rf_model = all_models['Random Forest']
        if hasattr(rf_model, 'feature_importances_'):
            feature_names = ['Age', 'BMI', 'Children', 'Smoker', 'Sex', 'NE', 'NW', 'SE', 'SW']
            importances = rf_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#ff4b4b' if f == 'Smoker' else '#1f77b4' for f in importance_df['Feature']]
            ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors, alpha=0.8)
            ax.set_xlabel('Importance Score')
            ax.set_title('Random Forest: What Matters Most for Insurance Costs?', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            st.caption("🔴 Smoking is the #1 predictor (~65% importance), followed by Age (~18%) and BMI (~12%)")

st.divider()
st.caption("Built with Streamlit • Multiple ML Models • Educational Project")