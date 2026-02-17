# 🏥 Medical Insurance Cost Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://insapp-j8sxgwpkhsswzxqsargwch.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

&gt; **An interactive Machine Learning web application that predicts medical insurance costs with 90% accuracy using multiple regression models.**

🌐 **[Live Demo](https://insapp-j8sxgwpkhsswzxqsargwch.streamlit.app/)**

---

## ✨ Features

- 🎯 **4 ML Models**: Compare Linear, Ridge, Lasso, and Random Forest regressors in real-time
- 🏆 **90% Accuracy**: Random Forest achieves R² = 0.897, MAE = $2,049
- 📊 **Interactive Comparison**: Side-by-side model predictions with visualizations
- 🔥 **Risk Analysis**: Personalized health risk assessment with actionable insights
- ⚡ **Smart Predictions**: Handles non-linear relationships (smoking + BMI interactions)
- 🎨 **Beautiful UI**: Modern Streamlit interface with responsive design

---

## 🚀 Quick Start

### Live Application
Simply visit: **[https://insapp-j8sxgwpkhsswzxqsargwch.streamlit.app/](https://insapp-j8sxgwpkhsswzxqsargwch.streamlit.app/)**

### Local Installation

```bash
# Clone repository
git clone https://github.com/Ragunathan28/MED_APP.git
cd MED_APP

# Install dependencies
pip install -r app/requirements.txt

# Run app
streamlit run app/app.py