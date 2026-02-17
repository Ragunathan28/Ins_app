import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import os
import gdown

# Default classes
CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'clothes', 'green-glass', 'metal', 'paper',
    'plastic', 'shoes', 'trash', 'white-glass'
]

# Google Drive file ID (replace with your actual file ID)
# Get this from your Google Drive share link
# Example: https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I0J/view?usp=sharing
# File ID: 1A2B3C4D5E6F7G8H9I0J
GDRIVE_FILE_ID = "1AGJNTCTx406d14nmbsgs1rxPF1829hoy"  # ← REPLACE THIS!

# Local path (works on both Windows and Linux)
MODEL_PATH = r"C:\Users\Ragu\garbage app\best_garbage_model.pth"

# Page config
st.set_page_config(
    page_title="♻️ Smart Garbage Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    .confidence-bar-bg {
        height: 30px;
        background: rgba(0,0,0,0.2);
        border-radius: 15px;
        overflow: hidden;
        position: relative;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 15px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .eco-tip {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Model class definition
class GarbageClassifier(nn.Module):
    def __init__(self, num_classes=12, pretrained=False):
        super(GarbageClassifier, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def download_model():
    """Download model from Google Drive if not exists"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model... This may take a minute (~18MB)"):
            try:
                url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success("✅ Model downloaded successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                return False
    return True

@st.cache_resource
def load_model():
    """Load model (download if needed)"""
    device = torch.device('cpu')
    
    # Download if not exists (for Streamlit Cloud)
    if not os.path.exists(MODEL_PATH):
        if not download_model():
            return None, CLASSES, False
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = GarbageClassifier(num_classes=12, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        classes = checkpoint.get('classes', CLASSES)
        return model, classes, True
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, CLASSES, False

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def predict_image(model, image, classes):
    transform = get_transforms()
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    probs = probabilities.squeeze().numpy()
    pred_class = classes[predicted.item()]
    confidence_score = confidence.item() * 100
    
    return pred_class, confidence_score, probs

def get_class_info(class_name):
    info = {
        'battery': {
            'color': '#e74c3c', 'icon': '🔋', 'category': 'Hazardous Waste',
            'instructions': 'Take to designated battery recycling points. Never throw in regular trash!',
            'tip': 'Batteries contain heavy metals that can contaminate soil and water.'
        },
        'biological': {
            'color': '#795548', 'icon': '🍎', 'category': 'Organic Waste',
            'instructions': 'Compost if possible. Use biodegradable bags for collection.',
            'tip': 'Organic waste can be turned into nutrient-rich compost for gardens.'
        },
        'brown-glass': {
            'color': '#8d6e63', 'icon': '🍺', 'category': 'Recyclable',
            'instructions': 'Rinse and place in glass recycling bin. Remove caps and lids.',
            'tip': 'Brown glass protects contents from UV light - great for beer and medicine bottles!'
        },
        'cardboard': {
            'color': '#d4a373', 'icon': '📦', 'category': 'Recyclable',
            'instructions': 'Flatten boxes to save space. Keep dry and clean.',
            'tip': 'Recycling cardboard saves 25% of the energy needed to make new cardboard.'
        },
        'clothes': {
            'color': '#e91e63', 'icon': '👕', 'category': 'Reusable/Donate',
            'instructions': 'Donate if in good condition. Otherwise, textile recycling.',
            'tip': 'The fashion industry produces 10% of global carbon emissions.'
        },
        'green-glass': {
            'color': '#4caf50', 'icon': '🍾', 'category': 'Recyclable',
            'instructions': 'Rinse thoroughly. Remove corks and caps before recycling.',
            'tip': 'Green glass is often used for wine bottles and can be recycled infinitely.'
        },
        'metal': {
            'color': '#607d8b', 'icon': '🥫', 'category': 'Recyclable',
            'instructions': 'Rinse cans. Crush to save space. Remove labels if possible.',
            'tip': 'Aluminum cans can be recycled and back on shelves in just 60 days!'
        },
        'paper': {
            'color': '#ffeb3b', 'icon': '📄', 'category': 'Recyclable',
            'instructions': 'Keep dry and clean. Remove plastic windows from envelopes.',
            'tip': 'Recycling one ton of paper saves 17 trees and 7,000 gallons of water.'
        },
        'plastic': {
            'color': '#ff9800', 'icon': '🥤', 'category': 'Recyclable',
            'instructions': 'Check recycling number. Rinse containers. Crush bottles.',
            'tip': 'Only 9% of all plastic ever made has been recycled. Do your part!'
        },
        'shoes': {
            'color': '#9c27b0', 'icon': '👟', 'category': 'Donate/Reuse',
            'instructions': 'Donate to charity if wearable. Some brands offer recycling programs.',
            'tip': 'Shoes can take 30-40 years to decompose in landfills.'
        },
        'trash': {
            'color': '#616161', 'icon': '🗑️', 'category': 'Landfill',
            'instructions': 'Last resort. Ensure no recyclable materials are mixed in.',
            'tip': 'Aim for zero waste - reduce and reuse before throwing away.'
        },
        'white-glass': {
            'color': '#f5f5f5', 'icon': '🥛', 'category': 'Recyclable',
            'instructions': 'Rinse well. Remove lids and caps. Keep separate from colored glass.',
            'tip': 'Clear glass is the most valuable for recycling as it can be made into any color.'
        }
    }
    return info.get(class_name, {
        'color': '#9e9e9e', 'icon': '❓', 'category': 'Unknown',
        'instructions': 'Please check local guidelines.',
        'tip': 'When in doubt, check with your local waste management authority.'
    })

def get_bin_color(category):
    bins = {
        'Recyclable': '#2196f3', 'Organic Waste': '#4caf50',
        'Hazardous Waste': '#f44336', 'Reusable/Donate': '#ff9800',
        'Donate/Reuse': '#ff9800', 'Landfill': '#616161', 'Unknown': '#9e9e9e'
    }
    return bins.get(category, '#9e9e9e')

def main():
    st.markdown('<h1 class="main-header">♻️ Smart Garbage Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">AI-Powered Waste Sorting for a Cleaner Planet 🌍</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ App Info")
        st.write(f"Model: `{MODEL_PATH}`")
        st.write(f"Exists locally: `{os.path.exists(MODEL_PATH)}`")
        
        st.divider()
        st.header("🌍 Environmental Impact")
        facts = [
            ("🌳", "17 trees saved per ton of paper"),
            ("💧", "7,000 gallons of water saved"),
            ("⚡", "95% less energy for aluminum"),
            ("🕒", "60 days for can to return to shelf"),
            ("♾️", "Glass recycles infinitely")
        ]
        for icon, fact in facts:
            st.write(f"{icon} {fact}")
        
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
            st.rerun()
    
    # Load model
    model, classes, loaded = load_model()
    
    if not loaded:
        st.error("⚠️ Model failed to load!")
        st.info("Please check your Google Drive file ID is correct.")
        st.stop()
    
    st.sidebar.success(f"✅ Loaded: {len(classes)} classes")
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📸 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Drop your waste item image here...",
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        
        camera_input = st.camera_input("Or take a photo now 📷")
        
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
        elif camera_input is not None:
            image = Image.open(camera_input).convert('RGB')
        
        if image is not None:
            st.image(image, caption="Your Image", use_container_width=True)
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("Width", f"{image.size[0]}px")
            with cols[1]:
                st.metric("Height", f"{image.size[1]}px")
            with cols[2]:
                st.metric("Mode", image.mode)
    
    with col2:
        if image is not None:
            st.subheader("🤖 AI Analysis")
            
            with st.spinner("🔍 Analyzing waste type..."):
                pred_class, confidence, all_probs = predict_image(model, image, classes)
            
            info = get_class_info(pred_class)
            
            st.markdown(f"""
            <div class="prediction-card" style="background: linear-gradient(135deg, {info['color']} 0%, {info['color']}dd 100%);">
                <div style="font-size: 4rem; margin-bottom: 0.5rem;">{info['icon']}</div>
                <h2 style="margin: 0; font-size: 2rem; text-transform: uppercase; letter-spacing: 2px;">{pred_class.replace('-', ' ')}</h2>
                <div style="font-size: 3rem; font-weight: bold; margin: 1rem 0;">{confidence:.1f}%</div>
                <div style="font-size: 1.1rem; opacity: 0.9;">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
            bin_color = get_bin_color(info['category'])
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <span style="background: {bin_color}; color: white; padding: 0.5rem 1.5rem; 
                border-radius: 25px; font-weight: bold; font-size: 1.1rem;">
                    {info['category']}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📋 Disposal Instructions")
            st.info(info['instructions'])
            
            st.markdown(f"""
            <div class="eco-tip">
                <strong>💡 Did you know?</strong><br>
                {info['tip']}
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📊 Detailed Confidence Breakdown"):
                sorted_indices = np.argsort(all_probs)[::-1]
                
                for idx in sorted_indices[:5]:
                    cls = classes[idx]
                    prob = all_probs[idx] * 100
                    cls_info = get_class_info(cls)
                    
                    cols = st.columns([1, 4, 1])
                    with cols[0]:
                        st.write(f"{cls_info['icon']} {cls.replace('-', ' ')}")
                    with cols[1]:
                        st.markdown(f"""
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width: {prob}%; background: {cls_info['color']}; color: white;">
                                {prob:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with cols[2]:
                        st.write(f"{prob:.1f}%")
        else:
            st.info("👈 Upload an image or take a photo to classify your waste!")
            
            st.subheader("🗂️ Supported Categories")
            example_items = [
                ("🔋", "Battery", "#e74c3c"), ("🍎", "Biological", "#795548"),
                ("📦", "Cardboard", "#d4a373"), ("🥫", "Metal", "#607d8b"),
                ("📄", "Paper", "#ffeb3b"), ("🥤", "Plastic", "#ff9800"),
                ("🍾", "Glass", "#4caf50"), ("👕", "Clothes", "#e91e63")
            ]
            
            cols = st.columns(4)
            for i, (icon, name, color) in enumerate(example_items):
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: {color}20; 
                    border-radius: 10px; border: 2px solid {color}; margin: 0.5rem 0;">
                        <div style="font-size: 2rem;">{icon}</div>
                        <div style="font-weight: bold; color: {color};">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()