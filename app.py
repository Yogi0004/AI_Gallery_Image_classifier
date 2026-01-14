import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

# ====================== Page Config ======================
st.set_page_config(page_title="Gallery Image Detector", layout="centered")
st.title("🏛️ Gallery Image Detector")
#st.markdown("Upload an image to detect if it's a **gallery floor plan, section, or elevation drawing**")

# ====================== Model Definition (MUST MATCH train.py EXACTLY) ======================
class GalleryImageClassifier(nn.Module):
    """Exact same model class as in train.py"""
    def __init__(self, num_classes=2):
        super(GalleryImageClassifier, self).__init__()
        
        # Load pretrained ResNet-18 backbone
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        num_features = self.resnet.fc.in_features
        
        # Replace classifier head
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

# ====================== Model Loading ======================
@st.cache_resource
def load_model():
    model = GalleryImageClassifier(num_classes=2)

    # Possible saved model paths
    possible_paths = [
        "models/best_model.pth",
        "models/final_model.pth",
        "best_model.pth",
        "final_model.pth",
        "gallery_model.pth"
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        st.error("🚨 **Trained model not found!**")
        st.markdown("""
        ### Please fix this:
        1. Run `python train.py` and let it complete all epochs
        2. It will create `models/best_model.pth` and `models/final_model.pth`
        3. Then refresh/re-run this app
        
        Current directory files:
        """)
        try:
            files = os.listdir(".")
            st.code("\n".join(files))
            if os.path.exists("models"):
                st.code("models/ folder: " + "\n".join(os.listdir("models")))
        except:
            pass
        st.stop()

    st.info(f"✅ Loading model from: `{model_path}`")

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model()

# ====================== Transform ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====================== Prediction ======================
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Processing..."):
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                confidence, pred_idx = torch.max(probabilities, 0)

            confidence_pct = confidence.item() * 100

            if pred_idx.item() == 1:
                st.success("✅ **YES – This is a Gallery Technical Drawing**")
                st.balloons()
            else:
                st.warning("❌ **NO – This is not a Gallery Technical Drawing**")

            st.markdown(f"**Confidence: {confidence_pct:.1f}%**")

            st.progress(probabilities[1].item())
            st.write(f"**Gallery Probability:** {probabilities[1].item()*100:.2f}%")
            st.write(f"**Non-Gallery Probability:** {probabilities[0].item()*100:.2f}%")

            st.markdown("### Explanation:")
            if pred_idx.item() == 1:
                st.info("The image shows typical features of architectural gallery drawings: geometric lines, structured layout, minimal color, technical style.")
            else:
                st.info("The image appears to be a photograph, 3D render, artwork, or unrelated content without technical drawing characteristics.")

# ====================== Sidebar ======================
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
Detects **architectural technical drawings** (floor plans, sections, elevations) of art galleries/museums.

**Gallery examples:** Black-and-white line drawings with geometry and labels  
**Non-gallery:** Photos, 3D renders, paintings, landscapes
""")

st.sidebar.header("🛠️ Setup Steps")
st.sidebar.markdown("""
1. Run `python train.py` → wait for completion  
2. Model saves to `models/best_model.pth`  
3. Then run `streamlit run app.py`
""")