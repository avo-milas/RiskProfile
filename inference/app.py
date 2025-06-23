import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from PIL import Image

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    body, html, .css-18e3th9 {
        font-size: 20px !important;
    }
    h1 {
        font-size: 3rem !important;
    }
    h2 {
        font-size: 2.5rem !important;
    }
    label, .stSlider label, .stSelectbox label, .stButton button, .stTextInput label {
        font-size: 22px !important;
    }
    div[data-baseweb="select"] > div {
        font-size: 20px !important;
    }
    .stNumberInput input {
        font-size: 20px !important;
    }
    div[data-testid="stImage"] > figcaption {
        font-size: 1.2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('label_mappings.json', 'r') as f:
    label_mappings = json.load(f)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


num_text_features = 3 * 384
num_numeric_features = 1
num_categorical_features = 4
input_dim = num_text_features + num_numeric_features + num_categorical_features

model = MLPRegressor(input_dim=input_dim)
model.load_state_dict(torch.load('mlp_model.pth', map_location='cpu'))
model.eval()


st.title("Investor Risk Score Estimator")
st.markdown("---")
st.subheader("Select 3 images in order of your preference:")

image_ids = list(range(1, 17))
cols = st.columns(4)
selection = {}

for i, img_id in enumerate(image_ids):
    col = cols[i % 4]
    image_path = f"../images/{img_id}.png"
    image = Image.open(image_path)
    col.image(image, caption=f"ID {img_id}", use_container_width=True)
    selection[img_id] = col.checkbox(f"Select ID {img_id}")

selected = [img_id for img_id, chosen in selection.items() if chosen]

if len(selected) < 3:
    st.info(f"Please select {3 - len(selected)} more image(s).")
    st.stop()
elif len(selected) > 3:
    st.error("You have selected more than 3 images. Please select exactly 3.")
    st.stop()
else:
    st.success("✅ 3 images selected. You can proceed!")

st.subheader("Set preference order for selected images:")
pref_1 = st.selectbox("Preference 1", selected, key="p1")
remaining_2 = [x for x in selected if x != pref_1]
pref_2 = st.selectbox("Preference 2", remaining_2, key="p2")
pref_3 = [x for x in remaining_2 if x != pref_2][0]
st.write(f"Preference 3: {pref_3}")

st.header("Sociodemographic Information:")

age = st.slider("Age", 18, 100, 30)

gender = st.selectbox("Gender", list(label_mappings['gender'].keys()))
education = st.selectbox("Education", list(label_mappings['education'].keys()))
income = st.selectbox("Income", list(label_mappings['income'].keys()))
marital_status = st.selectbox("Marital Status", list(label_mappings['marital_status'].keys()))

emb1 = embeddings[pref_1]
emb2 = embeddings[pref_2]
emb3 = embeddings[pref_3]

emb_features = np.hstack([emb1, emb2, emb3])

numeric_features = np.array([age])

categorical_features = np.array([
    label_mappings['gender'][gender],
    label_mappings['education'][education],
    label_mappings['income'][income],
    label_mappings['marital_status'][marital_status]
])

final_input = np.hstack([emb_features, numeric_features, categorical_features]).astype(np.float32)
final_tensor = torch.tensor(final_input).unsqueeze(0)

if st.button("Calculate Risk Score"):
    with torch.no_grad():
        prediction = model(final_tensor).item()
    st.success(f"Predicted Risk Score: {prediction:.3f}")
