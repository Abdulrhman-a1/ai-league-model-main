import streamlit as st
from PIL import Image
from predictor import predict_density
from utils import render_density_map

st.set_page_config(layout="wide")
st.title("Crowd Counter - CSRNet")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    count, density_map = predict_density(image)
    heatmap = render_density_map(density_map)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(heatmap, caption=f"Density Map - Count: {count}", use_container_width=True)
