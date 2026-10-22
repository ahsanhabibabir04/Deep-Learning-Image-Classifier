
import streamlit as st
from PIL import Image

from src.predict import predict_image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered",
)

st.title("Cat vs Dog Image Classifier")
st.write(
    "Upload an image and let the deep learning model "
    "predict whether it contains a cat or a dog."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing image..."):
            result = predict_image(image)

        st.subheader(f"Prediction: {result['label']}")
        st.metric(
            "Model confidence",
            f"{result['confidence']:.2%}",
        )
        st.caption(
            "This is a model prediction, not a guarantee. "
            "Results may vary on unfamiliar images."
        )
