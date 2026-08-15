import streamlit as st
import numpy as np
import pickle

from PIL import Image
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array


@st.cache_resource
def load_model():
    return MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )


model = load_model()


with open("data/product_metadata.pkl", "rb") as f:
    payload = pickle.load(f)

feature_list = payload["features"]
image_paths = payload["paths"]


neighbors = NearestNeighbors(
    n_neighbors=25,
    algorithm="brute",
    metric="cosine"
)

neighbors.fit(feature_list)


def extract_embedding(img):
    img = img.resize((224, 224))

    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    embedding = model.predict(img_array, verbose=0)

    return embedding


st.title("Jewelry Visual Search")

uploaded_file = st.file_uploader(
    "Upload a jewelry image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    query_img = Image.open(uploaded_file).convert("RGB")

    st.image(query_img, caption="Query Image")

    query_embedding = extract_embedding(query_img)

    distances, indices = neighbors.kneighbors(query_embedding)

    st.subheader("Similar Jewelry")

    cols = st.columns(5)

    for i, idx in enumerate(indices[0]):

        with cols[i % 5]:
            st.image(
                image_paths[idx],
                caption=f"Distance: {distances[0][i]:.2f}"
            )
