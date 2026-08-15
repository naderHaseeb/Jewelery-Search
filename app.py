import streamlit as st
import numpy as np
import pickle
import zipfile
import os

from PIL import Image
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import img_to_array


ZIP_PATH = "archive (4).zip"
EXTRACT_PATH = "data"

if not os.path.exists("data/Jewellery_Data"):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)


@st.cache_resource
def load_model():
    return MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )


model = load_model()


with open("data/product_metadata_500.pkl", "rb") as f:
    payload = pickle.load(f)

feature_list = payload["features"]
old_image_paths = payload["paths"]


image_paths = []

for old_path in old_image_paths:

    filename = os.path.basename(old_path)

    necklace_path = os.path.join(
        "data",
        "Jewellery_Data",
        "necklace",
        filename
    )

    ring_path = os.path.join(
        "data",
        "Jewellery_Data",
        "ring",
        filename
    )

    if os.path.exists(necklace_path):
        image_paths.append(necklace_path)

    elif os.path.exists(ring_path):
        image_paths.append(ring_path)

    else:
        image_paths.append(None)


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

    embedding = model.predict(
        img_array,
        verbose=0
    )

    return embedding


st.title("Jewelry Visual Search")

uploaded_file = st.file_uploader(
    "Upload a jewelry image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    query_img = Image.open(uploaded_file).convert("RGB")

    st.image(query_img, width=300)

    query_embedding = extract_embedding(query_img)

    distances, indices = neighbors.kneighbors(query_embedding)

    st.subheader("Top Similar Jewelry")

    cols = st.columns(5)

    valid_results = 0

    for i, idx in enumerate(indices[0]):

        image_path = image_paths[idx]

        if image_path is None:
            continue

        similarity = 1 - distances[0][i]

        if similarity < 0.30:
            continue

        with cols[valid_results % 5]:

            st.image(
                image_path,
                use_container_width=True
            )

            st.caption(
                f"Similarity: {similarity:.2f}"
            )

        valid_results += 1

    if valid_results == 0:
        st.warning("No sufficiently similar jewelry was found.")
