# Jewelry Visual Search Engine

## Overview

This project is a visual similarity search engine built using Deep Learning and Streamlit. Instead of classifying jewelry into categories, the system retrieves visually similar jewelry items based on an uploaded image.

A pretrained MobileNetV2 model is used to extract feature embeddings from jewelry images. These embeddings are then compared using Nearest Neighbors with cosine similarity to find the most similar products in the dataset.

---

## Features

- Upload a jewelry image
- Extract image embeddings using MobileNetV2
- Search for the 25 most visually similar jewelry items
- Display similarity scores for each result
- Filter out low-similarity matches using a similarity threshold
- Simple web interface built with Streamlit

---

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- Scikit-learn
- Streamlit
- NumPy
- Pillow

---

## Project Structure

```
jewelry-search/
│
├── app.py
├── requirements.txt
├── README.md
│
└── data/
    ├── archive (4).zip
    └── product_metadata_500.pkl
```

---

## How It Works

1. A user uploads a jewelry image.
2. The image is resized and preprocessed.
3. MobileNetV2 extracts a feature embedding.
4. The embedding is compared with the precomputed database embeddings.
5. Nearest Neighbors retrieves the most similar jewelry items.
6. The results are displayed together with their similarity scores.

---

## Model

The project uses a pretrained **MobileNetV2** model trained on the ImageNet dataset. The model is used only for feature extraction through transfer learning. No additional model training or fine-tuning was performed.

---

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Future Improvements

- Fine-tune the model using a jewelry-specific dataset.
- Replace MobileNetV2 with a stronger backbone such as Vision Transformer.
- Add filtering by jewelry type or category.
- Improve retrieval quality using metric learning techniques.
