# Spotify Song Recommender using PyTorch

This project builds a content-based music recommender using PyTorch.

## Features
- Learns song embeddings using an autoencoder
- Recommends similar songs using cosine similarity
- Supports:
  - Song-based queries
  - Mood-based queries (e.g., "sad", "high energy", "chill acoustic")

## Tech Stack
- Python
- PyTorch
- pandas, scikit-learn
- NumPy

## How it works
1. Spotify dataset(from here: https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset/code) is preprocessed (audio + metadata features).
2. Autoencoder learns compressed song embeddings. 
3. Similar songs are retrieved using cosine similarity in embedding space. 

## Run locally
Train model:
bash
python spotify_recommender.py

Run recommender:
bash
python recommend.py

## Example Queries
song → You belong with me
mood → sad
mood → high energy
mood → chill acoustic

## Notes
Dataset not included due to size.