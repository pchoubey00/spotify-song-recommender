# Spotify Song Recommender (PyTorch + React)

Full-stack music recommender system using PyTorch for embeddings and React for UI.

## Features

- Song-based recommendations
- Mood-based recommendations ("sad", "high energy", "chill acoustic")
- Learned embeddings using PyTorch autoencoder
- FastAPI backend
- React frontend

## Tech Stack

Backend:
- Python
- PyTorch
- FastAPI
- scikit-learn

Frontend:
- React (Vite)
- JavaScript

## How it works

1. Spotify dataset is preprocessed (audio + metadata features)
2. Autoencoder learns 32-dimensional song embeddings
3. Recommendations are generated using cosine similarity

## Run locally(separate terminals for frontend and backend)

### backend from terminal
cd backend
pip install -r ../requirements.txt
uvicorn app:app --reload

### frontend from terminal
cd frontend
npm install
npm run dev

### Open from browser
http://localhost:5173

### Example Queries

- Song: Comedy
- Mood: sad
- Mood: high energy
- Mood: chill acoustic



