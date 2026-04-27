from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from recommend import recommend_songs, recommend_by_mood

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend/song")
def song_recommendation(query: str):
    results = recommend_songs(query, top_k=10, return_results=True)
    return {"results": results}

@app.get("/recommend/mood")
def mood_recommendation(query: str):
    results = recommend_by_mood(query, top_k=10, return_results=True)
    return {"results": results}