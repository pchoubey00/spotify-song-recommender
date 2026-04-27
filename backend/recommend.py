import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SongAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return reconstructed


# 1. Load data
df = pd.read_csv("spotify-tracks-dataset.csv")
df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], errors="ignore")
df = df.dropna(subset=["track_id", "track_name", "artists", "track_genre"])

numeric_features = [
    "popularity", "duration_ms", "danceability", "energy", "key",
    "loudness", "mode", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "time_signature",
]

categorical_features = ["track_genre", "explicit"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X = preprocessor.fit_transform(df[numeric_features + categorical_features])

if hasattr(X, "toarray"):
    X = X.toarray()

X_tensor = torch.tensor(X, dtype=torch.float32)

# 2. Load trained model
input_dim = X_tensor.shape[1]
model = SongAutoencoder(input_dim=input_dim, embedding_dim=32)
model.load_state_dict(torch.load("song_autoencoder.pt"))
model.eval()

# 3. Generate song embeddings
with torch.no_grad():
    song_embeddings = model.encoder(X_tensor).numpy()


def recommend_songs(song_name, top_k=10, return_results=False):
    matches = df[df["track_name"].str.lower() == song_name.lower()]

    if matches.empty:
        print(f"No exact match found for: {song_name}")
        return

    song_index = matches.index[0]

    query_embedding = song_embeddings[song_index].reshape(1, -1)

    similarities = cosine_similarity(query_embedding, song_embeddings)[0]

    similar_indices = similarities.argsort()[::-1][1:top_k + 1]

    print(f"\nSongs similar to: {df.loc[song_index, 'track_name']} by {df.loc[song_index, 'artists']}\n")

    results = df.iloc[similar_indices][
        ["track_name", "artists", "track_genre", "popularity", "danceability", "energy", "valence"]
    ]
    if return_results:
        return results.to_dict(orient="records")
    print(results.to_string(index=False))


def recommend_by_mood(query, top_k=10, return_results=False):
    query = query.lower()

    mood_profile = {
        "danceability": 0.5,
        "energy": 0.5,
        "valence": 0.5,
        "acousticness": 0.5,
        "instrumentalness": 0.2,
        "speechiness": 0.1,
        "liveness": 0.2,
        "tempo": 120,
        "popularity": 60,
        "duration_ms": df["duration_ms"].median(),
        "key": df["key"].median(),
        "loudness": df["loudness"].median(),
        "mode": df["mode"].median(),
        "time_signature": df["time_signature"].median(),
    }

    if "sad" in query:
        mood_profile["valence"] = 0.15
        mood_profile["energy"] = 0.35
        mood_profile["danceability"] = 0.35

    if "happy" in query:
        mood_profile["valence"] = 0.85
        mood_profile["energy"] = 0.65
        mood_profile["danceability"] = 0.65

    if "high energy" in query or "workout" in query:
        mood_profile["energy"] = 0.9
        mood_profile["tempo"] = 145
        mood_profile["danceability"] = 0.75
        mood_profile["valence"] = 0.65

    if "chill" in query:
        mood_profile["energy"] = 0.25
        mood_profile["tempo"] = 90
        mood_profile["danceability"] = 0.35
        mood_profile["acousticness"] = 0.65

    if "acoustic" in query:
        mood_profile["acousticness"] = 0.9
        mood_profile["energy"] = min(mood_profile["energy"], 0.45)
        mood_profile["instrumentalness"] = 0.1

    genre_match = None
    for genre in df["track_genre"].unique():
        if genre in query:
            genre_match = genre
            break

    query_row = pd.DataFrame([{**mood_profile, "track_genre": genre_match or "pop", "explicit": False}])

    query_vector = preprocessor.transform(query_row[numeric_features + categorical_features])

    if hasattr(query_vector, "toarray"):
        query_vector = query_vector.toarray()

    query_tensor = torch.tensor(query_vector, dtype=torch.float32)

    with torch.no_grad():
        query_embedding = model.encoder(query_tensor).numpy()

    similarities = cosine_similarity(query_embedding, song_embeddings)[0]
    similar_indices = similarities.argsort()[::-1][:top_k]

    print(f"\nSongs matching mood: {query}\n")

    results = df.iloc[similar_indices][
        ["track_name", "artists", "track_genre", "popularity", "danceability", "energy", "valence", "acousticness", "tempo"]
    ]
    if return_results:
        return results.to_dict(orient="records")


    print(results.to_string(index=False))
# Try it
#while True:
    #user_input = input("\nEnter a song name or mood query, or type 'quit' to exit: ")

    #if user_input.lower() == "quit":
    #    print("Goodbye!")
    #    break

    #mode = input("Search by song or mood? Type 'song' or 'mood': ").lower()

    #if mode == "song":
    #    recommend_songs(user_input, top_k=10)
    #elif mode == "mood":
    #    recommend_by_mood(user_input, top_k=10)
    #else:
    #    print("Please type either 'song' or 'mood'.")