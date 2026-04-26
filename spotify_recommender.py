import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("spotify-tracks-dataset.csv")

# 2. Drop useless index columns
df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], errors="ignore")

# 3. Drop rows missing important display info
df = df.dropna(subset=["track_id", "track_name", "artists", "track_genre"])

# 4. Choose features for the recommender
numeric_features = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

categorical_features = [
    "track_genre",
    "explicit",
]

# 5. Preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X = preprocessor.fit_transform(df[numeric_features + categorical_features])

# Convert sparse matrix to dense if needed
if hasattr(X, "toarray"):
    X = X.toarray()

# 6. Convert to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32)

print("Dataset shape:", df.shape)
print("PyTorch tensor shape:", X_tensor.shape)
print("Example song:", df.iloc[0][["track_name", "artists", "track_genure" if False else "track_genre"]])

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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


# 7. Create DataLoader
dataset = TensorDataset(X_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# 8. Initialize model
input_dim = X_tensor.shape[1]
model = SongAutoencoder(input_dim=input_dim, embedding_dim=32)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 9. Train model
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for batch in loader:
        x_batch = batch[0]

        reconstructed = model(x_batch)
        loss = loss_fn(reconstructed, x_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# 10. Save model
torch.save(model.state_dict(), "song_autoencoder.pt")
print("Model saved as song_autoencoder.pt")