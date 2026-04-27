import { useState } from "react";
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("mood");
  const [results, setResults] = useState([]);

  async function getRecommendations() {
    if (!query.trim()) return;

    const endpoint =
      mode === "song"
        ? "http://localhost:8000/recommend/song"
        : "http://localhost:8000/recommend/mood";

    const response = await fetch(`${endpoint}?query=${encodeURIComponent(query)}`);
    const data = await response.json();

    setResults(data.results || []);
  }

  return (
    <div className="app">
      <h1>Spotify Song Recommender</h1>
      <p>Search by song name or mood.</p>

      <div className="controls">
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="mood">Mood</option>
          <option value="song">Song</option>
        </select>

        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={mode === "mood" ? "sad, high energy, chill acoustic" : "Comedy"}
        />

        <button onClick={getRecommendations}>Recommend</button>
      </div>

      <div className="results">
        {results.map((song, index) => (
          <div className="card" key={index}>
            <h3>{song.track_name}</h3>
            <p>{song.artists}</p>
            <p className="genre">{song.track_genre}</p>
            <div className="stats">
              <span>Energy: {song.energy}</span>
              <span>Valence: {song.valence}</span>
              <span>Danceability: {song.danceability}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;