# 🎬 CineMatch AI — Movie Recommendation System

An intelligent, high-performance Movie Recommendation System powered by **FAISS vector indexing**, **Sentence Transformers (MiniLM)**, and **TMDB metadata** spanning over **187,000+ movies**. Built with an interactive, cinematic dark-mode **Streamlit** user interface.

---

## 🌟 Key Features

- **🎯 Dual Recommendation Engines**:
  - **Movie-to-Movie Vector Similarity ("More Like This")**: Select or search any film from the 187k+ catalog or use one-click quick picks (*Interstellar*, *Inception*, *The Dark Knight*, *Fight Club*, etc.) to discover nearest neighbor recommendations in 384-dimensional embedding space.
  - **Natural Language Semantic Search ("Plot & Vibe Search")**: Type free-form plot descriptions, moods, or themes (e.g., *"rainy neo-noir cyberpunk detective in futuristic city"* or *"mind-bending dream heist"*). The app encodes your query in real-time with `all-MiniLM-L6-v2` and queries FAISS instantaneously.
- **🖼️ Rich Movie Cards & Live TMDB Posters**:
  - Fetches real-time posters directly from TMDB's CDN (`https://image.tmdb.org/t/p/w500`).
  - Built-in zero-dependency SVG fallback placeholders for missing posters.
  - Interactive expanders for full movie synopses and direct TMDB page links.
- **🎛️ Interactive Filters & Controls**:
  - Dynamic slider for number of recommendations ($4$ to $20$).
  - Minimum rating threshold filter (⭐ $0.0$ - $9.0+$).
  - Multi-select genre filtering (Action, Sci-Fi, Thriller, Animation, Drama, etc.).
- **⚡ High Performance & Low Memory**:
  - Uses a compressed **31 MB Parquet** dataset with PyArrow for sub-millisecond metadata lookups.
  - Employs **memory-mapped NumPy embeddings** (`mmap_mode="r"`) to minimize RAM consumption.
  - Vector similarity search via **FAISS** (`IndexFlatIP`).

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend / Web UI** | [Streamlit](https://streamlit.io/) |
| **Vector Search Engine** | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) |
| **NLP Embeddings** | [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`, 384 dimensions) |
| **Data Engine & Formats** | [Pandas](https://pandas.pydata.org/), [PyArrow](https://arrow.apache.org/docs/python/) (Parquet), [NumPy](https://numpy.org/) |
| **Package Management** | [uv](https://github.com/astral-sh/uv) |
| **Python Version** | Python `>= 3.12` |

---

## 📁 Project Structure

```
MRS/
├── app.py                          # Main Streamlit web application
├── pyproject.toml                  # Project metadata and dependencies
├── uv.lock                         # Locked dependency versions
├── README.md                       # Documentation
├── movie_recommender_files/        # Data artifacts
│   ├── movies.parquet              # Cleaned TMDB catalog (187,501 movies)
│   ├── movies.faiss                # FAISS vector similarity index
│   ├── movie_embeddings.npy        # 384-d precomputed dense embeddings
│   └── movies_meta.pkl             # Bidirectional index & title mappings
└── src/
    └── mrs/
        └── __init__.py
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.12+** and [uv](https://docs.astral.sh/uv/) installed:
```powershell
# Install uv (if not already installed)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Installation
Clone the repository and install dependencies using `uv`:
```powershell
git clone <your-repository-url>
cd MRS
uv sync
```

*(Alternatively, with standard pip: `pip install -r pyproject.toml` or `pip install streamlit faiss-cpu sentence-transformers pandas pyarrow numpy`)*

### 3. Running the Application
Launch the Streamlit dashboard:
```powershell
uv run streamlit run app.py
```

Once started, open your web browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 💡 How to Use

### 1. "More Like This" (Movie Similarity)
1. Pick a popular movie from the **Quick Picks** buttons or search for any movie in the dropdown.
2. The **Hero Spotlight** will display the chosen movie's poster, release year, rating, genres, and synopsis.
3. Browse the recommended movies below, each showing match similarity score (`🎯 % Match`), star ratings, genres, and an expandable synopsis.

### 2. "Plot & Vibe Search" (Semantic NLP)
1. Switch to the **🔮 Plot & Vibe Search** tab.
2. Click any of the prompt inspiration presets or type your own custom scenario into the text box (e.g., *"solitary astronaut stranded on an alien planet"*).
3. Click **🚀 Search Semantic Vectors** to retrieve semantically matching movies.

### 3. Adjusting Filters
Use the **Sidebar** to:
- Adjust recommendation count ($4$ to $20$).
- Filter out movies below a certain rating (e.g., ⭐ $7.0+$).
- Filter recommendations by one or more genres.

---

## 🔑 API Keys & Configuration (Optional)

> [!NOTE]
> **No API key is required** to run the app! Movie posters load directly via TMDB's public CDN, and embeddings/vector searches execute completely on your local machine.

If you wish to configure optional credentials (such as a TMDB API Key for live trailers or a Hugging Face token):
1. Create a `.streamlit/secrets.toml` file:
   ```toml
   TMDB_API_KEY = "your_tmdb_api_key_here"
   HF_TOKEN = "your_huggingface_token_here"
   ```
2. Access them within Streamlit:
   ```python
   import streamlit as st
   tmdb_key = st.secrets.get("TMDB_API_KEY", "")
   ```

---

## 👤 Author
- **Moksh Shah**
