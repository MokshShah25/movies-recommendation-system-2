import urllib.parse
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch AI | Smart Movie Recommender",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS FOR SLEEK CINEMATIC DARK UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .hero-banner {
        padding: 2rem 2.2rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        margin-bottom: 1.8rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f43f5e 0%, #fb7185 30%, #a855f7 70%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0;
        max-width: 800px;
    }

    .badge-score {
        display: inline-block;
        background: rgba(16, 185, 129, 0.18);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .badge-rating {
        display: inline-block;
        background: rgba(234, 179, 8, 0.18);
        color: #facc15;
        border: 1px solid rgba(234, 179, 8, 0.35);
        border-radius: 6px;
        padding: 2px 7px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .badge-genre {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 9999px;
        padding: 1px 7px;
        font-size: 0.68rem;
        margin: 2px 2px 2px 0;
    }

    .hero-spotlight {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(244, 63, 94, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    /* Streamlit widget tweaks */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# RESOURCE & DATA CACHING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading FAISS index...")
def load_vector_store():
    index = faiss.read_index("movie_recommender_files/movies.faiss")
    return index

@st.cache_resource(show_spinner="⏳ Loading Sentence Transformer NLP Model (all-MiniLM-L6-v2)...")
def load_transformer_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner="⏳ Indexing movie catalog...")
def load_movies_catalog():
    # Strictly load only the columns used in the UI to save ~80MB+ of RAM
    columns_to_load = [
        "id", "title", "release_date", "vote_average", 
        "vote_count", "genres", "poster_path", "overview"
    ]
    df = pd.read_parquet("movie_recommender_files/movies.parquet", columns=columns_to_load)
    df["title"] = df["title"].fillna("Unknown Title")
    
    # Extract release year
    df["year"] = df["release_date"].apply(
        lambda d: str(d)[:4] if pd.notna(d) and len(str(d)) >= 4 else ""
    )
    
    # Pre-build display labels: "Title (Year)"
    df["display_title"] = df.apply(
        lambda r: f"{r['title']} ({r['year']})" if r['year'] else r['title'], axis=1
    )
    
    # Dictionary lookup for fast index resolution
    id_to_index = {int(mid): idx for idx, mid in enumerate(df["id"])}
    
    # Extract unique genres safely
    all_genres = set()
    for g in df["genres"].dropna():
        for item in g.split(","):
            item = item.strip()
            if item:
                all_genres.add(item)
                
    return df, id_to_index, sorted(list(all_genres))

# Initialize Data & Models
index = load_vector_store()
model = load_transformer_model()
df, id_to_index, available_genres = load_movies_catalog()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_poster_url(poster_path: str, title: str) -> str:
    """Returns official TMDB image URL or a styled SVG fallback data URI."""
    if pd.notna(poster_path) and str(poster_path).strip():
        return f"https://image.tmdb.org/t/p/w500{poster_path.strip()}"
    
    clean_title = str(title).replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
    short_title = clean_title[:24] + "..." if len(clean_title) > 24 else clean_title
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="500" height="750" viewBox="0 0 500 750">
        <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#1e293b"/>
                <stop offset="100%" stop-color="#0f172a"/>
            </linearGradient>
        </defs>
        <rect width="500" height="750" fill="url(#grad)"/>
        <circle cx="250" cy="300" r="60" fill="#334155" opacity="0.5"/>
        <text x="250" y="320" font-family="sans-serif" font-size="54" fill="#94a3b8" text-anchor="middle">🎬</text>
        <text x="250" y="420" font-family="sans-serif" font-size="20" font-weight="bold" fill="#f8fafc" text-anchor="middle">{short_title}</text>
        <text x="250" y="460" font-family="sans-serif" font-size="14" fill="#64748b" text-anchor="middle">Poster Unavailable</text>
    </svg>"""
    return f"data:image/svg+xml;utf8,{urllib.parse.quote(svg)}"

def search_similar_by_movie(
    target_idx: int,
    k: int = 8,
    min_rating: float = 0.0,
    selected_genres: list = None
):
    """Searches FAISS using the target movie's precomputed vector."""
    query_vec = index.reconstruct(int(target_idx)).reshape(1, -1)
    k_fetch = min(len(df), max(50, k * 6))
    scores, indices = index.search(query_vec, k_fetch)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == target_idx:
            continue  # Exclude self
        row = df.iloc[idx]
        
        # Rating filter
        if row["vote_average"] < min_rating:
            continue
            
        # Genre filter
        if selected_genres:
            row_genres = [g.strip() for g in str(row["genres"]).split(",") if g.strip()]
            if not any(g in row_genres for g in selected_genres):
                continue
                
        results.append((row, float(score)))
        if len(results) >= k:
            break
            
    return results

def search_similar_by_query(
    query_text: str,
    k: int = 8,
    min_rating: float = 0.0,
    selected_genres: list = None
):
    """Encodes custom user prompt with SentenceTransformer and queries FAISS."""
    query_vec = model.encode([query_text], convert_to_numpy=True)
    k_fetch = min(len(df), max(50, k * 6))
    scores, indices = index.search(query_vec, k_fetch)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        row = df.iloc[idx]
        
        # Rating filter
        if row["vote_average"] < min_rating:
            continue
            
        # Genre filter
        if selected_genres:
            row_genres = [g.strip() for g in str(row["genres"]).split(",") if g.strip()]
            if not any(g in row_genres for g in selected_genres):
                continue
                
        results.append((row, float(score)))
        if len(results) >= k:
            break
            
    return results

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Discovery Filters")
    st.caption("Fine-tune your recommendation algorithm")
    
    k_slider = st.slider("Number of Recommendations", min_value=4, max_value=20, value=8, step=2)
    min_rating = st.slider("Minimum Rating (⭐)", min_value=0.0, max_value=9.0, value=5.0, step=0.5)
    
    genre_filter = st.multiselect(
        "Filter by Genre (Optional)",
        options=available_genres,
        default=[]
    )
    
    st.divider()
    st.markdown("### 📊 Catalog Statistics")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Movies", f"{len(df):,}")
    with col_stat2:
        st.metric("Vector Dims", f"{index.d}")
        
    st.caption("Engine: FAISS IndexFlatIP + all-MiniLM-L6-v2")
    
    st.markdown("---")
    st.markdown(
        "<div style='color: #64748b; font-size: 0.8rem; text-align: center;'>"
        "Built with Streamlit, FAISS & HuggingFace 🤗"
        "</div>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# HERO HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">CineMatch AI 🍿</div>
    <p class="hero-subtitle">
        High-dimensional semantic movie discovery powered by FAISS vector search, 
        sentence transformers, and TMDB metadata across 187,000+ cinematic titles.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DISCOVERY TABS (MODE 1: MOVIE-BASED, MODE 2: NATURAL LANGUAGE)
# -----------------------------------------------------------------------------
tab_movie, tab_semantic = st.tabs(["🎯 More Like This (Movie Similarity)", "🔮 Plot & Vibe Search (Natural Language)"])

# =============================================================================
# TAB 1: MOVIE-TO-MOVIE RECOMMENDATIONS
# =============================================================================
with tab_movie:
    st.markdown("### 🎬 Choose a reference movie")
    
    # Popular Quick-Pick Buttons
    st.caption("Quick picks:")
    quick_cols = st.columns(6)
    popular_picks = [
        ("🪐 Interstellar", 157336),
        ("🌀 Inception", 27205),
        ("🦇 The Dark Knight", 155),
        ("🥊 Fight Club", 550),
        ("🕶️ The Matrix", 603),
        ("🕷️ Spider-Man", 557)
    ]
    
    if "selected_movie_id" not in st.session_state:
        st.session_state["selected_movie_id"] = 157336  # Default: Interstellar
        
    for i, (label, p_id) in enumerate(popular_picks):
        with quick_cols[i]:
            if st.button(label, key=f"quick_btn_{p_id}", use_container_width=True):
                st.session_state["selected_movie_id"] = p_id
                st.session_state["search_text"] = ""
                st.rerun()

    # Search & Selector
    col_search, col_select = st.columns([1, 2])
    with col_search:
        search_query = st.text_input(
            "🔎 Filter catalog by title:",
            value=st.session_state.get("search_text", ""),
            placeholder="Type e.g. Avatar, Avengers, Batman..."
        )
        st.session_state["search_text"] = search_query
        
    # Filter candidates based on search
    if search_query.strip():
        matched_df = df[df["title"].str.contains(search_query, case=False, na=False)].head(50)
        if matched_df.empty:
            matched_df = df.head(50)
            st.warning(f"No movies matching '{search_query}'. Showing popular titles.")
    else:
        # Default to top voted movies for instant accessibility
        matched_df = df.sort_values(by="vote_count", ascending=False).head(100)

    # Ensure current selected movie is in options
    current_selected_id = st.session_state["selected_movie_id"]
    if current_selected_id not in matched_df["id"].values and current_selected_id in id_to_index:
        current_row = df.iloc[[id_to_index[current_selected_id]]]
        matched_df = pd.concat([current_row, matched_df]).drop_duplicates(subset=["id"])

    options_dict = dict(zip(matched_df["id"], matched_df["display_title"]))
    selected_id = col_select.selectbox(
        "Select Movie:",
        options=list(options_dict.keys()),
        index=list(options_dict.keys()).index(current_selected_id) if current_selected_id in options_dict else 0,
        format_func=lambda x: options_dict.get(x, "Unknown"),
        key="movie_select_box"
    )
    st.session_state["selected_movie_id"] = selected_id

    # -------------------------------------------------------------------------
    # SELECTED MOVIE SPOTLIGHT
    # -------------------------------------------------------------------------
    if selected_id in id_to_index:
        sel_idx = id_to_index[selected_id]
        sel_row = df.iloc[sel_idx]
        
        st.markdown("<div class='hero-spotlight'>", unsafe_allow_html=True)
        col_post, col_info = st.columns([1, 4])
        with col_post:
            st.image(get_poster_url(sel_row["poster_path"], sel_row["title"]), use_container_width=True)
        with col_info:
            st.markdown(f"## {sel_row['title']} <span style='color:#94a3b8; font-size:1.4rem;'>({sel_row['year']})</span>", unsafe_allow_html=True)
            
            # Badges
            rating_val = f"⭐ {sel_row['vote_average']:.1f}/10 ({sel_row['vote_count']:,} votes)" if sel_row['vote_count'] > 0 else "⭐ N/A"
            st.markdown(f"<span class='badge-rating'>{rating_val}</span>", unsafe_allow_html=True)
            
            if pd.notna(sel_row["genres"]) and str(sel_row["genres"]).strip():
                genres_html = " ".join([f"<span class='badge-genre'>{g.strip()}</span>" for g in str(sel_row["genres"]).split(",") if g.strip()])
                st.markdown(f"<div style='margin-top: 8px;'>{genres_html}</div>", unsafe_allow_html=True)
                
            st.markdown(f"<p style='color: #cbd5e1; margin-top: 12px; line-height: 1.6;'>{sel_row['overview'] if pd.notna(sel_row['overview']) and sel_row['overview'] else 'No synopsis available.'}</p>", unsafe_allow_html=True)
            
            tmdb_link = f"https://www.themoviedb.org/movie/{sel_row['id']}"
            st.markdown(f"<a href='{tmdb_link}' target='_blank' style='color:#38bdf8; text-decoration:none; font-size:0.9rem; font-weight:600;'>🔗 View on TMDB ↗</a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # RECOMMENDATIONS GRID
        # ---------------------------------------------------------------------
        st.markdown(f"### ✨ Top Recommendations similar to *{sel_row['title']}*")
        
        with st.spinner("Finding vector nearest neighbors..."):
            recommendations = search_similar_by_movie(
                target_idx=sel_idx,
                k=k_slider,
                min_rating=min_rating,
                selected_genres=genre_filter
            )

        if not recommendations:
            st.warning("No matching recommendations found with the active filters. Try lowering the minimum rating or clearing genre filters.")
        else:
            # Display in 4-column cards
            num_cols = 4
            for row_start in range(0, len(recommendations), num_cols):
                cols = st.columns(num_cols)
                chunk = recommendations[row_start : row_start + num_cols]
                for c_idx, (m_row, score) in enumerate(chunk):
                    with cols[c_idx]:
                        st.image(get_poster_url(m_row["poster_path"], m_row["title"]), use_container_width=True)
                        st.markdown(f"**{m_row['title']}** <span style='color:#94a3b8; font-size:0.85rem;'>({m_row['year']})</span>", unsafe_allow_html=True)
                        
                        # Score pill & Rating
                        match_pct = max(0, min(100, int(score * 100)))
                        st.markdown(
                            f"<span class='badge-score'>🎯 {match_pct}% Match</span> "
                            f"<span class='badge-rating'>⭐ {m_row['vote_average']:.1f}</span>",
                            unsafe_allow_html=True
                        )
                        
                        # Genre tags
                        if pd.notna(m_row["genres"]) and str(m_row["genres"]).strip():
                            tags = [g.strip() for g in str(m_row["genres"]).split(",") if g.strip()][:3]
                            tags_html = " ".join([f"<span class='badge-genre'>{t}</span>" for t in tags])
                            st.markdown(f"<div style='margin-top: 4px; margin-bottom: 6px;'>{tags_html}</div>", unsafe_allow_html=True)
                            
                        # Overview expander
                        with st.expander("📖 Synopsis"):
                            st.write(m_row["overview"] if pd.notna(m_row["overview"]) and m_row["overview"] else "No synopsis available.")
                            st.markdown(f"[TMDB Link ↗](https://www.themoviedb.org/movie/{m_row['id']})")
                st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

# =============================================================================
# TAB 2: NATURAL LANGUAGE & PLOT SEARCH
# =============================================================================
with tab_semantic:
    st.markdown("### 🔮 Search by Plot, Mood, or Vibe")
    st.caption("Describe any concept, storyline, atmosphere, or style in plain English:")
    
    # Preset Prompt Inspiration
    prompt_col1, prompt_col2, prompt_col3 = st.columns(3)
    with prompt_col1:
        if st.button("🌌 Space expedition with black holes"):
            st.session_state["semantic_input"] = "A deep space expedition through a wormhole near a black hole to save humanity"
            st.rerun()
    with prompt_col2:
        if st.button("🕵️ Rainy neo-noir cyberpunk detective"):
            st.session_state["semantic_input"] = "A gritty cyberpunk detective in a dark rainy futuristic neon city hunting rogue synthetics"
            st.rerun()
    with prompt_col3:
        if st.button("⏳ Mind-bending dream heist & puzzles"):
            st.session_state["semantic_input"] = "A thief who steals corporate secrets through dream-sharing technology inside subconsciousness"
            st.rerun()

    user_query = st.text_area(
        "Enter your plot description or vibe query:",
        value=st.session_state.get("semantic_input", "A deep space expedition through a wormhole near a black hole to save humanity"),
        height=100,
        placeholder="e.g. A solitary astronaut stranded on Mars trying to survive and communicate with Earth..."
    )
    st.session_state["semantic_input"] = user_query
    
    btn_search = st.button("🚀 Search Semantic Vectors", type="primary")
    
    if user_query.strip():
        with st.spinner("🧠 Encoding query with SentenceTransformer & querying FAISS index..."):
            semantic_results = search_similar_by_query(
                query_text=user_query,
                k=k_slider,
                min_rating=min_rating,
                selected_genres=genre_filter
            )
            
        st.markdown(f"### 🎯 Results for: *\"{user_query}\"*")
        
        if not semantic_results:
            st.warning("No matches found matching your filters. Try relaxing the rating or genre filters.")
        else:
            num_cols = 4
            for row_start in range(0, len(semantic_results), num_cols):
                cols = st.columns(num_cols)
                chunk = semantic_results[row_start : row_start + num_cols]
                for c_idx, (m_row, score) in enumerate(chunk):
                    with cols[c_idx]:
                        st.image(get_poster_url(m_row["poster_path"], m_row["title"]), use_container_width=True)
                        st.markdown(f"**{m_row['title']}** <span style='color:#94a3b8; font-size:0.85rem;'>({m_row['year']})</span>", unsafe_allow_html=True)
                        
                        match_pct = max(0, min(100, int(score * 100)))
                        st.markdown(
                            f"<span class='badge-score'>🎯 {match_pct}% Match</span> "
                            f"<span class='badge-rating'>⭐ {m_row['vote_average']:.1f}</span>",
                            unsafe_allow_html=True
                        )
                        
                        if pd.notna(m_row["genres"]) and str(m_row["genres"]).strip():
                            tags = [g.strip() for g in str(m_row["genres"]).split(",") if g.strip()][:3]
                            tags_html = " ".join([f"<span class='badge-genre'>{t}</span>" for t in tags])
                            st.markdown(f"<div style='margin-top: 4px; margin-bottom: 6px;'>{tags_html}</div>", unsafe_allow_html=True)
                            
                        with st.expander("📖 Synopsis"):
                            st.write(m_row["overview"] if pd.notna(m_row["overview"]) and m_row["overview"] else "No synopsis available.")
                            st.markdown(f"[TMDB Link ↗](https://www.themoviedb.org/movie/{m_row['id']})")
                st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)