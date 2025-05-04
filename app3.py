import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# Backgrounds for moods
background_map = {
    'Happy': 'https://images.unsplash.com/photo-1508436134971-8d78db686559?auto=format&fit=crop&w=2000&q=80',
    'Sad': 'https://images.unsplash.com/photo-1520975911656-325fc4f136b6?auto=format&fit=crop&w=2000&q=80',
    'Relaxed': 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=2000&q=80',
    'Thrilling': 'https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=2000&q=80',
    'Romantic': 'https://images.unsplash.com/photo-1512678089-9a8b32bda1e4?auto=format&fit=crop&w=2000&q=80',
    'Intrigued': 'https://images.unsplash.com/photo-1497801636351-daf78e2a3144?auto=format&fit=crop&w=2000&q=80',
    'Neutral': 'https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=2000&q=80'
}

# Streamlit page config
st.set_page_config(page_title="MoodFlixx", layout="wide")

# Initialize session state
if 'user_mood' not in st.session_state:
    st.session_state.user_mood = 'Neutral'
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'embed_model' not in st.session_state:
    st.session_state.embed_model = None
if 'overview_embeddings' not in st.session_state:
    st.session_state.overview_embeddings = None

# Function to set background
def set_background(mood):
    bg_url = background_map.get(mood, '')
    if bg_url:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("{bg_url}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(0,0,0,0.6);
                padding: 2rem;
                border-radius: 10px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Set initial background
set_background(st.session_state.user_mood)

# Sidebar - Mood Selection
with st.sidebar:
    st.header("🎉 Let's Get This Movie Party Started!")
    mood_map = {
        "☀️ Feeling sunny!": "Happy",
        "🌧️ Bit cloudy, could use a lift.": "Sad",
        "🛋️ Just wanna chill and veg out.": "Relaxed",
        "⚡ Buzzing! Need some excitement.": "Thrilling",
        "💘 Got love on the brain.": "Romantic",
        "🤔 Thinking cap on, ready for twists.": "Intrigued",
        "😐 Just... neutral? Show me anything good.": "Neutral"
    }
    
    mood_choice = st.selectbox(
        "1️⃣ What's your vibe right now?", 
        list(mood_map.keys()),
        key="mood_selector"
    )
    
    if st.session_state.mood_selector != mood_map.get(st.session_state.user_mood, ""):
        st.session_state.user_mood = mood_map[mood_choice]
        set_background(st.session_state.user_mood)

# Load data function with progress bar
@st.cache_data(show_spinner=False)
def load_data(path='enhanced_movies.csv'):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Could not find '{path}'. Please place your CSV in the same folder as app.py.")
        st.stop()
    
    def rt_to_min(rt):
        hrs = re.search(r'(\d+)\s*hrs?', str(rt))
        mins = re.search(r'(\d+)\s*min', str(rt))
        return (int(hrs.group(1))*60 if hrs else 0) + (int(mins.group(1)) if mins else 0)
    
    df['runtime_mins'] = df['runtime'].apply(rt_to_min)
    df['genres_list'] = df['genres'].fillna('').apply(lambda s: [g.strip() for g in s.split(',') if g])
    df['release_date'] = pd.to_datetime(df['release_date'].astype('Int64').astype(str) + '-01-01')
    df['overview'] = df['overview'].fillna('')
    return df

# Load embeddings only when needed
def load_embeddings(df):
    progress_text = "Preparing recommendations engine..."
    my_bar = st.progress(0, text=progress_text)
    
    # Load model (20% progress)
    my_bar.progress(20, text="Loading AI model...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings (80% progress)
    my_bar.progress(60, text="Processing movie descriptions...")
    overview_embeddings = embed_model.encode(df['overview'].tolist(), convert_to_tensor=True)
    
    my_bar.progress(100, text="Ready to recommend!")
    my_bar.empty()
    
    return embed_model, overview_embeddings

# Load data only once
if not st.session_state.data_loaded:
    with st.spinner("Loading movie database..."):
        st.session_state.df = load_data()
    st.session_state.data_loaded = True

# Main content
st.title("🎬 MoodFlixx Movie Recommender")

# Sidebar - Other inputs (only show after data is loaded)
if st.session_state.data_loaded:
    with st.sidebar:
        # Q2: Genre cravings
        genre_labels = {
            'Action': 'Action (Pow! Bang!)',
            'Comedy': 'Comedy (Giggles & Snorts)',
            'Drama': 'Drama (Feel all the feels)',
            'Romance': 'Romance (Love is in the air)',
            'Sci-Fi': 'Sci-Fi (Mind Bender)',
            'Horror': 'Horror (Hide behind the couch)',
            'Thriller': 'Thriller (On the edge of your seat)',
            'Animation': 'Animation (Cartoon therapy)',
            'Adventure': 'Adventure (Epic Journey)',
            'Crime': 'Crime (Whodunit?)'
        }
        user_genres = st.multiselect(
            "2️⃣ If your mood was a movie genre, what's the main flavor?",
            options=list(genre_labels.keys()),
            format_func=lambda x: genre_labels[x],
            default=list(genre_labels.keys())[:2],
            max_selections=2
        )
        # Q3: Runtime preference
        rt_choice = st.radio(
            "3️⃣ Got time for an epic saga, a regular movie night, or just a quick flick?",
            options=['Long (>120 min)', 'Standard (90-120 min)', 'Quick (<90 min)']
        )
        user_time = 10000 if rt_choice=='Long (>120 min)' else (120 if rt_choice=='Standard (90-120 min)' else 90)
        # Q4: Star power threshold
        rating_map = {'Only the highly-rated gems! (>7.5)':7.5, 'Solid and popular picks (6.5-7.5)':6.5, 'Feeling adventurous, surprise me!':0.0}
        user_rating = rating_map[st.selectbox("4️⃣ How brave are we feeling?", list(rating_map.keys()))]
        # Q5: Language vibe
        user_lang = st.selectbox("5️⃣ Yo, what language you vibin' with?", ["Any"]+sorted(st.session_state.df['original_language'].unique()))
        # Q6: Vintage or new
        years = sorted(st.session_state.df['release_date'].dt.year.unique())
        yr_min, yr_max = st.select_slider("6️⃣ Vintage or New? Pick a year range:", options=years, value=(years[0], years[-1]))
        # Q7: Favorite cast/director
        user_actor = st.text_input("7️⃣ Any must-see actors or directors you're always game for? (optional)")
        # Q8: Free-text feelings
        user_text = st.text_area("8️⃣ Spill the beans! Anything else on your mind? (optional)")
        st.button("🚀 Show me the flicks!", key="rec_button")

# Recommendation logic
if st.session_state.get("rec_button") and st.session_state.data_loaded:
    # Only load embeddings if text input is provided
    if user_text.strip() and st.session_state.embed_model is None:
        st.session_state.embed_model, st.session_state.overview_embeddings = load_embeddings(st.session_state.df)
    
    filtered = st.session_state.df[st.session_state.df['final_mood']==st.session_state.user_mood].copy()
    if user_genres:
        filtered = filtered[filtered['genres_list'].apply(lambda gl: any(g in gl for g in user_genres))]
    filtered = filtered[(filtered['runtime_mins']<=user_time)&(filtered['vote_average']>=user_rating)&(filtered['release_date'].dt.year.between(yr_min,yr_max))]
    if user_lang!="Any": filtered = filtered[filtered['original_language']==user_lang]
    if user_actor.strip(): filtered = filtered[filtered['Cast'].str.contains(user_actor,case=False,na=False)]
    
    if user_text.strip() and st.session_state.embed_model is not None:
        user_emb = st.session_state.embed_model.encode(user_text, convert_to_tensor=True)
        sim_scores = util.cos_sim(user_emb, st.session_state.overview_embeddings)[0].cpu().numpy()
        sim_series = pd.Series(sim_scores,index=st.session_state.df.index)
        sim_series = (sim_series - sim_series.min())/(sim_series.max()-sim_series.min())
    else:
        sim_series=None
    
    base_score = filtered['popularity']*0.6 + filtered['vote_average']*0.4
    
    if sim_series is not None:
        combined_score = base_score.reindex(sim_series.index).fillna(0)*0.7 + sim_series*0.3
        filtered['score']=combined_score.loc[filtered.index]
    else:
        filtered['score']=base_score
    
    recs=filtered.sort_values('score',ascending=False).head(5)
    
    if recs.empty:
        st.warning("No matches found 😢 Try expanding your filters.")
    else:
        cols=st.columns(len(recs))
        for idx,(_,row) in enumerate(recs.iterrows()):
            with cols[idx]:
                if 'poster_url' in row: st.image(row['poster_url'],use_column_width=True)
                st.markdown(f"**{row['title']}** ({row['release_date'].year})")
                st.caption(f"Mood: {row['final_mood']} | ⭐ {row['vote_average']}")
                st.write(row['overview'][:150]+"…")