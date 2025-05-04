# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import time
import os
import requests
import ast # To parse LLM list output safely
import json # For potentially cleaner LLM output parsing
import uuid # For generating unique user IDs
import datetime # For tracking timestamps
import base64 # Potentially for icons if needed later

# --- Attempt to import Google Generative AI ---
try:
    import google.generativeai as genai
except ModuleNotFoundError:
    print("CRITICAL ERROR: 'google-generativeai' library not found...")
    try: st.set_page_config(page_title="Error", layout="wide"); st.error("CRITICAL ERROR: 'google-generativeai' library not found...")
    except Exception: pass
    st.stop()

# --- Attempt to import NLTK sentiment analyzer ---
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except ModuleNotFoundError:
    print("CRITICAL ERROR: NLTK components not found...")
    try: st.set_page_config(page_title="Error", layout="wide"); st.error("CRITICAL ERROR: NLTK components not found...")
    except Exception: pass
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="MoodFlixx ✨ | Chat Edition",
    page_icon="💬", # Chat bubble icon
    layout="centered", # Centered layout often works well for chat
    initial_sidebar_state="collapsed" # No sidebar needed initially
)

# --- NLTK VADER Lexicon Check ---
try: nltk.data.find('sentiment/vader_lexicon.zip')
except (nltk.downloader.DownloadError, LookupError):
    with st.spinner("Setting up language tools... 🧠"):
        try: nltk.download('vader_lexicon', quiet=True); st.toast("Language tools ready!", icon="👍")
        except Exception as e: st.warning(f"NLTK download failed: {e}")

# --- API Key Loading ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not GOOGLE_API_KEY or not TMDB_API_KEY:
    st.error("⚠️ API Keys Missing! Need both Google & TMDb keys. Set env vars & restart.")
    st.stop()

# --- Configure Google AI ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    llm = genai.GenerativeModel('gemini-1.5-flash')
    # llm_pro = genai.GenerativeModel('gemini-pro') # Optional: Use pro for the final summary/rec step
except Exception as e:
    st.error(f"Fatal Error configuring Google AI: {e}")
    st.stop()

# --- Initialize Sentiment Analyzer (Optional for this flow, but keep) ---
analyzer = None
try: analyzer = SentimentIntensityAnalyzer()
except Exception as e: st.warning(f"Couldn't start Sentiment Analyzer: {e}")

# --- Helper Functions ---
@st.cache_data(ttl=3600)
def generate_llm_response(_llm_model, prompt, temperature=0.8): # Slightly higher temp default
    """Generates text using the provided LLM model."""
    try:
        response = _llm_model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=temperature))
        if hasattr(response, 'text'): return response.text.strip()
        if response.parts: return "".join(part.text for part in response.parts).strip()
        if response.prompt_feedback and response.prompt_feedback.block_reason:
             return f"Oof, filter said no ({response.prompt_feedback.block_reason}). Let's rephrase? 🤔"
        return "My AI brain just lagged out... what were we saying? 😵‍💫"
    except Exception as e:
        print(f"LLM Error: {e} for prompt: {prompt[:100]}...")
        return "Ugh, connection's ghosting me. 👻 Try asking again?"

TMDB_BASE_IMAGE_URL = "https://image.tmdb.org/t/p/"
POSTER_SIZE = "w500"
PLACEHOLDER_IMAGE = "https://placehold.co/500x750/333333/777777?text=Poster+MIA+%3A%2F"
@st.cache_data(ttl=86400)
def get_poster_url(movie_title, api_key):
    if not api_key or not movie_title or pd.isna(movie_title): return PLACEHOLDER_IMAGE
    search_url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": str(movie_title), "include_adult": False, "language": "en-US", "page": 1}
    try:
        response = requests.get(search_url, params=params, timeout=8); response.raise_for_status()
        data = response.json()
        if data.get('results'):
            poster_path = data['results'][0].get('poster_path')
            if poster_path: return f"{TMDB_BASE_IMAGE_URL.rstrip('/')}/{POSTER_SIZE.strip('/')}{poster_path}"
        return PLACEHOLDER_IMAGE
    except Exception as e:
        # print(f"TMDB Error fetching poster for '{movie_title}': {e}") # Optional: log TMDB errors
        return PLACEHOLDER_IMAGE # Simplified error handling for this function

@st.cache_data
def parse_runtime(runtime_str):
    if pd.isna(runtime_str) or not isinstance(runtime_str, str): return 0
    runtime_str = runtime_str.lower(); hours, minutes = 0, 0
    try:
        h_match = re.search(r'(\d+)\s*h(?:rs?|our)?', runtime_str)
        m_match = re.search(r'(\d+)\s*m(?:in)?', runtime_str)
        if h_match: hours = int(h_match.group(1))
        if m_match: minutes = int(m_match.group(1))
        if hours == 0 and minutes == 0:
             num_match = re.match(r'^\s*(\d+)\s*$', runtime_str.strip())
             if num_match: minutes = int(num_match.group(1))
    except Exception: return 0
    return hours * 60 + minutes

# --- Load Data ---
@st.cache_data
def load_data(filepath='enhanced_movies.csv'):
    try:
        data = pd.read_csv(filepath)
        data['Title'] = data['Title'].fillna('Unknown Title').astype(str)
        data['runtime'] = data['runtime'].fillna('0 min')
        data['runtime_minutes'] = data['runtime'].apply(parse_runtime)
        data['genres'] = data['genres'].fillna('').astype(str)
        data['final_mood'] = data['final_mood'].fillna('Unknown').astype(str)
        data['vote_average'] = pd.to_numeric(data['vote_average'], errors='coerce').fillna(0.0)
        data['original_language'] = data['original_language'].fillna('Unknown').astype(str)
        data['Cast'] = data['Cast'].fillna('')
        data['Director'] = data['Director'].fillna('')
        data['overview'] = data['overview'].fillna('')
        cols_to_keep = ['Title', 'genres', 'final_mood', 'vote_average', 'runtime',
                         'runtime_minutes', 'Director', 'Cast', 'overview', 'original_language']
        # Ensure all cols_to_keep exist, create empty ones if not
        for col in cols_to_keep:
             if col not in data.columns:
                 # Use empty string for object types, 0 for numeric
                 data[col] = '' if data[cols_to_keep[0]].dtype == 'object' else 0 # Using first col type as proxy, assuming common sense data
                 # More robust: Check expected types if known, or default to object
                 # data[col] = '' if col in ['Title', 'genres', ...] else 0
        # Select only the columns we need
        data = data[cols_to_keep]
        # Drop rows where Title is completely missing or effectively empty after cleaning
        data.dropna(subset=['Title'], inplace=True)
        data = data[data['Title'].str.strip() != '']
        return data.reset_index(drop=True) # Reset index after dropping rows
    except FileNotFoundError: st.error(f"FATAL: Movie data file MIA at '{filepath}' 💀"); return pd.DataFrame()
    except Exception as e: st.error(f"FATAL: Error loading data: {e}"); return pd.DataFrame()


movie_data = load_data()
if movie_data.empty: st.warning("Can't give recs without the movie data..."); st.stop()

# --- New Helper: Find Movies by Title from LLM ---
def find_movies_by_title(movie_df, titles_list):
    """Finds movies from the dataframe matching titles suggested by LLM."""
    if movie_df.empty or not titles_list:
        return pd.DataFrame()

    # Clean LLM titles (remove potential quotes/numbering)
    cleaned_titles = []
    for t in titles_list:
        # Robust cleaning: remove leading numbers/dots/whitespace, quotes, and trim
        t_clean = re.sub(r'^\s*\d+\.\s*', '', t).strip().strip('"').strip("'").strip()
        if t_clean: # Only add non-empty strings
             cleaned_titles.append(t_clean.lower()) # Use lower case for matching

    if not cleaned_titles:
        return pd.DataFrame() # Return empty if no valid titles parsed

    # Find matches (case-insensitive)
    # Use a boolean mask for efficiency
    mask = movie_df['Title'].str.lower().isin(cleaned_titles)
    matches = movie_df[mask].copy()

    # Try to preserve order from LLM list (best effort)
    title_to_order = {title: i for i, title in enumerate(cleaned_titles)}
    # Use .get() with a default value for titles not found in cleaned_titles (shouldn't happen with mask)
    matches['sort_order'] = matches['Title'].str.lower().apply(lambda x: title_to_order.get(x, len(cleaned_titles)))
    matches = matches.sort_values('sort_order').drop(columns='sort_order')


    # Ensure expected columns are present for display
    cols_expected = ['Title', 'genres', 'final_mood', 'vote_average', 'runtime', 'Director', 'Cast', 'overview', 'original_language']
    # Select columns ensuring they exist in matches.columns
    cols_present = [c for c in cols_expected if c in matches.columns]

    return matches[cols_present]


# --- Buddy Data File Management ---
BUDDY_DATA_FILE = "mood_buddies.json"
BUDDY_TIMEOUT_MINUTES = 15 # How long a user is considered "active"

def load_buddy_data(filename):
    """Loads buddy data from a JSON file, handling file not found."""
    if not os.path.exists(filename):
        print(f"Buddy data file not found: {filename}. Starting fresh.")
        return []
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            # Basic validation
            if not isinstance(data, list):
                print(f"Warning: Buddy data file {filename} has invalid format (not a list). Starting fresh.")
                return []
            return data
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading buddy data file {filename}: {e}. Starting fresh.")
        # Optionally, you might want to back up the bad file and start fresh
        # try: os.rename(filename, f"{filename}.corrupted_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
        # except Exception as rename_e: print(f"Could not backup corrupted file: {rename_e}")
        return [] # Return empty list on error

def save_buddy_data(data, filename):
    """Saves buddy data to a JSON file."""
    # This is a very basic attempt at handling concurrent access.
    # A real application requires proper file locking (like 'flock' on Unix)
    # or using a database designed for concurrency.
    try:
        lock_file = f"{filename}.lock"
        # Simple check and skip write if lock exists. Not foolproof.
        # In a real app, implement a proper loop with waits and retries
        if os.path.exists(lock_file):
             print(f"Warning: {lock_file} exists. Skipping save.")
             return

        # Create lock file (basic, could still race - 'x' mode is slightly better)
        try:
            with open(lock_file, "x") as f: # Use 'x' mode to fail if file exists
                 f.write("")
        except FileExistsError:
            print(f"Warning: {lock_file} appeared just before write. Skipping save.")
            return # Skip if lock already exists
        except Exception as e:
             print(f"Error creating lock file {lock_file}: {e}. Skipping save.")
             return


        # Save data
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        # Remove lock file
        if os.path.exists(lock_file):
             os.remove(lock_file)

    except Exception as e:
        print(f"Error saving buddy data file {filename}: {e}")
        # Clean up lock file if error occurred after creating it
        if os.path.exists(lock_file):
             try: os.remove(lock_file)
             except: pass # Ignore errors during cleanup

def add_or_update_buddy(filename, user_id, mood):
    """Adds or updates a user's entry in the buddy data file."""
    if not user_id or not mood or mood == 'Unknown':
         return # Don't add invalid entries

    current_time = datetime.datetime.utcnow().isoformat()
    buddies = load_buddy_data(filename)

    found = False
    # Create a new list to avoid modifying during iteration and ensure clean data structure
    updated_buddies = []
    for buddy in buddies:
        if buddy.get("id") == user_id:
            # Update existing user
            updated_buddies.append({"id": user_id, "mood": mood, "timestamp": current_time})
            found = True
        elif buddy.get("id") is not None and buddy.get("mood") is not None and buddy.get("timestamp") is not None:
            # Keep other valid buddy entries
            updated_buddies.append(buddy)
        else:
             # Discard invalid entries found during load
             print(f"Discarding invalid buddy entry: {buddy}")


    if not found:
        # Add new user if not found
        updated_buddies.append({"id": user_id, "mood": mood, "timestamp": current_time})

    save_buddy_data(updated_buddies, filename)

def clean_old_buddies(filename, timeout_minutes):
    """Removes buddies whose timestamp is older than the timeout."""
    buddies = load_buddy_data(filename)
    current_time = datetime.datetime.utcnow()
    timeout_delta = datetime.timedelta(minutes=timeout_minutes)

    active_buddies = []
    for buddy in buddies:
        try:
            timestamp_str = buddy.get("timestamp")
            if timestamp_str:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                # Keep if within timeout and has required keys
                if current_time - timestamp < timeout_delta and buddy.get("id") and buddy.get("mood"):
                    active_buddies.append(buddy)
            # Discard entries without valid timestamps or missing keys
        except (ValueError, TypeError, KeyError) as e:
            # print(f"Discarding invalid or old entry during clean: {buddy} - Error: {e}")
            continue # Discard invalid entries silently during clean

    # Only save if changes were made to avoid unnecessary writes
    if len(active_buddies) < len(buddies):
        print(f"Cleaning buddy data: Removed {len(buddies) - len(active_buddies)} old/invalid entries.")
        save_buddy_data(active_buddies, filename)
    # else:
        # print("No old entries to clean.")


def find_matching_buddies(filename, current_user_id, target_mood):
    """Finds active buddies with the same target mood, excluding the current user."""
    if not target_mood or target_mood == 'Unknown':
        return [] # Don't match if mood is invalid

    # Clean old buddies before finding matches to ensure list is fresh
    clean_old_buddies(filename, BUDDY_TIMEOUT_MINUTES)

    buddies = load_buddy_data(filename)
    matching_buddies = [
        buddy for buddy in buddies
        if buddy.get("mood") == target_mood
        and buddy.get("id") != current_user_id # Exclude current user
        and buddy.get("id") is not None # Ensure ID exists
        and buddy.get("mood") is not None # Ensure mood exists
    ]
    return matching_buddies


# --- Define Conversation Steps & Questions ---
conversation_steps = [
    {"key": "intro", "question": "Yo! MoodFlixx here ✨ Ready to find ur next obsession? First up, how's ur day been treating ya?"},
    {"key": "day_vibe", "question": "Gotcha. So, spill the tea ☕ – what kinda movies/shows usually hit different for you? Fave genres? Anything you watched recently & loved (or hated lol)?"},
    {"key": "taste", "question": "Interesting taste! 👀 Now, for *tonight*... what's the main character energy we're going for? Chill vibes? Edge of ur seat? Need a good cry? 😭"},
    {"key": "current_mood", "question": "Okay, channeling that energy! 🙏 Any hard passes? Like actors u can't stand, genres that are a total 'ick', or maybe a time limit? (e.g., 'nothing over 2 hrs', 'no horror plz')"},
    {"key": "dealbreakers", "question": "Noted! 📝 Alright, I think I've got the picture. Lemme cook for a sec... 🧑‍🍳"},
]

# --- Initialize Session State ---
if "current_step" not in st.session_state: st.session_state.current_step = 0
if "conversation_history" not in st.session_state: st.session_state.conversation_history = []
if "user_profile_info" not in st.session_state: st.session_state.user_profile_info = {}
# Add states for LLM output
if "llm_summary" not in st.session_state: st.session_state.llm_summary = ""
if "llm_titles" not in st.session_state: st.session_state.llm_titles = []
if "llm_pick_text" not in st.session_state: st.session_state.llm_pick_text = ""
if "final_recommendations" not in st.session_state: st.session_state.final_recommendations = None
if "processing_complete" not in st.session_state: st.session_state.processing_complete = False
# New: Initialize a unique user ID for this session
if "user_id" not in st.session_state: st.session_state.user_id = str(uuid.uuid4())
# New: State to store potential moods for buddy matching based on recs
if "available_buddy_moods" not in st.session_state: st.session_state.available_buddy_moods = []
# New: State to store the user's selected mood for buddy matching
if "selected_buddy_mood" not in st.session_state: st.session_state.selected_buddy_mood = None
# New: State to store the list of found buddies for the selected mood
if "current_buddies_list" not in st.session_state: st.session_state.current_buddies_list = []


# --- Display Chat History ---
st.markdown("### Let's Chat Movies! 💬")
chat_container = st.container(height=400, border=True) # Fixed height scrollable container
with chat_container:
    # Display initial welcome message if starting
    if not st.session_state.conversation_history and st.session_state.current_step == 0:
         st.session_state.conversation_history.append({"role": "assistant", "content": conversation_steps[0]["question"]})
         # st.rerun() # No rerun needed here, will display on first run

    # Display messages from history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            # Check if it's a special message like the processing one we updated
            # and handle display accordingly if needed, otherwise just markdown
            st.markdown(message["content"])


# --- Main Interaction Logic ---
current_step_index = st.session_state.current_step

# --- Step 1-N: Ask Questions ---
if current_step_index < len(conversation_steps):
    current_q_spec = conversation_steps[current_step_index]
    q_key = current_q_spec["key"]
    q_text = current_q_spec["question"]

    # Display current question if it's the next one and hasn't been displayed
    # Prevent re-adding the same question on rerun after user input
    if not st.session_state.conversation_history or \
       st.session_state.conversation_history[-1]["content"] != q_text or \
       st.session_state.conversation_history[-1]["role"] != "assistant":
         st.session_state.conversation_history.append({"role": "assistant", "content": q_text})
         st.rerun() # Rerun to display the new question immediately (this helps ensure the chat displays correctly)

    # Get user input - only show input if it's the user's turn to answer
    prompt = st.chat_input("Your thoughts go here...", key=f"chat_input_{current_step_index}") # Add a unique key
    if prompt:
        # Store user answer
        st.session_state.user_profile_info[q_key] = prompt
        st.session_state.conversation_history.append({"role": "user", "content": prompt})

        # Advance to next step
        st.session_state.current_step += 1
        st.rerun() # Rerun to trigger the next step (either next question or processing)


# --- Step N+1: Process & Summarize with LLM ---
# This step triggers AFTER the last user input is recorded (current_step == len(conversation_steps))
# and only if processing_complete is False
elif current_step_index == len(conversation_steps) and not st.session_state.processing_complete:
    # Add a placeholder message to the chat to indicate processing started
    # Check if the last message is NOT the processing message
    if not st.session_state.conversation_history or \
       st.session_state.conversation_history[-1]["role"] != "assistant" or \
       "Analyzing your vibe" not in st.session_state.conversation_history[-1]["content"]:
         st.session_state.conversation_history.append({"role": "assistant", "content": "Analyzing your vibe... 🤔✨"})
         st.rerun() # Rerun to show the processing message


    # The actual processing happens here after the rerun that shows the processing message
    # Add a small delay to ensure the spinner/message is seen
    time.sleep(0.5) # Give UI a moment to update with the spinner/message

    with st.spinner("Analyzing your vibe... 🤔✨"):
        # Compile user info
        day_info = st.session_state.user_profile_info.get("day_vibe", "N/A")
        taste_info = st.session_state.user_profile_info.get("taste", "N/A")
        mood_info = st.session_state.user_profile_info.get("current_mood", "N/A")
        dealbreakers_info = st.session_state.user_profile_info.get("dealbreakers", "N/A")

        # Construct the detailed prompt for LLM
        llm_prompt = f"""
You are MoodFlixx, a witty, Gen Z AI movie recommender. A user just told you this:
- How their day was: "{day_info}"
- Their general taste: "{taste_info}"
- What they want right NOW: "{mood_info}"
- Dealbreakers/Constraints: "{dealbreakers_info}"

Based *only* on this conversation and the provided user info:
1. Generate a super short (max 2 lines), fun, slightly sassy summary of what kind of movie vibe this user is looking for *right now*. Address the user directly. Use emojis appropriately.
2. Recommend a list of 7 specific movie titles that fit this synthesized vibe *extremely well*. Prioritize movies released after 2000 if possible but include variety. Format this strictly as a Python list of strings. Example: ["Movie Title 1", "Movie Title 2", "Movie Title 3", "Movie Title 4", "Movie Title 5", "Movie Title 6", "Movie Title 7"]
3. Pick ONE movie from your recommended list (or a closely related wildcard) as your 'personal recommendation'. Explain *briefly* (1 sentence) in a fun/Gen Z way why *you* think they'll specifically dig this one. Format as: Personal Pick: [Movie Title] - [Your witty reason]

Output *only* the summary, the Python list, and the personal pick line, structured *exactly* like this:

SUMMARY: [Your 2-line summary here]
TITLES: ["Movie Title 1", "Movie Title 2", "Movie Title 3", "Movie Title 4", "Movie Title 5", "Movie Title 6", "Movie Title 7"]
PICK: Personal Pick: [Movie Title] - [Your witty reason here]
"""
        synthesis_model = llm # Use flash for speed, or switch to llm_pro if defined and desired
        llm_response_raw = generate_llm_response(synthesis_model, llm_prompt, temperature=0.7)

        # --- Parse LLM Response ---
        summary = "Hmm, couldn't quite summarize... my AI brain got lost in the matrix. 👾"
        titles = []
        pick = "Couldn't pick one, sorry! All the options were too good/bad? 🤔"

        try:
            # Use regex to find the parts reliably
            summary_match = re.search(r"SUMMARY:\s*(.*)", llm_response_raw, re.IGNORECASE | re.DOTALL)
            titles_match = re.search(r"TITLES:\s*(\[.*?\])", llm_response_raw, re.IGNORECASE | re.DOTALL)
            pick_match = re.search(r"PICK:\s*(.*)", llm_response_raw, re.IGNORECASE | re.DOTALL)

            if summary_match:
                summary = summary_match.group(1).strip()
                # Clean up potential bleed-over from subsequent sections
                summary = re.split(r'TITLES:|PICK:', summary, flags=re.IGNORECASE)[0].strip()


            if titles_match:
                titles_str = titles_match.group(1)
                try:
                    # Use ast.literal_eval for safe evaluation of the list string
                    titles = ast.literal_eval(titles_str)
                    if not isinstance(titles, list): titles = [] # Ensure it's a list
                     # Clean up each title string just in case
                    titles = [re.sub(r'^\s*\d+\.\s*', '', t).strip().strip('"').strip("'").strip() for t in titles]
                    titles = [t for t in titles if t] # Remove any empty strings or titles that are just numbers/quotes
                except (ValueError, SyntaxError) as e:
                    print(f"LLM Parsing Error (Titles): {e}. String was: {titles_str}")
                    titles = [] # Fallback to empty list


            if pick_match:
                pick = pick_match.group(1).strip()
                # Clean up potential bleed-over
                pick = re.split(r'SUMMARY:|TITLES:', pick, flags=re.IGNORECASE)[0].strip()


            st.session_state.llm_summary = summary
            st.session_state.llm_titles = titles
            st.session_state.llm_pick_text = pick
            st.session_state.processing_complete = True
            st.session_state.current_step += 1 # Move to display step

            # Update the processing message in chat history with the summary
            # Find and replace the last 'Analyzing...' message
            if st.session_state.conversation_history:
                 last_msg_index = -1
                 # Search backwards for the 'Analyzing...' message
                 for i in range(len(st.session_state.conversation_history)-1, -1, -1):
                     if st.session_state.conversation_history[i]["role"] == "assistant" and "Analyzing your vibe" in st.session_state.conversation_history[i]["content"]:
                         last_msg_index = i
                         break
                 if last_msg_index != -1:
                     st.session_state.conversation_history[last_msg_index]["content"] = summary # Replace with summary
                     # Add a caption below the summary in the chat history if needed
                     # st.session_state.conversation_history.insert(last_msg_index + 1, {"role": "assistant", "content": "(According to my AI brain 🧠)", "is_caption": True}) # Custom flag for caption
                 else:
                     # Fallback: just append the summary if the original message wasn't found
                      st.session_state.conversation_history.append({"role": "assistant", "content": summary})

            st.rerun() # Rerun to display results


        except Exception as e:
            st.error(f"Error processing the AI response: {e}")
            print(f"Raw LLM Response causing error:\n{llm_response_raw}") # Print raw output for debugging
            st.session_state.processing_complete = True # Mark as complete even on error to stop loop
            st.session_state.current_step += 1 # Move to display step (will show error message)
            st.rerun()


# --- Step N+2: Display Results & Buddy Finder ---
# This step triggers AFTER processing_complete is True
elif st.session_state.processing_complete:

    # --- Display LLM Summary ---
    # Summary is now handled in the previous step and added to chat history.

    # --- Fetch and Display Movie Recs ---
    # Only fetch/process recommendations if not already done in the same 'complete' state
    if st.session_state.final_recommendations is None and st.session_state.llm_titles:
         with st.spinner("Grabbing the movie posters & deets... 🍿"):
              st.session_state.final_recommendations = find_movies_by_title(movie_data, st.session_state.llm_titles)
         # Rerun once after fetching to display immediately, but only if recs were found
         if st.session_state.final_recommendations is not None and not st.session_state.final_recommendations.empty:
              # Determine available moods from these recommendations upon first fetch
              unique_moods = sorted(list(st.session_state.final_recommendations['final_mood'].unique()))
              # Filter out 'Unknown' and any empty strings/None
              st.session_state.available_buddy_moods = [m for m in unique_moods if m and m != 'Unknown']
              # Set a default selected mood if available
              if st.session_state.available_buddy_moods and st.session_state.selected_buddy_mood not in st.session_state.available_buddy_moods:
                   st.session_state.selected_buddy_mood = st.session_state.available_buddy_moods[0]

              st.rerun() # Rerun to display recs and buddy UI


    recs = st.session_state.final_recommendations # Get recs from state


    if recs is not None and not recs.empty:
         st.markdown("---") # Separator before recommendations
         st.subheader("Okay, based on our chat, check these out! 👇")

         num_columns = 2 # Fewer columns for centered layout might look better
         cols = st.columns(num_columns)
         col_index = 0

         for index, movie in recs.iterrows():
             # Ensure we don't go out of column bounds if recs count is odd
             current_col = cols[col_index % num_columns]
             with current_col:
                 with st.container(border=True):
                     movie_title = movie.get('Title', 'N/A')
                     # Caching the poster URL separately
                     poster_url = get_poster_url(movie_title, TMDB_API_KEY)
                     st.image(poster_url, use_column_width=True)
                     st.markdown(f"**{movie_title}**")

                     movie_rating_val = movie.get('vote_average', 0)
                     movie_rating = f"{movie_rating_val:.1f} ⭐" if pd.notna(movie_rating_val) and movie_rating_val > 0 else "N/A"
                     movie_runtime = movie.get('runtime', 'N/A')
                     genres_list = movie.get('genres', 'N/A').split(',')
                     top_genre = genres_list[0].strip() if genres_list and genres_list[0].strip() else "Misc"
                     st.caption(f"{movie_rating} | {movie_runtime} | {top_genre}")

                     with st.expander("More Deets 👀"):
                         st.markdown(f"**Mood Tag:** `{movie.get('final_mood', 'N/A')}`")
                         st.markdown(f"**Director:** {movie.get('Director', 'N/A')}")
                         cast_list = [c.strip() for c in movie.get('Cast', '').split(',') if c.strip()]
                         cast_display = ", ".join(cast_list[:3]) + ('...' if len(cast_list) > 3 else '')
                         if cast_display: st.markdown(f"**Starring:** {cast_display}")
                         st.caption(f"**Synopsis:** {movie.get('overview', 'N/A')}")
             col_index += 1

         # --- Watch with Mood Buddies Feature (Interactive) ---
         st.markdown("---")
         st.subheader("👯 Watch with Mood Buddies")

         available_moods = st.session_state.available_buddy_moods

         if not available_moods:
             st.info("Couldn't find specific moods for buddy matching from these recommendations. 🤔")
         else:
             # Add a default 'Select a Mood' option if nothing is selected yet
             display_moods = available_moods[:]
             if st.session_state.selected_buddy_mood is None:
                  display_moods.insert(0, "Select a Mood...")
                  selected_mood_index = 0 # Default to 'Select a Mood...'
             else:
                 # Find the index of the currently selected mood
                 try: selected_mood_index = display_moods.index(st.session_state.selected_buddy_mood)
                 except ValueError:
                      # If the previously selected mood is no longer available, default
                      display_moods.insert(0, "Select a Mood...")
                      selected_mood_index = 0


             # UI for selecting mood and triggering search
             col_select, col_button = st.columns([0.7, 0.3])
             with col_select:
                 selected_mood_for_search = st.selectbox(
                     "Which mood are you looking for buddies for?",
                     options=display_moods,
                     index=selected_mood_index,
                     key="buddy_mood_select"
                 )
             with col_button:
                  # Align button vertically with selectbox
                  st.markdown("<br>", unsafe_allow_html=True) # Add some space
                  find_buddies_button = st.button("Find Buddies! ✨", key="find_buddies_button")


             # Logic to find and display buddies when a mood is selected or button is clicked
             if selected_mood_for_search and selected_mood_for_search != "Select a Mood..." :

                 # Update session state if mood selection changed via selectbox
                 if st.session_state.selected_buddy_mood != selected_mood_for_search:
                      st.session_state.selected_buddy_mood = selected_mood_for_search
                      # Clear previous results when mood changes
                      st.session_state.current_buddies_list = []
                      # No rerun here, the button click or subsequent interaction will trigger

                 # Trigger buddy search if button clicked OR if mood was just selected and it's valid
                 if find_buddies_button or (st.session_state.selected_buddy_mood == selected_mood_for_search and not st.session_state.current_buddies_list): # Added check for empty list to auto-find on select if not already found

                      st.session_state.selected_buddy_mood = selected_mood_for_search # Ensure state is correct
                      with st.spinner(f"Finding buddies for '{selected_mood_for_search}' mood..."):
                           # Add current user with the selected mood
                           add_or_update_buddy(BUDDY_DATA_FILE, st.session_state.user_id, selected_mood_for_search)
                           # Find matches
                           matching_buddies = find_matching_buddies(BUDDY_DATA_FILE, st.session_state.user_id, selected_mood_for_search)
                           st.session_state.current_buddies_list = matching_buddies # Store results

                      # Rerun to display results after finding buddies
                      st.rerun() # Essential to update the UI with the results


             # Display the found buddies (if any and a search was performed)
             if st.session_state.current_buddies_list:
                 st.write(f"✨ Found buddies currently looking for a **`{st.session_state.selected_buddy_mood}`** vibe! ✨")
                 st.write("Potential Movie Buddies:")
                 # Display anonymous IDs (or generate simple names)
                 # Generate a simple name for each buddy based on their ID hash for consistency
                 buddy_display_names = []
                 for buddy in st.session_state.current_buddies_list:
                      buddy_id = buddy.get('id', 'unknown')
                      # Simple hash to get a number for anonymous display
                      unique_num = hash(buddy_id) % 10000 # Get a number between 0-9999
                      buddy_display_names.append(f"Buddy-{unique_num:04d}") # Pad with zeros

                 buddy_list_display = ", ".join(buddy_display_names)
                 st.markdown(f"🔗 Potential connections: **{buddy_list_display}**")
                 st.caption(f"*(These users looked for the **`{st.session_state.selected_buddy_mood}`** mood in the last {BUDDY_TIMEOUT_MINUTES} mins)*")
                 st.caption("*(Note: This is a simple prototype feature. There's no direct chat or linking mechanism here.)*")

             elif st.session_state.selected_buddy_mood is not None and st.session_state.selected_buddy_mood != "Select a Mood...":
                 # Display this message if a mood was selected and the list is empty
                 st.info(f"No other buddies currently looking for a **`{st.session_state.selected_buddy_mood}`** vibe. You're a trendsetter! 😎 Check back in a bit or try a different mood.")


         # --- Display LLM Personal Pick ---
         if st.session_state.llm_pick_text:
               st.markdown("---") # Separator before personal pick
               with st.container(border=True):
                   st.subheader("🔥 My Personal Hot Take:")
                   st.markdown(f"> {st.session_state.llm_pick_text}")


    elif st.session_state.llm_titles: # If LLM gave titles but find_movies_by_title returned empty
         st.warning("The AI suggested some titles, but I couldn't find them in my database. Weird! 😕 Maybe try again?")

    else: # If llm_titles was empty
         st.warning("The AI couldn't come up with specific titles this time. Maybe try rephrasing your answers?")


    # --- Start Over Button ---
    st.markdown("---") # Separator before Start Over
    if st.button("🔄 Start a New Vibe Search"):
        # Reset relevant state variables
        keys_to_reset = [
            "current_step", "conversation_history", "user_profile_info",
            "llm_summary", "llm_titles", "llm_pick_text",
            "final_recommendations", "processing_complete",
            "available_buddy_moods", "selected_buddy_mood", "current_buddies_list" # Added buddy states
            ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


# --- Footer --- (Optional for chat interface)
# st.caption("MoodFlixx Chat Edition ✨")