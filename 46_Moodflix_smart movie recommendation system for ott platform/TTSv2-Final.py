import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import pygame
import os
import tempfile
import time
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from io import BytesIO
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize pygame mixer
pygame.mixer.init()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file")
    st.stop()

# Initialize Google AI
genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 3000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.5
        self.is_speaking = False
        self.is_listening = False
        self.mic = None
        self._setup_audio_devices()
    
    def _setup_audio_devices(self):
        """Check microphone and speaker availability"""
        try:
            self.mic = sr.Microphone()
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.mic_available = True
        except Exception as e:
            st.error(f"Microphone setup failed: {str(e)}")
            self.mic_available = False
        
        try:
            pygame.mixer.music.set_volume(0.8)
            self.speaker_available = True
        except Exception as e:
            st.error(f"Speaker setup failed: {str(e)}")
            self.speaker_available = False
    
    def speak(self, text, lang='en'):
        """Convert text to speech and play it"""
        if not self.speaker_available or not text:
            return False
        
        self.is_speaking = True
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmpfile:
                temp_path = tmpfile.name
            
            try:
                tts = gTTS(text=text, lang=lang, slow=False)
                tts.save(temp_path)
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                return True
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass
        except Exception as e:
            st.error(f"Speech error: {str(e)}")
            return False
        finally:
            self.is_speaking = False
    
    def listen(self):
        """Listen to microphone and return transcribed text"""
        if not self.mic_available or not self.mic:
            return None
        
        self.is_listening = True
        try:
            with self.mic as source:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=8)
                try:
                    return self.recognizer.recognize_google(audio)
                except sr.UnknownValueError:
                    return ""
                except Exception as e:
                    st.error(f"Recognition error: {str(e)}")
                    return None
        except sr.WaitTimeoutError:
            return None
        except Exception as e:
            st.error(f"Listening error: {str(e)}")
            return None
        finally:
            self.is_listening = False

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = VoiceAssistant()
    st.session_state.conversation_step = 0
    st.session_state.user_responses = {}
    st.session_state.processing = False
    st.session_state.recommendations = None
    st.session_state.movie_posters = []
    st.session_state.last_message = ""
    st.session_state.waiting_for_response = False
    st.session_state.waiting_for_feedback = False
    st.session_state.conversation_active = False
    st.session_state.conversation_history = []

# Conversation flow
QUESTIONS = [
    "Hello! I'm your movie recommendation assistant. How are you feeling today?",
    "What kind of movies do you typically enjoy? Any favorite genres?",
    "What mood are you in for a movie tonight? Exciting? Relaxing? Thought-provoking?",
    "Any dealbreakers? Actors you don't like? Genres you want to avoid?"
]

def generate_feedback(response, step):
    """Generate personalized feedback using AI"""
    feedback_prompts = {
        0: f"The user said: '{response}' about how they're feeling. Generate a warm, empathetic response that acknowledges their mood and naturally transitions to asking about movie preferences.",
        1: f"The user's movie preferences are: '{response}'. Create a thoughtful response showing you understand their tastes, then ask what mood they're in for a movie tonight.",
        2: f"The user wants a movie that matches this mood: '{response}'. Respond appropriately to their desired mood, then ask about any dealbreakers.",
        3: f"The user's movie dealbreakers are: '{response}'. Acknowledge these preferences and say you'll now find perfect recommendations."
    }
    
    try:
        feedback = llm.generate_content(feedback_prompts.get(step, "Thank you! I'll now find recommendations."))
        return feedback.text if feedback.text else "Interesting! Let's continue."
    except Exception as e:
        st.error(f"Feedback error: {str(e)}")
        return "Thanks for sharing! Let's move on."

def get_movie_poster(movie_title):
    """Fetch movie poster from TMDB API"""
    if not TMDB_API_KEY:
        return None
    
    try:
        # First search for the movie to get its ID
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        search_response = requests.get(search_url).json()
        
        if search_response.get('results'):
            movie_id = search_response['results'][0]['id']
            
            # Get movie details including images
            movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
            movie_response = requests.get(movie_url).json()
            
            if movie_response.get('poster_path'):
                poster_url = f"https://image.tmdb.org/t/p/w500{movie_response['poster_path']}"
                return poster_url
    except Exception as e:
        st.error(f"Couldn't fetch poster for {movie_title}: {str(e)}")
    return None

def generate_recommendations():
    """Generate and display recommendations with posters"""
    st.session_state.processing = True
    st.session_state.assistant.speak("Analyzing your preferences...")
    
    # Collect all user responses
    prompt = "Based on these user responses:\n"
    for i, response in st.session_state.user_responses.items():
        prompt += f"{QUESTIONS[i]}\nUser: {response}\n\n"
    
    prompt += """
    Please recommend 3 movies that would be perfect for this user.
    For each movie, provide ONLY:
    1. Exact movie title (very important for poster search)
    2. Release year
    3. Main genre
    4. One sentence why it's a good match
    
    Format exactly like this example:
    The Shawshank Redemption (1994) - Drama: An uplifting story about hope and friendship.
    The Dark Knight (2008) - Action: Gripping superhero film with deep themes.
    Inception (2010) - Sci-Fi: Mind-bending exploration of dreams and reality.
    """
    
    try:
        response = llm.generate_content(prompt)
        if response.text:
            # Clean and store recommendations
            st.session_state.recommendations = response.text
            
            # Extract movie titles for poster search
            movie_lines = [line for line in response.text.split('\n') if line.strip()]
            movie_titles = [line.split('(')[0].strip() for line in movie_lines if '(' in line]
            
            # Fetch posters
            st.session_state.movie_posters = []
            for title in movie_titles[:3]:  # Get max 3 posters
                poster_url = get_movie_poster(title)
                if poster_url:
                    st.session_state.movie_posters.append(poster_url)
            
            # Speak recommendations
            st.session_state.assistant.speak("Here are my recommendations!")
            st.session_state.assistant.speak(response.text)
            
            # Add to conversation history
            st.session_state.conversation_history.append(("Assistant", "Here are my recommendations:"))
            st.session_state.conversation_history.append(("Assistant", response.text))
    except Exception as e:
        error_msg = "Sorry, I encountered an error while generating recommendations."
        st.error(f"Error: {str(e)}")
        st.session_state.assistant.speak(error_msg)
        st.session_state.conversation_history.append(("Assistant", error_msg))
    
    st.session_state.processing = False
    st.rerun()

# Main app interface
st.title("🎬 Interactive Movie Recommender")

# Check microphone permission
if not st.session_state.assistant.mic_available:
    st.warning("""
    Microphone access is required for this app to work.
    Please:
    1. Refresh this page
    2. Click "Allow" when prompted for microphone access
    3. Make sure your microphone is properly connected
    """)
    st.stop()

# Status display
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Microphone:** {'🎤 Active' if st.session_state.assistant.is_listening else '🎙️ Ready'}")
with col2:
    st.markdown(f"**Speaker:** {'🔊 Active' if st.session_state.assistant.is_speaking else '🔇 Ready'}")

# Start conversation
if not st.session_state.conversation_active:
    if st.button("🎤 Start Conversation"):
        st.session_state.conversation_active = True
        st.session_state.conversation_step = 0
        st.session_state.user_responses = {}
        st.session_state.recommendations = None
        st.session_state.movie_posters = []
        st.session_state.waiting_for_response = True
        st.session_state.conversation_history = []
        st.rerun()

# Conversation logic
if st.session_state.conversation_active:
    # Ask current question
    if st.session_state.waiting_for_response and st.session_state.conversation_step < len(QUESTIONS):
        question = QUESTIONS[st.session_state.conversation_step]
        st.session_state.last_message = question
        st.session_state.conversation_history.append(("Assistant", question))
        if st.session_state.assistant.speak(question):
            st.session_state.waiting_for_response = False
            st.rerun()
    
    # Listen for response
    elif not st.session_state.waiting_for_response and not st.session_state.waiting_for_feedback and st.session_state.conversation_step < len(QUESTIONS):
        response = st.session_state.assistant.listen()
        if response is not None:
            if response.strip():  # Only store non-empty responses
                st.session_state.user_responses[st.session_state.conversation_step] = response
                st.session_state.conversation_history.append(("You", response))
                st.session_state.waiting_for_feedback = True
            st.rerun()
    
    # Generate feedback
    elif st.session_state.waiting_for_feedback and st.session_state.conversation_step < len(QUESTIONS):
        feedback = generate_feedback(
            st.session_state.user_responses.get(st.session_state.conversation_step, ""),
            st.session_state.conversation_step
        )
        
        # Speak feedback
        st.session_state.assistant.speak(feedback)
        st.session_state.conversation_history.append(("Assistant", feedback))
        
        # Move to next question or generate recommendations
        st.session_state.conversation_step += 1
        st.session_state.waiting_for_feedback = False
        
        if st.session_state.conversation_step < len(QUESTIONS):
            st.session_state.waiting_for_response = True
        st.rerun()
    
    # Generate recommendations after last question
    elif st.session_state.conversation_step == len(QUESTIONS) and not st.session_state.recommendations:
        generate_recommendations()

# Display conversation
st.markdown("---")
st.subheader("Conversation History")

for sender, message in st.session_state.conversation_history:
    if sender == "Assistant":
        with st.chat_message("Assistant"):
            st.write(message)
    else:
        with st.chat_message("User"):
            st.write(message)

# Show recommendations
if st.session_state.recommendations:
    st.markdown("---")
    st.subheader("🎉 Your Recommendations")
    st.write(st.session_state.recommendations)
    
    # Display posters if available
    if st.session_state.movie_posters:
        st.markdown("### Movie Posters")
        cols = st.columns(len(st.session_state.movie_posters))
        for i, poster_url in enumerate(st.session_state.movie_posters):
            try:
                response = requests.get(poster_url)
                img = Image.open(BytesIO(response.content))
                cols[i].image(img, use_column_width=True)
            except:
                pass

# Repeat button
if st.session_state.last_message and st.button("🔁 Repeat Last Message"):
    st.session_state.assistant.speak(st.session_state.last_message)
    st.rerun()

# Initial instructions
if not st.session_state.conversation_active and st.session_state.assistant.speaker_available:
    st.session_state.assistant.speak("Click the Start Conversation button to begin")