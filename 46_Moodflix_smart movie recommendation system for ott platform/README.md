
 🎬 MoodFlixx: Mood-Based Movie Recommendation System

**Hackathon Project - AITech_IITMandi_Hackathon25**

---

 🧾 Objective

To solve the problem of overwhelming choices on OTT platforms, this system provides movie recommendations based on the **user's current mood** using AI techniques.  
This aligns with **Problem 9**, which requires:

- Mood-based questionnaire and input
- Integration of LLMs for recommendations
- Movie metadata analysis (genre, cast, overview)
- Lightweight UX (under 10 questions)
- Optional voice input/output and camera use

---

 ✅ Achievements (Problem 9 Coverage)

| Requirement                                | Achieved |
|--------------------------------------------|:--------:|
| Mood-based interactive questionnaire       | ✅        |
| Less than 10 questions                     | ✅        |
| Voice input and TTS output                 | ✅        |
| Free-text mood expression                  | ✅        |
| Movie metadata + NLP + Gemini AI used      | ✅        |
| Poster-based result display                | ✅        |
| Camera-ready for facial emotion detection  | 🟡 (In progress)       |
| LLM-based personalization with Gemini      | ✅        |

---

 🚀 How to Run the Project

 Step 1: Install the Required Libraries

Use pip to install the dependencies:

```bash
pip install streamlit gTTS SpeechRecognition pygame google-generativeai python-dotenv requests pandas nltk
```

---

 Step 2: Setup API Keys

Create a file named `.env` in the root directory and add:

```
GOOGLE_API_KEY=your_google_api_key
TMDB_API_KEY=your_tmdb_api_key
```

---

 Step 3: Run the Modules

 1. 🗨️ Text Chat Version (Chat-based mood Q&A)
```bash
streamlit run buddies.py
```

 2. 🎙️ Voice-Based Version (with speech input/output)
```bash
streamlit run TTSv2-Final.py
```

Allow microphone access in your browser when prompted.

---

 📁 Files Included

```
📂 MoodFlixx/
├── buddies.py               Chat-based mood Q&A assistant
├── TTSv2-Final.py           Voice-based assistant using Gemini
├── enhanced_movies.csv      Dataset with genre, mood, ratings
├── .env                     API keys file (you create this)
```

---

 💡 Notes

- Use Chrome or Firefox for best microphone support.
- Output includes movie title, overview, rating, and posters.
- Voice UI speaks the recommendation aloud.

---

Developed as part of the Hackathon at IIT Mandi.
