import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import speech_recognition as sr
import io
import tempfile
from pydub import AudioSegment
import html

import os

# Page configuration
st.set_page_config(
    page_title="Emotion Based Music | Spotify Style",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "muse_v3.csv"))


df = df[df['spotify_id'].notna() & (df['spotify_id'] != '')].copy()


df['link'] = df['spotify_id'].apply(lambda x: f"https://open.spotify.com/track/{x}" if pd.notna(x) and x != '' else None)
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name','emotional','pleasant','link','artist']]

df = df[df['link'].notna()].copy()

df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

total_songs = len(df)
df_sad     = df[:min(18000, total_songs//5)]
df_fear    = df[min(18000, total_songs//5):min(36000, 2*total_songs//5)]
df_angry   = df[min(36000, 2*total_songs//5):min(54000, 3*total_songs//5)]
df_neutral = df[min(54000, 3*total_songs//5):min(72000, 4*total_songs//5)]
df_happy   = df[min(72000, 4*total_songs//5):]

def recommend(emotions):
    data = pd.DataFrame()
    mapping = {
        "Neutral": df_neutral,
        "Angry": df_angry,
        "fear": df_fear,
        "happy": df_happy,
        "Sad": df_sad
    }
    counts = [30, 20, 15, 10, 5]

    for i, emo in enumerate(emotions[:5]):
        emotion_df = mapping.get(emo, df_sad)
        sample_size = min(counts[i], len(emotion_df))
        if sample_size > 0:
            sampled = emotion_df.sample(n=sample_size)
            sampled = sampled[sampled['link'].notna() & (sampled['link'].str.contains('open.spotify.com/track', na=False))]
            if len(sampled) > 0:
                data = pd.concat([data, sampled], ignore_index=True)
    return data

def most_common(emotion_list):
    return [e for e, _ in Counter(emotion_list).most_common()]

def parse_text_to_emotions(text):
    """
    Parse text input to extract emotions using keyword matching.
    Returns a list of emotions matching the existing format.
    """
    if not text:
        return []
    
    text_lower = text.lower()
    emotions = []
    
    # Keyword mapping - case insensitive
    emotion_keywords = {
        "happy": ["happy", "happiness", "joy", "joyful", "cheerful", "glad", "pleased"],
        "Sad": ["sad", "sadness", "sorrow", "unhappy", "depressed", "melancholy", "gloomy"],
        "Angry": ["angry", "anger", "mad", "furious", "irritated", "annoyed", "rage"],
        "fear": ["fear", "fearful", "afraid", "scared", "anxious", "worried", "nervous"],
        "Neutral": ["neutral", "calm", "peaceful", "relaxed", "normal", "fine", "okay"]
    }
    
    # Check for each emotion
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                emotions.append(emotion)
                break  # Only add each emotion once
    
    # If no emotions found, default to neutral
    if not emotions:
        emotions.append("Neutral")
    
    return emotions

def transcribe_audio(audio_data):
    """
    Convert audio data from Streamlit audio_input to text using speech recognition.
    Returns the transcribed text or None if transcription fails.
    """
    if audio_data is None:
        return None
    
    tmp_path = None
    wav_path = None
    
    try:
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Get the file extension from Streamlit's audio input (usually webm or wav)
        file_extension = audio_data.name.split('.')[-1] if hasattr(audio_data, 'name') else 'webm'
        
        # Save audio data to temporary file with original extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            audio_bytes = audio_data.read()
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Load audio file and convert to WAV format (speech_recognition expects WAV)
        audio = AudioSegment.from_file(tmp_path)
        
        # Export to WAV format
        wav_path = tmp_path.replace(f'.{file_extension}', '.wav')
        audio.export(wav_path, format="wav")
        
        # Use speech recognition
        with sr.AudioFile(wav_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data_sr = recognizer.record(source)
        
        # Try to recognize speech using Google Speech Recognition
        try:
            text = recognizer.recognize_google(audio_data_sr)
            return text
        except sr.UnknownValueError:
            st.error("Could not understand the audio. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results from speech recognition service: {e}")
            return None
                
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.load_weights(
    os.path.join(BASE_DIR, "model.h5")
)

emotion_dict = {
    0:"Angry", 1:"Disgusted", 2:"Fearful",
    3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"
}

EMOTION_MAP = {
    "Disgusted": "Sad",
    "Surprised": "happy",
    "Fearful": "fear",
    "Happy": "happy",
    "Angry": "Angry",
    "Neutral": "Neutral",
    "Sad": "Sad"
}

# Spotify-inspired CSS styling
st.markdown("""
<style>
    /* Spotify Color Palette */
    :root {
        --spotify-black: #121212;
        --spotify-dark: #191414;
        --spotify-green: #1DB954;
        --spotify-light-green: #1ed760;
        --spotify-gray: #b3b3b3;
        --spotify-dark-gray: #535353;
        --spotify-card: #181818;
        --spotify-hover: #282828;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #1a1a1a 0%, #121212 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(180deg, rgba(29, 185, 84, 0.1) 0%, rgba(18, 18, 18, 1) 100%);
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .main-subtitle {
        color: var(--spotify-gray);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--spotify-card);
        color: var(--spotify-gray);
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--spotify-green);
        color: #000000;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--spotify-hover);
        color: #ffffff;
    }
    
    /* Buttons styling */
    .stButton > button {
        background-color: var(--spotify-green);
        color: #000000;
        border: none;
        border-radius: 500px;
        padding: 14px 32px;
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: var(--spotify-light-green);
        transform: scale(1.05);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: var(--spotify-card);
        color: #ffffff;
        border: 1px solid var(--spotify-dark-gray);
        border-radius: 4px;
        padding: 12px 16px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--spotify-green);
        outline: none;
    }
    
    .stTextInput label {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Cards */
    .emotion-card {
        background-color: var(--spotify-card);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
    }
    
    .emotion-card:hover {
        background-color: var(--spotify-hover);
    }
    
    /* Song list styling */
    .song-item {
        background-color: transparent;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 0;
        display: flex !important;
        align-items: center;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        cursor: pointer;
        width: 100%;
        box-sizing: border-box;
        text-decoration: none;
        color: inherit;
    }
    
    .song-item:hover {
        background-color: var(--spotify-hover);
    }
    
    .song-item:hover .song-number {
        color: transparent;
    }
    
    .song-item:hover .song-number::before {
        content: "▶";
        color: #ffffff;
        font-size: 14px;
    }
    
    .song-number {
        color: var(--spotify-gray);
        font-weight: 600;
        min-width: 40px;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .song-info {
        flex: 1;
        margin-left: 16px;
    }
    
    .song-title {
        color: #ffffff;
        font-weight: 500;
        font-size: 16px;
        margin: 0;
        transition: color 0.2s ease;
    }
    
    .song-item:hover .song-title {
        color: #ffffff;
    }
    
    .song-artist {
        color: var(--spotify-gray);
        font-size: 14px;
        margin: 4px 0 0 0;
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(29, 185, 84, 0.1);
        border-left: 4px solid var(--spotify-green);
        color: var(--spotify-light-green);
    }
    
    .stWarning {
        background-color: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        color: #ffc107;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.1);
        border-left: 4px solid #dc3545;
        color: #dc3545;
    }
    
    /* Tab content */
    .tab-content {
        padding: 2rem 0;
    }
    
    /* Audio input styling */
    .stAudioInput {
        background-color: var(--spotify-card);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--spotify-green) transparent transparent transparent;
    }
    
    /* Recommendations section */
    .recommendations-section {
        background: linear-gradient(180deg, rgba(29, 185, 84, 0.05) 0%, transparent 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--spotify-black);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--spotify-dark-gray);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--spotify-gray);
    }
    
    /* Container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Better spacing */
    .element-container {
        margin-bottom: 1.5rem;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Audio input container */
    [data-testid="stAudioInput"] {
        background-color: var(--spotify-card);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Better focus states */
    *:focus {
        outline: 2px solid var(--spotify-green);
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1 class="main-title">🎵 Emotion Based Music</h1>
    <p class="main-subtitle">Discover music that matches your mood</p>
</div>
""", unsafe_allow_html=True)


if 'emotion_list' not in st.session_state:
    st.session_state.emotion_list = []


tab1, tab2, tab3 = st.tabs(["Face Scan", "Text Input", "Speech Input"])

with tab1:
    st.markdown("""
    <div class="tab-content">
        <div class="emotion-card">
            <h3 style='color: #ffffff; margin-bottom: 1rem;'>Scan Your Face</h3>
            <p style='color: #b3b3b3; margin-bottom: 1.5rem;'>Position your face in front of the camera and click the button below to detect your emotion.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    frame_box = st.empty()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("SCAN EMOTION", key="scan_emotion"):
            cap = cv2.VideoCapture(0)
            face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            temp_emotion_list = []

            for _ in range(20):
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(gray, 1.3, 5)

                for (x,y,w,h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    roi = cv2.resize(roi, (48,48))
                    roi = roi.reshape(1,48,48,1)

                    pred = model.predict(roi, verbose=0)
                    raw_emotion = emotion_dict[int(np.argmax(pred))]
                    emotion = EMOTION_MAP[raw_emotion]
                    temp_emotion_list.append(emotion)

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(29,185,84),2)
                    cv2.putText(frame, emotion, (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2)

                frame_box.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

            cap.release()
            st.session_state.emotion_list = temp_emotion_list
            if temp_emotion_list:
                detected_emotions = ', '.join(set(temp_emotion_list))
                st.markdown(f"""
                <div class="emotion-card" style="background: linear-gradient(135deg, rgba(29, 185, 84, 0.2) 0%, rgba(18, 18, 18, 1) 100%);">
                    <h4 style='color: #1DB954; margin: 0;'>✓ Emotion Detected Successfully</h4>
                    <p style='color: #ffffff; margin-top: 0.5rem;'>Detected emotions: <strong>{detected_emotions}</strong></p>
                </div>
                """, unsafe_allow_html=True)


with tab2:
    st.markdown("""
    <div class="tab-content">
        <div class="emotion-card">
            <h3 style='color: #ffffff; margin-bottom: 1rem;'>Enter Your Emotion</h3>
            <p style='color: #b3b3b3; margin-bottom: 0.5rem;'>Type how you're feeling and we'll recommend music for you.</p>
            <p style='color: #535353; font-size: 0.9rem; margin: 0;'>Examples: "happy song", "I feel sad", "play angry music", "feeling neutral"</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    text_input = st.text_input("", key="text_input", placeholder="e.g., happy song, sad music, I feel angry...")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("SUBMIT", key="submit_text"):
            if text_input:
                parsed_emotions = parse_text_to_emotions(text_input)
                if parsed_emotions:
                    # Simulate multiple detections like face scan (for consistency)
                    st.session_state.emotion_list = parsed_emotions * 5
                    detected_emotions = ', '.join(set(parsed_emotions))
                    st.markdown(f"""
                    <div class="emotion-card" style="background: linear-gradient(135deg, rgba(29, 185, 84, 0.2) 0%, rgba(18, 18, 18, 1) 100%);">
                        <h4 style='color: #1DB954; margin: 0;'>✓ Emotions Detected</h4>
                        <p style='color: #ffffff; margin-top: 0.5rem;'>Detected: <strong>{detected_emotions}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Could not detect emotions from the text. Please try again.")
            else:
                st.warning("Please enter some text.")

# Tab 3: Speech Input
with tab3:
    st.markdown("""
    <div class="tab-content">
        <div class="emotion-card">
            <h3 style='color: #ffffff; margin-bottom: 1rem;'>Speak Your Emotion</h3>
            <p style='color: #b3b3b3; margin-bottom: 1.5rem;'>Click the microphone button below, speak your emotion, and we'll transcribe and analyze it.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    audio_data = st.audio_input("", key="audio_input")
    
    if audio_data is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(" TRANSCRIBE & DETECT", key="transcribe_audio"):
                with st.spinner("Transcribing audio..."):
                    transcribed_text = transcribe_audio(audio_data)
                    
                if transcribed_text:
                    st.markdown(f"""
                    <div class="emotion-card" style="margin-bottom: 1rem;">
                        <p style='color: #1DB954; margin: 0; font-weight: 600;'>Transcribed:</p>
                        <p style='color: #ffffff; margin-top: 0.5rem;'>{transcribed_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    parsed_emotions = parse_text_to_emotions(transcribed_text)
                    if parsed_emotions:
                        # Simulate multiple detections like face scan (for consistency)
                        st.session_state.emotion_list = parsed_emotions * 5
                        detected_emotions = ', '.join(set(parsed_emotions))
                        st.markdown(f"""
                        <div class="emotion-card" style="background: linear-gradient(135deg, rgba(29, 185, 84, 0.2) 0%, rgba(18, 18, 18, 1) 100%);">
                            <h4 style='color: #1DB954; margin: 0;'>✓ Emotions Detected</h4>
                            <p style='color: #ffffff; margin-top: 0.5rem;'>Detected: <strong>{detected_emotions}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Could not detect emotions from the transcribed text.")
                else:
                    st.error("Failed to transcribe audio. Please try again.")

emotion_list = st.session_state.emotion_list

if emotion_list:
    final_emotions = most_common(emotion_list)
    rec_df = recommend(final_emotions)
    
    rec_df = rec_df[rec_df['link'].notna() & (rec_df['link'].str.contains('open.spotify.com/track', na=False))].copy()
    
    if len(rec_df) > 0:
        # Header section
        st.markdown("""
        <div class="recommendations-section">
            <h2 style='color: #ffffff; font-size: 2rem; font-weight: 900; margin-bottom: 0.5rem;'>Made for You</h2>
            <p style='color: #b3b3b3; font-size: 1.1rem; margin-bottom: 2rem;'>Based on your detected emotions: <strong style='color: #1DB954;'>{}</strong></p>
        </div>
        """.format(', '.join(final_emotions[:3])), unsafe_allow_html=True)
        
        # Playlist header
        st.markdown("""
        <div style='background-color: #181818; padding: 16px; border-radius: 8px 8px 0 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1);'>
            <div style='display: flex; align-items: center; color: #b3b3b3; font-size: 14px; font-weight: 600;'>
                <div style='min-width: 40px; text-align: center;'>#</div>
                <div style='flex: 1; margin-left: 16px;'>TITLE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Build song list HTML
        songs_html = ""
        for i, (l, a, n) in enumerate(zip(rec_df["link"], rec_df["artist"], rec_df["name"]), 1):
            if pd.notna(l) and 'open.spotify.com/track' in str(l):
                # Escape HTML special characters to prevent breaking HTML
                song_title = html.escape(str(n)) if pd.notna(n) else ""
                song_artist = html.escape(str(a)) if pd.notna(a) else ""
                song_link = str(l) if pd.notna(l) else "#"
                
                # Make entire song item a clickable link
                songs_html += f'<a href="{song_link}" target="_blank" class="song-item"><div class="song-number">{i}</div><div class="song-info"><p class="song-title">{song_title}</p><p class="song-artist">{song_artist}</p></div></a>'
        
        # Render all songs in one container
        st.markdown(
            f'<div style="background-color: #181818; padding: 0 16px 16px 16px; border-radius: 0 0 8px 8px;">{songs_html}</div>',
            unsafe_allow_html=True
        )
        
        # Footer note
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem; padding: 1rem; background-color: rgba(29, 185, 84, 0.1); border-radius: 8px;'>
            <p style='color: #b3b3b3; font-size: 0.9rem; margin: 0;'>
                Click on any song to open it in Spotify. Make sure you have Spotify installed or use the web player.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="emotion-card" style="text-align: center; padding: 3rem;">
            <h3 style='color: #ffffff; margin-bottom: 1rem;'>No Songs Found</h3>
            <p style='color: #b3b3b3;'>We couldn't find songs with valid Spotify links. Please try detecting your emotion again.</p>
        </div>
        """, unsafe_allow_html=True)
