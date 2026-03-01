# Emotion-Based Music Recommendation System

A modern web application that recommends music based on your emotions using multiple input methods. Built with a beautiful Spotify-inspired UI, this app analyzes your emotions through face scanning, text input, or speech recognition, and provides personalized music recommendations from Spotify.

## Features

- **Face Scan**: Real-time emotion detection using your webcam and a trained deep learning model
- **Text Input**: Type your emotions (e.g., "happy song", "I feel sad") for instant recommendations
- **Speech Input**: Speak your emotions and let the app transcribe and analyze them
- **Spotify-Style UI**: Beautiful dark theme interface inspired by Spotify's design
- **Spotify Integration**: Direct links to songs on Spotify for seamless music discovery
- **Smart Emotion Mapping**: Advanced emotion detection with keyword matching and ML-based face analysis

## Installation & Setup

### Prerequisites

- Python 3.7 or higher
- Webcam (for face scanning feature)
- Microphone (for speech input feature)
- Internet connection (for speech recognition API)

### Installation Steps

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: On some systems, you may need to install `pyaudio` separately:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install portaudio19-dev python3-pyaudio
   
   # On macOS
   brew install portaudio
   pip install pyaudio
   
   # On Windows
   pip install pipwin
   pipwin install pyaudio
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open in your default browser
   - If not, navigate to `http://localhost:8501`

## How to Use

### Method 1: Face Scan
1. Click on the **"Face Scan"** tab
2. Position your face in front of your webcam
3. Click the **"SCAN EMOTION"** button
4. The app will capture multiple frames and detect your emotion
5. View your personalized music recommendations

### Method 2: Text Input
1. Click on the **"Text Input"** tab
2. Type how you're feeling in the text box
   - Examples: "happy song", "I feel sad", "play angry music", "feeling neutral"
3. Click **"SUBMIT"**
4. The app will analyze your text and recommend music

### Method 3: Speech Input
1. Click on the **"Speech Input"** tab
2. Click the microphone button to record your voice
3. Speak your emotion (e.g., "I'm feeling happy", "play sad music")
4. Click **"TRANSCRIBE & DETECT"**
5. The app will transcribe your speech and recommend music

### Viewing Recommendations
- After detecting your emotion, scroll down to see your personalized playlist
- Click on any song to open it directly in Spotify
- Songs are sorted by emotion relevance

## Technologies Used

### Core Libraries
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision for face detection
- **TensorFlow & Keras**: Deep learning model for emotion recognition
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis

### Additional Libraries
- **SpeechRecognition**: Speech-to-text conversion
- **pydub**: Audio processing
- **pyaudio**: Audio I/O operations
- **Pillow**: Image processing
- **scikit-learn**: Machine learning utilities

## Project Structure

```
Emotion-based-music-recommendation-system/
├── app.py                          # Main application file
├── model.h5                        # Trained emotion detection model
├── haarcascade_frontalface_default.xml  # Face detection cascade
├── muse_v3.csv                     # English music dataset with Spotify links
├── tamil_songs.csv                 # Tamil songs dataset (optional - create this file)
├── tamil_songs_template.csv        # Template for Tamil songs CSV
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Adding Tamil Songs

The app supports Tamil songs! To add Tamil songs:

1. **Create a CSV file** named `tamil_songs.csv` in the project directory

2. **Required columns**:
   - `spotify_id`: Spotify track ID
   - `track`: Song name
   - `artist`: Artist name
   - `number_of_emotion_tags`: Emotion intensity (0-5, where 0=sad, 5=very happy)
   - `valence_tags`: Valence score (0-1, where 0=negative, 1=positive)

3. **Example format**:
   ```csv
   spotify_id,track,artist,number_of_emotion_tags,valence_tags
   4r7spK3M05WxR6GP35yzjJ,Enna Solla Pogirai,Anirudh Ravichander,4,0.8
   5K1m5c9M1zJqoUFIQJd2PD,Why This Kolaveri Di,Dhanush,5,0.9
   ```

4. **Getting Spotify IDs**:
   - Open Spotify and find the song
   - Right-click → Share → Copy Song Link
   - Extract the ID from the URL (the part after `/track/`)
   - Example: `https://open.spotify.com/track/4r7spK3M05WxR6GP35yzjJ` → ID is `4r7spK3M05WxR6GP35yzjJ`

5. **Using the app**:
   - Select "Tamil" from the language selector at the top
   - Scan your emotion or enter text/speech
   - Get Tamil song recommendations!

**Note**: If `tamil_songs.csv` doesn't exist, the app will show English songs by default. You can still select "All" to see both languages if Tamil songs are added.

## How It Works

1. **Emotion Detection**:
   - **Face Scan**: Uses OpenCV for face detection and a CNN model for emotion classification
   - **Text Input**: Keyword matching to identify emotions from text
   - **Speech Input**: Converts speech to text, then uses keyword matching

2. **Emotion Processing**:
   - Multiple emotion detections are collected and sorted by frequency
   - Emotions are mapped to standardized categories (Happy, Sad, Angry, Fear, Neutral)

3. **Music Recommendation**:
   - Songs are selected from a curated dataset based on detected emotions
   - Recommendations are weighted by emotion frequency
   - All songs include direct Spotify links
