# ================== Imports ==================
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# ================== Load Dataset ==================
df = pd.read_csv("/home/madhan/Mini Projects/Emotion-based-music-recommendation-system/muse_v3.csv")

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name','emotional','pleasant','link','artist']]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

df_sad     = df[:18000]
df_fear    = df[18000:36000]
df_angry   = df[36000:54000]
df_neutral = df[54000:72000]
df_happy   = df[72000:]

# ================== Recommendation Logic ==================
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
        data = pd.concat(
            [data, mapping.get(emo, df_sad).sample(n=counts[i])],
            ignore_index=True
        )
    return data

def most_common(emotion_list):
    return [e for e, _ in Counter(emotion_list).most_common()]

# ================== Load Emotion Model ==================
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
    "/home/madhan/Mini Projects/Emotion-based-music-recommendation-system/model.h5"
)

emotion_dict = {
    0:"Angry", 1:"Disgusted", 2:"Fearful",
    3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"
}

# 🔑 NORMALIZATION (CRITICAL FIX)
EMOTION_MAP = {
    "Disgusted": "Sad",
    "Surprised": "happy",
    "Fearful": "fear",
    "Happy": "happy",
    "Angry": "Angry",
    "Neutral": "Neutral",
    "Sad": "Sad"
}

# ================== UI ==================
st.markdown("<h2 style='text-align:center'>Emotion Based Music Recommendation</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center;color:grey'>Scan your face to detect emotion</h5>", unsafe_allow_html=True)

frame_box = st.empty()
emotion_list = []

if st.button("SCAN EMOTION"):
    cap = cv2.VideoCapture(0)
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
            emotion_list.append(emotion)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2)

        frame_box.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

    cap.release()
    st.success("Emotion detected successfully")

# ================== Recommendations ==================
if emotion_list:
    final_emotions = most_common(emotion_list)
    rec_df = recommend(final_emotions)

    st.markdown("<h4 style='text-align:center'>Recommended Songs</h4>", unsafe_allow_html=True)

    for i,(l,a,n) in enumerate(zip(rec_df["link"], rec_df["artist"], rec_df["name"])):
        st.markdown(f"<h5 style='text-align:center'><a href='{l}'>{i+1}. {n}</a></h5>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;color:grey'>{a}</p>", unsafe_allow_html=True)
