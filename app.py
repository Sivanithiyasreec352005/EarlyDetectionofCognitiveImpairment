import streamlit as st
import sqlite3
import pickle
import numpy as np
import cv2
import tensorflow as tf

# ===========================
# Load Models
# ===========================
with open("cognitive_nb_model.pkl", "rb") as f:
    cognitive_model = pickle.load(f)

face_model = tf.keras.models.load_model("FACE.model")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ===========================
# Database Setup
# ===========================
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT
            )""")
conn.commit()

# ===========================
# Likert Scale & Questions
# ===========================
likert_map = {
    "Never / No difficulty": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always / Severe difficulty": 5
}

questions = [
    "Do you have difficulty remembering names of people you just met?",
    "Do you often lose track of time while doing tasks?",
    "Do you forget why you entered a room?",
    "Do you find it hard to follow conversations?",
    "Do you misplace items (keys, phone, etc.) frequently?",
    "Do you struggle to concentrate on reading material?",
    "Do you forget appointments or scheduled tasks?",
    "Do you have trouble learning new things?",
    "Do you find it difficult to multitask?",
    "Do you forget familiar routes or directions?",
    "Do you lose focus during meetings or classes?",
    "Do you have trouble recalling recent events?",
    "Do you often repeat questions because you forgot the answer?",
    "Do you struggle to find the right words in conversations?",
    "Do you forget birthdays, dates, or important events?"
]

st.set_page_config(page_title="Cognitive + Stress Assessment", page_icon="🧠", layout="centered")

# ===========================
# Session State
# ===========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "responses" not in st.session_state:
    st.session_state.responses = []

# ===========================
# Sidebar: Login/Register
# ===========================
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# --------------------------
# Register
# --------------------------
if choice == "Register":
    st.subheader("Create a New Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Register"):
        if new_user and new_pass:
            try:
                c.execute("INSERT INTO users VALUES (?, ?)", (new_user, new_pass))
                conn.commit()
                st.success("Registered successfully! Please login.")
            except sqlite3.IntegrityError:
                st.error("Username already exists!")
        else:
            st.warning("Enter username and password")

# --------------------------
# Login
# --------------------------
elif choice == "Login" and not st.session_state.logged_in:
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        result = c.fetchone()
        if result:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid username or password!")

# --------------------------
# Main Assessment Page
# --------------------------
if st.session_state.logged_in:
    st.subheader("📝 Cognitive Questionnaire")
    
    # Display dropdowns for questions
    user_answers = []
    for i, q in enumerate(questions):
        ans = st.selectbox(f"{i+1}. {q}", options=list(likert_map.keys()), key=f"q{i}")
        user_answers.append(likert_map[ans])
    
    if st.button("Submit Questionnaire"):
        st.session_state.responses = user_answers
        st.success("Cognitive questions submitted successfully!")

        # --------------------------
        # Webcam capture
        # --------------------------
        st.info("📸 Webcam will open. Press 'Q' to capture each frame. Capture at least 20 frames for accuracy.")
        cap = cv2.VideoCapture(0)
        face_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to open webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

            cv2.imshow("Press Q to Capture Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                if len(faces) > 0:
                    (x,y,w,h) = faces[0]  # take first face
                    face_img = gray[y:y+h, x:x+w]
                    im_array = cv2.resize(face_img, (50,50))
                    im_array = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)
                    im_array = np.expand_dims(im_array, axis=0)
                    pred = face_model.predict(im_array)
                    face_scores.append(pred[0,0])
                    st.success(f"Captured frame #{len(face_scores)}")
                else:
                    st.warning("No face detected! Try again.")

            if key == 27:  # ESC key to quit
                break

            if cv2.getWindowProperty("Press Q to Capture Frame", cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

        # --------------------------
        # Compute Overall Score
        # --------------------------
        cognitive_pred = cognitive_model.predict(np.array(st.session_state.responses).reshape(1,-1))[0]
        cognitive_score = cognitive_pred*5 + np.sum(st.session_state.responses)/15*5/5  # scale 0-5

        if len(face_scores) > 0:
            avg_face_score = np.mean(face_scores) * 5
        else:
            avg_face_score = 0

        overall_score = cognitive_score + avg_face_score
        overall_score = np.clip(overall_score, 0, 10)
        status = "Impaired" if overall_score >= 5 else "Normal"

        st.success(f"✅ Overall Cognitive-Stress Score: {overall_score:.2f}/10")
        st.info(f"Status: {status}")
