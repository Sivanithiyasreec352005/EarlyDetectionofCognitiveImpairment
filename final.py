import tkinter as tk
from tkinter import messagebox, ttk
import pickle
import numpy as np
import cv2
import tensorflow as tf
import time

# ===========================
# Load Saved Models
# ===========================
# Naive Bayes model for cognitive questionnaire
with open("cognitive_nb_model.pkl", "rb") as f:
    cognitive_model = pickle.load(f)

# Face stress model
face_model = tf.keras.models.load_model("FACE.model")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ===========================
# Likert scale & questions
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

# ===========================
# GUI Setup
# ===========================
root = tk.Tk()
root.title("Cognitive + Stress Assessment")
root.geometry("600x650")
answers = []

# ===========================
# Function to capture face and calculate stress level
# ===========================
def capture_face_stress():
    cam = cv2.VideoCapture(0)
    sample_frames = 50
    frame_counter = 0
    image_samples = []

    messagebox.showinfo("Camera", "Camera will open now. Please look at the camera.")

    while frame_counter < sample_frames:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
            im = gray[y:y+h, x:x+w]

        cv2.imshow('Capturing Face Frames', img)
        if 'im' in locals():
            im_array = cv2.resize(im, (50,50))
            im_array = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)
            im_array = np.expand_dims(im_array, axis=0)
            image_samples.append(im_array)
            frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if len(image_samples) == 0:
        return 0.0  # no faces captured

    image_samples = np.concatenate(image_samples, axis=0)
    predictions = face_model.predict(image_samples)
    stress_predictions = predictions[:, 0]  # Assuming index 0 corresponds to stress level
    avg_stress = np.mean(stress_predictions)
    return avg_stress  # Value between 0 and 1

# ===========================
# Function to submit questionnaire and combine score
# ===========================
def submit_responses():
    responses = []
    for combo in answers:
        selected = combo.get()
        if selected == "":
            messagebox.showwarning("Warning", "Please answer all questions.")
            return
        responses.append(likert_map[selected])

    responses = np.array(responses).reshape(1, -1)
    cognitive_pred = cognitive_model.predict(responses)[0]  # 0=Normal,1=Impaired

    # Convert cognitive prediction to 0-5 scale
    cognitive_score = cognitive_pred * 5 + np.sum(responses)/15 * 5 / 5  # roughly 0-5 scale

    # Capture face and get stress level
    face_score = capture_face_stress() * 5  # scale to 0-5

    # Combine both for 0-10 scale
    overall_score = cognitive_score + face_score
    overall_score = np.clip(overall_score, 0, 10)

    status = "Impaired" if overall_score >= 5 else "Normal"
    messagebox.showinfo("Result", f"Overall Cognitive-Stress Score: {overall_score:.2f}/10\nStatus: {status}")

# ===========================
# Scrollable GUI with dropdowns
# ===========================
canvas = tk.Canvas(root)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
frame = tk.Frame(canvas)

options = list(likert_map.keys())

for i, q in enumerate(questions):
    tk.Label(frame, text=f"{i+1}. {q}", wraplength=550, justify="left", anchor="w", font=("Arial", 10, "bold")).pack(anchor="w", pady=5)
    combo = ttk.Combobox(frame, values=options, state="readonly")
    combo.pack(fill="x", padx=10, pady=2)
    answers.append(combo)

tk.Button(frame, text="Submit", command=submit_responses, bg="green", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

canvas.create_window(0,0, anchor="nw", window=frame)
canvas.update_idletasks()
canvas.configure(scrollregion=canvas.bbox("all"), yscrollcommand=scroll_y.set)
canvas.pack(fill="both", expand=True, side="left")
scroll_y.pack(fill="y", side="right")

root.mainloop()
