import tkinter as tk
from tkinter import messagebox, ttk
import pickle
import numpy as np

# ===========================
# 1. Load Saved Model
# ===========================
with open("cognitive_nb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Likert scale mapping
likert_map = {
    "Never / No difficulty": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always / Severe difficulty": 5
}

# List of questions
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
# 2. GUI Setup
# ===========================
root = tk.Tk()
root.title("Cognitive Impairment Test")
root.geometry("600x650")

answers = []  # Store selected answers (combobox variables)

def submit_responses():
    responses = []
    for combo in answers:
        selected = combo.get()
        if selected == "":
            messagebox.showwarning("Warning", "Please answer all questions before submitting.")
            return
        responses.append(likert_map[selected])

    responses = np.array(responses).reshape(1, -1)
    prediction = model.predict(responses)[0]
    result = "🧠 Cognitive Status: Impaired" if prediction == 1 else "🧠 Cognitive Status: Normal"
    messagebox.showinfo("Result", result)

# Scrollable canvas
canvas = tk.Canvas(root)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
frame = tk.Frame(canvas)

# Add questions with dropdowns
options = list(likert_map.keys())

for i, q in enumerate(questions):
    tk.Label(frame, text=f"{i+1}. {q}", wraplength=550, justify="left", anchor="w", font=("Arial", 10, "bold")).pack(anchor="w", pady=5)
    combo = ttk.Combobox(frame, values=options, state="readonly")
    combo.pack(fill="x", padx=10, pady=2)
    answers.append(combo)

tk.Button(frame, text="Submit", command=submit_responses, bg="green", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

canvas.create_window(0, 0, anchor="nw", window=frame)
canvas.update_idletasks()
canvas.configure(scrollregion=canvas.bbox("all"), yscrollcommand=scroll_y.set)

canvas.pack(fill="both", expand=True, side="left")
scroll_y.pack(fill="y", side="right")

root.mainloop()
