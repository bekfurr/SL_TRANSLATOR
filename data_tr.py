import pickle
import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, Button, Label, Entry, Text, messagebox
import time
import PIL.Image
import PIL.ImageTk
import threading

class RealtimeTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.model_dict = pickle.load(open('model.p', 'rb'))
        self.model = self.model_dict['model']
        self.data_dict = pickle.load(open('data.pickle', 'rb'))
        self.class_names = self.data_dict['class_names']
        self.labels_dict = {i: name for i, name in enumerate(self.class_names)}
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Kamera ochilmadi!")
            self.root.quit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)  # Tasvir o'lchamini kichraytirish
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.setup_gui()
        self.text_output = ""
        self.last_update_time = time.time()
        self.waiting_for_hand = False
        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        self.start_video_thread()
        self.update_frame()

    def setup_gui(self):
        Label(self.root, text="Enter update interval (seconds, e.g., 3):").pack()
        self.interval_entry = Entry(self.root)
        self.interval_entry.insert(0, "3")
        self.interval_entry.pack()
        Button(self.root, text="Start", command=self.start_translation).pack()
        self.video_label = Label(self.root)
        self.video_label.pack()
        Label(self.root, text="Output:").pack()
        self.output_text = Text(self.root, height=2, width=30)
        self.output_text.pack()

    def start_video_thread(self):
        self.video_thread = threading.Thread(target=self.capture_video, daemon=True)
        self.video_thread.start()

    def capture_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()

    def extract_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                data_aux = []
                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                if x_ and y_:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                    return data_aux
        return None

    def update_frame(self):
        with self.lock:
            if self.frame is not None:
                frame = self.frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = PIL.Image.fromarray(frame)
                self.photo = PIL.ImageTk.PhotoImage(image=self.current_image)
                self.video_label.config(image=self.photo)
                self.video_label.image = self.photo
        self.root.after(50, self.update_frame)  # 50 msda yangilash

    def start_translation(self):
        try:
            self.interval = float(self.interval_entry.get())
            if self.interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Iltimos, musbat vaqt kiriting!")
            return

        self.last_update_time = time.time()
        self.translate_loop()

    def translate_loop(self):
        if not self.running:
            return
        with self.lock:
            if self.frame is None:
                return
            frame = self.frame.copy()
        current_time = time.time()
        landmarks = self.extract_landmarks(frame)
        if landmarks and not self.waiting_for_hand:
            prediction = self.model.predict([np.asarray(landmarks)])
            predicted_character = self.labels_dict.get(int(prediction[0]), "Unknown")
            if current_time - self.last_update_time >= self.interval:
                self.text_output += predicted_character
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, f"Translating: {self.text_output}")
                self.last_update_time = current_time
        elif current_time - self.last_update_time >= self.interval:
            if not self.waiting_for_hand:
                self.text_output += " "  # Probel qo'shish
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, f"Translating: {self.text_output}")
                self.last_update_time = current_time
            else:
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, f"Translating: {self.text_output} (Waiting for hand...)")
            self.waiting_for_hand = True
        elif landmarks and self.waiting_for_hand:
            self.waiting_for_hand = False
            self.last_update_time = current_time
        self.root.after(50, self.translate_loop)  # 50 msda takrorlash

    def on_closing(self):
        self.running = False
        self.video_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = RealtimeTranslator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
