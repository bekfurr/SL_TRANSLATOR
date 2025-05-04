import pickle
import cv2
import mediapipe as mp
import numpy as np
from tkinter import Tk, Button, Label, Text, messagebox
import time

class RealtimeTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Translator")
        self.model_dict = pickle.load(open('model.p', 'rb'))
        self.model = self.model_dict['model']
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Kamera ochilmadi!")
            self.root.quit()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.labels_dict = {}  # Dinamik ravishda to'ldiriladi
        self.setup_gui()
        self.text_output = ""
        self.last_detection_time = 0

    def setup_gui(self):
        Label(self.root, text="Mode:").pack()
        Button(self.root, text="Test Mode", command=self.test_mode).pack()
        Button(self.root, text="Translate Mode", command=self.translate_mode).pack()
        self.output_text = Text(self.root, height=2, width=30)
        self.output_text.pack()

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

    def test_mode(self):
        self.text_output = ""
        self.output_text.delete(1.0, "end")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            landmarks = self.extract_landmarks(frame)
            if landmarks:
                prediction = self.model.predict([np.asarray(landmarks)])
                predicted_character = str(prediction[0])
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, f"Test: {predicted_character}")
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow('frame')

    def translate_mode(self):
        self.text_output = ""
        self.output_text.delete(1.0, "end")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            landmarks = self.extract_landmarks(frame)
            current_time = time.time()
            if landmarks:
                self.last_detection_time = current_time
                prediction = self.model.predict([np.asarray(landmarks)])
                predicted_character = str(prediction[0])
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, f"Translating: {self.text_output + predicted_character}")
            elif current_time - self.last_detection_time >= 3 and self.last_detection_time > 0:
                self.text_output += " "  # Harakat tugaganidan keyin bo'sh joy qo'shish
                self.output_text.delete(1.0, "end")
                self.output_text.insert(1.0, f"Translating: {self.text_output}")
                self.last_detection_time = 0
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow('frame')

    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = RealtimeTranslator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()