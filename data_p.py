import os
import pickle
import cv2
import mediapipe as mp
from tkinter import Tk, Button, Label, Entry, filedialog, messagebox

class DataProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processor")
        self.dataset_path = ""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.setup_gui()

    def setup_gui(self):
        Label(self.root, text="Select Dataset Folder:").pack()
        self.path_entry = Entry(self.root)
        self.path_entry.pack()
        Button(self.root, text="Browse", command=self.browse_folder).pack()
        Button(self.root, text="Process Data", command=self.process_data).pack()

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, folder)
            self.dataset_path = folder

    def process_data(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Iltimos, papkani tanlang!")
            return
        data = []
        labels = []
        for dir_ in os.listdir(self.dataset_path):
            for img_path in os.listdir(os.path.join(self.dataset_path, dir_)):
                data_aux = []
                x_ = []
                y_ = []
                img = cv2.imread(os.path.join(self.dataset_path, dir_, img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
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
                        data.append(data_aux)
                        labels.append(int(dir_))
        with open('data.pickle', 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        messagebox.showinfo("Success", "Data processed and saved as data.pickle")

if __name__ == "__main__":
    root = Tk()
    app = DataProcessor(root)
    root.mainloop()