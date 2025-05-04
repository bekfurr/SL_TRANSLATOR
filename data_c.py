import os
import cv2
import numpy as np
from datasets import load_dataset
from tkinter import Tk, Button, Label, Entry, filedialog, messagebox
import mediapipe as mp
import pickle

class DatasetCreator:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Creator")
        self.dataset_path = ""
        self.hf_dataset_name = ""
        # MediaPipe qo'lni aniqlash uchun sozlamalar
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.setup_gui()

    def setup_gui(self):
        Label(self.root, text="Hugging Face Dataset Name (e.g., sign_language_dataset):").pack()
        self.hf_entry = Entry(self.root)
        self.hf_entry.pack()

        Label(self.root, text="Local Image Folder (optional):").pack()
        self.folder_entry = Entry(self.root)
        self.folder_entry.pack()
        Button(self.root, text="Browse Folder", command=self.browse_folder).pack()

        Button(self.root, text="Create Dataset", command=self.create_dataset).pack()

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_entry.delete(0, 'end')
            self.folder_entry.insert(0, folder)

    def extract_landmarks(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                return landmarks
        return None

    def create_dataset(self):
        self.hf_dataset_name = self.hf_entry.get()
        self.dataset_path = self.folder_entry.get()
        output_dir = "sign_language_dataset"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data = []
        labels = []
        class_names = []

        # Hugging Face datasetini yuklash
        if self.hf_dataset_name:
            try:
                dataset = load_dataset(self.hf_dataset_name)
                for split in dataset:
                    for item in dataset[split]:
                        image = np.array(item['image'])
                        label = item['label']
                        landmarks = self.extract_landmarks(image)
                        if landmarks:
                            data.append(landmarks)
                            labels.append(label)
                            if str(label) not in class_names:
                                class_names.append(str(label))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Hugging Face dataset: {e}")

        # Mahalliy suratlarni qayta ishlash
        if self.dataset_path:
            for label in os.listdir(self.dataset_path):
                label_path = os.path.join(self.dataset_path, label)
                if os.path.isdir(label_path):
                    for img_name in os.listdir(label_path):
                        img_path = os.path.join(label_path, img_name)
                        img = cv2.imread(img_path)
                        if img is not None:
                            landmarks = self.extract_landmarks(img)
                            if landmarks:
                                data.append(landmarks)
                                labels.append(label)
                                if label not in class_names:
                                    class_names.append(label)

        # Ma'lumotlarni saqlash
        with open(os.path.join(output_dir, "landmarks_data.pkl"), "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
        with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
            f.write("\n".join(sorted(class_names)))

        messagebox.showinfo("Success", f"Dataset created at {output_dir}")

if __name__ == "__main__":
    root = Tk()
    app = DatasetCreator(root)
    root.mainloop()
