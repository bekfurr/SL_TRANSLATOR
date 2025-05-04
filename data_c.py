import os
import cv2
from tkinter import Tk, Button, Label, Entry, Listbox, messagebox

class DataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Collector")
        self.classes = []
        self.dataset_size = 100
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Kamera ochilmadi!")
            self.root.quit()
        self.setup_gui()

    def setup_gui(self):
        Label(self.root, text="Enter class name (e.g., A):").pack()
        self.class_entry = Entry(self.root)
        self.class_entry.pack()
        Button(self.root, text="Add Class", command=self.add_class).pack()

        self.class_list = Listbox(self.root)
        self.class_list.pack()

        Button(self.root, text="Start Collecting", command=self.start_collecting).pack()
        Button(self.root, text="Finish", command=self.finish).pack()

    def add_class(self):
        class_name = self.class_entry.get().strip()
        if class_name and class_name not in self.classes:
            self.classes.append(class_name)
            self.class_list.insert("end", class_name)
            self.class_entry.delete(0, "end")
            with open("class_names.txt", "w") as f:
                f.write("\n".join(self.classes))

    def start_collecting(self):
        if not self.classes:
            messagebox.showwarning("Warning", "Iltimos, kamida bitta sinf qo'shing!")
            return
        for idx, class_name in enumerate(self.classes):
            if not os.path.exists(os.path.join("data", str(idx))):
                os.makedirs(os.path.join("data", str(idx)))
            print(f'Collecting data for class {class_name} (index {idx})')
            self.collect_data(idx)

    def collect_data(self, class_idx):
        counter = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        while counter < self.dataset_size:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join("data", str(class_idx), f'{counter}.jpg'), frame)
            counter += 1
        print(f'Collected {counter} images for class {class_idx}')

    def finish(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = DataCollector(root)
    root.mainloop()
