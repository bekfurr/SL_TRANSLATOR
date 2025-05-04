import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tkinter import Tk, Button, Label, messagebox

class ModelTrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Trainer")
        Button(self.root, text="Train Model", command=self.train_model).pack()

    def train_model(self):
        try:
            data_dict = pickle.load(open('data.pickle', 'rb'))
            data = np.asarray(data_dict['data'])
            labels = np.asarray(data_dict['labels'])
            unique_labels = np.unique(labels)
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            labels = np.array([label_map[label] for label in labels])
            x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
            score = accuracy_score(y_predict, y_test)
            print('{}% of samples were classified correctly !'.format(score * 100))
            with open('model.p', 'wb') as f:
                pickle.dump({'model': model}, f)
            messagebox.showinfo("Success", f"Model trained with accuracy {score * 100:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {e}")

if __name__ == "__main__":
    root = Tk()
    app = ModelTrainer(root)
    root.mainloop()