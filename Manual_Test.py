from Logic.predict import predict_image
import tkinter as tk
from tkinter import filedialog


def manual_test(model):
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an X-ray Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if file_path:
        result, confidence = predict_image(file_path, model)
        print("Prediction:", result)
        print("Confidence:", confidence * 100, "%")