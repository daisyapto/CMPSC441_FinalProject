from Logic.predict import predict_image
import tkinter as tk
from tkinter import filedialog


def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an X-ray Image",
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    if file_path:
        result = predict_image(file_path)
        print("Prediction:", result)
        # print("Confidence:", round(confidence * 100, 2), "%")


if __name__ == "__main__":
    main()
