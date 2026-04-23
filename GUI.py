import customtkinter as ctk
from Logic.predict import predict

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("AI Tumor Detection")
        self.geometry("1000x700")
        self.resizable(False, False)

        # ===================== HEADER =====================
        self.header_frame = ctk.CTkFrame(self, height=150)
        self.header_frame.pack(fill="x", padx=20, pady=20)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Brain Tumor Classification",
            font=("Arial", 32, "bold")
        )
        self.title_label.pack(anchor="w", padx=20, pady=(10, 5))

        self.desc_label = ctk.CTkLabel(
            self.header_frame,
            text="First, choose an AI model: CNN 1, CNN 2, or the Ensemble Model Second, upload an xray of a brain to test or use our database of xrays.",
            font=("Arial", 16)
        )
        self.desc_label.pack(anchor="w", padx=20)

        # ===================== MAIN CONTENT =====================
        self.content_frame = ctk.CTkFrame(self)
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # ---------- LEFT: MODEL SELECTION ----------
        self.model_frame = ctk.CTkFrame(self.content_frame)
        self.model_frame.pack(side="left", expand=True, padx=20, pady=20)

        self.model_label = ctk.CTkLabel(
            self.model_frame,
            text="Select Model Type",
            font=("Arial", 16, "bold")
        )
        self.model_label.pack(pady=10)

        self.model_var = ctk.StringVar(value="ensemble")

        self.cnn1_radio = ctk.CTkRadioButton(
            self.model_frame,
            text="CNN Model 1",
            variable=self.model_var,
            value="cnn1"
        )
        self.cnn1_radio.pack(anchor="w", padx=20, pady=5)

        self.cnn2_radio = ctk.CTkRadioButton(
            self.model_frame,
            text="CNN Model 2",
            variable=self.model_var,
            value="cnn2"
        )
        self.cnn2_radio.pack(anchor="w", padx=20, pady=5)

        self.ensemble_radio = ctk.CTkRadioButton(
            self.model_frame,
            text="Ensemble Model",
            variable=self.model_var,
            value="ensemble"
        )
        self.ensemble_radio.pack(anchor="w", padx=20, pady=5)

        # ---------- RIGHT: TEST SELECTION ----------
        self.test_frame = ctk.CTkFrame(self.content_frame)
        self.test_frame.pack(side="right", expand=True, padx=20, pady=20)

        self.test_label = ctk.CTkLabel(
            self.test_frame,
            text="Select Test",
            font=("Arial", 16, "bold")
        )
        self.test_label.pack(pady=10)

        self.test_var = ctk.StringVar(value="manual")

        self.manual_radio = ctk.CTkRadioButton(
            self.test_frame,
            text="Manual Upload",
            variable=self.test_var,
            value="manual",
            command=self.toggle_dataset_input
        )
        self.manual_radio.pack(anchor="w", padx=20, pady=5)

        self.database_radio = ctk.CTkRadioButton(
            self.test_frame,
            text="Our Database",
            variable=self.test_var,
            value="database",
            command=self.toggle_dataset_input
        )
        self.database_radio.pack(anchor="w", padx=20, pady=5)

        # Text box for max images
        self.dataset_label = ctk.CTkLabel(self.test_frame, text="Number of Test Images:")
        self.dataset_label.pack(pady=(10, 0), anchor="w", padx=20)

        self.dataset_entry = ctk.CTkEntry(self.test_frame, placeholder_text="e.g. 100")
        self.dataset_entry.pack(pady=5, anchor="w", padx=20)

        # ===================== PERFORMANCE PANEL =====================
        self.performance_frame = ctk.CTkFrame(self, height=120)
        self.performance_frame.pack(fill="x", padx=20, pady=10)

        self.performance_label = ctk.CTkLabel(
            self.performance_frame,
            text="Performance",
            font=("Arial", 16, "bold")
        )
        self.performance_label.pack(pady=(10, 5))

        self.performance_value = ctk.CTkLabel(
            self.performance_frame,
            text="No results yet",
            font=("Arial", 20)
        )
        self.performance_value.pack(pady=5)

        # ===================== SUBMIT BUTTON =====================
        self.submit_button = ctk.CTkButton(
            self,
            text="Go",
            command=self.on_submit
        )
        self.submit_button.pack(pady=20)

        self.toggle_dataset_input()

    def toggle_dataset_input(self):
        if self.test_var.get() == "database":
            self.dataset_label.pack(pady=(10, 0))
            self.dataset_entry.pack(pady=5)
        else:
            self.dataset_label.pack_forget()
            self.dataset_entry.pack_forget()

    # ===================== CALLBACK =====================
    def on_submit(self):
        test_type = self.test_var.get()
        model_type = self.model_var.get()

        max_images = self.dataset_entry.get()
        try:
            max_images = int(max_images)
        except:
            max_images = None

        output = predict(model_type=model_type, test_type=test_type, max_images=max_images)

        if output["mode"] == "manual":
            # Get return data
            prediction = output["prediction"]
            confidence = output["confidence"]

            # Clean
            confidence = round(confidence * 100, 2)

            # Print
            self.performance_value.configure(text=f"Prediction Label: {prediction} with {confidence}% confidence")
        else:
            print(output)
            # Get return data
            accuracy = output["accuracy"]
            correct = output["correct"]
            total_tests = output["total"]

            # Clean
            accuracy = round(accuracy * 100, 2)

            # Print
            self.performance_value.configure(
                text=f"Correct Predictions {correct} / {total_tests}  ({accuracy}% accuracy)")


if __name__ == "__main__":
    app = App()
    app.mainloop()
