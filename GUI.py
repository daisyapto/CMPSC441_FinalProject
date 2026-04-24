import customtkinter as ctk
from Logic.predict import predict
import threading
import sys
from tkinter import filedialog

# -------------------------
# CALM COLOR THEME
# -------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

CALM_BLUE = "#4A6FA5"
CALM_LIGHT = "#E8EEF7"
CALM_GRAY = "#6B7280"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("AI Tumor Detection")
        self.geometry("1000x700")
        self.resizable(False, False)

        # Proper close behavior
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # ===================== HEADER =====================
        self.header_frame = ctk.CTkFrame(self, corner_radius=12, fg_color=CALM_LIGHT)
        self.header_frame.pack(fill="x", padx=20, pady=(20, 10))

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Brain Tumor Classification",
            font=("Arial", 30, "bold"),
            text_color=CALM_BLUE
        )
        self.title_label.pack(anchor="w", padx=25, pady=(15, 5))

        self.desc_label = ctk.CTkLabel(
            self.header_frame,
            text="Select a model, then choose how you want to test it.\nUpload an X-ray or evaluate against the dataset.",
            font=("Arial", 14),
            text_color=CALM_GRAY
        )
        self.desc_label.pack(anchor="w", padx=25, pady=(0, 15))

        # ===================== MAIN CONTENT =====================
        self.content_frame = ctk.CTkFrame(self, corner_radius=12)
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # ---------- LEFT: MODEL ----------
        self.model_frame = ctk.CTkFrame(self.content_frame, corner_radius=12, border_width=1)
        self.model_frame.pack(side="left", expand=True, fill="both", padx=(20, 10), pady=20)

        ctk.CTkLabel(
            self.model_frame,
            text="Model Selection",
            font=("Arial", 16, "bold")
        ).pack(anchor="w", padx=20, pady=(15, 10))

        self.model_var = ctk.StringVar(value="ensemble")

        for text, value in [
            ("CNN Model 1", "cnn1"),
            ("CNN Model 2", "cnn2"),
            ("Ensemble Model", "ensemble")
        ]:
            ctk.CTkRadioButton(
                self.model_frame,
                text=text,
                variable=self.model_var,
                value=value
            ).pack(anchor="w", padx=25, pady=5)

        # ---------- RIGHT: TEST ----------
        self.test_frame = ctk.CTkFrame(self.content_frame, corner_radius=12, border_width=1)
        self.test_frame.pack(side="right", expand=True, fill="both", padx=(10, 20), pady=20)

        ctk.CTkLabel(
            self.test_frame,
            text="Test Method",
            font=("Arial", 16, "bold")
        ).pack(anchor="w", padx=20, pady=(15, 10))

        self.test_var = ctk.StringVar(value="manual")

        ctk.CTkRadioButton(
            self.test_frame,
            text="Manual Upload",
            variable=self.test_var,
            value="manual",
            command=self.toggle_dataset_input
        ).pack(anchor="w", padx=25, pady=5)

        ctk.CTkRadioButton(
            self.test_frame,
            text="Dataset Evaluation",
            variable=self.test_var,
            value="database",
            command=self.toggle_dataset_input
        ).pack(anchor="w", padx=25, pady=5)

        # Dataset input
        self.dataset_label = ctk.CTkLabel(self.test_frame, text="Number of Test Images:")
        self.dataset_label.pack(anchor="w", padx=25, pady=(15, 0))

        self.dataset_entry = ctk.CTkEntry(self.test_frame, placeholder_text="e.g. 100", width=200)
        self.dataset_entry.pack(anchor="w", padx=25, pady=8)

        # ===================== RESULTS =====================
        self.performance_frame = ctk.CTkFrame(self, corner_radius=12, border_width=1)
        self.performance_frame = ctk.CTkScrollableFrame(self)
        self.performance_frame.pack(fill="x", padx=20, pady=(10, 5))

        ctk.CTkLabel(
            self.performance_frame,
            text="Results",
            font=("Arial", 16, "bold")
        ).pack(anchor="w", padx=20, pady=(10, 5))

        self.performance_value = ctk.CTkLabel(
            self.performance_frame,
            text="No results yet",
            font=("Arial", 22, "bold"),
            justify="left"
        )
        self.performance_value.pack(anchor="w", padx=20, pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            self.performance_frame,
            text="Status: Idle",
            font=("Arial", 12),
            text_color=CALM_GRAY
        )
        self.status_label.pack(anchor="w", padx=20)

        self.progress_bar = ctk.CTkProgressBar(self.performance_frame)
        self.progress_bar.pack(fill="x", padx=20, pady=(5, 10))
        self.progress_bar.set(0)

        # ===================== BUTTONS =====================
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(pady=(10, 20))

        self.submit_button = ctk.CTkButton(
            self.button_frame,
            text="Run Prediction",
            command=self.start_prediction,
            height=45,
            width=200,
            fg_color=CALM_BLUE,
            hover_color="#3B5A8A",
            font=("Arial", 16, "bold")
        )
        self.submit_button.pack(side="left", padx=10)

        self.reset_button = ctk.CTkButton(
            self.button_frame,
            text="Run Again",
            command=self.reset_ui,
            height=45,
            width=200,
            fg_color="#9CA3AF",
            hover_color="#6B7280",
            font=("Arial", 16)
        )
        self.reset_button.pack(side="left", padx=10)

        self.toggle_dataset_input()

    # -------------------------
    def toggle_dataset_input(self):
        if self.test_var.get() == "database":
            self.dataset_label.pack(anchor="w", padx=25, pady=(15, 0))
            self.dataset_entry.pack(anchor="w", padx=25, pady=8)
        else:
            self.dataset_label.pack_forget()
            self.dataset_entry.pack_forget()

    # -------------------------
    def start_prediction(self):
        test_type = self.test_var.get()

        file_path = None
        if test_type == "manual":
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
            )
            if not file_path:
                return

        self.submit_button.configure(state="disabled", text="Running...")
        self.reset_button.configure(state="normal")
        self.status_label.configure(text="Status: Running...")
        self.progress_bar.set(0.3)

        threading.Thread(
            target=self.run_prediction,
            args=(file_path,),
            daemon=True
        ).start()

    # -------------------------
    def run_prediction(self, file_path=None):
        test_type = self.test_var.get()
        model_type = self.model_var.get()

        max_images = self.dataset_entry.get()
        try:
            max_images = int(max_images)
        except:
            max_images = None

        output = predict(
            model_type=model_type,
            test_type=test_type,
            max_images=max_images,
            file_path=file_path
        )

        self.after(0, self.finish_prediction_from_thread, output)

    # -------------------------
    def finish_prediction_from_thread(self, output):
        self.progress_bar.set(1.0)

        if output["mode"] == "manual":
            prediction = output["prediction"]
            confidence = round(output["confidence"] * 100, 2)
            text = f"{prediction}\n{confidence}% confidence"
        else:
            accuracy = output["accuracy"]
            correct = output["correct"]
            total_tests = output["total"]

            recall = output['recall']
            precision = output['precision']
            f1 = output['f1']

            text = (
                f"{accuracy * 100:.2f}% Accuracy\n"
                f"{correct} / {total_tests} correct\n\n"
                f"Recall: {recall * 100:.2f}%\n"
                f"Precision: {precision * 100:.2f}%\n"
                f"F1 Score: {f1 * 100:.2f}%"
            )

        self.performance_value.configure(text=text)
        self.status_label.configure(text="Status: Complete")
        self.submit_button.configure(state="normal", text="Run Prediction")
        self.reset_button.configure(state="normal")

    # -------------------------
    def reset_ui(self):
        self.performance_value.configure(text="No results yet")
        self.status_label.configure(text="Status: Idle")
        self.progress_bar.set(0)
        self.submit_button.configure(state="normal", text="Run Prediction")

    # -------------------------
    def on_close(self):
        self.destroy()
        sys.exit()


if __name__ == "__main__":
    app = App()
    app.mainloop()
