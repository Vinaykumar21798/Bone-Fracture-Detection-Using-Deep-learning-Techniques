"""
Main GUI application for Bone Fracture Detection
Clean clinical output layout (NO heatmap / NO ROI)
Includes displacement angle
"""

import threading
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image

from predictions import predict_full_enhanced
from config import (
    PROJECT_ROOT,
    GUI_TITLE,
    GUI_WIDTH,
    GUI_HEIGHT,
    GUI_MIN_WIDTH,
    GUI_MIN_HEIGHT,
    SUPPORTED_IMAGE_FORMATS,
)

filename = ""


class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(GUI_TITLE)
        self.geometry(f"{GUI_WIDTH}x{GUI_HEIGHT}")
        self.minsize(GUI_MIN_WIDTH, GUI_MIN_HEIGHT)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._create_widgets()

    # --------------------------------------------------
    # UI
    # --------------------------------------------------

    def _create_widgets(self):
        self.main = ctk.CTkFrame(self)
        self.main.pack(fill="both", expand=True, padx=40, pady=30)

        # Upload
        self.upload_btn = ctk.CTkButton(
            self.main, text="Upload X-ray Image", command=self.upload_image
        )
        self.upload_btn.pack(pady=(0, 15))

        self.img_label = ctk.CTkLabel(self.main, text="No image selected")
        self.img_label.pack(pady=(0, 20))

        self.predict_btn = ctk.CTkButton(
            self.main, text="Predict", command=self.predict_gui, state="disabled"
        )
        self.predict_btn.pack(pady=(0, 20))

        self.status_label = ctk.CTkLabel(self.main, text="")
        self.status_label.pack()

        # ---------------- RESULTS (initially hidden) ----------------
        self.bone_label = ctk.CTkLabel(
            self.main, font=ctk.CTkFont("Roboto", 16, "bold")
        )
        self.result_label = ctk.CTkLabel(
            self.main, font=ctk.CTkFont("Roboto", 18)
        )
        self.angle_label = ctk.CTkLabel(
            self.main, font=ctk.CTkFont("Roboto", 16)
        )
        self.severity_label = ctk.CTkLabel(
            self.main, font=ctk.CTkFont("Roboto", 16)
        )
        self.recommendation_label = ctk.CTkLabel(
            self.main,
            wraplength=620,
            justify="left",
            font=ctk.CTkFont("Roboto", 14)
        )

    # --------------------------------------------------
    # Upload
    # --------------------------------------------------

    def upload_image(self):
        global filename
        filename = filedialog.askopenfilename(
            filetypes=SUPPORTED_IMAGE_FORMATS,
            initialdir=str(PROJECT_ROOT / "test")
        )

        if not filename:
            return

        img = Image.open(filename).resize((260, 260))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(260, 260))

        self.img_label.configure(image=ctk_img, text="")
        self.img_label.image = ctk_img
        self.predict_btn.configure(state="normal")

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict_gui(self):
        self.status_label.configure(text="Processing...")
        self.predict_btn.configure(state="disabled")
        threading.Thread(target=self._run_prediction, daemon=True).start()

    def _run_prediction(self):
        result = predict_full_enhanced(filename)
        self.after(0, self._update_results, result)

    # --------------------------------------------------
    # Update Results
    # --------------------------------------------------

    def _update_results(self, result: dict):
        self.status_label.configure(text="")
        self.predict_btn.configure(state="normal")

        # Clear previous content
        self.bone_label.pack_forget()
        self.result_label.pack_forget()
        self.angle_label.pack_forget()
        self.severity_label.pack_forget()
        self.recommendation_label.pack_forget()

        # -------- Bone Type --------
        self.bone_label.configure(
            text=f"Bone Type: {result.get('body_part', 'Unknown')}"
        )
        self.bone_label.pack(anchor="w", pady=(25, 5))

        # -------- Result --------
        self.result_label.configure(
            text=f"Result: {result['fracture_status'].upper()} "
                 f"({result['fracture_confidence']:.1%})"
        )
        self.result_label.pack(anchor="w", pady=5)

        # -------- Displacement Angle --------
        angle = result.get("displacement_angle")
        if angle is not None:
            self.angle_label.configure(
                text=f"Estimated Displacement Angle: {angle}°"
            )
            self.angle_label.pack(anchor="w", pady=5)

        # -------- Severity --------
        if result.get("severity_level"):
            self.severity_label.configure(
                text=f"Severity: {result['severity_level']}"
            )
            self.severity_label.pack(anchor="w", pady=5)

        # -------- Recommendation --------
        if result.get("recommendation_text"):
            self.recommendation_label.configure(
                text="Treatment & Recommendation:\n\n"
                     + result["recommendation_text"]
            )
            self.recommendation_label.pack(anchor="w", pady=(15, 0))


if __name__ == "__main__":
    App().mainloop()
