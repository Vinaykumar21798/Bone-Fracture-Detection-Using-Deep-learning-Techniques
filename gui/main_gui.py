"""
Desktop GUI for Explainable Bone Fracture Detection System
Academic-grade interface for visualizing fracture detection, explainability, and recommendations.
"""
import os
import threading
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import customtkinter as ctk
from PIL import ImageTk, Image
import numpy as np
import cv2

# Import project modules
# Adjust import path since we're in gui/ subdirectory
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from predictions import predict_full_enhanced
from config import (
    PROJECT_ROOT, SUPPORTED_IMAGE_FORMATS,
    GUI_TITLE
)


class FractureDetectionGUI(ctk.CTk):
    """
    Main GUI application for bone fracture detection.
    Implements the full analysis pipeline: CNN → Grad-CAM → ROI → Severity → Recommendation
    """
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title(GUI_TITLE)
        self.geometry("900x1000")
        self.minsize(800, 900)
        
        # Set appearance
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Store current image path
        self.current_image_path = None
        self.current_result = None
        
        # Create GUI layout
        self._create_layout()
        
    def _create_layout(self):
        """Create the GUI layout matching the exact specification."""
        
        # Main container
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # ============================================
        # Section 1: Upload X-ray & Image Preview
        # ============================================
        upload_frame = ctk.CTkFrame(main_container)
        upload_frame.pack(fill="x", padx=10, pady=10)
        
        # Upload button and preview in same row
        upload_preview_container = ctk.CTkFrame(upload_frame)
        upload_preview_container.pack(fill="x", padx=10, pady=10)
        
        # Upload button (left side)
        upload_btn_frame = ctk.CTkFrame(upload_preview_container)
        upload_btn_frame.pack(side="left", padx=10, pady=10)
        
        self.upload_btn = ctk.CTkButton(
            upload_btn_frame,
            text="Upload X-ray",
            command=self._upload_image,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.upload_btn.pack(pady=5)
        
        # Image preview (right side)
        preview_frame = ctk.CTkFrame(upload_preview_container)
        preview_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        preview_label = ctk.CTkLabel(
            preview_frame,
            text="Image Preview",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        preview_label.pack(pady=5)
        
        self.image_preview_label = ctk.CTkLabel(
            preview_frame,
            text="No image selected",
            width=400,
            height=300,
            fg_color="gray90"
        )
        self.image_preview_label.pack(pady=5, padx=10)
        
        # ============================================
        # Section 2: Analyze Button
        # ============================================
        analyze_frame = ctk.CTkFrame(main_container)
        analyze_frame.pack(fill="x", padx=10, pady=10)
        
        self.analyze_btn = ctk.CTkButton(
            analyze_frame,
            text="Analyze",
            command=self._analyze_image,
            width=200,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled"
        )
        self.analyze_btn.pack(pady=15)
        
        # ============================================
        # Section 3: Prediction Output
        # ============================================
        prediction_frame = ctk.CTkFrame(main_container)
        prediction_frame.pack(fill="x", padx=10, pady=10)
        
        # Fracture status (no title, direct display)
        self.fracture_status_label = ctk.CTkLabel(
            prediction_frame,
            text="Fracture Status: -",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.fracture_status_label.pack(pady=5)
        
        # Confidence
        self.confidence_label = ctk.CTkLabel(
            prediction_frame,
            text="Confidence: -",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_label.pack(pady=5)
        
        # ============================================
        # Section 4: Explainability (Grad-CAM)
        # ============================================
        explainability_frame = ctk.CTkFrame(main_container)
        explainability_frame.pack(fill="x", padx=10, pady=10)
        
        # Grad-CAM label (no separate title)
        self.gradcam_label = ctk.CTkLabel(
            explainability_frame,
            text="Grad-CAM Heatmap (Image)\nNo heatmap available",
            width=400,
            height=300,
            fg_color="gray90",
            font=ctk.CTkFont(size=12)
        )
        self.gradcam_label.pack(pady=5, padx=10)
        
        # Store image reference to prevent garbage collection
        self.gradcam_image_ref = None

        # Edges (OpenCV) preview for ROI analysis
        self.edges_label = ctk.CTkLabel(
            explainability_frame,
            text="Edge Map (ROI)\nNo edges available",
            width=200,
            height=150,
            fg_color="gray95",
            font=ctk.CTkFont(size=11)
        )
        self.edges_label.pack(pady=5, padx=10)
        self.edges_image_ref = None
        
        # ============================================
        # Section 5: Severity Estimation
        # ============================================
        severity_frame = ctk.CTkFrame(main_container)
        severity_frame.pack(fill="x", padx=10, pady=10)
        
        # Severity level (no title, direct display)
        self.severity_level_label = ctk.CTkLabel(
            severity_frame,
            text="Severity Level: -",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.severity_level_label.pack(pady=5)
        
        # Recommendation (in same section as severity per spec)
        self.recommendation_label = ctk.CTkLabel(
            severity_frame,
            text="Recommendation: -",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.recommendation_label.pack(pady=5)

        # Severity metrics (concise) and details button
        self.metrics_label = ctk.CTkLabel(
            severity_frame,
            text="",
            font=ctk.CTkFont(size=12),
            wraplength=700,
            justify="left"
        )
        self.metrics_label.pack(pady=5)

        self.details_btn = ctk.CTkButton(
            severity_frame,
            text="View Recommendation Details",
            command=self._show_recommendation_details,
            width=220,
            state="disabled"
        )
        self.details_btn.pack(pady=5)
        
        # ============================================
        # Section 7: Disclaimer (MANDATORY)
        # ============================================
        disclaimer_frame = ctk.CTkFrame(main_container, fg_color="yellow")
        disclaimer_frame.pack(fill="x", padx=10, pady=10)
        
        self.disclaimer_label = ctk.CTkLabel(
            disclaimer_frame,
            text="⚠️ This system provides decision support only and does not replace medical diagnosis.",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="red",
            wraplength=700,
            justify="center"
        )
        self.disclaimer_label.pack(pady=10, padx=10)
        
        # Loading indicator (hidden initially)
        self.loading_label = ctk.CTkLabel(
            main_container,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.loading_label.pack(pady=5)
    
    def _upload_image(self):
        """Handle image upload from file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=SUPPORTED_IMAGE_FORMATS,
            initialdir=str(PROJECT_ROOT / 'test')
        )
        
        if not file_path:
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Selected file does not exist.")
            return
        
        self.current_image_path = file_path
        
        # Display image in preview
        try:
            img = Image.open(file_path)
            # Resize for preview (maintain aspect ratio)
            max_size = (400, 300)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img)
            self.image_preview_label.configure(
                image=img_tk,
                text=""
            )
            self.image_preview_label.image = img_tk  # Keep a reference
            
            # Enable analyze button
            self.analyze_btn.configure(state="normal")
            
            # Clear previous results
            self._clear_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
            self.analyze_btn.configure(state="disabled")
    
    def _analyze_image(self):
        """Trigger the full analysis pipeline."""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return
        
        # Disable analyze button during processing
        self.analyze_btn.configure(state="disabled")
        self.loading_label.configure(
            text="Analyzing... Please wait",
            text_color="blue"
        )
        self.update()
        
        # Run analysis in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._run_analysis, args=(self.current_image_path,))
        thread.daemon = True
        thread.start()
    
    def _run_analysis(self, img_path: str):
        """
        Run the full analysis pipeline:
        CNN (ResNet50) → Grad-CAM → ROI Extraction → OpenCV Severity → Recommendation
        """
        try:
            # Call the enhanced prediction function
            result = predict_full_enhanced(img_path)
            self.current_result = result
            
            # Update GUI in main thread
            self.after(0, self._update_results, result)
            
        except FileNotFoundError as e:
            self.after(0, lambda: messagebox.showerror(
                "Error",
                f"Model file not found: {str(e)}\nPlease ensure model files are in the weights/ directory."
            ))
            self.after(0, self._reset_analyze_button)
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)  # Print to console for debugging
            self.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.after(0, self._reset_analyze_button)
    
    def _update_results(self, result: dict):
        """Update GUI with analysis results."""
        # Debug: Print available keys
        print(f"DEBUG: Result keys: {list(result.keys())}")
        print(f"DEBUG: Has gradcam_overlay: {'gradcam_overlay' in result}")
        if 'gradcam_overlay' in result:
            print(f"DEBUG: gradcam_overlay value: {result['gradcam_overlay'] is not None}")
        print(f"DEBUG: Has recommendation: {'recommendation' in result}")
        print(f"DEBUG: Has recommendation_text: {'recommendation_text' in result}")
        print(f"DEBUG: Has severity_level: {'severity_level' in result}, value: {result.get('severity_level')}")
        
        # Update prediction results
        fracture_status = result.get('fracture_status', 'unknown')
        fracture_confidence = result.get('fracture_confidence', 0.0)
        
        if fracture_status == 'fractured':
            status_text = "Fracture Status: FRACTURE DETECTED"
            status_color = "red"
        else:
            status_text = "Fracture Status: NO FRACTURE"
            status_color = "green"
        
        self.fracture_status_label.configure(
            text=status_text,
            text_color=status_color
        )
        
        self.confidence_label.configure(
            text=f"Confidence: {fracture_confidence:.1%}"
        )
        
        # Update Grad-CAM heatmap if available
        if fracture_status == 'fractured' and 'gradcam_overlay' in result and result['gradcam_overlay'] is not None:
            try:
                overlay_img = result['gradcam_overlay']
                
                print(f"DEBUG: Grad-CAM overlay type: {type(overlay_img)}, shape: {overlay_img.shape if isinstance(overlay_img, np.ndarray) else 'N/A'}")
                
                # Convert numpy array to PIL Image
                if isinstance(overlay_img, np.ndarray):
                    # Ensure values are in 0-255 range
                    if overlay_img.max() <= 1.0:
                        overlay_img = (overlay_img * 255).astype(np.uint8)
                    else:
                        overlay_img = overlay_img.astype(np.uint8)
                    
                    # Ensure image is RGB (3 channels)
                    if len(overlay_img.shape) == 2:
                        # Grayscale - convert to RGB
                        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2RGB)
                    elif len(overlay_img.shape) == 3:
                        # Check if it's BGR and convert to RGB
                        if overlay_img.shape[2] == 3:
                            # Assume it's RGB already (from gradcam.py it should be RGB)
                            pass
                        elif overlay_img.shape[2] == 4:
                            # RGBA - convert to RGB
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGBA2RGB)
                    
                    print(f"DEBUG: After processing - shape: {overlay_img.shape}, dtype: {overlay_img.dtype}, min: {overlay_img.min()}, max: {overlay_img.max()}")
                    
                    # Convert to PIL Image
                    pil_img = Image.fromarray(overlay_img, 'RGB')
                    
                    # Resize for display
                    max_size = (400, 300)
                    pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    
                    img_tk = ImageTk.PhotoImage(pil_img)
                    # Store reference to prevent garbage collection
                    self.gradcam_image_ref = img_tk
                    self.gradcam_label.configure(
                        image=img_tk,
                        text="Grad-CAM Heatmap (Image)"
                    )
                    print("DEBUG: Grad-CAM overlay displayed successfully")
            except Exception as e:
                import traceback
                error_msg = f"Warning: Could not display Grad-CAM overlay: {e}\n{traceback.format_exc()}"
                print(error_msg)
                self.gradcam_label.configure(
                    text="Grad-CAM Heatmap (Image)\nHeatmap unavailable",
                    image=""
                )
        else:
            self.gradcam_label.configure(
                text="Grad-CAM Heatmap (Image)\nNo fracture detected",
                image=""
            )

        # Display edges image (if provided)
        if fracture_status == 'fractured' and 'edges_image' in result and result['edges_image'] is not None:
            try:
                edges = result['edges_image']
                if isinstance(edges, np.ndarray):
                    # edges is single-channel (binary) - convert to RGB for display
                    if edges.dtype != np.uint8:
                        edges = (edges * 255).astype(np.uint8) if edges.max() <= 1.0 else edges.astype(np.uint8)
                    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    pil_edges = Image.fromarray(edges_rgb)
                    max_size = (200, 150)
                    pil_edges.thumbnail(max_size, Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(pil_edges)
                    self.edges_image_ref = img_tk
                    self.edges_label.configure(image=img_tk, text="Edge Map (ROI)")
            except Exception as e:
                print(f"Warning: Could not display edges image: {e}")
                self.edges_label.configure(text="Edge Map (ROI)\nUnavailable", image="")
        else:
            self.edges_label.configure(text="Edge Map (ROI)\nNo edges available", image="")
        
        # Update severity estimation if available
        if fracture_status == 'fractured':
            severity_level = result.get('severity_level')
            severity_metrics = result.get('severity_metrics', {})
            
            if severity_level:
                # Set color based on severity
                severity_colors = {
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }
                color = severity_colors.get(severity_level, 'gray')
                
                self.severity_level_label.configure(
                    text=f"Severity Level: {severity_level.upper()}",
                    text_color=color
                )
            else:
                self.severity_level_label.configure(
                    text="Severity Level: Unable to determine",
                    text_color="gray"
                )
            
            # Update recommendation based on severity
            severity_level = result.get('severity_level')
            recommendation = result.get('recommendation')
            recommendation_text = result.get('recommendation_text')
            
            # Debug output
            print(f"DEBUG: Recommendation check - severity_level: {severity_level}")
            print(f"DEBUG: Recommendation check - has recommendation dict: {recommendation is not None}")
            print(f"DEBUG: Recommendation check - has recommendation_text: {recommendation_text is not None}")
            if recommendation:
                print(f"DEBUG: Recommendation dict keys: {list(recommendation.keys())}")
                print(f"DEBUG: Recommendation title: {recommendation.get('title', 'N/A')}")
            
            # Map severity to concise recommendation text (as per specification)
            rec_text = None
            
            if severity_level:
                # Use severity level to get concise recommendation
                severity_rec_map = {
                    'Low': 'Rest and follow-up',
                    'Medium': 'Orthopedic consultation',
                    'High': 'Immediate medical attention'
                }
                rec_text = severity_rec_map.get(severity_level, 'Consult healthcare provider')
                print(f"DEBUG: Using severity-based recommendation: {rec_text}")
            elif recommendation:
                # Fallback: extract from recommendation dict
                title = recommendation.get('title', '')
                if 'Low' in title or 'low' in title.lower():
                    rec_text = 'Rest and follow-up'
                elif 'Medium' in title or 'medium' in title.lower():
                    rec_text = 'Orthopedic consultation'
                elif 'High' in title or 'high' in title.lower():
                    rec_text = 'Immediate medical attention'
                else:
                    # Extract from recommendation text
                    rec_text = recommendation.get('recommendation', 'Consult healthcare provider')
                    # Extract first sentence for concise display
                    rec_text = rec_text.split('.')[0] if '.' in rec_text else rec_text
                print(f"DEBUG: Using recommendation dict: {rec_text}")
            elif recommendation_text:
                # Fallback: use recommendation text (extract first sentence)
                rec_text = recommendation_text.split('\n')[0] if recommendation_text else "Consult healthcare provider"
                rec_text = rec_text.split('.')[0] if '.' in rec_text else rec_text
                print(f"DEBUG: Using recommendation_text: {rec_text}")
            else:
                # Last resort: provide default recommendation for fracture
                rec_text = "Consult healthcare provider for treatment recommendations"
                print(f"DEBUG: Using default recommendation: {rec_text}")
            
            # Display recommendation
            self.recommendation_label.configure(
                text=f"Recommendation: {rec_text}",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            # Display concise severity metrics
            if severity_metrics:
                metrics_text = (
                    f"Crack length: {severity_metrics.get('crack_length', 0):.1f} px  |  "
                    f"Displacement: {severity_metrics.get('displacement', 0):.2f}  |  "
                    f"Density: {severity_metrics.get('crack_density', 0):.3f}"
                )
                self.metrics_label.configure(text=metrics_text)
            else:
                self.metrics_label.configure(text="")

            # Enable recommendation details button when full text is available
            full_text = recommendation_text or (recommendation.get('title') + '\n\n' + recommendation.get('recommendation') if recommendation else None)
            if full_text:
                self.details_btn.configure(state="normal")
                # Store current full recommendation for details view
                self._current_recommendation_text = full_text
            else:
                self.details_btn.configure(state="disabled")
                self._current_recommendation_text = None
            print(f"DEBUG: Recommendation displayed: {rec_text}")
        else:
            # No fracture detected
            self.severity_level_label.configure(
                text="Severity Level: N/A",
                text_color="gray"
            )
            self.recommendation_label.configure(
                text="Recommendation: No fracture detected. No treatment recommendations needed.",
                font=ctk.CTkFont(size=14, weight="bold")
            )
        
        # Clear loading message and re-enable button
        self.loading_label.configure(text="")
        self.analyze_btn.configure(state="normal")

    def _show_recommendation_details(self):
        """Show a popup with the full recommendation text (including disclaimer)."""
        full_text = getattr(self, '_current_recommendation_text', None)
        if not full_text:
            messagebox.showinfo("Recommendation Details", "No recommendation details available.")
            return

        # Show details in a simple Toplevel window
        details = ctk.CTkToplevel(self)
        details.title("Recommendation Details")
        details.geometry("700x500")

        text_box = ctk.CTkTextbox(details, width=660, height=440)
        text_box.pack(padx=10, pady=10, fill="both", expand=True)
        text_box.insert("0.0", full_text)
        text_box.configure(state="disabled")

        close_btn = ctk.CTkButton(details, text="Close", command=details.destroy)
        close_btn.pack(pady=6)
    
    def _clear_results(self):
        """Clear all result displays."""
        self.fracture_status_label.configure(text="Fracture Status: -")
        self.confidence_label.configure(text="Confidence: -")
        self.gradcam_label.configure(text="Grad-CAM Heatmap (Image)\nNo heatmap available", image="")
        self.edges_label.configure(text="Edge Map (ROI)\nNo edges available", image="")
        self.metrics_label.configure(text="")
        self.details_btn.configure(state="disabled")
        self._current_recommendation_text = None
        self.severity_level_label.configure(text="Severity Level: -")
        self.recommendation_label.configure(text="Recommendation: -")
    
    def _reset_analyze_button(self):
        """Reset analyze button state after error."""
        self.analyze_btn.configure(state="normal")
        self.loading_label.configure(text="")


def main():
    """Main entry point for the GUI application."""
    try:
        app = FractureDetectionGUI()
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

