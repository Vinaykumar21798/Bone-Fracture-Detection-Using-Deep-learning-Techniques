# Bone Fracture Detection GUI

## Overview
Desktop GUI application for explainable bone fracture detection system. Provides a clean, academic interface for visualizing fracture detection results, explainability, severity estimation, and decision-support outputs.

## Features

### Layout Sections
1. **Upload Section**: Upload X-ray button and Image Preview panel
2. **Analyze Button**: Triggers full analysis pipeline
3. **Prediction Output**: Displays fracture status and confidence
4. **Explainability**: Shows Grad-CAM heatmap overlay
5. **Severity Estimation**: Displays severity level and explanation
6. **Recommendation**: Shows treatment recommendations based on severity
7. **Disclaimer**: Mandatory medical disclaimer

### Analysis Pipeline
The GUI implements the complete analysis pipeline:
- **CNN (ResNet50)**: Body part classification and fracture detection
- **Grad-CAM**: Explainability visualization
- **ROI Extraction**: Fracture localization using OpenCV
- **Severity Estimation**: OpenCV-based crack and displacement analysis
- **Rule-Based Recommendation**: Treatment suggestions based on severity

## Usage

### Running the GUI

**Option 1: Using the launcher script**
```bash
python run_gui.py
```

**Option 2: Direct import**
```bash
python -m gui.main_gui
```

**Option 3: From Python**
```python
from gui.main_gui import main
main()
```

### Workflow

1. **Upload Image**: Click "Upload X-ray" button and select an X-ray image file
2. **Analyze**: Click "Analyze" button to run the full analysis pipeline
3. **View Results**: 
   - Fracture status and confidence
   - Grad-CAM heatmap showing model attention
   - Severity level (Low/Medium/High)
   - Treatment recommendations

### Supported Image Formats
- PNG (.png)
- JPEG (.jpg, .jpeg, .jfif)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Requirements

All dependencies are listed in `requirements.txt`. Key packages:
- `customtkinter` - Modern GUI framework
- `Pillow` - Image processing
- `tensorflow` - Deep learning models
- `opencv-python` - Image processing and severity analysis
- `numpy` - Numerical operations

## Code Structure

```
gui/
├── __init__.py          # Package initialization
├── main_gui.py          # Main GUI application
└── README.md           # This file
```

### Main Components

- **FractureDetectionGUI**: Main application class
  - `_create_layout()`: Creates GUI layout
  - `_upload_image()`: Handles image upload
  - `_analyze_image()`: Triggers analysis
  - `_run_analysis()`: Runs analysis pipeline in background thread
  - `_update_results()`: Updates GUI with results

## Recommendations by Severity

- **Low Severity**: Rest and follow-up
- **Medium Severity**: Orthopedic consultation
- **High Severity**: Immediate medical attention

## Disclaimer

⚠️ **This system provides decision support only and does not replace medical diagnosis.**

All results should be reviewed by qualified healthcare professionals. This tool is intended for research and educational purposes only.

## Technical Details

- **Framework**: CustomTkinter (modern Tkinter wrapper)
- **Threading**: Analysis runs in background thread to prevent GUI freezing
- **Image Processing**: PIL/Pillow for image display, OpenCV for analysis
- **Model Integration**: Uses existing `predict_full_enhanced()` function

## Troubleshooting

### Model Files Not Found
Ensure model files are in the `weights/` directory:
- `ResNet50_BodyParts.h5`
- `ResNet50_Elbow_frac.h5`
- `ResNet50_Hand_frac.h5`
- `ResNet50_Shoulder_frac.h5`

### Import Errors
Make sure you're running from the project root directory and all dependencies are installed:
```bash
pip install -r requirements.txt
```

### GUI Not Displaying
- Check that CustomTkinter is properly installed
- Verify Python version (3.8+)
- Check console for error messages

