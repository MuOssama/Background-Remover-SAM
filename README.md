# Background-Remover-SAM


## Overview

SAM Background Remover is a desktop application that leverages Meta's Segment Anything Model (SAM) to easily remove backgrounds from images. The tool provides an intuitive interface for selecting foreground objects and removing backgrounds with high precision.

## Features

- **Interactive Segmentation**: Click to add or remove areas from the selection
- **Multiple Editing Tools**: Segment, erase, and recover tools for precise control
- **Post-Processing Options**:
  - Remove small isolated regions (islands)
  - Apply anti-aliasing for smoother edges
  - Edge smoothing with adjustable kernel size
- **Batch Processing**: Queue multiple images for efficient workflow
- **GPU Acceleration**: Utilizes CUDA if available for faster processing
- **Transparency Control**: Adjust overlay opacity for better visualization
- **High-Quality Output**: Save results as JPG or PNG with white background

## System Requirements

- Python 3.7 or higher
- CUDA-compatible GPU recommended (but not required)
- At least 8GB RAM (16GB recommended for large images)
- Operating system: Windows, macOS, or Linux

## Installation

### 1. Clone or download the repository

```bash
git clone https://github.com/yourusername/sam-background-remover.git
cd sam-background-remover
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The requirements include:
- PyQt5
- OpenCV (cv2)
- NumPy
- PyTorch
- segment-anything

### 4. Download SAM model checkpoint

The application will automatically prompt you to download the required model file on first launch. Alternatively, you can download it manually:

```bash
mkdir -p checkpoints
# For the lite model (faster, less accurate):
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O checkpoints/sam_vit_l_0b3195.pth
# OR for the heavy model (slower, more accurate):
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth
```

## Running the Application

```bash
python main.py
```

## Usage Guide

### Loading Images

1. Click the **Load Image** button to select one or more images
2. If multiple images are selected, they will be queued for processing
3. Use the **Next Image** button to move to the next image in the queue

### Segmenting the Image

1. Select the **Segment** tool (default)
2. **Left-click** on areas you want to **keep** (foreground)
3. **Right-click** on areas you want to **remove** (background)
4. The segmentation updates automatically after each click
5. Points are color-coded:
   - Green circles: Areas to keep (foreground)
   - Red circles: Areas to remove (background)

### Refining the Selection

1. Use the **Erase** tool to manually remove parts of the selection:
   - Adjust brush size with the slider
   - Click and drag to erase (paint white) parts of the image
   
2. Use the **Recover** tool to restore parts of the image:
   - Adjust brush size with the slider
   - Click and drag to recover previously removed areas

### Post-Processing Options

1. **Remove Islands**: Eliminate small isolated regions
   - Adjust the minimum island size with the spinbox
   - Click "Remove Islands" to apply

2. **Anti-aliasing**: Smooth the edges of the selection
   - Adjust the strength with the spinbox
   - Click "Add Anti-aliasing" to apply

3. **Smoothing**: Apply morphological operations to smooth the mask
   - Adjust the kernel size with the spinbox
   - Click "Apply Smoothing" to apply

### Overlay Adjustments

- Use the **Overlay Opacity** slider to adjust the transparency of the original image in the preview

### Saving Results

1. Click the **Save Result** button
2. Choose a location and file format (JPG or PNG)
3. The result will be saved with a white background and the selected foreground objects

### Additional Controls

- **Clear Points**: Remove all selection points and start over
- **Model Info**: Displays the currently loaded SAM model type

## Troubleshooting

### Common Issues

1. **Application is slow or crashes**:
   - Try using the lighter model (vit_l instead of vit_h)
   - Reduce image resolution before loading
   - Ensure you have enough free RAM

2. **GPU not being utilized**:
   - Verify that CUDA is properly installed
   - Check that PyTorch was installed with CUDA support
   - Run `torch.cuda.is_available()` in Python to verify CUDA detection

3. **Model download fails**:
   - Download the model manually and place it in the `checkpoints` folder
   - Check your internet connection and try again
   - Ensure you have enough disk space

4. **Selection is not accurate**:
   - Add more points to guide the segmentation
   - Try adding points near complex edges
   - Use the Erase and Recover tools for fine adjustments

## Advanced Usage

### Modifying Default Settings

You can adjust default settings by editing the following variables at the beginning of the `SAMBackgroundRemover` class:

- `self.brush_size`: Default brush size for erase/recover tools
- `self.min_island_size`: Default minimum island size in pixels
- `self.kernel_size`: Default kernel size for smoothing
- `self.anti_aliasing_strength`: Default anti-aliasing strength
- `self.model_type`: Model type ('vit_l' or 'vit_h')
- `self.sam_checkpoint`: Path to model checkpoint file

### Command Line Arguments

The application currently doesn't support command line arguments, but you can modify the code to add this functionality as needed.

## Credits

- **Segment Anything Model (SAM)**: Developed by Meta AI Research
- **PyQt5**: Qt framework for Python
- **OpenCV**: Computer vision library
- **NumPy**: Scientific computing library
