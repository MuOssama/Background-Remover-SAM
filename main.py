import os
import sys
import numpy as np
import torch
import cv2
import requests
import threading
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QSlider, QMessageBox, QProgressDialog, QSpinBox, QGroupBox, QButtonGroup, QRadioButton
import time 
from PyQt5.QtCore import pyqtSignal
class SAMBackgroundRemover(QMainWindow):
    # Define this at class level, outside __init__
    preprocessing_finished_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S.A.M Background Remover")
        self.setGeometry(100, 100, 1000, 600)
        
        # Initialize variables
        self.image_path = None
        self.image = None
        self.display_image = None
        self.original_pixmap = None
        self.segmented_pixmap = None
        
        # Points for segmentation
        self.points = []
        self.labels = []  # 1 for foreground (keep), 0 for background (remove)
        self.mask = None
        self.original_mask = None  # Store original mask for recovery purposes
        self.predictor = None
        self.image_queue = []
        self.preprocessed_images = {}
        self.preprocessed_images_lock = threading.Lock()




        self.preprocessing_finished_signal.connect(self.on_preprocessing_finished)
        # Set model info
        #self.sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
        self.sam_checkpoint = "checkpoints/sam_vit_l_0b3195.pth"
        self.model_type = "vit_l"
        #self.model_type = "vit_h"

        self.model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        #self.model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("cuda: "+str(torch.cuda.is_available()) )
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
        # Add these in __init__ after initializing other variables
        # Create large tensors on the GPU
        a = torch.randn(1000, 1000, device=self.device)
        b = torch.randn(1000, 1000, device=self.device)

        # Run some heavy computation
        start = time.time()
        for _ in range(100):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print("Finished in", time.time() - start, "seconds")
        # Post-processing options
        self.min_island_size = 100  # Default minimum island size in pixels
        self.kernel_size = 5  # Default kernel size for smoothing
        self.anti_aliasing_strength = 2  # Default anti-aliasing strength
        
        # Brush settings
        self.brush_size = 20
        self.is_drawing = False
        self.last_pos = QPoint()
        self.current_tool = "segment"  # Default tool: "segment", "erase", or "recover"
        
        # Setup UI
        self.init_ui()
        
        # Ensure the model is available and load it
        self.ensure_model_available()
        
    def init_ui(self):
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel (image display)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 500)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.mousePressEvent = self.handle_mouse_press
        self.image_label.mouseMoveEvent = self.handle_mouse_move
        self.image_label.mouseReleaseEvent = self.handle_mouse_release
        
        # Right panel (controls)
        right_panel = QVBoxLayout()
        
        # Buttons
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        
        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)
        
        self.clear_button = QPushButton("Clear Points")
        self.clear_button.clicked.connect(self.clear_points)
        
        # Tool selection
        tool_group_box = QGroupBox("Tools")
        tool_layout = QVBoxLayout()
        
        self.tool_group = QButtonGroup(self)
        
        self.segment_tool = QRadioButton("Segment")
        self.segment_tool.setChecked(True)
        self.erase_tool = QRadioButton("Erase")
        self.recover_tool = QRadioButton("Recover")
        
        self.tool_group.addButton(self.segment_tool)
        self.tool_group.addButton(self.erase_tool)
        self.tool_group.addButton(self.recover_tool)
        
        self.segment_tool.toggled.connect(lambda: self.set_tool("segment"))
        self.erase_tool.toggled.connect(lambda: self.set_tool("erase"))
        self.recover_tool.toggled.connect(lambda: self.set_tool("recover"))
        
        tool_layout.addWidget(self.segment_tool)
        tool_layout.addWidget(self.erase_tool)
        tool_layout.addWidget(self.recover_tool)
        # In the init_ui method, add this after the load_button
        self.next_image_button = QPushButton("Next Image")
        self.next_image_button.clicked.connect(self.load_next_image)
        self.next_image_button.setEnabled(False)  # Disabled by default

        # Add to the layout
        right_panel.addWidget(self.next_image_button)
        # Brush size slider for erase/recover
        brush_layout = QHBoxLayout()
        brush_label = QLabel("Brush Size:")
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(5, 50)
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.setTickPosition(QSlider.TicksBelow)
        self.brush_slider.valueChanged.connect(self.update_brush_size)
        brush_layout.addWidget(brush_label)
        brush_layout.addWidget(self.brush_slider)
        
        tool_layout.addLayout(brush_layout)
        tool_group_box.setLayout(tool_layout)
        
        # Counter for points
        self.add_point_counter = QLabel("Add Points: 0")
        self.remove_point_counter = QLabel("Remove Points: 0")
        
        # Transparency slider for overlay
        overlay_layout = QHBoxLayout()
        overlay_label = QLabel("Overlay Opacity:")
        self.overlay_slider = QSlider(Qt.Horizontal)
        self.overlay_slider.setRange(0, 100)
        self.overlay_slider.setValue(30)  # 30% opacity
        self.overlay_slider.setTickPosition(QSlider.TicksBelow)
        self.overlay_slider.valueChanged.connect(self.update_segmentation_display)
        overlay_layout.addWidget(overlay_label)
        overlay_layout.addWidget(self.overlay_slider)
        
        # Threshold slider for mask refinement
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Mask Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(30)
        self.threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.threshold_slider.valueChanged.connect(self.update_segmentation_display)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        
        # Post-processing controls group
        post_processing_group = QGroupBox("Post-Processing")
        post_processing_layout = QVBoxLayout()
        
        # 1. Remove Islands/Noise control
        island_layout = QHBoxLayout()
        self.remove_islands_button = QPushButton("Remove Islands")
        self.remove_islands_button.clicked.connect(self.remove_islands)
        self.remove_islands_button.setEnabled(False)
        
        island_label = QLabel("Min Size:")
        self.island_size_spinbox = QSpinBox()
        self.island_size_spinbox.setRange(10, 1000)
        self.island_size_spinbox.setValue(self.min_island_size)
        self.island_size_spinbox.setSingleStep(10)
        
        island_layout.addWidget(self.remove_islands_button)
        island_layout.addWidget(island_label)
        island_layout.addWidget(self.island_size_spinbox)
        
        # 2. Anti-aliasing control
        anti_aliasing_layout = QHBoxLayout()
        self.anti_aliasing_button = QPushButton("Add Anti-aliasing")
        self.anti_aliasing_button.clicked.connect(self.apply_anti_aliasing)
        self.anti_aliasing_button.setEnabled(False)
        
        aa_label = QLabel("Strength:")
        self.aa_strength_spinbox = QSpinBox()
        self.aa_strength_spinbox.setRange(1, 5)
        self.aa_strength_spinbox.setValue(self.anti_aliasing_strength)
        
        anti_aliasing_layout.addWidget(self.anti_aliasing_button)
        anti_aliasing_layout.addWidget(aa_label)
        anti_aliasing_layout.addWidget(self.aa_strength_spinbox)
        
        # 3. Smoothing control
        smoothing_layout = QHBoxLayout()
        self.smoothing_button = QPushButton("Apply Smoothing")
        self.smoothing_button.clicked.connect(self.apply_smoothing)
        self.smoothing_button.setEnabled(False)
        
        smoothing_label = QLabel("Kernel Size:")
        self.smoothing_spinbox = QSpinBox()
        self.smoothing_spinbox.setRange(3, 21)
        self.smoothing_spinbox.setValue(self.kernel_size)
        self.smoothing_spinbox.setSingleStep(2)  # Ensure odd values for kernel
        
        smoothing_layout.addWidget(self.smoothing_button)
        smoothing_layout.addWidget(smoothing_label)
        smoothing_layout.addWidget(self.smoothing_spinbox)
        
        # Add layouts to post-processing group
        post_processing_layout.addLayout(island_layout)
        post_processing_layout.addLayout(anti_aliasing_layout)
        post_processing_layout.addLayout(smoothing_layout)
        post_processing_group.setLayout(post_processing_layout)
        
        # Status label
        self.status_label = QLabel("Loading SAM model...")
        self.model_info_label = QLabel(f"Model: {self.model_type} ({os.path.basename(self.sam_checkpoint)})")
        
        # Add widgets to right panel
        right_panel.addWidget(self.load_button)
        right_panel.addWidget(self.save_button)
        right_panel.addWidget(self.clear_button)
        right_panel.addWidget(tool_group_box)
        right_panel.addWidget(self.add_point_counter)
        right_panel.addWidget(self.remove_point_counter)
        right_panel.addLayout(overlay_layout)
        right_panel.addLayout(threshold_layout)
        right_panel.addWidget(post_processing_group)
        right_panel.addStretch()
        right_panel.addWidget(self.model_info_label)
        right_panel.addWidget(self.status_label)
        
        # Instructions label
        self.instructions_label = QLabel("Instructions: Left-click to add areas to keep. Right-click to mark areas to remove.")
        self.instructions_label.setStyleSheet("font-weight: bold;")
        right_panel.addWidget(self.instructions_label)
        
        # Add panels to main layout
        main_layout.addWidget(self.image_label, 3)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget, 1)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def on_preprocessing_finished(self, image_path):
        """Handle completion of background preprocessing"""
        if self.image_queue and image_path in self.image_queue:
            self.status_label.setText(f"Preprocessing finished for upcoming image")
        elif image_path == self.image_path:
            self.status_label.setText("Current image preprocessing completed")       
        
    def set_tool(self, tool):
        self.current_tool = tool
        if tool == "segment":
            self.instructions_label.setText("Instructions: Left-click to add areas to keep. Right-click to mark areas to remove.")
        elif tool == "erase":
            self.instructions_label.setText("Instructions: Click and drag to erase (paint white) parts of the image.")
        elif tool == "recover":
            self.instructions_label.setText("Instructions: Click and drag to recover parts of the image that were removed.")
    
    def update_brush_size(self):
        self.brush_size = self.brush_slider.value()
    
    def handle_mouse_press(self, event):
        if self.image is None:
            return
            
        if self.current_tool == "segment":
            self.get_point_and_segment(event)
        elif self.current_tool in ["erase", "recover"]:
            self.is_drawing = True
            self.last_pos = event.pos()
            self.apply_brush(event.pos())
    
    def handle_mouse_move(self, event):
        if self.image is None or not self.is_drawing:
            return
            
        if self.current_tool in ["erase", "recover"]:
            current_pos = event.pos()
            self.draw_line(self.last_pos, current_pos)
            self.last_pos = current_pos
    
    def handle_mouse_release(self, event):
        if self.image is None:
            return
            
        self.is_drawing = False
    
    def apply_brush(self, pos):
        if self.mask is None or self.image is None:
            return
            
        # Calculate scaling ratios and offsets to map screen coordinates to image coordinates
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
            
        scale_x = self.image.shape[1] / pixmap.width()
        scale_y = self.image.shape[0] / pixmap.height()
        
        offset_x = (self.image_label.width() - pixmap.width()) // 2
        offset_y = (self.image_label.height() - pixmap.height()) // 2
        
        # Calculate the position on the original image
        img_x = int((pos.x() - offset_x) * scale_x)
        img_y = int((pos.y() - offset_y) * scale_y)
        
        # Check if the point is within the image bounds
        if 0 <= img_x < self.image.shape[1] and 0 <= img_y < self.image.shape[0]:
            # Apply brush effect
            self.apply_brush_effect(img_x, img_y)
    
    def draw_line(self, start_pos, end_pos):
        """Draw a line between two points using the brush"""
        if self.mask is None or self.image is None:
            return
            
        # Calculate scaling ratios and offsets
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
            
        scale_x = self.image.shape[1] / pixmap.width()
        scale_y = self.image.shape[0] / pixmap.height()
        
        offset_x = (self.image_label.width() - pixmap.width()) // 2
        offset_y = (self.image_label.height() - pixmap.height()) // 2
        
        # Calculate start and end positions on the original image
        start_x = int((start_pos.x() - offset_x) * scale_x)
        start_y = int((start_pos.y() - offset_y) * scale_y)
        end_x = int((end_pos.x() - offset_x) * scale_x)
        end_y = int((end_pos.y() - offset_y) * scale_y)
        
        # Check if within bounds
        if (0 <= start_x < self.image.shape[1] and 0 <= start_y < self.image.shape[0] and
            0 <= end_x < self.image.shape[1] and 0 <= end_y < self.image.shape[0]):
            
            # Calculate points along the line using Bresenham's algorithm
            dx = abs(end_x - start_x)
            dy = abs(end_y - start_y)
            sx = 1 if start_x < end_x else -1
            sy = 1 if start_y < end_y else -1
            err = dx - dy
            
            x, y = start_x, start_y
            
            # Apply brush at each point along the line
            while True:
                self.apply_brush_effect(x, y)
                
                if x == end_x and y == end_y:
                    break
                    
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy
    
    def apply_brush_effect(self, x, y):
        """Apply brush effect at the specified position"""
        if self.mask is None or self.image is None:
            return
            
        # Create a circular brush mask
        y_grid, x_grid = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
        dist_from_center = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
        brush_mask = dist_from_center <= self.brush_size
        
        # Apply the effect based on the current tool
        if self.current_tool == "erase":
            # Set mask to False (remove) within brush area
            self.mask[brush_mask] = False
        elif self.current_tool == "recover":
            # Set mask to True (keep) within brush area
            # This recovers the original image in areas where it was masked out
            self.mask[brush_mask] = True
        
        # Update the display
        self.update_segmentation_display()
    
    def ensure_model_available(self):
        if not os.path.isfile(self.sam_checkpoint):
            reply = QMessageBox.question(
                self, 
                "Model Download", 
                f"The SAM model file {self.sam_checkpoint} is not found.\nWould you like to download it now?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.download_model()
            else:
                self.status_label.setText("Model not available. Please download it manually.")
        else:
            self.load_sam_model()
    
    def download_model(self):
        # Create progress dialog
        progress = QProgressDialog("Downloading SAM model...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Downloading")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        def download_thread():
            try:
                # Create a temporary file path
                temp_file = self.sam_checkpoint + ".download"
                
                # Request with timeout and larger chunk size
                response = requests.get(self.model_url, stream=True, timeout=30)
                response.raise_for_status()  # Raise exception for bad responses
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024 * 1024  # 1MB chunks
                downloaded = 0
                
                with open(temp_file, 'wb') as file:
                    for data in response.iter_content(block_size):
                        if progress.wasCanceled():
                            break
                        downloaded += len(data)
                        file.write(data)
                        # Update progress
                        if total_size > 0:
                            percent = int(100 * downloaded / total_size)
                            progress.setValue(percent)
                
                # Only rename the file if download completed
                if not progress.wasCanceled() and downloaded == total_size:
                    # Replace the final file only after successful download
                    if os.path.exists(self.sam_checkpoint):
                        os.remove(self.sam_checkpoint)
                    os.rename(temp_file, self.sam_checkpoint)
                    
                    progress.setValue(100)
                    progress.close()
                    
                    # Load model after successful download
                    self.load_sam_model()
                else:
                    progress.close()
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    QMessageBox.warning(self, "Download Incomplete", 
                        "The download was cancelled or incomplete. Please try again.")
                    
            except Exception as e:
                progress.close()
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                QMessageBox.critical(self, "Download Error", f"Error downloading model: {str(e)}")
                self.status_label.setText(f"Error downloading model: {str(e)}")
        
        # Start download in a separate thread
        download_thread = threading.Thread(target=download_thread)
        download_thread.daemon = True
        download_thread.start()
    
    def load_sam_model(self):
        try:
            # Check if model file exists
            if not os.path.isfile(self.sam_checkpoint):
                self.status_label.setText(f"Model file not found: {self.sam_checkpoint}")
                return
            
            # Import here to avoid errors if segment_anything is not installed
            from segment_anything import sam_model_registry, SamPredictor
            
            # Load the model
            self.status_label.setText("Loading SAM model... This may take a moment.")
            QApplication.processEvents()  # Update UI
            
            sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            
            self.status_label.setText(f"SAM model loaded successfully on {self.device}")
            self.status_label.setText("Load an image to start")
            
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency", 
                "The segment_anything package is not installed.\n"
                "Please install it with: pip install segment-anything")
            self.status_label.setText("Error: segment_anything not installed")
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"Error loading SAM model: {str(e)}")
            self.status_label.setText(f"Error loading SAM model: {str(e)}")
    
    def load_image(self):
        try:
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
            file_dialog.setFileMode(QFileDialog.ExistingFiles)  # Allow multiple file selection
            
            if file_dialog.exec_():
                selected_files = file_dialog.selectedFiles()
                
                if not selected_files:
                    return
                    
                # Load the first image
                self.image_path = selected_files[0]
                
                # Queue other images for preprocessing
                if len(selected_files) > 1:
                    self.image_queue = selected_files[1:]
                    for next_image in self.image_queue:
                        self.preprocess_image_in_background(next_image)
                    self.status_label.setText(f"Preprocessing {len(self.image_queue)} additional images in background...")
                
                # Check if this image is already preprocessed
                with self.preprocessed_images_lock:
                    if self.image_path in self.preprocessed_images:
                        # Use the preprocessed data
                        preprocessed_data = self.preprocessed_images[self.image_path]
                        self.image = preprocessed_data['image']
                        self.predictor = preprocessed_data['predictor']
                        self.status_label.setText("Using preprocessed image data")
                    else:
                        # Process normally
                        self.image = cv2.imread(self.image_path)
                        if self.image is None:
                            QMessageBox.warning(self, "Error", "Could not read the image file.")
                            return
                            
                        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                        
                        # Set image for SAM
                        if self.predictor:
                            self.status_label.setText("Setting image for segmentation...")
                            QApplication.processEvents()  # Update UI
                            self.predictor.set_image(self.image)
                            self.status_label.setText("Left-click to add areas to keep, right-click for areas to remove")
                
                # Update display
                self.display_image = self.image.copy()
                self.update_image_display()
                
                # Reset points
                self.points = []
                self.labels = []
                self.update_point_counters()
                self.save_button.setEnabled(False)
                self.remove_islands_button.setEnabled(False)
                self.anti_aliasing_button.setEnabled(False)
                self.smoothing_button.setEnabled(False)
                self.mask = None
                self.original_mask = None
                
                # Reset to segment tool
                self.segment_tool.setChecked(True)
                self.current_tool = "segment"
                
                # Enable next image button if we have more images in queue
                self.next_image_button.setEnabled(len(self.image_queue) > 0)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image: {str(e)}")
            self.status_label.setText(f"Error loading image: {str(e)}")

    # Function to switch to the next image
    def load_next_image(self):
        if not self.image_queue:
            self.next_image_button.setEnabled(False)
            return
            
        # Get the next image
        next_image_path = self.image_queue.pop(0)
        self.image_path = next_image_path
        
        # Check if it's preprocessed
        with self.preprocessed_images_lock:
            if next_image_path in self.preprocessed_images:
                # Use the preprocessed data
                preprocessed_data = self.preprocessed_images[next_image_path]
                self.image = preprocessed_data['image']
                self.predictor = preprocessed_data['predictor']
                self.status_label.setText("Using preprocessed image data")
            else:
                # Process normally
                self.image = cv2.imread(next_image_path)
                if self.image is None:
                    QMessageBox.warning(self, "Error", "Could not read the next image file.")
                    return
                    
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                
                # Set image for SAM
                if self.predictor:
                    self.status_label.setText("Setting image for segmentation...")
                    QApplication.processEvents()  # Update UI
                    self.predictor.set_image(self.image)
                    self.status_label.setText("Ready for segmentation")
        
        # Update display
        self.display_image = self.image.copy()
        self.update_image_display()
        
        # Reset points
        self.points = []
        self.labels = []
        self.update_point_counters()
        self.save_button.setEnabled(False)
        self.remove_islands_button.setEnabled(False)
        self.anti_aliasing_button.setEnabled(False)
        self.smoothing_button.setEnabled(False)
        self.mask = None
        self.original_mask = None
        self.next_image_button.setEnabled(len(self.image_queue) > 0)
        self.status_label.setText(f"Loaded next image. {len(self.image_queue)} remaining in queue.")
        
    def update_image_display(self):
        if self.display_image is not None:
            height, width, channel = self.display_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit, maintaining aspect ratio
            scaled_pixmap = self.original_pixmap.scaled(self.image_label.width(), 
                                                        self.image_label.height(), 
                                                        Qt.KeepAspectRatio, 
                                                        Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
    
    def get_point_and_segment(self, event):
        if self.image is None or self.predictor is None or self.current_tool != "segment":
            return
        
        # Get click position on the image
        pos = event.pos()
        pixmap = self.image_label.pixmap()
        
        if pixmap is None:
            return
            
        # Calculate scaling ratios
        scale_x = self.image.shape[1] / pixmap.width()
        scale_y = self.image.shape[0] / pixmap.height()
        
        # Calculate offset to center the image in the label
        offset_x = (self.image_label.width() - pixmap.width()) // 2
        offset_y = (self.image_label.height() - pixmap.height()) // 2
        
        # Calculate the position on the original image
        img_x = int((pos.x() - offset_x) * scale_x)
        img_y = int((pos.y() - offset_y) * scale_y)
        
        # Check if the point is within the image bounds
        if 0 <= img_x < self.image.shape[1] and 0 <= img_y < self.image.shape[0]:
            # Determine if this is a left click (add) or right click (remove)
            is_add = event.button() == Qt.LeftButton
            is_remove = event.button() == Qt.RightButton
            
            if is_add or is_remove:
                self.points.append((img_x, img_y))
                # 1 for foreground (left click/add), 0 for background (right click/remove)
                self.labels.append(1 if is_add else 0)
                
                # Update points counter
                self.update_point_counters()
                
                point_type = "add" if is_add else "remove"
                self.status_label.setText(f"{point_type.capitalize()} point added at ({img_x}, {img_y}). Processing segmentation...")
                QApplication.processEvents()  # Update UI
                
                # Automatically perform segmentation after adding a point
                self.segment_image()
    
    def update_point_counters(self):
        add_points = sum(1 for label in self.labels if label == 1)
        remove_points = sum(1 for label in self.labels if label == 0)
        self.add_point_counter.setText(f"Add Points: {add_points}")
        self.remove_point_counter.setText(f"Remove Points: {remove_points}")
    
    def segment_image(self):
        if not self.points or self.predictor is None:
            return
        
        try:
            # Convert points for SAM
            input_points = np.array(self.points)
            input_labels = np.array(self.labels)
            
            # Generate mask
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True
            )
            
            # Select the mask with the highest score
            mask_idx = np.argmax(scores)
            self.mask = masks[mask_idx]
            
            # Store original mask for recovery tool
            if self.original_mask is None:
                self.original_mask = self.mask.copy()
            
            # Create display with white background and semi-transparent original
            self.update_segmentation_display()
            
            # Enable post-processing buttons
            self.save_button.setEnabled(True)
            self.remove_islands_button.setEnabled(True)
            self.anti_aliasing_button.setEnabled(True)
            self.smoothing_button.setEnabled(True)
            
            self.status_label.setText("Segmentation complete. Add more points or apply post-processing.")
            
        except Exception as e:
            QMessageBox.critical(self, "Segmentation Error", f"Error during segmentation: {str(e)}")
            self.status_label.setText(f"Error during segmentation: {str(e)}")
    
    def remove_islands(self):
        """Remove small isolated regions from the mask"""
        if self.mask is None:
            return
            
        try:
            # Get minimum island size from spinbox
            min_size = self.island_size_spinbox.value()
            
            # Create a copy of the mask
            mask_copy = self.mask.copy().astype(np.uint8)
            
            # Find all connected components
            num_labels, labels = cv2.connectedComponents(mask_copy)
            
            # Count pixels in each component
            sizes = np.bincount(labels.flatten())
            
            # Create a new mask with only the components larger than min_size
            new_mask = np.zeros_like(mask_copy)
            
            # Skip the background (index 0)
            for i in range(1, num_labels):
                if sizes[i] >= min_size:
                    new_mask[labels == i] = 1
            
            # Update mask and display
            self.mask = new_mask.astype(bool)
            self.update_segmentation_display()
            self.status_label.setText(f"Removed islands smaller than {min_size} pixels")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error removing islands: {str(e)}")
            self.status_label.setText(f"Error removing islands: {str(e)}")
            
    def apply_anti_aliasing(self):
            """Apply anti-aliasing to mask edges"""
            if self.mask is None:
                return
                
            try:
                # Get anti-aliasing strength
                strength = self.aa_strength_spinbox.value()
                
                # Convert binary mask to uint8
                mask_uint8 = self.mask.astype(np.uint8) * 255
                
                # Apply Gaussian blur to soften edges (anti-aliasing)
                blurred_mask = cv2.GaussianBlur(mask_uint8, (0, 0), sigmaX=strength)
                
                # Threshold back to binary but keep it as float for smoother edges
                _, thresholded = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)
                
                # Update mask
                self.mask = (thresholded > 0)
                self.update_segmentation_display()
                self.status_label.setText(f"Applied anti-aliasing with strength {strength}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error applying anti-aliasing: {str(e)}")
                self.status_label.setText(f"Error applying anti-aliasing: {str(e)}")
    
    def apply_smoothing(self):
        """Apply smoothing to the mask"""
        if self.mask is None:
            return
            
        try:
            # Get kernel size from spinbox
            kernel_size = self.smoothing_spinbox.value()
            
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Convert mask to uint8
            mask_uint8 = self.mask.astype(np.uint8) * 255
            
            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            # Apply morphological operations to smooth the mask
            # First dilate to fill gaps, then erode to shrink back
            dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # Further smoothing with median blur
            smoothed = cv2.medianBlur(eroded, kernel_size)
            
            # Update mask
            self.mask = (smoothed > 0)
            self.update_segmentation_display()
            self.status_label.setText(f"Applied smoothing with kernel size {kernel_size}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error applying smoothing: {str(e)}")
            self.status_label.setText(f"Error applying smoothing: {str(e)}")
    
    def update_segmentation_display(self):
        if self.mask is None or self.image is None:
            return
            
        # Get overlay opacity from slider
        overlay_opacity = self.overlay_slider.value() / 100.0
        
        # Create a white background
        white_bg = np.ones_like(self.image) * 255
        
        # Create a new display image with white background
        display_result = white_bg.copy()
        
        # Apply mask to copy the foreground from original image
        display_result[self.mask] = self.image[self.mask]
        
        # For visualization, blend the original image with lower opacity
        # This helps show the context for placing more points
        blended = cv2.addWeighted(
            self.image, overlay_opacity,  # Original image with lower opacity
            display_result, 1 - overlay_opacity,  # Result with higher opacity
            0
        )
        
        # Find contours of the mask for drawing outlines
        mask_uint8 = self.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw green contours around segmented areas
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)
        
        # Draw points on top of the blended result
        for i, (x, y) in enumerate(self.points):
            if self.labels[i] == 1:  # Add point (foreground)
                cv2.circle(blended, (x, y), 5, (0, 255, 0), -1)  # Green dot
                cv2.circle(blended, (x, y), 10, (0, 255, 0), 2)  # Green circle
            else:  # Remove point (background)
                cv2.circle(blended, (x, y), 5, (255, 0, 0), -1)  # Red dot
                cv2.circle(blended, (x, y), 10, (255, 0, 0), 2)  # Red circle
        
        # Update display
        height, width, channel = blended.shape
        bytes_per_line = 3 * width
        q_image = QImage(blended.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.segmented_pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit, maintaining aspect ratio
        scaled_pixmap = self.segmented_pixmap.scaled(self.image_label.width(), 
                                                    self.image_label.height(), 
                                                    Qt.KeepAspectRatio, 
                                                    Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
    
    def save_result(self):
        if self.mask is None:
            return
        
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG Files  (*.jpg);; PNG Files(*.png);;All Files (*)")
            
            if file_path:
                # Create output image with white background
                white_bg = np.ones_like(self.image) * 255
                
                # Copy only the foreground from original image
                result = white_bg.copy()
                result[self.mask] = self.image[self.mask]
                
                # Convert to BGR for OpenCV
                result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                
                # Save image
                cv2.imwrite(file_path, result_bgr)
                self.status_label.setText(f"Result saved to {file_path}")
                QMessageBox.information(self, "Success", f"Image saved successfully to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving image: {str(e)}")
            self.status_label.setText(f"Error saving image: {str(e)}")
    
    def clear_points(self):
        self.points = []
        self.labels = []
        self.update_point_counters()
        self.mask = None
        self.original_mask = None
        if self.image is not None:
            self.display_image = self.image.copy()
            self.update_image_display()
            self.status_label.setText("Points cleared. Left-click to add areas, right-click to remove areas.")
            
            # Disable post-processing buttons
            self.save_button.setEnabled(False)
            self.remove_islands_button.setEnabled(False)
            self.anti_aliasing_button.setEnabled(False)
            self.smoothing_button.setEnabled(False)

            
    def preprocess_image_in_background(self, image_path):
        """
        Process an image in background thread to create SAM embeddings
        """
        try:
            # Create a background thread for preprocessing
            preprocessing_thread = threading.Thread(
                target=self._do_image_preprocessing,
                args=(image_path,)
            )
            preprocessing_thread.daemon = True
            preprocessing_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting preprocessing thread: {str(e)}")
            return False

    # Worker function that will run in background thread
    def _do_image_preprocessing(self, image_path):
        """
        Worker function that does the actual preprocessing
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                return
                
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create a new predictor instance for this thread if needed
            from segment_anything import sam_model_registry, SamPredictor
            
            # Use a thread-local predictor
            sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            predictor = SamPredictor(sam)
            
            # Generate embeddings - this is the slow part we're precomputing
            predictor.set_image(image)
            
            # Store the results in a dictionary that's accessible from main thread
            with self.preprocessed_images_lock:
                self.preprocessed_images[image_path] = {
                    'image': image,
                    'predictor': predictor,
                    'embeddings': predictor.get_image_embedding()  # Store the embeddings
                }
                
            # Update UI from main thread
            self.preprocessing_finished_signal.emit(image_path)
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAMBackgroundRemover()
    window.show()
    sys.exit(app.exec_())