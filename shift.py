import sys
import os
import numpy as np

# Version identifier for update system
VERSION = "1.0.2"

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, 
                            QFileDialog, QSlider, QGroupBox, QGridLayout, QDialog, QLineEdit, QMessageBox, QRadioButton, QProgressDialog)
from PyQt5.QtCore import Qt, QPoint
import registration
import cv2
from skimage import exposure
from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
import io

class MaskVisualizationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mask_img = None
        self.mask_file_path = None
        self.bg_img = None
        self.colored_mask = None
        self.processed_bg = None
        self.offset_x = 0
        self.offset_y = 0
        self.last_pos = None
        self.drag_enabled = False
        self.cmaps = plt.colormaps()  # Get all available colormaps
        self.current_cmap = 'viridis'
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # Set initially on top
        
        # Use a more explicit approach to manage the plots
        self.bg_plot = None
        self.mask_plot = None
        
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Shift Analyzer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Main widget and layout (horizontal orientation)
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left sidebar for controls
        left_sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setSpacing(10)
        
        # Mask controls group
        mask_group = QGroupBox("Mask Controls")
        mask_layout = QVBoxLayout()
        
        # Mask file selection
        mask_file_layout = QHBoxLayout()
        mask_file_layout.addWidget(QLabel("Mask:"))
        self.mask_file_btn = QPushButton("Select")
        self.mask_file_btn.clicked.connect(self.select_mask_file)
        mask_file_layout.addWidget(self.mask_file_btn)
        mask_layout.addLayout(mask_file_layout)
        
        self.mask_file_label = QLabel("No file selected")
        self.mask_file_label.setWordWrap(True)
        mask_layout.addWidget(self.mask_file_label)
        
        # Colormap selection
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        # Add the most common colormaps at the top
        top_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'rainbow', 'tab10', 'tab20', 'Spectral']
        self.cmap_combo.addItems(top_cmaps)
        self.cmap_combo.addItems([cm for cm in self.cmaps if cm not in top_cmaps])
        self.cmap_combo.currentTextChanged.connect(self.update_colormap)
        cmap_layout.addWidget(self.cmap_combo)
        mask_layout.addLayout(cmap_layout)
        
        # Mask opacity
        opacity_layout = QVBoxLayout()
        opacity_layout.addWidget(QLabel("Mask Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self.update_display)
        opacity_layout.addWidget(self.opacity_slider)
        mask_layout.addLayout(opacity_layout)
        
        mask_group.setLayout(mask_layout)
        sidebar_layout.addWidget(mask_group)
        
        # Background image controls group
        bg_group = QGroupBox("Background Image Controls")
        bg_layout = QVBoxLayout()
        
        # Background file selection
        bg_file_layout = QHBoxLayout()
        bg_file_layout.addWidget(QLabel("Background:"))
        self.bg_file_btn = QPushButton("Select")
        self.bg_file_btn.clicked.connect(self.select_bg_file)
        bg_file_layout.addWidget(self.bg_file_btn)
        bg_layout.addLayout(bg_file_layout)
        
        self.bg_file_label = QLabel("No file selected")
        self.bg_file_label.setWordWrap(True)
        bg_layout.addWidget(self.bg_file_label)
        
        # Brightness and contrast
        brightness_layout = QVBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_bg_adjustment)
        brightness_layout.addWidget(self.brightness_slider)
        bg_layout.addLayout(brightness_layout)
        
        contrast_layout = QVBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-100)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.update_bg_adjustment)
        contrast_layout.addWidget(self.contrast_slider)
        bg_layout.addLayout(contrast_layout)
        
        self.auto_adjust_btn = QPushButton("Auto Brightness/Contrast")
        self.auto_adjust_btn.clicked.connect(self.auto_adjust_bg)
        bg_layout.addWidget(self.auto_adjust_btn)
        
        bg_group.setLayout(bg_layout)
        sidebar_layout.addWidget(bg_group)
        
        # Offset and Positioning Controls
        offset_group = QGroupBox("Positioning Controls")
        offset_layout = QVBoxLayout()
        
        self.offset_label = QLabel("Offset: X=0.0, Y=0.0")
        offset_layout.addWidget(self.offset_label)
        
        # Add step size adjustment
        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("Step Size:"))
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.1", "0.5", "1", "5", "10", "20"])
        self.step_combo.setCurrentIndex(1)  # Default to 1 pixel step
        step_layout.addWidget(self.step_combo)
        offset_layout.addLayout(step_layout)
        
        # Arrow controls
        arrow_grid = QGridLayout()
        
        # Up button
        self.up_btn = QPushButton("↑")
        self.up_btn.clicked.connect(self.move_up)
        arrow_grid.addWidget(self.up_btn, 0, 1)
        
        # Left button
        self.left_btn = QPushButton("←")
        self.left_btn.clicked.connect(self.move_left)
        arrow_grid.addWidget(self.left_btn, 1, 0)
        
        # Right button
        self.right_btn = QPushButton("→")
        self.right_btn.clicked.connect(self.move_right)
        arrow_grid.addWidget(self.right_btn, 1, 2)
        
        # Down button
        self.down_btn = QPushButton("↓")
        self.down_btn.clicked.connect(self.move_down)
        arrow_grid.addWidget(self.down_btn, 2, 1)
        
        offset_layout.addLayout(arrow_grid)
        
        # Reset button
        self.reset_offset_btn = QPushButton("Reset Position")
        self.reset_offset_btn.clicked.connect(self.reset_offset)
        offset_layout.addWidget(self.reset_offset_btn)
        
        offset_group.setLayout(offset_layout)
        sidebar_layout.addWidget(offset_group)
        
        # Apply & Registration buttons
        apply_layout = QHBoxLayout()
        
        # Save Shifted Mask button
        self.save_btn = QPushButton("Save Shifted Mask")
        self.save_btn.clicked.connect(self.save_shifted_mask)
        apply_layout.addWidget(self.save_btn)
        
        # Apply Registration button (open registration dialog)
        self.apply_btn = QPushButton("Batch Registration")
        self.apply_btn.clicked.connect(self.apply_registration)
        apply_layout.addWidget(self.apply_btn)
        
        sidebar_layout.addLayout(apply_layout)
        
        # Stretch at the bottom to push controls up
        sidebar_layout.addStretch()
        
        # Set fixed width for sidebar
        left_sidebar.setLayout(sidebar_layout)
        left_sidebar.setFixedWidth(300)
        main_layout.addWidget(left_sidebar)
        
        # Right area for visualization (expanded)
        viz_widget = QWidget()
        viz_layout = QVBoxLayout()
        
        # Visualization area
        self.figure = plt.figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        viz_layout.addWidget(self.canvas)
        
        viz_widget.setLayout(viz_layout)
        main_layout.addWidget(viz_widget, stretch=1)  # Make it expandable
        
        # Mouse events for dragging the mask
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Keyboard events for arrow key navigation
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initialize empty display
        self.bg_plot = None
        self.mask_plot = None
        self.update_display()
        
    def get_step_size(self):
        """Get the current step size from the combo box"""
        return float(self.step_combo.currentText())
        
    def move_up(self):
        """Move mask up by the step size"""
        step = self.get_step_size()
        self.offset_y -= step  # Subtract because Y is inverted in image coordinates
        self.update_offset_display()
        self.update_display()
    
    def move_down(self):
        """Move mask down by the step size"""
        step = self.get_step_size()
        self.offset_y += step  # Add because Y is inverted in image coordinates
        self.update_offset_display()
        self.update_display()
    
    def move_left(self):
        """Move mask left by the step size"""
        step = self.get_step_size()
        self.offset_x -= step
        self.update_offset_display()
        self.update_display()
    
    def move_right(self):
        """Move mask right by the step size"""
        step = self.get_step_size()
        self.offset_x += step
        self.update_offset_display()
        self.update_display()
    
    def update_offset_display(self):
        """Update the offset display label"""
        self.offset_label.setText(f"Offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
    
    def on_key_press(self, event):
        """Handle key press events for arrow keys"""
        if event.key == 'up':
            self.move_up()
        elif event.key == 'down':
            self.move_down()
        elif event.key == 'left':
            self.move_left()
        elif event.key == 'right':
            self.move_right()
        
    def reset_offset(self):
        self.offset_x = 0
        self.offset_y = 0
        self.update_offset_display()
        self.update_display()
        
    def select_mask_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Mask File", "", 
                                                 "Image Files (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)")
        if file_path:
            try:
                # Clear previous mask data completely
                if self.mask_plot is not None:
                    try:
                        self.mask_plot.remove()
                    except:
                        pass
                    self.mask_plot = None
                
                # Clear the colored mask to ensure no residual data
                self.colored_mask = None
                
                # Load the new mask
                self.mask_img = imread(file_path, as_gray=True)
                self.mask_file_path = file_path
                self.mask_file_label.setText(os.path.basename(file_path))
                
                # Reset offset when loading a new mask
                self.offset_x = 0
                self.offset_y = 0
                self.offset_label.setText("Offset: X=0, Y=0")
                
                # Force a complete redraw
                self.update_colormap()
                
                # Print mask info for debugging
                print(f"Loaded mask: {os.path.basename(file_path)}")
                print(f"Shape: {self.mask_img.shape}")
                print(f"Data type: {self.mask_img.dtype}")
                print(f"Min value: {np.min(self.mask_img)}, Max value: {np.max(self.mask_img)}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.mask_file_label.setText(f"Error: {str(e)}")
                self.mask_file_path = None
    
    def select_bg_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Background Image(s)", "", 
                                              "Image Files (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)")
        if files:
            try:
                # Check if a single file is selected
                if len(files) == 1:
                    # Try to load as multi-page TIFF (Z-stack)
                    img = imread(files[0])
                    
                    # If the image is a 3D stack
                    if len(img.shape) == 3 and img.shape[2] > 4:
                        # It's a Z-stack with multiple slices
                        max_projection = np.max(img, axis=0)
                        self.bg_file_label.setText(f"Z-stack with {img.shape[0]} slices")
                    else:
                        # Single 2D image or RGB image
                        max_projection = img
                        self.bg_file_label.setText(os.path.basename(files[0]))
                else:
                    # Multiple files selected
                    images = []
                    for f in files:
                        img = imread(f)
                        # Handle case where image itself is a stack
                        if len(img.shape) == 3 and img.shape[2] <= 4:  # RGB/RGBA image
                            images.append(img)
                        elif len(img.shape) == 3:  # Multi-page image
                            # For each slice in the stack
                            for i in range(img.shape[0]):
                                images.append(img[i])
                        else:  # Regular 2D image
                            images.append(img)
                    
                    # Ensure all images have the same dimensions
                    first_shape = images[0].shape
                    images = [img for img in images if img.shape == first_shape]
                    
                    # Perform max projection
                    if images:
                        max_projection = np.max(np.array(images), axis=0)
                        self.bg_file_label.setText(f"{len(images)} slices merged")
                    else:
                        raise ValueError("No compatible images found for merging")
                
                # Convert 16-bit to 8-bit if needed with proper scaling
                if max_projection.dtype == np.uint16:
                    # Scale properly from uint16 to uint8
                    max_val = np.max(max_projection)
                    if max_val > 0:  # Avoid division by zero
                        scaled = (max_projection.astype(np.float32) * 255 / max_val).astype(np.uint8)
                    else:
                        scaled = max_projection.astype(np.uint8)
                    max_projection = scaled
                
                self.bg_img = max_projection
                
                # Reset background plot to force redraw
                self.bg_plot = None
                
                # Print background image info for debugging
                print(f"Loaded background image")
                print(f"Shape: {self.bg_img.shape}")
                print(f"Data type: {self.bg_img.dtype}")
                print(f"Min value: {np.min(self.bg_img)}, Max value: {np.max(self.bg_img)}")
                
                # Auto adjust brightness/contrast
                self.auto_adjust_bg()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.bg_file_label.setText(f"Error: {str(e)}")
    
    def update_colormap(self):
        if self.mask_img is not None:
            cmap_name = self.cmap_combo.currentText()
            self.current_cmap = cmap_name
            
            # Normalize the mask to range 0-1
            mask_normalized = img_as_float(self.mask_img)
            
            # Apply colormap
            cmap = plt.get_cmap(cmap_name)
            colored = cmap(mask_normalized)
            
            # Set alpha channel based on mask values
            # Where mask is 0, make fully transparent
            colored[..., 3] = np.where(mask_normalized > 0, 1.0, 0.0)
            
            self.colored_mask = colored
            
            # If colormap is updated, reset the mask_plot to force redraw
            self.mask_plot = None
            
            self.update_display()
    
    def update_bg_adjustment(self):
        if self.bg_img is not None:
            try:
                brightness = self.brightness_slider.value() / 100.0
                contrast = 1.0 + (self.contrast_slider.value() / 100.0)
                
                # Convert to float for processing
                img_float = img_as_float(self.bg_img)
                
                # Apply contrast first (around mean)
                mean = np.mean(img_float)
                adjusted = (img_float - mean) * contrast + mean
                
                # Then apply brightness
                adjusted = adjusted + brightness
                
                # Clip to valid range
                adjusted = np.clip(adjusted, 0, 1)
                
                self.processed_bg = img_as_ubyte(adjusted)
                self.update_display()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Adjustment error: {str(e)}")
    
    def auto_adjust_bg(self):
        if self.bg_img is not None:
            try:
                # Apply adaptive histogram equalization for auto brightness/contrast
                if len(self.bg_img.shape) == 3 and self.bg_img.shape[2] >= 3:  # RGB/RGBA
                    # Process each channel
                    adjusted = np.zeros_like(self.bg_img)
                    for i in range(min(3, self.bg_img.shape[2])):  # Process up to 3 channels
                        # Use CLAHE for better results with microscopy images
                        adjusted[..., i] = exposure.equalize_adapthist(self.bg_img[..., i], clip_limit=0.03)
                    self.processed_bg = img_as_ubyte(adjusted)
                else:  # Grayscale
                    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    equalized = exposure.equalize_adapthist(self.bg_img, clip_limit=0.03)
                    self.processed_bg = img_as_ubyte(equalized)
                
                # Reset sliders
                self.brightness_slider.setValue(0)
                self.contrast_slider.setValue(0)
                
                self.update_display()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Auto-adjust error: {str(e)}")
    
    def update_display(self):
        # Clear and recreate the figure only if needed
        if not hasattr(self, 'ax') or self.figure.axes == []:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_axis_off()
        
        # First ensure all previously created plots are removed
        if self.bg_plot is not None:
            try:
                self.bg_plot.remove()
            except:
                pass
            self.bg_plot = None
            
        if self.mask_plot is not None:
            try:
                self.mask_plot.remove()
            except:
                pass
            self.mask_plot = None
        
        # Display background if available - background is fixed
        if self.processed_bg is not None:
            bg_cmap = 'gray' if len(self.processed_bg.shape) == 2 else None
            self.bg_plot = self.ax.imshow(
                self.processed_bg, 
                cmap=bg_cmap,
                extent=[0, self.processed_bg.shape[1], self.processed_bg.shape[0], 0],
                zorder=1  # Lower zorder means it's drawn first (behind)
            )
        
        # Display mask overlay if available - mask moves with offset
        if self.colored_mask is not None:
            # Get mask opacity
            opacity = self.opacity_slider.value() / 100.0
            
            # Create a copy with adjusted opacity
            overlay = self.colored_mask.copy()
            # Only adjust opacity where mask isn't transparent
            overlay[..., 3] = np.where(overlay[..., 3] > 0, opacity, 0)
            
            mask_h, mask_w = overlay.shape[:2]
            
            # Calculate the extent with the current offset
            # This is critical - extent defines the coordinates for the mask
            x_min = self.offset_x
            x_max = self.offset_x + mask_w
            y_min = self.offset_y
            y_max = self.offset_y + mask_h
            
            self.mask_plot = self.ax.imshow(
                overlay, 
                extent=[x_min, x_max, y_max, y_min],
                interpolation='nearest',
                zorder=2  # Higher zorder means it's drawn last (on top)
            )
            
            # Print debug info about the mask
            print(f"Mask dimensions: {mask_w}x{mask_h}")
            print(f"Mask displayed at: X={x_min:.1f}-{x_max:.1f}, Y={y_min:.1f}-{y_max:.1f}")
        
        # Set appropriate limits to see all content
        if self.processed_bg is not None:
            bg_h, bg_w = self.processed_bg.shape[:2]
            self.ax.set_xlim(0, bg_w)
            self.ax.set_ylim(bg_h, 0)  # Inverted y-axis for image coordinates
        
        self.canvas.draw()
    
    def on_press(self, event):
        if not event.inaxes or self.colored_mask is None:
            return
            
        # Get the mask dimensions
        mask_h, mask_w = self.colored_mask.shape[:2]
            
        # Calculate mask boundaries in the display
        x_min = self.offset_x
        x_max = self.offset_x + mask_w
        y_min = self.offset_y
        y_max = self.offset_y + mask_h
            
        # Check if click is within the mask boundaries
        if (x_min <= event.xdata <= x_max and 
            y_min <= event.ydata <= y_max):
                
            # Convert click position to mask array coordinates
            mask_x = int(event.xdata - self.offset_x)
            mask_y = int(event.ydata - self.offset_y)
                
            # Ensure we're within the mask array bounds
            if (0 <= mask_x < mask_w and 0 <= mask_y < mask_h):
                # Check if we clicked on a non-transparent part
                alpha = self.colored_mask[mask_y, mask_x, 3]
                if alpha > 0:
                    self.drag_enabled = True
                    self.last_pos = (event.xdata, event.ydata)
                    # Print debug info
                    print(f"Drag started at: {event.xdata:.1f}, {event.ydata:.1f}")
                    print(f"Current mask offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
    
    def on_release(self, event):
        if self.drag_enabled:
            print(f"Drag ended. Final mask offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
        self.drag_enabled = False
    
    def on_motion(self, event):
        if self.drag_enabled and event.inaxes and self.last_pos:
            # Calculate the distance moved
            dx = event.xdata - self.last_pos[0]
            dy = event.ydata - self.last_pos[1]
            
            # Update the mask position by changing its offset
            self.offset_x += dx
            self.offset_y += dy
            
            # Debug info
            print(f"Drag delta: dx={dx:.1f}, dy={dy:.1f}")
            print(f"New offset: X={self.offset_x:.1f}, Y={self.offset_y:.1f}")
            
            # Update offset display
            self.update_offset_display()
            
            # Update the last position for the next movement calculation
            self.last_pos = (event.xdata, event.ydata)
            
            # Redraw the display with the new offset
            self.update_display()
    
    def shift_without_warp(self, image, x_shift, y_shift):
        """
        Shift an image without using warpAffine, preserving exact pixel values
        """
        # Get dimensions
        if image.ndim == 3:
            height, width, channels = image.shape
        else:
            height, width = image.shape
            channels = 1
            
        # Create output image of same shape and data type
        if channels == 1 and image.ndim == 2:
            result = np.zeros((height, width), dtype=image.dtype)
        else:
            result = np.zeros((height, width, channels), dtype=image.dtype)
            
        # Calculate source and destination regions
        x_shift = int(round(x_shift))
        y_shift = int(round(y_shift))
        
        # Source region in the original image
        src_x_start = max(0, -x_shift)
        src_y_start = max(0, -y_shift)
        src_x_end = min(width, width - x_shift)
        src_y_end = min(height, height - y_shift)
        
        # Destination region in the result image
        dst_x_start = max(0, x_shift)
        dst_y_start = max(0, y_shift)
        dst_x_end = min(width, width + x_shift)
        dst_y_end = min(height, height + y_shift)
        
        # Make sure the regions have the same size
        width_to_copy = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
        height_to_copy = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
        
        if width_to_copy <= 0 or height_to_copy <= 0:
            return result  # Nothing to copy
            
        src_x_end = src_x_start + width_to_copy
        src_y_end = src_y_start + height_to_copy
        dst_x_end = dst_x_start + width_to_copy
        dst_y_end = dst_y_start + height_to_copy
        
        # Copy the relevant region
        if channels == 1 and image.ndim == 2:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                image[src_y_start:src_y_end, src_x_start:src_x_end]
        else:
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, :] = \
                image[src_y_start:src_y_end, src_x_start:src_x_end, :]
                
        return result
    
    def apply_registration(self):
        """Open registration tool for batch processing."""
        try:
            # Create an instance of RegistrationTool without starting a new QApplication
            reg_tool = registration.RegistrationTool(parent=self, offset_x=self.offset_x, offset_y=self.offset_y)
            
            # Show the dialog non-modally (won't block the main window)
            reg_tool.show()
            
            # Optional: Connect the dialog's closed signal to a handler if needed
            reg_tool.finished.connect(lambda: print("Registration dialog closed"))
            
        except ImportError:
            QMessageBox.critical(self, "Error", "Registration module not found. Make sure registration.py is in the same directory or in your Python path.")
            return
    
    def save_shifted_mask(self):
        """Apply the current offset to the original mask and save it."""
        if self.mask_img is None or self.mask_file_path is None:
            QMessageBox.warning(self, "Warning", "Please load a mask file first.")
            return

        # Suggest a default output filename based on the original
        base, ext = os.path.splitext(os.path.basename(self.mask_file_path))
        default_save_path = os.path.join(os.path.dirname(self.mask_file_path), f"{base}_shifted{ext}")

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Shifted Mask As", default_save_path,
                                                 "Image Files (*.png *.jpg *.tif *.tiff *.bmp);;All Files (*)")

        if not save_path:
            return # User cancelled

        try:
            # Read the original mask image again to ensure we have the raw data
            # This is important if the mask was converted to float for colormapping
            original_mask = imread(self.mask_file_path) 
            
            # Apply the shift using the current offsets
            x_shift = int(round(self.offset_x))
            y_shift = int(round(self.offset_y))
            
            shifted_mask = self.shift_without_warp(original_mask, x_shift, y_shift)
            
            # Save the shifted mask using skimage.io.imsave to preserve data type
            imsave(save_path, shifted_mask, check_contrast=False)
            
            QMessageBox.information(self, "Success", f"Shifted mask saved to:\n{save_path}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save shifted mask: {str(e)}")

    def showEvent(self, event):
        super().showEvent(event)
        # After showing, remove the always-on-top flag
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MaskVisualizationTool()
    window.show()
    sys.exit(app.exec_())