# Mosaic Tool

Feature detection algorithm for balloon imagery, designed to process images without metadata and generate mosaics using computer vision techniques. The code leverages feature detection, matching, and homography estimation to create a panoramic mosaic from a series of high-altitude images.

## Features
- **Automatic image loading**: Load images from a folder for processing.
- **Feature detection**: Supports both SIFT and ORB detectors to find and match keypoints between images.
- **Mosaic creation**: Combines images into a single mosaic using cumulative homography transformations.
- **Visualization tools**: Display feature matches and final mosaic.

## Requirements
OpenCV (cv2): This library is used for computer vision tasks such as reading images, detecting features, and processing images.

Install using: pip install opencv-python
NumPy (numpy): NumPy is used for numerical operations such as array manipulations and transformations.

Install using: pip install numpy
Matplotlib (pyplot): Matplotlib is used for plotting and visualizing images and other data.

Install using: pip install matplotlib
OS Module (os): This is part of the Python standard library, so no additional installation is needed. It provides functions for interacting with the operating system (e.g., listing directories, handling file paths).

Time Module (time): This is part of the Python standard library. It is used for timing events and measuring the duration of code execution.

USE:
pip install opencv-python numpy matplotlib
![Mosaic_v5 1_canvastes222t](https://github.com/user-attachments/assets/2e5241f3-2c54-4e05-b210-d4739faaa1ad)
