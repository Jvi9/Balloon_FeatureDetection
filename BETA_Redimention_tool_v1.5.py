# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 22:46:54 2024

@author: Jhon
"""

import cv2
import os

def resize_image(image_path, reduction_factor, method, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load the image {image_path}.")
        return
    
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    new_height = int(height * reduction_factor)
    new_width = int(width * reduction_factor)
    dim = (new_width, new_height)
    
    # Choose the interpolation method
    if method == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'bicubic':
        interpolation = cv2.INTER_CUBIC
    elif method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'area':
        interpolation = cv2.INTER_AREA
    else:
        raise ValueError("Unsupported method. Use 'bilinear', 'bicubic', 'nearest', or 'area'.")
    
    # Resize the image
    resized_image = cv2.resize(image, dim, interpolation=interpolation)
    
    # Save the resized image to the output folder
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        cv2.imwrite(output_path, resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # Adjust quality if necessary
    else:
        cv2.imwrite(output_path, resized_image)

def process_folder(folder_path, output_folder, reduction_factor, method):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Supported extensions
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)
            
            # Resize and save the image in the output folder
            resize_image(image_path, reduction_factor, method, output_path)
            print(f'Processed: {image_path} -> Saved in: {output_path}')

# Specify the path of the folders, the reduction factor, and the interpolation method
folder_path = r'C:\Users\Jhon\Desktop\High_Altitude_project\Benchmark test\High_images'
# Name the directory to create or save the data
output_folder = r'C:\Users\Jhon\Desktop\High_Altitude_project\Benchmark test\Low_images30percent'
reduction_factor = 0.10  # 50% reduction in the original dimensions
method = 'area'  # Can be 'bilinear', 'bicubic', 'nearest', or 'area'

process_folder(folder_path, output_folder, reduction_factor, method)
