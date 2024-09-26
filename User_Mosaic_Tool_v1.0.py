# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:32:01 2024

@author: Jhon
"""

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time

# Variables to control visualization
detector_type = 'SIFT'  # Change to 'ORB' if you need
plot_matches = True  # Change to True if you want to visualize matches
plot_homography = False  # Change to True if you want to visualize homography

# Variables to select the image range
start_image_index = 0  # Use zero to run from the first photo
end_image_index = None  # Final photo index, use `None` to process till the final photo

# List to store the number of matches between images
num_matches_list = []
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def calculate_rotation_angle(H):
    h11 = H[0, 0]
    h12 = H[0, 1]
    angle_rad = np.arctan2(h12, h11)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated_image

def estimate_transform(keypoints1, keypoints2, matches):
    start_time = time.time()
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
    if H is not None:
        # print("Homography matrix:\n", H)
        angle = calculate_rotation_angle(H)
        print(f"Rotation angle: {angle:.2f} grades")
    else:
        print("The Homography Matrix coudn't be calculated")
    
    end_time = time.time()
    print(f"Time to get the transformation: {end_time - start_time:.2f} seconds")
    
    return H, mask

def apply_homography(image, H, canvas_shape):
    start_time = time.time()
    h, w = canvas_shape[:2]
    warped_image = cv2.warpPerspective(image, H, (w, h))
    end_time = time.time()
    print(f"Time to apply the homography: {end_time - start_time:.2f} seconds")
    return warped_image

def get_rotated_dimensions(image, H):
    h, w = image.shape[:2]
    corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)
    corners = np.expand_dims(corners, axis=1)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    min_x = np.min(transformed_corners[:, 0, 0])
    max_x = np.max(transformed_corners[:, 0, 0])
    min_y = np.min(transformed_corners[:, 0, 1])
    max_y = np.max(transformed_corners[:, 0, 1])
    
    new_width = int(np.ceil(max_x - min_x))
    new_height = int(np.ceil(max_y - min_y))    
    return new_width, new_height, min_x, min_y

def calculate_final_canvas_size(images, cumulative_transforms):
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for i in range(len(images)):
        h, w = images[i].shape[:2]
        corners = np.array([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]], dtype=np.float32)
        corners = np.expand_dims(corners, axis=1)
        
        # Verify that cumulative_transforms has the proper lenght
        if i < len(cumulative_transforms):
            transformed_corners = cv2.perspectiveTransform(corners, cumulative_transforms[i])
            
            # Update the coordinates min and max
            min_x = min(min_x, np.min(transformed_corners[:, 0, 0]))
            max_x = max(max_x, np.max(transformed_corners[:, 0, 0]))
            min_y = min(min_y, np.min(transformed_corners[:, 0, 1]))
            max_y = max(max_y, np.max(transformed_corners[:, 0, 1]))
        else:
            print(f"Warning: Cumulative Transformation for the image {i} not available.")
    
    # Calculate the final canvas dimensions
    final_width = int(np.ceil(max_x - min_x))
    final_height = int(np.ceil(max_y - min_y))

    return final_width, final_height, min_x, min_y

def expand_canvas(canvas, new_width, new_height):
    new_canvas = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    new_canvas[:canvas.shape[0], :canvas.shape[1]] = canvas
    return new_canvas

def visualize_homography(img1, img2, H):
    start_time = time.time()
    h, w = img1.shape[:2]
    new_width, new_height, min_x, min_y = get_rotated_dimensions(img2, H)
    
    img2_warped = apply_homography(img2, H, (new_height, new_width))
    
    angle = calculate_rotation_angle(H)
    
    print(f"Rotation angle calculated: {angle:.2f} grades")
    
    # Rotar la imagen original con el ángulo negativo calculado
    img2_rotated = rotate_image(img2, -angle)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Base Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2RGB))
    plt.title('Rotaded Image')
    plt.axis('off')
    
    plt.show()
    
    end_time = time.time()
    print(f"Time to visualize the homography: {end_time - start_time:.2f} seconds")

def load_images(image_dir):
    start_time = time.time()
    images = []
    filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    filenames.sort()  # Sort them if it's necessary
    
    print(f"Total photos found in the folder: {len(filenames)}")
    for i, filename in enumerate(filenames):
        print(f"{i}: {filename}")  # Enumerate and show image names

    selected_filenames = filenames[start_image_index:end_image_index]  # Selection of the image range
    
    for filename in selected_filenames:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    end_time = time.time()
    print(f"Time to load the images: {end_time - start_time:.2f} seconds")
    return images, selected_filenames

def show_image(img, title="Image"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def detect_features(image, detector_type):
    start_time = time.time()
    gray = preprocess_image(image)
    
    if detector_type == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=1000)
    elif detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=500000)
    else:
        raise ValueError("Detector type not supported.")
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    end_time = time.time()
    print(f"Time to detect features with {detector_type}: {end_time - start_time:.2f} seconds")
    return keypoints, descriptors

def match_features(descriptors1, descriptors2, detector_type):
    start_time = time.time()
    
    if detector_type == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    elif detector_type == 'ORB':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError("Detector type not supported.")
    
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    end_time = time.time()
    print(f"Time to get the matches with {detector_type}: {end_time - start_time:.2f} seconds")
    return good_matches
def analyze_deformation(H, max_aspect_ratio_change=0.2, max_deformation=0.15):
    """
    Analyzes the homography matrix H for deformation.
    
    :param H: Homography matrix
    :param max_aspect_ratio_change: Maximum allowed change in aspect ratio
    :param max_deformation: Maximum allowed skew or non-uniform scaling
    :return: True if the transformation is within allowable limits, otherwise False
    """
    # Extract scaling factors (singular values from SVD)
    U, S, Vt = np.linalg.svd(H[:2, :2])

    scale_x, scale_y = S[0], S[1]

    # Aspect ratio change (difference between scaling factors)
    aspect_ratio_change = abs(scale_x - scale_y) / max(scale_x, scale_y)

    # Deformation (non-uniform scaling/skew detection)
    deformation = np.abs(scale_x - 1) + np.abs(scale_y - 1)

    if aspect_ratio_change > max_aspect_ratio_change:
        print(f"Aspect ratio change {aspect_ratio_change:.2f} exceeds threshold of {max_aspect_ratio_change}")
        return False
    
    if deformation > max_deformation:
        print(f"Deformation {deformation:.2f} exceeds threshold of {max_deformation}")
        return False

    return True

def visualize_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Matches')
    plt.axis('off')
    plt.show()
def find_mosaics(images):
    mosaics = []  # List to store groups of images
    current_mosaic = []  # Current group of images
    
    for i in range(len(images) - 1):
        keypoints1, descriptors1 = detect_features(images[i], detector_type)
        keypoints2, descriptors2 = detect_features(images[i + 1], detector_type)
        matches = match_features(descriptors1, descriptors2, detector_type)
        
        if len(matches) > 10:
            # If there are matches, add the image to the current group
            current_mosaic.append(i)
        else:
            # If no matches and there's a group being built, save the group
            if current_mosaic:
                current_mosaic.append(i)  # Add the current image
                mosaics.append(current_mosaic)  # Save the mosaic
                current_mosaic = []  # Reset the current group
    
    # Check if there's a group being built at the end
    if current_mosaic:
        current_mosaic.append(len(images) - 1)
        mosaics.append(current_mosaic)

    return mosaics

def create_mosaic(images):
    start_time = time.time()
    
    if len(images) < 2:
        raise ValueError("Two images at least are needed to create a mosaic.")
    
    cumulative_transform = np.eye(3)
    cumulative_transforms = [np.eye(3)]
    
    for i in range(1, len(images)):
        successful_match = False
        attempts = 0
        max_attempts = min(len(images) - i, 5)  # Try up to 5 matches ahead
        
        # Try to find a match between image[i-1] and image[i], up to 5 times with different images
        while not successful_match and attempts < max_attempts:
            next_image_index = i + attempts  # Start with i and increment for subsequent attempts
            print(f"Trying to match image {i-1} with image {next_image_index} (Attempt {attempts + 1})")
            
            keypoints1, descriptors1 = detect_features(images[i-1], detector_type)
            keypoints2, descriptors2 = detect_features(images[next_image_index], detector_type)
            matches = match_features(descriptors1, descriptors2, detector_type)
            
            num_matches_list.append(len(matches))
            
            if len(matches) > 10:
                H, mask = estimate_transform(keypoints1, keypoints2, matches)
                
                if H is not None:
                    # Analyze the homography matrix for deformation
                    if analyze_deformation(H):
                        # If the homography is valid, apply the transformation
                        translation_rotation = np.linalg.inv(H)[:2, :]
                        cumulative_transform = cumulative_transform @ np.vstack([translation_rotation, [0, 0, 1]])
                        cumulative_transforms.append(cumulative_transform)
                        
                        if plot_matches:
                            visualize_matches(images[i-1], keypoints1, images[next_image_index], keypoints2, matches)
                        
                        if plot_homography:
                            visualize_homography(images[i-1], images[next_image_index], H)
                        
                        successful_match = True  # Mark the match as successful
                    else:
                        print(f"Deformation detected in the homography matrix between image {i-1} and image {next_image_index}.")
                else:
                    print(f"The homography matrix between image {i-1} and image {next_image_index} could not be calculated.")
            else:
                print(f"Not enough matches between image {i-1} and image {next_image_index}.")
            
            attempts += 1  # Increment attempts
        
        if not successful_match:
            print(f"Unable to find a good match for image {i-1} after {max_attempts} attempts.")
            break  # Optionally stop the process if no match is found after 5 tries
    
    if len(cumulative_transforms) != len(images):
        print("!Warning: Not all cumulative transformations were calculated.")
    
    final_width, final_height, min_x, min_y = calculate_final_canvas_size(images, cumulative_transforms)
    canvas = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255
    
    translation_matrix = np.array([[1, 0, -min_x],
                                   [0, 1, -min_y],
                                   [0, 0, 1]])
    
    for i in range(len(images)):
        if i >= len(cumulative_transforms):
            print(f"Warning: Cumulative transformation for image {i} not available.")
            continue
        
        H = translation_matrix @ cumulative_transforms[i]
        warped_image = apply_homography(images[i], H, canvas.shape[:2])
        
        mask = warped_image > 0
        canvas[mask] = warped_image[mask]
        
    end_time = time.time()
    print(f"Total time to create the mosaic: {end_time - start_time:.2f} seconds")
    
    return canvas

if __name__ == "__main__":
    # Directories
    image_dir = r'C:\Users\Jhon\Desktop\High_Altitude_project\test_sorting'
    save_dir = 'C:/Users/Jhon/Desktop/High_Altitude_project/Mosaics/Recurrent2'

    # Load the images
    images, filenames = load_images(image_dir)
    
    # Find mosaics
    mosaics = find_mosaics(images)
    
    # Show the groups of images
    for idx, mosaic_indices in enumerate(mosaics):
        print(f"Mosaico {idx + 1}: {sorted(mosaic_indices)}")
        
        # Create and show the mosaic
        selected_images = [images[i] for i in mosaic_indices]
        mosaic_image = create_mosaic(selected_images)  # Create the mosaic
        show_image(mosaic_image, f"Mosaic {idx + 1}")
        
        # Save the mosaic
        mosaic_path = os.path.join(save_dir, f'Mosaic_{idx + 1}.jpg')
        cv2.imwrite(mosaic_path, mosaic_image)
        print(f"Mosaic saved in: {mosaic_path}")

    # Save a list of the number of matches between photos
    num_matches_path = os.path.join(save_dir, 'num_matches.txt')
    with open(num_matches_path, 'w') as file:
        for i in range(len(num_matches_list) - 1):
            file.write(f"{num_matches_list[i]},{num_matches_list[i+1]}\n")
    print(f"Matches list saved in: {num_matches_path}")
