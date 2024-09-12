# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 23:03:31 2024

@author: Jhon
"""


import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time

# Variables to control visualization
detector_type = 'SIFT'  # Change to 'ORB' if you need
plot_matches = False  # Change to True if you want to visualize matches
plot_homography = False  # Change to True if you want to visualize homografy

# Variables to select the image range
start_image_index = 0  # Use zero to run from the first photo
end_image_index = 30 #None  # Final photo index, use `None` to process till the final photo

# List to storage the number of matches between images
num_matches_list = []

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def load_images(image_dir):
    start_time = time.time()
    images = []
    filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    filenames.sort()  # Sort them if its necesary
    
    print(f"Total photos founded in the folder: {len(filenames)}")
    for i, filename in enumerate(filenames):
        print(f"{i}: {filename}")  # Enumerar y mostrar los nombres de las imágenes

    selected_filenames = filenames[start_image_index:end_image_index]  # Selecction of the imagery range
    
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

def detect_features(image,detector_type):
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
    if detector_type == 'ORB':
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    else:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    end_time = time.time()
    print(f"Time to get the matches with {detector_type}: {end_time - start_time:.2f} seconds")
    return good_matches

def visualize_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Matches')
    plt.axis('off')
    plt.show()

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

def create_mosaic(images):
    start_time = time.time()
    
    if len(images) < 2:
        raise ValueError("Two images at least are needed to create a mosaic.")
    
    cumulative_transform = np.eye(3)
    cumulative_transforms = [np.eye(3)]
    
    for i in range(1, len(images)):
        keypoints1, descriptors1 = detect_features(images[i-1], detector_type)
        keypoints2, descriptors2 = detect_features(images[i], detector_type)
        matches = match_features(descriptors1, descriptors2, detector_type)
        
        if len(matches) <= 10:
            print(f"There is not enough matches with {detector_type} between the image {i-1} and the image {i}.")
            print("Trying using ORB...")
            keypoints1, descriptors1 = detect_features(images[i-1], 'ORB')
            keypoints2, descriptors2 = detect_features(images[i], 'ORB')
            matches = match_features(descriptors1, descriptors2, 'ORB')
        
        num_matches_list.append(len(matches))
        
        if len(matches) > 10:
            H, mask = estimate_transform(keypoints1, keypoints2, matches)
            if H is not None:
                translation_rotation = np.linalg.inv(H)[:2, :]
                cumulative_transform = cumulative_transform @ np.vstack([translation_rotation, [0, 0, 1]])
                cumulative_transforms.append(cumulative_transform)
                
                if plot_matches:
                    visualize_matches(images[i-1], keypoints1, images[i], keypoints2, matches)
                
                if plot_homography:
                    visualize_homography(images[i-1], images[i], H)
            else:
                print(f"The homography matrix between image {i-1} and image {i} could not be calculated.")
        else:
            print(f"There are not enough matches even with ORB between image {i-1} and image {i}.")
    
    if len(cumulative_transforms) != len(images):
        print("!Warning: Not all cumulative transformations were calculated.")
    
    final_width, final_height, min_x, min_y = calculate_final_canvas_size(images, cumulative_transforms)
    canvas = np.ones((final_height, final_width, 3), dtype=np.uint8)*255
    
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
    # You should replace the image_dir with the folder which contains the imagery to process
    # You should replace the save_dir with the folder where you want to download the final mosaic
    # image_dir = r'C:\Users\Jhon\Desktop\High_Altitude_project\05_per_canon'
    image_dir = r'C:\Users\Jhon\Desktop\High_Altitude_project\Benchmark test\Low_images10percent'
    save_dir = 'C:/Users/Jhon/Desktop/High_Altitude_project/Mosaics/'

    # Load the images
    images, filenames = load_images(image_dir)
    
    # Mosaic creation
    mosaic = create_mosaic(images)

    # Plot the mosaic
    show_image(mosaic, "Final mosaic variation4")

    # Save the mosaic
    mosaic_path = os.path.join(save_dir, 'Mosaic_v5.1_canvastesting.jpg')
    cv2.imwrite(mosaic_path, mosaic)
    print(f"Mosaic saved in: {mosaic_path}")

    # Save a list of the number of matches between photos
    num_matches_path = os.path.join(save_dir, 'num_matches.txt')
    with open(num_matches_path, 'w') as file:
        for i in range(len(num_matches_list) - 1):
            file.write(f"{num_matches_list[i]},{num_matches_list[i+1]}\n")
    print(f"Matches list saved in: {num_matches_path}")