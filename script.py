import cv2
import numpy as np
import os

# Paths to input and output folders
input_folder = 'A'
output_folder = 'B'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

print(f"Processing files in {input_folder}...")

# Parameters
min_width = 500           # Minimum width of the extracted photograph
min_height = 500          # Minimum height of the extracted photograph
aspect_ratio_range = (0.5, 2)  # Allowed aspect ratio range (width/height)
max_photos_per_file = 6   # Maximum number of photographs to extract from a single file

# Parameters for edge detection and preprocessing
canny_threshold1 = 100   # Lower threshold for Canny edge detection
canny_threshold2 = 200   # Upper threshold for Canny edge detection
blur_kernel_size = (5, 5)  # Size of Gaussian blur kernel

def is_valid_photo(w, h, aspect_ratio):
    return w >= min_width and h >= min_height and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]

def extract_photos_from_file(image, filename):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess the image
    blurred = cv2.GaussianBlur(gray, blur_kernel_size, 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

    # Find contours (which correspond to the boundaries of each photo)
    contours, hierarchy = cv2.findContours(eroded_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    # Counter for number of photographs extracted
    photo_count = 0
    extracted_photos = []

    # Loop over contours to extract and save each photo
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:  # External contours only
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if is_valid_photo(w, h, aspect_ratio):
                # Check for overlap with previously saved photos
                overlap = False
                for (ex_x, ex_y, ex_w, ex_h) in extracted_photos:
                    if not (x > ex_x + ex_w or x + w < ex_x or y > ex_y + ex_h or y + h < ex_y):
                        overlap = True
                        break
                
                if not overlap:
                    if photo_count >= max_photos_per_file:
                        break
                    
                    cropped_image = image[y:y+h, x:x+w]
                    output_filename = f'{os.path.splitext(filename)[0]}_{photo_count + 1:02d}.tif'
                    output_path = os.path.join(output_folder, output_filename)
                    
                    # Print the output path for debugging
                    print(f"Saving extracted photo to: {output_path}")
                    
                    success = cv2.imwrite(output_path, cropped_image)
                    
                    if success:
                        print(f"Extracted and saved {output_filename} to {output_folder}")
                        photo_count += 1
                        extracted_photos.append((x, y, w, h))
                    else:
                        print(f"Failed to save {output_filename}")

print("Starting extraction...")
# Loop over all TIFF files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.tif', '.tiff')):
        print(f"Processing {filename}...")
        
        # Construct full file path
        input_path = os.path.join(input_folder, filename)
        print(f"Loading image from {input_path}...")
        
        # Load the TIFF image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to load image {input_path}. It might be corrupted or not exist.")
            continue

        print(f"Image loaded successfully. Shape: {image.shape}")

        extract_photos_from_file(image, filename)

print("Processing complete.")
