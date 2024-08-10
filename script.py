import cv2
import numpy as np
import os

# Paths to input and output folders
input_folder = 'A'
output_folder = 'B'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

print(f"Processing files in {input_folder}...")

# Parameters for all passes
min_width = 500           # Minimum width of the extracted photograph
min_height = 500          # Minimum height of the extracted photograph
aspect_ratio_range = (0.5, 2)  # Allowed aspect ratio range (width/height)
max_photos_per_file = 6   # Maximum number of photographs to extract from a single file
border_margin = 10        # Margin around the border to include edge photos

# Adjusted parameters for edge detection and preprocessing for each pass
passes = [
    {"canny_threshold1": 30, "canny_threshold2": 100, "blur_kernel_size": (3, 3), "dilate_iterations": 1},
    {"canny_threshold1": 15, "canny_threshold2": 60, "blur_kernel_size": (7, 7), "dilate_iterations": 2},
    {"canny_threshold1": 5, "canny_threshold2": 30, "blur_kernel_size": (11, 11), "dilate_iterations": 3},
]

def is_valid_photo(w, h, aspect_ratio):
    return w >= min_width and h >= min_height and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]

def extract_photos_from_file(image, filename, pass_num, extracted_photos):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocess the image
    params = passes[pass_num - 1]
    blurred = cv2.GaussianBlur(gray, params["blur_kernel_size"], 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, params["canny_threshold1"], params["canny_threshold2"])

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=params["dilate_iterations"])
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)

    # Find contours (which correspond to the boundaries of each photo)
    contours, hierarchy = cv2.findContours(eroded_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order
    contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)

    # Counter for number of photographs extracted
    photo_count = len(extracted_photos)
    photos_extracted_this_pass = 0

    # Loop over contours to extract and save each photo
    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:  # External contours only
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Adjust for photos close to the borders
            x = max(x - border_margin, 0)
            y = max(y - border_margin, 0)
            w = min(w + 2 * border_margin, image.shape[1] - x)
            h = min(h + 2 * border_margin, image.shape[0] - y)

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
                    success = cv2.imwrite(output_path, cropped_image)
                    
                    if success:
                        print(f"Extracted and saved {output_filename} to {output_folder}")
                        photo_count += 1
                        photos_extracted_this_pass += 1
                        extracted_photos.append((x, y, w, h))
                    else:
                        print(f"Failed to save {output_filename}")

    print(f"Pass {pass_num}: {photos_extracted_this_pass} photos extracted")
    return extracted_photos

print("Starting extraction...")
# Loop over all TIFF files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.tif'):
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

        extracted_photos = []
        for pass_num in range(1, 4):
            extracted_photos = extract_photos_from_file(image, filename, pass_num, extracted_photos)

print("Processing complete.")
