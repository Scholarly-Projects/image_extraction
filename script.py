import cv2
import numpy as np
import os

# Paths to input and output folders
input_folder = 'A'
output_folder = 'B'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop over all TIFF files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.tiff'):
        # Construct full file path
        input_path = os.path.join(input_folder, filename)
        
        # Load the TIFF image
        image = cv2.imread(input_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours (which correspond to the boundaries of each photo)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over contours to extract and save each photo
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = image[y:y+h, x:x+w]
            output_filename = f'{os.path.splitext(filename)[0]}_photo_{i}.tiff'
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, cropped_image)
            print(f"Extracted and saved {output_filename} to {output_folder}")

print("Processing complete.")
