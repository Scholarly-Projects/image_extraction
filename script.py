from skimage import io, color, filters, measure, morphology, util
from skimage.morphology import binary_closing, remove_small_objects
from scipy.ndimage import binary_fill_holes
import numpy as np
import os
from pathlib import Path
import time
from PIL import Image

# Configuration
input_folder = 'A'
output_folder = 'B'
os.makedirs(output_folder, exist_ok=True)

# Parameters
min_long_edge = 300
max_long_edge = 1000
margin = 0
max_photos = 10

def simple_extract(image_path, output_prefix):
    image = io.imread(image_path)
    if image.ndim == 2:
        image = color.gray2rgb(image)

    gray = color.rgb2gray(image)
    
    # Thresholding works better than edge detection for historic photos
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh
    
    # Close small gaps and fill holes
    closed = binary_closing(binary, footprint=morphology.rectangle(5, 5))
    filled = binary_fill_holes(closed)
    
    # Remove small objects
    cleaned = remove_small_objects(filled, min_size=(min_long_edge * min_long_edge)//4)
    
    # Find contours
    label_image = measure.label(cleaned)
    photos = []
    
    for region in measure.regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        width, height = maxc - minc, maxr - minr
        long_edge = max(width, height)
        aspect_ratio = width / height
        
        # Filter by size and reasonable photo aspect ratios (0.5-2.0)
        if (long_edge < min_long_edge or long_edge > max_long_edge or 
            aspect_ratio < 0.5 or aspect_ratio > 2.0):
            continue
            
        # Extract with margin
        x = max(minc - margin, 0)
        y = max(minr - margin, 0)
        w = min(width + 2 * margin, image.shape[1] - x)
        h = min(height + 2 * margin, image.shape[0] - y)

        crop = image[y:y+h, x:x+w]
        photos.append(util.img_as_ubyte(crop))

        if len(photos) >= max_photos:
            break

    # Save extracted photos
    for i, photo in enumerate(photos):
        im = Image.fromarray(photo)
        im.save(f"{output_prefix}_{i+1:03d}.jpg", format='JPEG', quality=90)

    return len(photos)

# === MAIN EXECUTION ===
print(f"Processing images from {input_folder}...")
start_time = time.time()
total_photos = 0

for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
        continue

    print(f"\nProcessing {filename}...")
    file_start = time.time()

    base_name = Path(filename).stem
    input_path = os.path.join(input_folder, filename)
    output_prefix = os.path.join(output_folder, base_name)

    found = simple_extract(input_path, output_prefix)
    total_photos += found

    print(f"  Extracted {found} photos in {time.time() - file_start:.2f}s")

print(f"\nFinished. Total {total_photos} photos extracted in {time.time() - start_time:.2f} seconds")
print(f"All photos saved to: {output_folder}")