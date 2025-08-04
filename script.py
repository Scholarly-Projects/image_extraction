import cv2
import numpy as np
import os
from pathlib import Path
import time

# Configuration
input_folder = 'A'
output_folder = 'B'
os.makedirs(output_folder, exist_ok=True)

# Parameters
min_area = 50000          # Minimum area for a photo (pixels)
min_dim = 250             # Minimum width/height (pixels)
max_aspect = 2.2          # Maximum width/height ratio
min_aspect = 0.45         # New minimum to exclude vertical slices
margin = 2                # Border margin (pixels)
max_photos = 12           # Max photos per image
output_quality = 99       # JPEG quality (1-100)

def deskew_image(image, contour):
    """Straighten a rotated photo using perspective transform"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    
    if len(approx) != 4:
        return None  # Can't deskew non-quadrilaterals
    
    # Order points consistently
    points = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    # Calculate dimensions
    (tl, tr, br, bl) = rect
    width = max(int(np.linalg.norm(tr - tl)), int(np.linalg.norm(br - bl)))
    height = max(int(np.linalg.norm(bl - tl)), int(np.linalg.norm(br - tr)))
    
    # Perspective transform
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))

def simple_extract(image_path, output_prefix):
    """Simplified photo extraction with deskewing"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: Failed to load {image_path}")
        return []
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    photos = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
            
        rect = cv2.minAreaRect(cnt)
        (_, _), (w, h), _ = rect
        
        # Skip if too small or aspect ratio too extreme
        if min(w, h) < min_dim or max(w/h, h/w) > max_aspect:
            continue
            
        # Get bounding box with margin
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # Modern replacement for np.int0
        x, y, w, h = cv2.boundingRect(box)
        
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2*margin, img.shape[1] - x)
        h = min(h + 2*margin, img.shape[0] - y)
        
        # Deskew and save
        deskewed = deskew_image(img, cnt)
        if deskewed is None:  # Fallback to simple crop
            deskewed = img[y:y+h, x:x+w]
        
        photos.append(deskewed)
        
        if len(photos) >= max_photos:
            break
    
    # Save extracted photos
    for i, photo in enumerate(photos):
        cv2.imwrite(f"{output_prefix}_{i+1:03d}.jpg", photo, 
                   [int(cv2.IMWRITE_JPEG_QUALITY), output_quality])
    
    return len(photos)

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
    output_path = os.path.join(output_folder, base_name)
    
    found = simple_extract(input_path, output_path)
    total_photos += found
    
    print(f"  Extracted {found} photos in {time.time() - file_start:.2f}s")

print(f"\nFinished. Total {total_photos} photos extracted in {time.time() - start_time:.2f} seconds")
print(f"All photos saved to: {output_folder}")