import cv2
import numpy as np
import os
from pathlib import Path
import time

# Paths to input and output folders
input_folder = 'A'
output_folder = 'B'
debug_folder = os.path.join(output_folder, 'debug')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

print(f"Processing files in {input_folder}...")

# Optimized parameters
min_width = 250           # Minimum width of the extracted photograph
min_height = 250          # Minimum height of the extracted photograph
aspect_ratio_range = (0.5, 2.0)  # Allowed aspect ratio range (width/height)
max_photos_per_file = 12   # Maximum number of photographs to extract
border_margin = 5         # Margin around the border to include edge photos
min_contour_area = 50000  # Minimum area for a contour to be considered
iou_threshold = 0.15      # Intersection-over-Union threshold for duplicates

# Multi-pass configuration with different detection strategies
passes = [
    # Pass 1: Standard Canny edge detection
    {
        "name": "Canny Edge Detection",
        "blur_kernel_size": (5, 5),
        "canny_threshold1": 30,
        "canny_threshold2": 100,
        "dilate_iterations": 2,
        "mask_after": False
    },
    # Pass 2: Adaptive thresholding for faint edges
    {
        "name": "Adaptive Thresholding",
        "blur_kernel_size": (3, 3),
        "thresh_block_size": 31,
        "thresh_c": 5,
        "close_kernel_size": 15,
        "mask_after": True
    },
    # Pass 3: Morphological gradient for structural detection
    {
        "name": "Morphological Gradient",
        "blur_kernel_size": (3, 3),
        "gradient_kernel": (15, 15),
        "mask_after": True
    }
]

def is_valid_photo(w, h, aspect_ratio, contour_area):
    """Check if detected region meets photo criteria with additional area check"""
    return (w >= min_width and 
            h >= min_height and 
            aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
            contour_area >= min_contour_area)

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    # Determine coordinates of intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    # Compute area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute area of both boxes
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def is_duplicate(new_box, existing_boxes):
    """Check if new box is a duplicate of existing ones using IoU"""
    for box in existing_boxes:
        if calculate_iou(new_box, box) > iou_threshold:
            return True
    return False

def create_mask_from_photos(image_shape, photos):
    """Create mask for already detected photos with careful dilation"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for (x, y, w, h) in photos:
        # Mask the photo area with small expansion
        cv2.rectangle(mask, (x-5, y-5), (x+w+10, y+h+10), 255, -1)
    return mask

def extract_photos(image, filename, debug=False):
    """Main extraction function with multiple detection strategies"""
    original_ext = Path(filename).suffix.lower()
    extracted_photos = []
    all_contours = []
    
    # Create working image
    working_img = image.copy()
    
    for pass_idx, params in enumerate(passes):
        pass_num = pass_idx + 1
        print(f"Processing pass {pass_num}: {params['name']}")
        
        # Apply mask if needed
        if pass_num > 1 and any(p.get("mask_after", False) for p in passes[:pass_idx]):
            mask = create_mask_from_photos(image.shape, extracted_photos)
            working_img = cv2.bitwise_and(working_img, working_img, mask=~mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing blur
        blurred = cv2.GaussianBlur(gray, params["blur_kernel_size"], 0)
        
        # Apply specific detection method for this pass
        if "canny_threshold1" in params:
            # Canny edge detection
            edges = cv2.Canny(blurred, params["canny_threshold1"], params["canny_threshold2"])
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.dilate(edges, kernel, iterations=params["dilate_iterations"])
        elif "thresh_block_size" in params:
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, params["thresh_block_size"], params["thresh_c"]
            )
            kernel = np.ones((params["close_kernel_size"], params["close_kernel_size"]), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        elif "gradient_kernel" in params:
            # Morphological gradient
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, params["gradient_kernel"])
            gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
            _, processed = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, hierarchy = cv2.findContours(processed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
            
        # Process contours
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:  # Skip inner contours
                continue
                
            # Get rotated rectangle for accurate dimensions
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int8(box)
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            contour_area = cv2.contourArea(contour)
            
            # Store all contours for debug
            if debug:
                all_contours.append((x, y, w, h, width, height, aspect_ratio, contour_area))
            
            # Check if valid photo
            if is_valid_photo(width, height, aspect_ratio, contour_area):
                # Expand bounding box
                x = max(x - border_margin, 0)
                y = max(y - border_margin, 0)
                w = min(w + 2 * border_margin, image.shape[1] - x)
                h = min(h + 2 * border_margin, image.shape[0] - y)
                
                # Create candidate box
                candidate = (x, y, w, h)
                
                # Check for duplicates
                if not is_duplicate(candidate, extracted_photos):
                    extracted_photos.append(candidate)
        
        # Stop if we've reached max photos
        if len(extracted_photos) >= max_photos_per_file:
            break
    
    # Create debug image if requested
    if debug:
        debug_img = image.copy()
        
        # Draw all detected contours
        for (x, y, w, h, width, height, aspect_ratio, area) in all_contours:
            color = (0, 0, 255)  # Red for invalid
            if is_valid_photo(width, height, aspect_ratio, area):
                color = (0, 255, 0)  # Green for valid
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 1)
        
        # Draw final extracted photos
        for i, (x, y, w, h) in enumerate(extracted_photos):
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_img, str(i+1), (x+5, y+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save debug image
        debug_path = os.path.join(debug_folder, f"{os.path.splitext(filename)[0]}_debug.jpg")
        cv2.imwrite(debug_path, debug_img)
    
    # Save extracted photos with original extension
    for i, (x, y, w, h) in enumerate(extracted_photos):
        output_path = os.path.join(
            output_folder, 
            f"{os.path.splitext(filename)[0]}_{i+1:03d}{original_ext}"
        )
        cropped_image = image[y:y+h, x:x+w]
        
        # Save with same quality as original
        if original_ext in ('.jpg', '.jpeg'):
            cv2.imwrite(output_path, cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        elif original_ext == '.png':
            cv2.imwrite(output_path, cropped_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        else:  # For TIFF and other formats
            cv2.imwrite(output_path, cropped_image)
    
    return extracted_photos

print("Starting optimized photo extraction...")
supported_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp')

# Process files
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(supported_extensions):
        continue
        
    print(f"\nProcessing {filename}...")
    start_time = time.time()
    
    # Load image
    image = cv2.imread(os.path.join(input_folder, filename))
    if image is None:
        print(f"Failed to load {filename}")
        continue
    
    # Extract photos
    extracted_photos = extract_photos(image, filename, debug=True)
    
    # Report results
    elapsed = time.time() - start_time
    print(f"Extracted {len(extracted_photos)} photos in {elapsed:.2f} seconds")
    print(f"Saved to {output_folder} with original file format")

print("\nProcessing complete. Summary:")
print(f"- Processed {len(os.listdir(input_folder))} files")
print(f"- Minimum photo size: {min_width}x{min_height} pixels")
print(f"- Aspect ratio range: {aspect_ratio_range[0]:.1f} to {aspect_ratio_range[1]:.1f}")
print(f"- Output files maintain original format from folder A")
print(f"- Debug images available in: {debug_folder}")