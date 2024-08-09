import os
from PIL import Image

def extract_images_from_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    # Loop through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(root, file)
                try:
                    # Open the image
                    with Image.open(file_path) as img:
                        print(f"Extracted image: {file_path}")
                        # Perform any required operations on the image
                        # For example: img.show() to display the image
                except Exception as e:
                    print(f"Failed to open image {file_path}: {e}")

if __name__ == "__main__":
    # Define the path to the A folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, 'A')
    extract_images_from_folder(folder_path)
