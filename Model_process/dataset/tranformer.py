# this is what cookie cuts the stock (synthetic) images (white is not removed yet)
# basically it allows the stock images to have the dimensions of the tiktok ones

import os
import csv
import cv2
import numpy as np

def resize_to_outer(image):
    """
    Resize the image to 347x347.
    """
    return cv2.resize(image, (347, 347))

def mask_outer(image):
    """
    Given a 347x347 image, create an outer image with a donut:
    - Outer circle: diameter 345 px (radius ≈172)
    - Inner hole: diameter 209 px (radius ≈104)
    Areas outside the donut (both outside the outer circle and inside the hole) are set to white.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    outer_radius = 345 // 2  # ≈172 px
    inner_radius = 209 // 2  # ≈104 px
    cv2.circle(mask, center, outer_radius, 255, -1)  # outer circle white
    cv2.circle(mask, center, inner_radius, 0, -1)      # inner hole black
    white_bg = np.full_like(image, 255)
    return np.where(mask[..., None] == 255, image, white_bg)

def crop_inner(image):
    """
    From a 347x347 image, crop the center 211x211 region.
    """
    h, w = image.shape[:2]
    start_x = (w - 211) // 2
    start_y = (h - 211) // 2
    return image[start_y:start_y+211, start_x:start_x+211]

def mask_inner(image):
    """
    Given a 211x211 image, create an inner image by keeping only the circle
    of 209 px diameter (radius ≈104) and setting pixels outside to white.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = 209 // 2  # ≈104 px
    cv2.circle(mask, center, radius, 255, -1)
    white_bg = np.full_like(image, 255)
    return np.where(mask[..., None] == 255, image, white_bg)

def process_image(image):
    """
    Process a single synthetic image:
    1. Resize to 347x347.
    2. For the outer image: apply a donut mask.
    3. For the inner image: crop the center 211x211 area then apply a circular mask.
    Returns (inner_processed, outer_processed).
    """
    resized = resize_to_outer(image)
    outer_processed = mask_outer(resized)
    inner_cropped = crop_inner(resized)
    inner_processed = mask_inner(inner_cropped)
    return inner_processed, outer_processed

def process_dataset(input_folder, output_folder, csv_filename, start_id=201, end_id=400):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    csv_path = os.path.join(output_folder, csv_filename)
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["unique_identifier", "inner", "outer"])
        
        files = sorted(os.listdir(input_folder))
        current_id = start_id
        
        for filename in files:
            filepath = os.path.join(input_folder, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
            inner_img, outer_img = process_image(image)
            inner_name = f"{current_id}_inner.jpg"
            outer_name = f"{current_id}_outer.jpg"
            inner_save_path = os.path.join(output_folder, inner_name)
            outer_save_path = os.path.join(output_folder, outer_name)
            cv2.imwrite(inner_save_path, inner_img)
            cv2.imwrite(outer_save_path, outer_img)
            writer.writerow([current_id, inner_name, outer_name])
            current_id += 1
            if current_id > end_id:
                break

if __name__ == "__main__":
    input_folder = "stock_images"  # Folder with 200 squared images
    output_folder = "transformed_stock_images"  # Folder to save transformed images
    csv_filename = "stock_images.csv"
    process_dataset(input_folder, output_folder, csv_filename, start_id=201, end_id=400)
