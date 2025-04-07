import os
import cv2
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

# Paths
INPUT_FOLDER = "transformed_sinthetic"
OUTPUT_FOLDER = "rotated_synthetic"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Output metadata
output_data = []

# Helper function to rotate image
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

# Get all _inner images
inner_files = sorted([f for f in os.listdir(INPUT_FOLDER) if "_inner" in f and f.endswith(".jpg")])

# Create synthetic rotated dataset
for idx, inner_filename in tqdm(enumerate(inner_files), total=len(inner_files)):
    # Get corresponding outer filename
    identifier = inner_filename.split("_")[0]
    outer_filename = f"{identifier}_outer.jpg"

    # Load images
    inner_path = os.path.join(INPUT_FOLDER, inner_filename)
    outer_path = os.path.join(INPUT_FOLDER, outer_filename)
    inner_img = cv2.imread(inner_path)
    outer_img = cv2.imread(outer_path)

    if inner_img is None or outer_img is None:
        print(f"Skipping {identifier}: image not found or corrupted.")
        continue

    # Generate random angles between -180 and 180
    inner_angle = random.uniform(-180, 180)
    outer_angle = random.uniform(-180, 180)
    total_angle = outer_angle - inner_angle

    # Rotate images
    rotated_inner = rotate_image(inner_img, inner_angle)
    rotated_outer = rotate_image(outer_img, outer_angle)

    # New filenames
    new_id = f"{idx:03d}"
    new_inner_name = f"{new_id}_inner.jpg"
    new_outer_name = f"{new_id}_outer.jpg"
    inner_out_path = os.path.join(OUTPUT_FOLDER, new_inner_name)
    outer_out_path = os.path.join(OUTPUT_FOLDER, new_outer_name)

    # Save images
    cv2.imwrite(inner_out_path, rotated_inner)
    cv2.imwrite(outer_out_path, rotated_outer)

    # Append row to output data
    output_data.append({
        "unique_identifier": new_id,
        "inner": new_inner_name,
        "outer": new_outer_name,
        "degree_rotation_inner": round(inner_angle, 2),
        "degree_rotation_outer": round(outer_angle, 2),
        "degree_rotation_total": round(total_angle, 2),
        "dataset": "rotated_synthetic",
        "correct_angle": round(total_angle, 2)
    })

# Save to CSV
df_output = pd.DataFrame(output_data)
df_output.to_csv("rotated_dataset_metadata.csv", index=False)

print("✅ Dataset creation complete! CSV and images saved.")
