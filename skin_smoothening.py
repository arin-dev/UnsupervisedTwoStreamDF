import cv2
import numpy as np
import os
import pandas as pd

def classify_skin_texture(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
    laplacian_var = cv2.Laplacian(blurred_image, cv2.CV_64F).var()
    return laplacian_var

def process_images_in_folder(folder_path, threshold=15.0):
    results = []
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        given_label = 0 if subfolder.count("id") == 2 else 1
        if os.path.isdir(subfolder_path):
            labels = []
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(subfolder_path, filename)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Error: Could not open image {image_path}.")
                        continue
                    label = classify_skin_texture(image)
                    labels.append(label)
            if labels:
                average_label = np.mean(labels)
                assigned_label = 1 if average_label > threshold else 0
                results.append((subfolder, assigned_label, average_label, given_label))
    return results

def main(folder_path):
    results = process_images_in_folder(folder_path)
    df = pd.DataFrame(results, columns=['Image', 'Assigned Label', 'Skin Texture Classification', 'Given Label'])
    df.to_csv('skin_texture_results.csv', index=False)

if __name__ == "__main__":
    main('./cropped_output_frames')
