import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from skimage import feature
from PIL import Image, ImageChops, ImageEnhance



def convert_to_ela_image(image, quality=90):
    """Compute ELA of an image."""
    im = Image.fromarray(image) if isinstance(image, np.ndarray) else Image.open(image).convert('RGB')
    resaved_filename = "temp_resaved.jpg"
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    return ImageEnhance.Brightness(ela_im).enhance(scale)

def extract_features(image):
    """Extract HOG features."""
    image_array = np.array(image.convert('L'))  # Convert to grayscale
    return feature.hog(image_array, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

def cluster_subfolders(main_folder, output_file="dbscan_clustering_results.csv"):
    """Cluster subfolders using DBSCAN."""
    subfolder_features = []
    subfolder_names = []

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        print(f"Processing subfolder: {subfolder}")
        folder_features = []

        for filename in sorted(os.listdir(subfolder_path))[:12]:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(subfolder_path, filename)
                ela_image = convert_to_ela_image(image_path)
                features = extract_features(ela_image)
                folder_features.append(features)

        if folder_features:
            aggregated_feature = np.mean(folder_features, axis=0)
            subfolder_features.append(aggregated_feature)
            subfolder_names.append(subfolder)

    if not subfolder_features:
        print("No features extracted from any subfolder. Check the dataset.")
        return

    subfolder_features = np.array(subfolder_features)
    dbscan = DBSCAN(eps=1.1, min_samples=5)
    # dbscan = DBSCAN(eps=0.05, min_samples=5, metric='cosine') ## UPDATED
    cluster_labels = dbscan.fit_predict(subfolder_features)

    results = [{"Subfolder": name, "Cluster": int(label)} for name, label in zip(subfolder_names, cluster_labels)]
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Clustering results saved to {output_file}")

def main():
    main_folder = "cropped_output_frames"
    output_file = "dbscan_clustering_results.csv"
    cluster_subfolders(main_folder, output_file=output_file)

if __name__ == "__main__":
    main()