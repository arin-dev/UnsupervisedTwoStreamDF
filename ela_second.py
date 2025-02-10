import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from keras.applications import ResNet50
from keras.models import Model
from keras.applications.resnet50 import preprocess_input


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import tensorflow as tf

# Function to visualize clustering results
def visualize_clustering(features, labels):
    """Visualize the clustering results using PCA."""
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()


# Function to compute ELA
def compute_ela(image, quality=96):
    """Compute Error Level Analysis (ELA) of an image."""
    _, compressed = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed = cv2.imdecode(compressed, 1)
    ela = cv2.absdiff(image, compressed)
    return ela

# Function to extract features using a pre-trained CNN
def extract_features(model, images):
    """Extract deep features using a pre-trained CNN model."""
    features = []
    for img in images:
        processed_img = preprocess_input(cv2.resize(img, (224, 224)))
        feature = model.predict(np.expand_dims(processed_img, axis=0))
        features.append(feature.flatten())
    return np.array(features)

# Function to cluster subfolders
def cluster_subfolders(main_folder, model, n_clusters=2, output_file="clustering_results.csv"):
    """Cluster subfolders into two clusters based on their aggregated features."""
    subfolder_features = []
    subfolder_names = []

    # Process each subfolder
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        print(f"Processing subfolder: {subfolder}")
        frames = []

        # Load up to 12 frames from the subfolder
        for frame_name in sorted(os.listdir(subfolder_path))[:12]:
            frame_path = os.path.join(subfolder_path, frame_name)
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to load frame: {frame_path}")
                continue
            ela_frame = compute_ela(frame)
            frames.append(ela_frame)

        if len(frames) == 0:
            print(f"No valid frames found in subfolder: {subfolder}")
            continue

        # Extract and aggregate features for the subfolder
        features = extract_features(model, frames)
        aggregated_feature = np.mean(features, axis=0)  # Aggregate features by averaging
        subfolder_features.append(aggregated_feature)
        subfolder_names.append(subfolder)

    if not subfolder_features:
        print("No features extracted from any subfolder. Check the dataset.")
        return

    # Perform clustering
    subfolder_features = np.array(subfolder_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
    cluster_labels = kmeans.fit_predict(subfolder_features)

    # Visualize clustering results
    visualize_clustering(subfolder_features, cluster_labels)
# def cluster_subfolders(main_folder, model, n_clusters=2, output_file="clustering_results.csv"):
#     """Cluster images from subfolders into clusters based on their individual features."""
#     all_features = []  # Store features for all images
#     all_images = []    # Store image paths for labeling

#     # Process each subfolder
#     for subfolder in os.listdir(main_folder):
#         subfolder_path = os.path.join(main_folder, subfolder)
#         if not os.path.isdir(subfolder_path):
#             continue

#         print(f"Processing subfolder: {subfolder}")

#         # Load up to 12 frames from the subfolder
#         for frame_name in sorted(os.listdir(subfolder_path))[:12]:
#             frame_path = os.path.join(subfolder_path, frame_name)
#             frame = cv2.imread(frame_path)
#             if frame is None:
#                 print(f"Failed to load frame: {frame_path}")
#                 continue
#             ela_frame = compute_ela(frame)
#             all_images.append(frame_path)  # Store image path

#             # Extract features for the individual image
#             feature = extract_features(model, [ela_frame])
#             all_features.append(feature.flatten())  # Flatten and store the feature

#     if not all_features:
#         print("No features extracted from any images. Check the dataset.")
#         return

#     # Perform clustering on all individual image features
#     all_features = np.array(all_features)
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(all_features)

#     # Save results for all images with their corresponding cluster labels
#     results = [{"Image": img, "Cluster": int(label)} for img, label in zip(all_images, cluster_labels)]
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(output_file, index=False)
#     print(f"Clustering results saved to {output_file}")

#     # Visualize clustering results
#     visualize_clustering(all_features, cluster_labels)

    # Save results
    results = [{"Subfolder": name, "Cluster": int(label)} for name, label in zip(subfolder_names, cluster_labels)]
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Clustering results saved to {output_file}")



# Main function
def main():
    # Check for GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("GPU is available.")
    else:
        print("GPU is not available. Using CPU.")
    # Paths
    main_folder = "cropped_output_frames"  # Replace with the path to your dataset
    output_file = "clustering_results_LD.csv"

    # Load pre-trained ResNet for feature extraction
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Perform clustering
    cluster_subfolders(main_folder, model, n_clusters=2, output_file=output_file)

if __name__ == "__main__":
    main()