import cv2
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def load_frames_from_directory(directory):
    """Load frames from a directory, assuming filenames are ordered."""
    frame_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.png'))])
    frames = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in frame_files]  # Load frames as grayscale
    return frames

def extract_features(frames):
    """Extract simple features (flattened pixel intensities) from frames."""
    return [frame.flatten() for frame in frames]

def compute_spearman_correlation(features):
    """Compute Spearman correlations for consecutive frames."""
    correlations = []
    for i in range(len(features) - 1):
        rho, _ = spearmanr(features[i], features[i + 1])  # Spearman correlation between two consecutive frames
        correlations.append(rho)
    return correlations

def classify_video(correlations, threshold=0.7):
    """Classify the video based on average correlation."""
    avg_correlation = np.mean(correlations)
    classification = "Real" if avg_correlation > threshold else "Deepfake"
    return classification, avg_correlation

def main(directory):
    """Main function to process frames in subdirectories and classify videos."""
    results = []
    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing frames from: {subfolder_path}")
            frames = load_frames_from_directory(subfolder_path)
            if len(frames) < 2:
                print(f"Not enough frames in {subfolder} to compute correlation.")
                results.append((subfolder, "Not enough frames", 0 if subfolder.count("id") == 2 else 1))
                continue
            
            features = extract_features(frames)
            correlations = compute_spearman_correlation(features)
            classification, avg_correlation = classify_video(correlations)
            
            print(f"Video Classification for {subfolder}: {classification}")
            print(f"Average Inter-Frame Correlation: {avg_correlation:.4f}")
            results.append((subfolder, classification, avg_correlation, 0 if subfolder.count("id") == 2 else 1))

    df = pd.DataFrame(results, columns=['Subfolder', 'Classification', 'Average Correlation', 'Label'])
    df.to_csv('video_classification_results.csv', index=False)

# Example usage
# video_frames_directory = "path_to_directory_with_frames"  # Replace with your frames directory
# video_frames_directory = "./frames_function_cross_test_data"  # Replace with your frames directory
video_frames_directory = "./cropped_frames"  # Replace with your frames directory
main(video_frames_directory)
