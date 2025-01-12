import os
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from data_loader import get_data_loaders
import torch

def calculate_difference(image1, image2):
    # print(type(image1), image1, image2)
    # kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Simple Sobel filter for edge detection
    # kernel = np.array([[2, 2, 2], [2, -1, -1], [2, -1, -2]])  # Simple Sobel filter for edge detection
    kernel = np.array([[1,1], [1,-5]])  # Simple Sobel filter for edge detection
    # diff1 = convolve(image1.astype(np.float32), kernel)
    # diff2 = convolve(image2.astype(np.float32), kernel)
    diff1 = convolve(image1.cpu().numpy(), kernel)
    diff2 = convolve(image2.cpu().numpy(), kernel)
    differences = np.abs(diff1 - diff2)
    sorted_differences = np.sort(differences, axis=None)
    return np.mean(sorted_differences[-20:])

def calculate_correlation(image1, image2):
    return np.corrcoef(image1.flatten(), image2.flatten())[0, 1]

def process_images_in_directory(directory, batch_size=1):
    results = []
    import json

    with open('labels_flipped_titles.json', 'r') as f:
        label_map = json.load(f)

    data_loader = get_data_loaders(directory, 'shape_predictor_68_face_landmarks.dat', batch_size)
    
    for data, video_names in data_loader:
        if data is None:
            continue
        
        # images = data.cpu().numpy()  # Convert to numpy and change shape
        images = data[0]
        
        # print(type(images), len(images))
        # print( type( images[0].numpy() ) )
        # print(data.shape)
        # print(images.shape)
        
        differences = [
            calculate_difference( images[i][0], images[i + 1][0] ) for i in range(len(images) - 1)
        ]

        # print("PASSED DIFFERENCES!!")
        # print("DIFFERENCES", differences)
        difference = np.max(differences)
        label = label_map.get(video_names[0], 'unknown')  # Default to 'unknown' if not found
        results.append([video_names[0], label, difference])
        # break
    df = pd.DataFrame(results, columns=['video_name', 'label', 'difference'])
    df.to_csv('correlation_results.csv', index=False)

def main():
    directory = './frames_function_cross_test_data'  # Update this path
    process_images_in_directory(directory)

if __name__ == "__main__":
    main()