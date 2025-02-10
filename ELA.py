import os
import pandas as pd
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import numpy as np
from skimage import feature
from sklearn.cluster import KMeans

count = 1

def convert_to_ela_image(filename, quality):
    global count
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    # Calculate the ELA image
    ela_im = ImageChops.difference(im, resaved_im)
    
    # Normalize the ELA image to enhance differences
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    # Convert ELA image to grayscale for better feature extraction
    ela_im_gray = ela_im.convert('L')

    # Optionally, apply Gaussian blur to reduce noise and enhance feature extraction
    ela_im_blurred = ela_im_gray.filter(ImageFilter.GaussianBlur(radius=1))

    if count % 500 == 0:
        ela_im_blurred.show()
        count = 1
    else:
        count += 1

    return ela_im_blurred

def apply_threshold(image, threshold=30):
    gray_image = image.convert('L')
    binary_image = gray_image.point(lambda p: p > threshold and 255)
    return binary_image

def extract_features(image):
    image_array = np.array(image)
    features = feature.hog(image_array, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualize=False)
    return features

def classify_subfolders(base_folder, n_clusters=2):
    subfolder_features = []
    subfolder_names = []
    global count
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            folder_features = []
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(subfolder_path, filename)
                    ela_image = convert_to_ela_image(image_path, 96)
                    thresholded_image = apply_threshold(ela_image)
                    # if count % 500 == 0:
                    #     thresholded_image.show()
                    features = extract_features(thresholded_image)
                    folder_features.append(features)
            if folder_features:
                aggregated_features = np.mean(folder_features, axis=0)
                subfolder_features.append(aggregated_features)
                subfolder_names.append(subfolder)
    
    if subfolder_features:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(subfolder_features)
        return subfolder_names, labels
    return [], []
# def classify_subfolders(base_folder, n_clusters=2):
#     subfolder_features = []
#     subfolder_names = []
    
#     for subfolder in os.listdir(base_folder):
#         subfolder_path = os.path.join(base_folder, subfolder)
#         if os.path.isdir(subfolder_path):
#             folder_features = []
#             for filename in os.listdir(subfolder_path):
#                 if filename.endswith('.jpg') or filename.endswith('.png'):
#                     image_path = os.path.join(subfolder_path, filename)
#                     ela_image = convert_to_ela_image(image_path, 96)
#                     thresholded_image = apply_threshold(ela_image)
#                     features = extract_features(thresholded_image)

#                     # Check the shape of the features
#                     if features is not None and features.size > 0:
#                         print(f"Extracted features from {filename}: {features.shape}")
#                         folder_features.append(features)
#                     else:
#                         print(f"No features extracted from {filename}. Skipping.")

#             if folder_features:
#                 aggregated_features = np.mean(folder_features, axis=0)
#                 subfolder_features.append(aggregated_features)
#                 subfolder_names.append(subfolder)
    
#     if subfolder_features:
#         subfolder_features = np.array(subfolder_features)  # Ensure it's a NumPy array
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = kmeans.fit_predict(subfolder_features)
#         return subfolder_names, labels
#     return [], []


def save_results_to_csv(folder_names, labels, output_file):
    results_df = pd.DataFrame({'Folder': folder_names, 'Cluster_Label': labels})
    results_df.to_csv(output_file, index=False)

# Sample usage
base_folder = r'cropped_output_frames'
output_file = 'ELA_classification_results.csv'

folder_names, labels = classify_subfolders(base_folder)
save_results_to_csv(folder_names, labels, output_file)



## OLD VERSION

# import os
# import pandas as pd
# from PIL import Image, ImageChops, ImageEnhance
# import numpy as np
# from skimage import feature

# def convert_to_ela_image(filename, quality):
#     resaved_filename = filename.split('.')[0] + '.resaved.jpg'
#     im = Image.open(filename).convert('RGB')
#     im.save(resaved_filename, 'JPEG', quality=quality)
#     resaved_im = Image.open(resaved_filename)
    
#     ela_im = ImageChops.difference(im, resaved_im)
    
#     extrema = ela_im.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
#     if max_diff == 0:
#         max_diff = 1
#     scale = 255.0 / max_diff
    
#     ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

#     # ela_im.show()  # Visualize the ELA image
    
#     return ela_im

# def apply_threshold(image, threshold=30):
#     gray_image = image.convert('L')
#     binary_image = gray_image.point(lambda p: p > threshold and 255)
#     return binary_image

# def extract_features(image):
#     image_array = np.array(image)
#     features = feature.hog(image_array, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
#     return features

# def classify_folder(folder_path):
#     features_list = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.jpg') or filename.endswith('.png'):
#             image_path = os.path.join(folder_path, filename)
#             ela_image = convert_to_ela_image(image_path, 96)
#             thresholded_image = apply_threshold(ela_image)
#             features = extract_features(thresholded_image)
#             features_list.append(features)
    
#     if features_list:
#         avg_features = np.mean(features_list, axis=0)
#         return features_list, avg_features
#     return [], []

# def process_folders(base_folder):
#     results = []
#     all_features = []
    
#     for subfolder in os.listdir(base_folder):
#         subfolder_path = os.path.join(base_folder, subfolder)
#         if os.path.isdir(subfolder_path):
#             features_list, avg_features = classify_folder(subfolder_path)
#             for features in features_list:
#                 all_features.append({'Folder': subfolder, 'Features': features.tolist()})
#             if avg_features.size > 0:
#                 results.append({'Folder': subfolder, 'Avg_Features': avg_features.tolist()})
    
#     return results, all_features

# def save_results_to_csv(results, all_features, output_file):
#     avg_df = pd.DataFrame(results)
#     avg_df.to_csv(f'avg_{output_file}', index=False)
    
#     features_df = pd.DataFrame(all_features)
#     features_df.to_csv(f'all_features_{output_file}', index=False)

# # Sample usage
# # base_folder = r'cropped_frames'  # Update this to your base folder
# base_folder = r'cropped_output_frames'  # Update this to your base folder
# output_file = 'ELA_classification_results.csv'

# results, all_features = process_folders(base_folder)
# save_results_to_csv(results, all_features, output_file)