import os
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.cluster import KMeans
import tensorflow as tf

def load_pretrained_model():
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    print(f"Using device: {device}")
    with tf.device(device):
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    return model

def extract_dnn_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    with tf.device(device):
        features = model.predict(img_array)
    return features.flatten()

def classify_images_with_dnn(base_folder, n_clusters=2):
    model = load_pretrained_model()
    subfolder_features = []
    subfolder_names = []

    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if os.path.isdir(subfolder_path):
            folder_features = []
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image_path = os.path.join(subfolder_path, filename)
                    features = extract_dnn_features(image_path, model)
                    folder_features.append(features)
            if folder_features:
                aggregated_features = np.mean(folder_features, axis=0)
                subfolder_features.append(aggregated_features)
                subfolder_names.append(subfolder)

    if subfolder_features:
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(subfolder_features)
        return subfolder_names, labels
    return [], []

def save_results_to_csv(folder_names, labels, output_file):
    results_df = pd.DataFrame({'Folder': folder_names, 'Cluster_Label': labels})
    results_df.to_csv(output_file, index=False)

base_folder = r'cropped_output_frames'
output_file = 'DNN_classification_results.csv'

folder_names, labels = classify_images_with_dnn(base_folder)
save_results_to_csv(folder_names, labels, output_file)
