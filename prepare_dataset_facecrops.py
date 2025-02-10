import os
import cv2
import dlib
import sys
from lib.vaf_util import get_crops_landmarks
from PIL import Image
from torchvision import transforms

def load_face_detector(face_detector_path):
    if not os.path.isfile(face_detector_path):
        print("Could not find shape_predictor_68_face_landmarks.dat")
        sys.exit()
    face_detector = dlib.get_frontal_face_detector()
    sp68 = dlib.shape_predictor(face_detector_path)
    return face_detector, sp68

def process_subfolder(subfolder_path, face_detector, sp68, transform, output_folder):
    frame_files = sorted(os.listdir(subfolder_path))
    for frame_file in frame_files:
        frame_path = os.path.join(subfolder_path, frame_file)
        if not frame_path.endswith('.jpg'):
            print(f"Invalid file type: {frame_path}. Skipping.")
            continue

        img = cv2.imread(frame_path)
        if img is None:
            print(f"Could not open image file: {frame_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_crops, _ = get_crops_landmarks(face_detector, sp68, img)

        if len(face_crops) == 0:
            print(f"No face detected in {frame_path}. Skipping subfolder {subfolder_path}.")
            return False

        face_crop = Image.fromarray(face_crops[0])
        if transform:
            face_crop = transform(face_crop)
            face_crop = transforms.ToPILImage()(face_crop)  # Convert back to PIL Image

        output_subfolder = os.path.join(output_folder, os.path.basename(subfolder_path))
        os.makedirs(output_subfolder, exist_ok=True)
        output_path = os.path.join(output_subfolder, frame_file)
        face_crop.save(output_path)
    return True

def process_main_folder(main_folder, face_detector_path, output_folder, transform=None):
    face_detector, sp68 = load_face_detector(face_detector_path)
    subfolders = sorted(os.listdir(main_folder))
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            if not process_subfolder(subfolder_path, face_detector, sp68, transform, output_folder):
                print(f"Skipping entire subfolder: {subfolder}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python prepare_dataset_facecrops.py <main_folder> <face_detector_path> <output_folder>")
        sys.exit(1)
    main_folder = sys.argv[1]
    face_detector_path = sys.argv[2]
    output_folder = sys.argv[3]
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    process_main_folder(main_folder, face_detector_path, output_folder, transform)

if __name__ == "__main__":
    main()