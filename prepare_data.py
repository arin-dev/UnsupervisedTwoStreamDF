import os
import cv2
import random
import sys

def extract_frames_from_videos(video_folder, output_folder):
    
    print("WORKING ! ")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # It yields a tuple (root, dirs, files) for each directory in the tree, where:
    # - root: the current directory path
    # - dirs: a list of subdirectories in the current directory
    # - files: a list of files in the current directory
    for root, _, files in os.walk(video_folder):
        for video_name in files:
            video_path = os.path.join(root, video_name)
            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():  # Check if the video was opened successfully
                print(f"Error opening video file: {video_path}")
                continue
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            available_starts = range(max(0, frame_count - 13))

            # Randomly select 4 starting points for the sets of 12 frames
            selected_starts = random.sample(available_starts, min(4, len(available_starts)))

            for start in selected_starts:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                frames = []
                for j in range(12):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                if len(frames) == 12:
                    output_video_folder = os.path.join(output_folder, f"{os.path.basename(video_name)}_{start}")
                    os.makedirs(output_video_folder, exist_ok=True)
                    for k, frame in enumerate(frames):
                        cv2.imwrite(os.path.join(output_video_folder, f"frame_{k}.jpg"), frame)
            cap.release()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python prepare_data.py <video_folder> <output_folder>")
        sys.exit(1)
    video_folder = sys.argv[1]
    output_folder = sys.argv[2]
    extract_frames_from_videos(video_folder, output_folder)

# To use this script from the terminal, run:
# python prepare_data.py <video_folder> <output_folder>
# python prepare_data.py ./function_test_data/ ./frames_function_test_data/