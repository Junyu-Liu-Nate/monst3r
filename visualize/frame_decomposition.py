import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    data_dir = '/Users/liujunyu/Desktop/Course/Brown/CSCI2951I/code/data'
    video_name = 'bear'

    video_path = os.path.join(data_dir, 'som_vid', video_name + '.mp4')
    output_folder = os.path.join(data_dir, 'som_vid', video_name)
    extract_frames(video_path, output_folder)