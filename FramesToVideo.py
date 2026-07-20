import cv2
import os

# Folders and output
frames_folder = "output"
output_video_file = "output_video.mp4"
fps = 24  # frames per second

# Get sorted frame files
frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(".png")])
if not frame_files:
    print("No frames found in", frames_folder)
    exit()

# Read the first frame to get size
first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
height, width, channels = first_frame.shape

# Create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

# Write all frames
for idx, f in enumerate(frame_files):
    frame = cv2.imread(os.path.join(frames_folder, f))
    out.write(frame)
    print(f"Added frame {idx+1}/{len(frame_files)}")

out.release()
print(f"Video saved as {output_video_file}")