import os                      # For file and folder operations
import numpy as np            # For numerical and array operations
from PIL import Image         # For image loading and saving
import time                   # For measuring execution time

# Input and output folders
video_folder = "frames"       # Folder containing input frames
out_folder = "output"     # Folder to save processed frames

# Create output folder if it does not exist
os.makedirs(out_folder, exist_ok=True)

# Load background image and convert to RGB format
bg_image_file = "background.png"
bg_img = Image.open(bg_image_file).convert("RGB")

# Function to load all frames from a folder into a NumPy array
def load_frames(folder):
    # Get all PNG files and sort them (ensures correct frame order)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
    
    frames = []
    for f in files:
        # Open image, convert to RGB, and store as NumPy array
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        frames.append(np.array(img))
    
    # Convert list of frames into a single NumPy array
    return np.array(frames)

# Load video frames
video_frames = load_frames(video_folder)

# Get number of frames and frame dimensions
num_frames, height, width, _ = video_frames.shape

# Resize background image to match frame size
bg_img = bg_img.resize((width, height))
bg_array = np.array(bg_img)

# --- Improved green detection mask ---
# Extract individual color channels
r = video_frames[..., 0]
g = video_frames[..., 1]
b = video_frames[..., 2]

# Create mask where green is dominant
# Conditions:
# 1. Green value is above threshold (100)
# 2. Green is at least 10% greater than red
# 3. Green is at least 10% greater than blue
mask = (g > 100) & (g > r * 1.1) & (g > b * 1.1)

# Expand background to match number of frames
# This creates a copy of background for each frame
bg_broadcast = np.tile(bg_array, (num_frames, 1, 1, 1))

# Apply mask:
# If pixel is green → replace with background
# Else → keep original frame pixel
output_frames = np.where(mask[..., None], bg_broadcast, video_frames)

# List to store CPU time per frame
frame_times = []

# Open file to store timing results
with open("cpu_time.txt", "w") as fp:
    # Process each frame individually for saving and timing
    for z in range(num_frames):
        
        # Start timing for current frame
        start = time.time()
        
        # Convert NumPy array back to image format
        img = Image.fromarray(output_frames[z])
        
        # Save processed frame
        img.save(os.path.join(out_folder, f"frame_{z:04d}.png"))
        
        # End timing
        end = time.time()
        
        # Compute time in milliseconds
        frame_time = (end - start) * 1000
        
        # Store and log time
        frame_times.append(frame_time)
        fp.write(f"{frame_time:.2f}\n")
        
        # Print progress
        print(f"Processed frame {z+1}/{num_frames} (CPU time: {frame_time:.2f} ms)")

    # Calculate total CPU time for all frames
    total_cpu_time = sum(frame_times)
    print(f"Total CPU Time: {total_cpu_time:.2f} ms")

    # Optionally write total time to file (currently commented)
    # fp.write(f"{total_cpu_time:.2f}\n")