import cv2              # OpenCV library for image processing
import numpy as np      # NumPy for array operations
import time             # Used to measure execution time

# Read the input image (foreground image)
img = cv2.imread("dog.png")

# Read the background image
bg = cv2.imread("background.png")

# Resize background image to match input image size
# img.shape gives (height, width, channels)
bg = cv2.resize(bg, (img.shape[1], img.shape[0]))

# Start timing (to measure CPU execution time)
start = time.time()

# Create an output image with same size and type as input image
# Initially filled with zeros (black image)
output = np.zeros_like(img)

# Extract height and width of image
# '_' is used to ignore number of channels
height, width, _ = img.shape

# Loop through each pixel in the image
for y in range(height):
    for x in range(width):
        
        # Get Blue, Green, Red values of pixel
        # OpenCV uses BGR format (not RGB)
        b, g, r = img[y, x]

        # Convert pixel values to integers (for safe comparison)
        b = int(b)
        g = int(g)
        r = int(r)

        # Check if pixel is "green enough"
        # Condition:
        # 1. Green is significantly higher than Red
        # 2. Green is significantly higher than Blue
        # 3. Green value is above threshold (to avoid dark pixels)
        if g > r + 30 and g > b + 30 and g > 100:
            
            # Replace green pixel with corresponding background pixel
            output[y, x] = bg[y, x]
        
        else:
            # Keep original pixel if not green
            output[y, x] = img[y, x]

# Save the processed image to file
cv2.imwrite("output_cpu.png", output)

# End timing
end = time.time()

# Calculate CPU execution time in milliseconds
cpu_time = (end - start) * 1000

# Print execution time
print("CPU Time:", cpu_time, "ms")

# Save execution time to a text file
with open("cpu_time.txt", "w") as f:
    f.write(str(cpu_time))