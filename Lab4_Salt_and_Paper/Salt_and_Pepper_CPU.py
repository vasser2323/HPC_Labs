import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def get_filtered_image(original_image):
    height, width = original_image.shape
    print(f"Processing image of size {width}x{height}")

    filtered_image = original_image.copy()

    for y in range(height):
        if y % 10 == 0:
            print(f"Processing row {y+1}/{height}")

        for x in range(width):
            delta = 1
            x_start, x_end = x - delta, x + delta
            y_start, y_end = y - delta, y + delta

            pixels_brightness = []
            for current_x in range(x_start, x_end + 1):
                for current_y in range(y_start, y_end + 1):
                    try:
                        pixels_brightness += [filtered_image[current_y, current_x]]
                    except IndexError as e:
                        pixels_brightness += [filtered_image[y, x]]

            for i in range(n := len(pixels_brightness)):
                for j in range(0, n - i - 1):
                    if pixels_brightness[j] > pixels_brightness[j + 1]:
                        pixels_brightness[j], pixels_brightness[j + 1] = (
                            pixels_brightness[j + 1],
                            pixels_brightness[j],
                        )

            average_index = len(pixels_brightness) // 2
            new_brightness = pixels_brightness[average_index]
            filtered_image[y, x] = new_brightness

    return filtered_image


image_path = os.path.join(os.path.dirname(__file__), "test(1024x683).png")
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

height, width = original_image.shape
print("--------------------------------")
print(f"Processing image of size {height}x{width}\n")

start_time = time.time()

# Processing image with CPU
blurred_image = get_filtered_image(original_image)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for CPU processing: {elapsed_time:.4f}")

# Show original image
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap="gray")
plt.title("Original image")

# Show blurred image
plt.subplot(1, 2, 2)
plt.imshow(blurred_image, cmap="gray")
plt.title(f"Blurred image (CPU)")

plt.show()
