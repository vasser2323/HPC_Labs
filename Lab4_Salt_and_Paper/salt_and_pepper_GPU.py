import os
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# CUDA kernel code
kernel_code = """
texture<unsigned char, 2, cudaReadModeElementType> tex;

__global__ void median_filter(unsigned char* input, unsigned char* output, int width, int height) {
    int main_pixel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int main_pixel_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (main_pixel_x < width && main_pixel_y < height) {
        int pixels[9];
        int texturePixels[9];
        int pixel_index = 0;

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                texturePixels[pixel_index] = tex2D(tex, main_pixel_x + j, main_pixel_y + i);
                pixels[pixel_index++] = input[width * (main_pixel_y + i) + (main_pixel_x + j)];
            }
        }

        for (int i = 0; i < 9 - 1; ++i) {
            for (int j = 0; j < 9 - i - 1; ++j) {
                if (pixels[j] > pixels[j + 1]) {
                    int temp = pixels[j];
                    pixels[j] = pixels[j + 1];
                    pixels[j + 1] = temp;
                }
            }
        }

        output[main_pixel_y * width + main_pixel_x] = pixels[4];
    }
}
"""

def get_filtered_image(original_image):
    height, width = original_image.shape

    module = SourceModule(kernel_code)
    median_filter_func = module.get_function("median_filter")

    original_image_gpu = cuda.mem_alloc(original_image.nbytes)
    cuda.memcpy_htod(original_image_gpu, original_image)

    texref = module.get_texref("tex")
    cuda.matrix_to_texref(original_image.astype(np.uint8), texref, order="C")

    output_image_gpu = cuda.mem_alloc(original_image.nbytes)

    block = (32, 32, 1)
    grid = ((width - 1) // block[0] + 1, (height - 1) // block[1] + 1, 1)

    median_filter_func(
        original_image_gpu,
        output_image_gpu,
        np.int32(width),
        np.int32(height),
        block=block,
        grid=grid,
    )

    output_image = np.empty_like(original_image)
    cuda.memcpy_dtoh(output_image, output_image_gpu)

    return output_image


image_path = "test(1024x683).png"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

start_time = time.time()
blurred_image = get_filtered_image(original_image)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for GPU processing: {elapsed_time:.4f} seconds")

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap="gray")
plt.title("Original image")

plt.subplot(1, 2, 2)
plt.imshow(blurred_image, cmap="gray")
plt.title("Blurred image (GPU)")

plt.show()