import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from matplotlib import pyplot as plt
from pycuda.compiler import SourceModule
import time

# Load the image
image = cv2.imread("test(1024x1024).png", cv2.IMREAD_GRAYSCALE)

# Copy the image data to the GPU
image_gpu = cuda.mem_alloc(image.nbytes)
cuda.memcpy_htod(image_gpu, image)

# Create an output array on the GPU
output_gpu = cuda.mem_alloc(image.nbytes)


gaussian_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
gaussian_kernel /= np.sum(gaussian_kernel)


cuda_kernel_code = """
__device__ const float M_PI = 3.14159265358979323846;

__device__ int get_f_value(int ai){
    return ai;
}

__device__ double get_g_value(double x, double y, double sigma){
    double exponent = exp(-((x*x) + (y*y)) / (2.0 * sigma * sigma));
    double result = exponent / (2.0 * M_PI * sigma * sigma);
    return result;
}

__device__ double get_r_value(int ai, double a0, double sigma){
    double term = (get_f_value(ai) - get_f_value(a0)) * (get_f_value(ai) - get_f_value(a0));

    if (term <-50)
        term = -50;
    if (term>50)
        term = 50;
    
    double log_exponent = -term / (2 * sigma * sigma);
    double log_result = log_exponent - 0.5 * log(2 * M_PI * sigma * sigma);
    double result = exp(log_result);

    return result;
}

__device__ double get_h_value(unsigned char* original_image, int x, int y, int width, int height, int filter_size, int sigma){
    int delta = filter_size / 2;
    int x_start = x - delta;
    int x_end = x + delta;
    int y_start = y - delta;
    int y_end = y + delta;

    double k = 0;
    double summ = 0;

    int main_pixel_index = y * width + x;
    int main_pixel_brightness = original_image[main_pixel_index];
    for (int y = y_start; y <= y_end; ++y) {
        for (int x = x_start; x <= x_end; ++x) {
            int current_pixel_brightness;
            if (0 <=x && x < width && 0 <= y && y < height) {
                int index = y * width + x;
                current_pixel_brightness = original_image[index];
            }
            else {
                current_pixel_brightness = 0;
            }

            double f = get_f_value(current_pixel_brightness);
            double g = get_g_value(abs(x - x_start), abs(y - y_start), sigma);
            double r = get_r_value(current_pixel_brightness, main_pixel_brightness, sigma);

            k += g * r;
            summ += f * g * r;
        }
    }

    if (k == 0)
        return original_image[main_pixel_index];
    else
        return summ / k;
}

__global__ void gaussian_filter(unsigned char* original_image, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int output_index = y * width + x;
        double new_brightness = get_h_value(original_image, x, y, width, height, 7, 10);
        output[output_index] = static_cast<unsigned char>(new_brightness);
    }
}
"""

# Compile the CUDA kernel
module = SourceModule(cuda_kernel_code)
gaussian_filter_kernel = module.get_function("gaussian_filter")

# Define block and grid sizes
block_size = (32, 32)
grid_size = (
    (image.shape[1] + block_size[0] - 1) // block_size[0],
    (image.shape[0] + block_size[1] - 1) // block_size[1],
)

# Measure the execution time
start_time = time.time()

# Launch the CUDA kernel to apply Gaussian filtering
gaussian_filter_kernel(
    image_gpu,
    output_gpu,
    np.int32(image.shape[1]),
    np.int32(image.shape[0]),
    block=(32, 32, 1),
    grid=grid_size,
)

# Synchronize GPU
cuda.Context.synchronize()

# Time measurement
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Время вычисления на GPU: {elapsed_time:.5f} секунд")

# Copying the result back to the CPU
output_image = np.empty_like(image)
cuda.memcpy_dtoh(output_image, output_gpu)

# Displaying output images
plt.imshow(output_image, cmap="gray")
plt.show()
