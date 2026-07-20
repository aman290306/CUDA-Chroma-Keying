#include <stdio.h>              // Standard input/output functions
#include <stdlib.h>             // Memory allocation functions
#include <cuda_runtime.h>       // CUDA runtime API

// stb libraries for image loading and saving
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel: each thread processes one pixel
__global__ void greenscreen_kernel(unsigned char *img, unsigned char *bg, unsigned char *out, int width, int height) {
    
    // Compute pixel coordinates handled by this thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check (avoid threads outside image)
    if(x >= width || y >= height) return;

    // Compute index in 1D array (3 values per pixel: R, G, B)
    int idx = (y * width + x) * 3;

    // Extract RGB values
    int r = img[idx + 0];
    int g = img[idx + 1];
    int b = img[idx + 2];

    // Green screen condition:
    // If green channel is significantly higher than red and blue
    if(g > r + 30 && g > b + 30 && g > 100) {
        
        // Replace pixel with background pixel
        out[idx + 0] = bg[idx + 0];
        out[idx + 1] = bg[idx + 1];
        out[idx + 2] = bg[idx + 2];
    
    } else {
        
        // Keep original pixel
        out[idx + 0] = r;
        out[idx + 1] = g;
        out[idx + 2] = b;
    }
}

int main() {
    // Input and output file names
    const char *input_file = "dog.png";
    const char *bg_file = "background.png";
    const char *output_file = "output_gpu.png";

    int width, height, channels;

    // CUDA events for measuring GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Load input image (force 3 channels RGB)
    unsigned char *img = stbi_load(input_file, &width, &height, &channels, 3);
    if(!img) {
        printf("Failed to load %s\n", input_file);
        return 1;
    }

    // Load background image
    int bg_width, bg_height, bg_channels;
    unsigned char *bg = stbi_load(bg_file, &bg_width, &bg_height, &bg_channels, 3);
    if(!bg) {
        printf("Failed to load %s\n", bg_file);
        stbi_image_free(img);
        return 1;
    }

    // Resize background if dimensions do not match input image
    if(bg_width != width || bg_height != height) {
        
        // Allocate memory for resized background
        unsigned char *bg_resized = (unsigned char*)malloc(width * height * 3);

        // Simple nearest-neighbor resizing
        for(int y = 0; y < height; y++){
            for(int x = 0; x < width; x++){
                
                int src_x = x * bg_width / width;
                int src_y = y * bg_height / height;

                int dst_idx = (y * width + x) * 3;
                int src_idx = (src_y * bg_width + src_x) * 3;

                bg_resized[dst_idx + 0] = bg[src_idx + 0];
                bg_resized[dst_idx + 1] = bg[src_idx + 1];
                bg_resized[dst_idx + 2] = bg[src_idx + 2];
            }
        }

        // Free old background and replace with resized version
        stbi_image_free(bg);
        bg = bg_resized;
    }

    // Allocate memory on GPU (device)
    unsigned char *d_img, *d_bg, *d_out;
    cudaMalloc(&d_img, width * height * 3);
    cudaMalloc(&d_bg, width * height * 3);
    cudaMalloc(&d_out, width * height * 3);

    // Copy data from CPU (host) to GPU (device)
    cudaMemcpy(d_img, img, width * height * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bg, bg, width * height * 3, cudaMemcpyHostToDevice);

    // Define CUDA execution configuration
    dim3 block(16, 16);  // 16x16 threads per block
    dim3 grid((width + 15) / 16, (height + 15) / 16); // number of blocks

    // Start timing
    cudaEventRecord(start, 0);

    // Launch kernel
    greenscreen_kernel<<<grid, block>>>(d_img, d_bg, d_out, width, height);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute elapsed time in milliseconds
    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Allocate memory for output on CPU
    unsigned char *output = (unsigned char*)malloc(width * height * 3);

    // Copy result from GPU to CPU
    cudaMemcpy(output, d_out, width * height * 3, cudaMemcpyDeviceToHost);

    // Save output image
    stbi_write_png(output_file, width, height, 3, output, width * 3);

    printf("Output written to %s\n", output_file);
    printf("GPU Time: %f ms\n", gpuTime);

    // Write GPU time to file
    FILE *f = fopen("gpu_time.txt", "w");
    if(f) {
        fprintf(f, "%f", gpuTime);
        fclose(f);
    }

    // Free memory (CPU + GPU)
    stbi_image_free(img);
    free(bg);
    free(output);
    cudaFree(d_img);
    cudaFree(d_bg);
    cudaFree(d_out);

    return 0;
}