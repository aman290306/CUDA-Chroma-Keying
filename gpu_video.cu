#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <direct.h>    // for _mkdir on Windows
#include <errno.h>     // for checking errno

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel
__global__ void greenscreen_kernel(unsigned char *frame, unsigned char *bg, unsigned char *out, int w, int h)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= w || y >= h) return;

    int idx = (y*w + x)*3;

    unsigned char r = frame[idx+0];
    unsigned char g = frame[idx+1];
    unsigned char b = frame[idx+2];

    if (g > 150 && g > r + 50 && g > b + 50) {
        out[idx+0] = bg[idx+0];
        out[idx+1] = bg[idx+1];
        out[idx+2] = bg[idx+2];
    } else {
        out[idx+0] = r;
        out[idx+1] = g;
        out[idx+2] = b;
    }
}

// Simple nearest-neighbor resize of background to frame size
unsigned char* resize_bg(unsigned char *bg, int bg_w, int bg_h, int frame_w, int frame_h)
{
    unsigned char *resized = (unsigned char*)malloc(frame_w*frame_h*3);
    for(int y=0; y<frame_h; y++) {
        int src_y = y * bg_h / frame_h;
        for(int x=0; x<frame_w; x++) {
            int src_x = x * bg_w / frame_w;
            for(int c=0; c<3; c++) {
                resized[(y*frame_w + x)*3 + c] = bg[(src_y*bg_w + src_x)*3 + c];
            }
        }
    }
    return resized;
}

int main() {
    // Create output folder
    if(_mkdir("out_frames") != 0 && errno != EEXIST) {
        printf("Failed to create output folder\n");
        return -1;
    }
    FILE *fp = fopen("gpu_time.txt", "w");


    // Load background
    int bg_w, bg_h, bg_c;
    unsigned char *h_bg = stbi_load("background.png", &bg_w, &bg_h, &bg_c, 3);
    if(!h_bg) { printf("Failed to load background\n"); return -1; }

    // Count frames
    int total_frames = 0;
    char fname[128];
    while(1) {
        sprintf(fname, "frames/frame_%04d.png", total_frames+1);
        FILE *fcheck = fopen(fname, "rb");
        if(!fcheck) break;
        fclose(fcheck);
        total_frames++;
    }
    if(total_frames==0) { printf("No frames found\n"); return -1; }
    printf("Total frames: %d\n", total_frames);

    // CUDA event variables for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalGPUTime = 0.0f;

    for(int f=0; f<total_frames; f++) {
        sprintf(fname, "frames/frame_%04d.png", f+1);
        int w,h,c;
        unsigned char *h_frame = stbi_load(fname, &w, &h, &c, 3);
        if(!h_frame) { printf("Failed to load frame %d\n", f+1); continue; }

        // Resize background to match this frame
        unsigned char *h_bg_resized = resize_bg(h_bg, bg_w, bg_h, w, h);

        size_t img_size = w*h*3;
        unsigned char *d_frame, *d_bg, *d_out;
        cudaMalloc(&d_frame, img_size);
        cudaMalloc(&d_bg, img_size);
        cudaMalloc(&d_out, img_size);

        cudaMemcpy(d_frame, h_frame, img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bg, h_bg_resized, img_size, cudaMemcpyHostToDevice);

        dim3 block(16,16);
        dim3 grid((w+15)/16, (h+15)/16);

        // Start GPU timer
        cudaEventRecord(start);

        greenscreen_kernel<<<grid, block>>>(d_frame, d_bg, d_out, w, h);
        cudaDeviceSynchronize();

        // Stop GPU timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalGPUTime += milliseconds;

        unsigned char *h_out = (unsigned char*)malloc(img_size);
        cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost);

        sprintf(fname, "output/out_frame%04d.png", f+1);
        stbi_write_png(fname, w, h, 3, h_out, w*3);

        printf("Frame %d processed (GPU time: %.3f ms)\n", f+1, milliseconds);
        fprintf(fp, "%f\n", milliseconds);
        free(h_frame);
        free(h_bg_resized);
        free(h_out);
        cudaFree(d_frame);
        cudaFree(d_bg);
        cudaFree(d_out);
    }

    free(h_bg);
    printf("All frames done! Total GPU time: %.3f ms\n", totalGPUTime);
   // fprintf(fp,"%f\n",totalGPUTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}