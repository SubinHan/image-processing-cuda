#include "BrightnessOld.cuh"
#include <algorithm>

void change_brightness_cpu(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp,
    float percent)
{
    for (int p = 0; p < width * height; p++) {
        for (int channel = 0; channel < bpp; channel++) {
            const int offset = p * bpp;

            float pixel;
            pixel = image[offset + channel];
            pixel = (pixel * percent) / 100.0f;

            if (pixel > 255.0f) 
                pixel = 255.0f;

            if (pixel < 0.0f) 
                pixel = 0.0;

            image[offset + channel] = (uint8_t)pixel;
        }
    }
}

void change_brightness_gpu(
    uint8_t* image,
    const int width,
    const int height,
    const int bpp,
    float percent
)
{
    const int buffer_size = sizeof(uint8_t) * width * height * bpp;
    const int image_size = width * height;

    uint8_t* d_image;
    cudaMalloc((void**)&d_image, buffer_size);
    cudaMemcpy(d_image, image, buffer_size, cudaMemcpyHostToDevice);

    int blocksx = ceil((image_size) / 256.0f);
    dim3 threads(256);
    dim3 grid(blocksx);

    change_brightness_gpu0 << <grid, threads >> > (d_image, width, height, bpp, percent);

    cudaMemcpy(image, d_image, buffer_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
}

__global__ void change_brightness_gpu0(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp,
    float percent)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = gridDim.x * blockDim.x * ty + tx;

    for (int channel = 0; channel < bpp; channel++) {
        const int offset = tid * bpp;

        float pixel;
        pixel = image[offset + channel];
        pixel = (pixel * percent) / 100.0f;

        if (pixel > 255.0)
            pixel = 255.0f;

        if (pixel < 0.0f)
            pixel = 0.0f;

        image[offset + channel] = (uint8_t)pixel;
    }
}
