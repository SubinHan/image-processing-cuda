﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION   
#define STB_IMAGE_WRITE_IMPLEMENTATION   
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdint.h>
#include <cmath>

constexpr int COLOR_SIZE = 256;
constexpr int BPP = 3;
constexpr int R = 0;
constexpr int G = 1;
constexpr int B = 2;
constexpr int BLOCK_SIZE_X = 16;
constexpr int BLOCK_SIZE_Y = 16;


uint8_t* get_pixel(uint8_t* const image, const int width, const int height, const int x, const int y, const int bpp)
{
	return &image[y * bpp * width + x * bpp];
}

void count_intensity(uint8_t* const image, const int width, const int height, const int bpp, int* result)
{
	for (int i = 0; i < bpp; i++)
	{
		for (int j = 0; j < COLOR_SIZE; j++)
		{
			*(result +i * COLOR_SIZE + j) = 0;
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t* pixel = get_pixel(image, width, height, j, i, bpp);
			
			(*(result + R * COLOR_SIZE + pixel[R]))++;
			(*(result + G * COLOR_SIZE + pixel[G]))++;
			(*(result + B * COLOR_SIZE + pixel[B]))++;
		}
	}
}

int find_min(int* counts)
{
	for (int i = 0; i < COLOR_SIZE; i++)
	{
		if (counts[i] > 0)
		{
			return i;
		}
	}
}

void calculate_cumulative(int* counts, int* cumulative)
{
	cumulative[0] = counts[0];
	for (int i = 1; i < COLOR_SIZE; i++)
	{
		cumulative[i] = cumulative[i - 1] + counts[i];
	}
}

int get_offset(const int x, const int y, const int width)
{
	return y * width + x;
}

__global__ void transform_gpu0(uint8_t* const image, const int width, const int bpp, uint8_t* const transformer, uint8_t* result)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int logical_tid = ty * width + tx;
	int image_offset = logical_tid * bpp;

	for (int color_offset = 0; color_offset < bpp; color_offset++)
	{
		uint8_t color = *(image + image_offset + color_offset);
		*(result + image_offset + color_offset) = *(transformer + COLOR_SIZE * color_offset + color);
	}
}

void transform_gpu(uint8_t* const image, const int width, const int height, const int bpp, uint8_t* const transformer, uint8_t* result)
{
	dim3 dg(width / BLOCK_SIZE_X + 1, height / BLOCK_SIZE_Y + 1, 1);
	dim3 db(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	
	uint8_t* dev_image;
	uint8_t* dev_transformer;
	uint8_t* dev_result;

	const int image_buffer_size = width * height * bpp * sizeof(uint8_t);
	const int transformer_buffer_size = bpp * COLOR_SIZE * sizeof(uint8_t);

	cudaMalloc((void**)&dev_image, image_buffer_size);
	cudaMalloc((void**)&dev_transformer, transformer_buffer_size);
	cudaMalloc((void**)&dev_result, image_buffer_size);

	cudaMemcpy(dev_image, image, image_buffer_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_transformer, transformer, transformer_buffer_size, cudaMemcpyHostToDevice);

	transform_gpu0 << <dg, db >> > (dev_image, width, bpp, dev_transformer, dev_result);

	cudaMemcpy(result, dev_result, image_buffer_size, cudaMemcpyDeviceToHost);
	cudaFree(dev_image);
	cudaFree(dev_transformer);
	cudaFree(dev_result);
}

void transform(uint8_t* const image, const int width, const int height, const int bpp, uint8_t* const transformer, uint8_t* result)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < bpp; k++)
			{
				int logical_offset = get_offset(j, i, width);
				int image_offset = logical_offset * bpp + k;
				uint8_t color = *(image + image_offset);

				*(result + image_offset) = *(transformer + COLOR_SIZE * k + color);
			}
		}
	}
}

void histogram_equalization(uint8_t* const image, const int width, const int height, const int bpp, uint8_t* result)
{
	int count[BPP][COLOR_SIZE];
	count_intensity(image, width, height, bpp, count[0]);

	uint8_t transformer[BPP][COLOR_SIZE];

	for (int i = 0; i < bpp; i++)
	{
		int min_intensity = find_min(count[i]);
		int count_min_intensity;

		int cumulative[COLOR_SIZE];
		calculate_cumulative(count[i], cumulative);
		count_min_intensity = cumulative[min_intensity];

		for (int j = 0; j < COLOR_SIZE; j++)
		{
			transformer[i][j] = round(((double)(cumulative[j] - count_min_intensity) / (double)(width * height - count_min_intensity) * (double)(COLOR_SIZE - 1)));
		}
	}

	transform_gpu(image, width, height, bpp, transformer[0], result);
}



void print_image(uint8_t* const image, const int width, const int height, const int bpp)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t* pixel = get_pixel(image, width, height, j, i, bpp);
			
			printf("[");
			for (int k = 0; k < bpp; k++)
			{
				printf("%d ", pixel[k]);
			}
			printf("] ");
		}
		printf("\n");
	}
}

int main() {
	int width, height, bpp;
	uint8_t* rgb_image = stbi_load("test/tower_of_pisa.jpg", &width, &height, &bpp, 0);
	uint8_t* output_image = (uint8_t*)malloc(width * height * bpp * sizeof(uint8_t));

	printf("%d %d %d\n", width, height, bpp);

	histogram_equalization(rgb_image, width, height, bpp, output_image);

	stbi_image_free(rgb_image);
	stbi_write_png("output_histogram_equ.png", width, height, bpp, output_image, width * bpp);

	free(output_image);
	return 0;
}

//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    PngImage image("test\\tiny_color.png");
//    Pixel32 pixel = image.getPixelAt(0, 0);
//    printf("%d %d %d %d", pixel.red, pixel.green, pixel.blue, pixel.alpha);
//    pixel.red = 0;
//    pixel.green = 0;
//    pixel.blue = 0;
//    image.setPixelAt(0, 0, pixel);
//    image.write("output2.png");
//
//    PngImage image2(32, 32);
//    image2.setPixelAt(0, 0, pixel);
//    image2.write("output3.png");
//
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}