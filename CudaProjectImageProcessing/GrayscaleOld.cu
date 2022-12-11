#include "GrayscaleOld.cuh"
#include <algorithm>

void convert_grayscale_cpu(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp)
{
	for (int p = 0; p < width * height; p++) {
		int offset = p * bpp;
		double gray = 
			0.2126 * image[offset + 0] 
			+ 0.7152 * image[offset + 1] 
			+ 0.0722 * image[offset + 2];

		image[offset + 0] = (uint8_t)gray;
		image[offset + 1] = (uint8_t)gray;
		image[offset + 2] = (uint8_t)gray;
	}
}

void convert_grayscale_gpu(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp)
{
	const int buffer_size = sizeof(uint8_t) * width * height * bpp;
	const int image_size = width * height;

	uint8_t* d_image;
	cudaMalloc((void**)&d_image, buffer_size);
	cudaMemcpy(d_image, image, buffer_size, cudaMemcpyHostToDevice);

	int blocksx = ceil((image_size) / 256.0f);
	dim3 threads(256);
	dim3 grid(blocksx);

	convert_grayscale_gpu0 << <grid, threads >> > (d_image, width, height, bpp);

	cudaMemcpy(image, d_image, buffer_size, cudaMemcpyDeviceToHost);

	cudaFree(d_image);
}

__global__ void convert_grayscale_gpu0(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp)
{
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	const int tid = gridDim.x * blockDim.x * ty + tx;

	const int size = width * height;

	if (tid >= size)
		return;

	const int offset = tid * bpp;
	double gray = 
		0.2126 * image[offset + 0]
		+ 0.7152 * image[offset + 1]
		+ 0.0722 * image[offset + 2];

	image[offset + 0] = (uint8_t)gray;
	image[offset + 1] = (uint8_t)gray;
	image[offset + 2] = (uint8_t)gray;
}

