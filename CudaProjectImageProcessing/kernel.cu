
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
//
//int main() {
//	int width, height, bpp;
//	uint8_t* rgb_image = stbi_load("test/tower_of_pisa.jpg", &width, &height, &bpp, 0);
//	uint8_t* output_image = (uint8_t*)malloc(width * height * bpp * sizeof(uint8_t));
//
//	printf("%d %d %d\n", width, height, bpp);
//
//	histogram_equalization(rgb_image, width, height, bpp, output_image);
//
//	stbi_image_free(rgb_image);
//	stbi_write_png("output_histogram_equ.png", width, height, bpp, output_image, width * bpp);
//
//	free(output_image);
//	return 0;
//}

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

#include <math.h>
#include "cufft.h"
#include "cufftXt.h"

// Pad data
__global__ void PadData(const cufftComplex* signal, cufftComplex** padded_signal, int signal_size,
	const cufftComplex* filter_kernel, cufftComplex** padded_filter_kernel,
	int filter_kernel_size) {
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	int new_size = signal_size + maxRadius;

	// Pad signal
	cufftComplex* new_data =
		reinterpret_cast<cufftComplex*>(malloc(sizeof(cufftComplex) * new_size));
	memcpy(new_data + 0, signal, signal_size * sizeof(cufftComplex));
	memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(cufftComplex));
	*padded_signal = new_data;

	// Pad filter
	new_data = reinterpret_cast<cufftComplex*>(malloc(sizeof(cufftComplex) * new_size));
	memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(cufftComplex));
	memset(new_data + maxRadius, 0,
		(new_size - filter_kernel_size) * sizeof(cufftComplex));
	memcpy(new_data + new_size - minRadius, filter_kernel,
		minRadius * sizeof(cufftComplex));
	*padded_filter_kernel = new_data;
}

__global__ void printComplex(cufftComplex* a, int input_width, int input_height)
{
	for (int row = 0; row < input_height; row++)
	{
		for (int col = 0; col < input_width; col++)
		{
			printf("%.1f,%.1f ", a[row * input_width + col].x, a[row * input_width + col].y);
		}
		printf("\n");
	}
}
//
//void blur(uint8_t* const image, const int width, const int height, const int bpp, uint8_t* output)
//{
//	const int input_width = width;
//	const int input_height = height;
//	const int kernel_width = 32;
//	const int kernel_height = 32;
//
//	float* A = new float[input_width * input_height];
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			A[i * width + j] = image[i * width + j];
//			printf("%.1f ", A[i * width + j]);
//		}
//		printf("\n");
//	}
//
//	float B[32][32] = {
//		{0.112, 0.112, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0.112},
//		{0.112, 0.112, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0.112},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		 																									
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		 																									
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		 																									
//		{0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0},
//		{0.112, 0.112, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  0, 0.112},
//	};
//
//	cufftReal* d_inA, * d_inB;
//	cufftComplex* d_outA, * d_outB;
//
//	size_t real_size = input_width * input_height * sizeof(cufftReal);
//	size_t complex_size = input_width * (input_height / 2 + 1) * sizeof(cufftComplex);
//
//	cudaMalloc((void**)&d_inA, real_size);
//	cudaMalloc((void**)&d_inB, real_size);
//
//	cudaMalloc((void**)&d_outA, complex_size);
//	cudaMalloc((void**)&d_outB, complex_size);
//
//	cudaMemset(d_inA, 0, real_size);
//	cudaMemset(d_inB, 0, real_size);
//
//	cudaMemcpy(d_inA, A, real_size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_inB, B, real_size, cudaMemcpyHostToDevice);
//
//
//
//	cufftHandle fwplanA, fwplanB, bwplan;
//	cufftPlan2d(&fwplanA, input_height, input_width, CUFFT_R2C);
//	cufftPlan2d(&fwplanB, kernel_height, kernel_width, CUFFT_R2C);
//	cufftPlan2d(&bwplan, input_height, input_width, CUFFT_C2R);
//
//	cufftExecR2C(fwplanA, d_inA, d_outA);
//	cufftExecR2C(fwplanB, d_inB, d_outB);
//
//
//
//	printComplex<<<1, 1>>>(d_outB, input_width, input_height);
//	
//	//////////////
//	// why 1/2 of real? : because the half of complex is conjugates. so, it's not necessary to keep these.
//	/////////////
//	int blocksx = ceil((input_width * (input_height / 2 + 1)) / 256.0f);
//	dim3 threads(256);
//	dim3 grid(blocksx);
//	// One complex product for each thread, scaled by the inverse of the
//	// number of elements involved in the FFT
//	pointwise_product << <grid, threads >> > (d_outA, d_outB, input_width * (input_height / 2 + 1), 1.0f / ((input_width * input_height)));
//
//	cufftExecC2R(bwplan, d_outA, d_inA);
//
//
//	cufftReal* result = new cufftReal[input_width * 2 * (input_height/2+1)];
//	cudaMemcpy(result, d_inA, real_size, cudaMemcpyDeviceToHost);
//
//	// Print result...
//
//	//for (int row = 0; row < input_height; row++)
//	//{
//	//	for (int col = 0; col < input_width; col++)
//	//	{
//	//		printf("%f ", result[row * input_width + col]);
//	//	}
//	//	printf("\n");
//	//}
//	for (int row = 0; row < input_height; row++)
//	{
//		for (int col = 0; col < input_width; col++)
//		{
//			printf("%.2f ", result[row * input_width + col]);
//		}
//		printf("\n");
//	}
//
//
//	for (int row = 0; row < input_height; row++)
//	{
//		for (int col = 0; col < input_width; col++)
//		{
//			output[row * input_width + col] = static_cast<uint8_t>(result[row * input_width + col]);
//			printf("%d ", output[row * input_width + col]);
//		}
//		printf("\n");
//	}
//
//	// Free memory...
//
//	cudaFree(d_inA);
//	cudaFree(d_inB);
//	cudaFree(d_outA);
//	cudaFree(d_outB);
//
//	delete A;
//	delete result;
//}
//

#include "ConvolutionOld.cuh"

int main() 
{
	int width, height, bpp;
	uint8_t* rgb_image = stbi_load("test/tower_of_pisa_almost_square.jpg", &width, &height, &bpp, 0);
	uint8_t* output_image = (uint8_t*)malloc(width * height * bpp * sizeof(uint8_t));
	
	printf("%d %d %d\n", width, height, bpp);

	//blur(rgb_image, width, height, bpp, output_image);
	
	//for (int k = 0; k < 3; k++)
	//{
	//	for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < width; j++)
	//		{
	//			printf("%3d ", rgb_image[(i * width + j) * bpp + k]);
	//		}
	//		printf("\n");
	//	}
	//	printf("\n");
	//}

	float kernel[3][3]{
		{0.11112, 0.11112, 0.11112},
		{0.11112, 0.11112, 0.11112},
		{0.1112, 0.11112, 0.11112}
	};

	float kernel5[5][5]{
		{0.04, 0.04, 0.04, 0.04, 0.04},
		{0.04, 0.04, 0.04, 0.04, 0.04},
		{0.04, 0.04, 0.04, 0.04, 0.04},
		{0.04, 0.04, 0.04, 0.04, 0.04},
		{0.04, 0.04, 0.04, 0.04, 0.04},
	};
	ConvolutionCalculator::convolution(rgb_image, kernel5[0], width, height, 5, 5, bpp, output_image);
	
	stbi_image_free(rgb_image);
	stbi_write_png("output_cufft_encapsulated.png", width, height, bpp, output_image, width * bpp);
	
	//for (int k = 0; k < bpp; k++)
	//{
	//	for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < width; j++)
	//		{
	//			printf("%3d ", output_image[(i * width + j) * bpp + k]);
	//		}
	//		printf("\n");
	//	}
	//	printf("\n");
	//}
	


	free(output_image);
	return 0;

	
}