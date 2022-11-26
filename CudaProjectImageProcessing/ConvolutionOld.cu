#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include "cufft.h"
#include "cufftXt.h"

#include "ConvolutionOld.cuh"

__global__ void write_data_to_image(uint8_t* image, int bpp, int offset, int size, cufftDoubleReal* data);
__global__ void collect_data(uint8_t* image_in, int bpp, int offset, int size, cufftDoubleReal* image_out);
__global__ void pad_kernel(
	cufftDoubleReal* kernel_input,
	const int image_width,
	const int image_height,
	const int kernel_size,
	cufftDoubleReal* kernel_output);
__global__ void pointwise_product(cufftDoubleComplex* a, cufftDoubleComplex* b, int size, double weight);

static const char* _cudaGetErrorEnum(cufftResult error) {
	switch (error) {
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";

	case CUFFT_INCOMPLETE_PARAMETER_LIST:
		return "CUFFT_INCOMPLETE_PARAMETER_LIST";

	case CUFFT_INVALID_DEVICE:
		return "CUFFT_INVALID_DEVICE";

	case CUFFT_PARSE_ERROR:
		return "CUFFT_PARSE_ERROR";

	case CUFFT_NO_WORKSPACE:
		return "CUFFT_NO_WORKSPACE";

	case CUFFT_NOT_IMPLEMENTED:
		return "CUFFT_NOT_IMPLEMENTED";

	case CUFFT_LICENSE_ERROR:
		return "CUFFT_LICENSE_ERROR";

	case CUFFT_NOT_SUPPORTED:
		return "CUFFT_NOT_SUPPORTED";
	}

	return "<unknown>";
}

static const char* _cudaGetErrorEnum(cudaError_t error) {
	return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const* const func, const char* const file,
	int const line) {
	if (result) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
			static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		exit(EXIT_FAILURE);
	}
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void collect_data(uint8_t* image_in, int bpp, int offset, int size, cufftDoubleReal* image_out)
{
	unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_index >= size)
		return;

	image_out[thread_index] = image_in[thread_index * bpp + offset];
}

__global__ void write_data_to_image(uint8_t* image, int bpp, int offset, int size, cufftDoubleReal* data)
{
	unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_index > size)
		return;

	if (data[thread_index] >= 255.f)
		data[thread_index] = 255.f;
	image[thread_index * bpp + offset] = data[thread_index];
}

__global__ void pad_kernel(cufftDoubleReal* kernel_input, const int image_width, const int image_height, const int kernel_size, cufftDoubleReal* kernel_output)
{
	unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = image_width * image_height;
	
	if (thread_index >= size)
		return;

	const int min_radius = kernel_size / 2;
	const int max_radius = kernel_size - min_radius;

	const int x = thread_index % image_width;
	const int y = thread_index / image_width;

	const bool is_x_left = x < max_radius;
	const bool is_x_right = x >= image_width - min_radius;
	const bool is_y_up = y < max_radius;
	const bool is_y_down = y >= image_height - min_radius;

	//printf("[%d] (%d, %d) (%d %d %d %d)\n", thread_index, x, y, is_x_left, is_x_right, is_y_up, is_y_down);
	//printf("[%d] [%d %d %d %d]\n", thread_index, is_x_left && is_y_up, is_x_right && is_y_up, is_x_left && is_y_down, is_x_right && is_y_down);
	//printf("[%d] [%d]\n", thread_index, (min_radius + y) * kernel_size + (x - (image_width - min_radius)));
	if (is_x_left && is_y_up)
	{
		int offset = (min_radius + y) * kernel_size + (min_radius + x);
		//printf("[%d] %d\n", thread_index, offset);
		kernel_output[thread_index] = kernel_input[offset];
		return;
	}

	if (is_x_right && is_y_up)
	{
		int offset = (min_radius + y) * kernel_size + (x - (image_width - min_radius));
		//printf("[%d] %d\n", thread_index, offset);
		kernel_output[thread_index] = kernel_input[offset];
		return;
	}

	if (is_x_left && is_y_down)
	{
		int offset = (y - (image_height - min_radius)) * kernel_size + (min_radius + x);
		//printf("[%d] %d\n", thread_index, offset);
		kernel_output[thread_index] = kernel_input[offset];
		return;
	}

	if (is_x_right && is_y_down)
	{
		int offset = (y - (image_height - min_radius)) * kernel_size + (x - (image_width - min_radius));
		//printf("[%d] %d\n", thread_index, offset);
		kernel_output[thread_index] = kernel_input[offset];
		return;
	}

	//printf("[%d] 0\n", thread_index);
	kernel_output[thread_index] = 0.f;
}

__global__ void scale(cufftDoubleComplex* complex, const int image_width, const int image_height, double weight)
{
	unsigned thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = image_width * image_height;

	if (thread_index >= size)
		return;

	complex[thread_index].x *= weight;
	complex[thread_index].y *= weight;
}

__global__ void convert_real_to_complex(cufftDoubleReal* real, const int image_width, const int image_height, cufftDoubleComplex* complex)
{
	unsigned thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = image_width * image_height;

	if (thread_index >= size)
		return;

	complex[thread_index].x = real[thread_index];
	complex[thread_index].y = 0.f;
}

__global__ void convert_complex_to_real(cufftDoubleComplex* complex, const int image_width, const int image_height, cufftDoubleReal* real)
{
	unsigned thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = image_width * image_height;

	if (thread_index >= size)
		return;

	real[thread_index] = complex[thread_index].x;
}

__global__ void pointwise_product2(cufftDoubleComplex* a, cufftDoubleComplex* b, int size, double weight)
{
	unsigned thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= size)
		return;

	a[thread_index].x = a[thread_index].x * b[thread_index].x;
	a[thread_index].y = a[thread_index].y * b[thread_index].y;
	a[thread_index].x *= weight;
	a[thread_index].y *= weight;
}

__global__ void pointwise_product(cufftDoubleComplex* a, cufftDoubleComplex* b, int size, double weight)
{
	unsigned thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= size)
		return;

	//printf("[%d] (%f, %f) * (%f, %f) = \n", thread_index, a[thread_index].x, a[thread_index].y, b[thread_index].x, b[thread_index].y);

	double a_real_original = a[thread_index].x;
	a[thread_index].x = a[thread_index].x * b[thread_index].x - a[thread_index].y * b[thread_index].y;
	a[thread_index].y = a_real_original * b[thread_index].y + a[thread_index].y * b[thread_index].x;
	//printf("[%d] (%f, %f) = \n", thread_index, a[thread_index].x, a[thread_index].y);
	a[thread_index].x *= weight;
	a[thread_index].y *= weight;

	//printf("[%d] = (%f, %f)\n", thread_index, a[thread_index].x, a[thread_index].y);
}

__global__ void print_2d_real(cufftDoubleReal* d_arr, int width, int height)
{
	for (int i = 0; i < height; i++)
	{
		//printf("%d: ", i);
		for (int j = 0; j < width; j++)
		{
			//printf("%4d", j);
			printf("%3.1f ", d_arr[i * width + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void ConvolutionCalculator::convolution(
	const uint8_t* image_in,
	const double* kernel,
	const int image_width,
	const int image_height,
	const int kernel_width,
	const int kernel_height,
	const int bpp,
	uint8_t* image_out
)
{
	const int image_size_with_bpp = image_width * image_height * bpp;
	const int image_size = image_width * image_height;
	const int kernel_size = kernel_width * kernel_height;

	const int complex_half_size = image_width * (image_height / 2 + 1);

	uint8_t* d_int8_image = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_int8_image, image_size_with_bpp * sizeof(uint8_t)));
	checkCudaErrors(cudaMemcpy(d_int8_image, image_in, image_size_with_bpp * sizeof(uint8_t), cudaMemcpyHostToDevice));


	cufftDoubleReal* d_real_kernel = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_real_kernel, kernel_size * sizeof(cufftDoubleReal)));
	checkCudaErrors(cudaMemcpy(d_real_kernel, kernel, kernel_size * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice));

	cufftDoubleReal* d_real_kernel_padded = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_real_kernel_padded, image_size * sizeof(cufftDoubleReal)));

	int blocksx = ceil((image_size) / 256.0f);
	dim3 threads(256);
	dim3 grid(blocksx);

	//printf("grid: %d, threads: %d\n", grid.x, threads.x);
	pad_kernel << <grid, threads >> > (d_real_kernel, image_width, image_height, kernel_width, d_real_kernel_padded);
	checkCudaErrors(cudaDeviceSynchronize());

	//print_2d_real << <1, 1 >> > (d_real_kernel_padded, image_width, image_height);

	cufftDoubleComplex* d_complex_kernel = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_complex_kernel, image_size * sizeof(cufftDoubleComplex)));
	convert_real_to_complex << <grid, threads >> > (d_real_kernel, image_width, image_height, d_complex_kernel);

	cufftHandle plan_complex_to_complex;
	checkCudaErrors(cufftPlan2d(&plan_complex_to_complex, image_height, image_width, CUFFT_Z2Z));

	cufftDoubleComplex* d_complex_kernel_frequency = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_complex_kernel_frequency, image_size * sizeof(cufftDoubleComplex)));
	cufftExecZ2Z(plan_complex_to_complex, d_complex_kernel, d_complex_kernel_frequency, CUFFT_FORWARD);
	checkCudaErrors(cudaDeviceSynchronize());

	//scale << <grid, threads >> > (d_complex_kernel_frequency, image_width, image_height, 1.0f / (image_width * image_height));
	checkCudaErrors(cudaDeviceSynchronize());

	for (int i = 0; i < bpp; i++)
	{
		cufftDoubleReal* d_real_image = nullptr;

		checkCudaErrors(cudaMalloc((void**)&d_real_image, image_size * sizeof(cufftDoubleReal)));

		collect_data<<<grid, threads>>>(d_int8_image, bpp, i, image_size, d_real_image);

		checkCudaErrors(cudaDeviceSynchronize());
		//print_2d_real << <1, 1 >> > (d_real_image, image_width, image_height);

		cufftDoubleComplex* d_complex_image = nullptr;
		checkCudaErrors(cudaMalloc((void**)&d_complex_image, image_size * sizeof(cufftDoubleComplex)));
		convert_real_to_complex << <grid, threads >> > (d_real_image, image_width, image_height, d_complex_image);

		cufftDoubleComplex* d_complex_image_frequency = nullptr;
		checkCudaErrors(cudaMalloc((void**)&d_complex_image_frequency, image_size * sizeof(cufftDoubleComplex)));
		checkCudaErrors(cufftExecZ2Z(plan_complex_to_complex, d_complex_image, d_complex_image_frequency, CUFFT_FORWARD));
		checkCudaErrors(cudaDeviceSynchronize());

		scale << <grid, threads >> > (d_complex_image_frequency, image_width, image_height, 1.0f / (image_width * image_height));
		checkCudaErrors(cudaDeviceSynchronize());


		//std::cout << multiplication_grid.x << multiplication_grid.y << multiplication_grid.z << multiplication_threads.x << multiplication_threads.y << multiplication_threads.z;
		//printf("%f", 1.0f / (image_width * image_height));
		pointwise_product2 << <grid, threads>> > (d_complex_image_frequency, d_complex_kernel_frequency, image_size, 1.0f);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cufftExecZ2Z(plan_complex_to_complex, d_complex_image_frequency, d_complex_image, CUFFT_INVERSE));
		checkCudaErrors(cudaDeviceSynchronize());

		convert_complex_to_real << <grid, threads >> > (d_complex_image, image_width, image_height, d_real_image);
		checkCudaErrors(cudaDeviceSynchronize());

		//print_2d_real << <1, 1 >> > (d_real_image, image_width, image_height);

		write_data_to_image << <grid, threads >> > (d_int8_image, bpp, i, image_size, d_real_image);
		checkCudaErrors(cudaDeviceSynchronize());

		cudaFree(d_real_image);
		cudaFree(d_complex_image);
		cudaFree(d_complex_image_frequency);
	}
	cufftDestroy(plan_complex_to_complex);

	checkCudaErrors(cudaMemcpy(image_out, d_int8_image, image_size_with_bpp * sizeof(uint8_t), cudaMemcpyDeviceToHost));

	cudaFree(d_real_kernel);
	cudaFree(d_real_kernel_padded);
	cudaFree(d_complex_kernel);
	cudaFree(d_complex_kernel_frequency);
	cudaFree(d_int8_image);

	checkCudaErrors(cudaPeekAtLastError());
}