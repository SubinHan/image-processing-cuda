#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <cmath>
#include <memory>

//constexpr int COLOR_SIZE = 256;
//constexpr int BPP = 3;
//constexpr int R = 0;
//constexpr int G = 1;
//constexpr int B = 2;
//constexpr int BLOCK_SIZE_X = 16;
//constexpr int BLOCK_SIZE_Y = 16;
//
//__global__ void transform_gpu0(uint8_t* const image, const int width, const int bpp, uint8_t* const transformer, uint8_t* result)
//{
//	int tx = blockDim.x * blockIdx.x + threadIdx.x;
//	int ty = blockDim.y * blockIdx.y + threadIdx.y;
//	int logical_tid = ty * width + tx;
//	int image_offset = logical_tid * bpp;
//
//	for (int color_offset = 0; color_offset < bpp; color_offset++)
//	{
//		uint8_t color = *(image + image_offset + color_offset);
//		*(result + image_offset + color_offset) = *(transformer + COLOR_SIZE * color_offset + color);
//	}
//}
//
//void transform_gpu(uint8_t* const image, const int width, const int height, const int bpp, uint8_t* const transformer, uint8_t* result)
//{
//	dim3 dg(width / BLOCK_SIZE_X + 1, height / BLOCK_SIZE_Y + 1, 1);
//	dim3 db(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
//	
//	uint8_t* dev_image;
//	uint8_t* dev_transformer;
//	uint8_t* dev_result;
//
//	const int image_buffer_size = width * height * bpp * sizeof(uint8_t);
//	const int transformer_buffer_size = bpp * COLOR_SIZE * sizeof(uint8_t);
//
//	cudaMalloc((void**)&dev_image, image_buffer_size);
//	cudaMalloc((void**)&dev_transformer, transformer_buffer_size);
//	cudaMalloc((void**)&dev_result, image_buffer_size);
//
//	cudaMemcpy(dev_image, image, image_buffer_size, cudaMemcpyHostToDevice);
//	cudaMemcpy(dev_transformer, transformer, transformer_buffer_size, cudaMemcpyHostToDevice);
//
//	transform_gpu0 << <dg, db >> > (dev_image, width, bpp, dev_transformer, dev_result);
//
//	cudaMemcpy(result, dev_result, image_buffer_size, cudaMemcpyDeviceToHost);
//	cudaFree(dev_image);
//	cudaFree(dev_transformer);
//	cudaFree(dev_result);
//}
//
//void transform(uint8_t* const image, const int width, const int height, const int bpp, uint8_t* const transformer, uint8_t* result)
//{
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			for (int k = 0; k < bpp; k++)
//			{
//				int logical_offset = get_offset(j, i, width);
//				int image_offset = logical_offset * bpp + k;
//				uint8_t color = *(image + image_offset);
//
//				*(result + image_offset) = *(transformer + COLOR_SIZE * k + color);
//			}
//		}
//	}
//}
//
//void histogram_equalization(uint8_t* const image, const int width, const int height, const int bpp, uint8_t* result)
//{
//	int count[BPP][COLOR_SIZE];
//	count_intensity(image, width, height, bpp, count[0]);
//
//	uint8_t transformer[BPP][COLOR_SIZE];
//
//	for (int i = 0; i < bpp; i++)
//	{
//		int min_intensity = find_min(count[i]);
//		int count_min_intensity;
//
//		int cumulative[COLOR_SIZE];
//		calculate_cumulative(count[i], cumulative);
//		count_min_intensity = cumulative[min_intensity];
//
//		for (int j = 0; j < COLOR_SIZE; j++)
//		{
//			transformer[i][j] = round(((double)(cumulative[j] - count_min_intensity) / (double)(width * height - count_min_intensity) * (double)(COLOR_SIZE - 1)));
//		}
//	}
//
//	transform_gpu(image, width, height, bpp, transformer[0], result);
//}
//
//void print_image(uint8_t* const image, const int width, const int height, const int bpp)
//{
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			uint8_t* pixel = get_pixel(image, width, height, j, i, bpp);
//			
//			printf("[");
//			for (int k = 0; k < bpp; k++)
//			{
//				printf("%d ", pixel[k]);
//			}
//			printf("] ");
//		}
//		printf("\n");
//	}
//}
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


#include "Image.h"
#include "Timer.h"
#include "GrayscaleOld.cuh"
#include "BrightnessOld.cuh"
#include "HsvOld.cuh"
#include "ConvolutionOld.cuh"
#include <vector>

void perform_comparison(std::vector<Image> images_to_test, void (*function_cpu)(Image), void (*function_gpu)(Image))
{
	for (Image image : images_to_test)
	{
		printf("image size: (%d, %d)\n", image.get_width(), image.get_height());
		Timer timer;
		function_cpu(image);
		float elapsed_time_cpu = timer.get_elapsed_seconds();
		printf("\tthe formmer's time: %f\n", elapsed_time_cpu);

		timer.reset();
		function_gpu(image);
		float elapsed_time_gpu = timer.get_elapsed_seconds();
		printf("\tthe latter's time: %f\n", elapsed_time_gpu);
	}
}

void grayscale_cpu(Image image)
{
	convert_grayscale_cpu(
		image.get_raw_data(),
		image.get_width(),
		image.get_height(),
		image.get_bpp()
	);
}

void grayscale_gpu(Image image)
{
	convert_grayscale_gpu(
		image.get_raw_data(),
		image.get_width(),
		image.get_height(),
		image.get_bpp()
	);
}

void brightness_cpu(Image image)
{
	change_brightness_cpu(
		image.get_raw_data(),
		image.get_width(),
		image.get_height(),
		image.get_bpp(),
		50.0f
	);
}

void brightness_gpu(Image image)
{
	change_brightness_gpu(
		image.get_raw_data(),
		image.get_width(),
		image.get_height(),
		image.get_bpp(),
		50.0f
	);
}

void hsv_cpu(Image image)
{
	const int width = image.get_width();
	const int height = image.get_height();
	const int bpp = image.get_bpp();

	Image red(width, height, bpp);
	Image green(width, height, bpp);
	Image blue(width, height, bpp);

	extract_color_cpu(
		image.get_raw_data(),
		width,
		height,
		bpp,
		red.get_raw_data(),
		green.get_raw_data(),
		blue.get_raw_data()
	);
}

void hsv_gpu(Image image)
{
	const int width = image.get_width();
	const int height = image.get_height();
	const int bpp = image.get_bpp();

	Image red(width, height, bpp);
	Image green(width, height, bpp);
	Image blue(width, height, bpp);

	extract_color_gpu(
		image.get_raw_data(),
		width,
		height,
		bpp,
		red.get_raw_data(),
		green.get_raw_data(),
		blue.get_raw_data()
	);
}

void conv_cpu(Image image)
{
	constexpr int KERNEL_SIZE = 7;

	const float blur[KERNEL_SIZE][KERNEL_SIZE] = { { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0} };

	convolution_cpu(
		image.get_raw_data(),
		image.get_width(),
		image.get_height(),
		blur[0],
		KERNEL_SIZE,
		image.get_bpp(),
		image.get_raw_data()
	);
}

void conv_gpu_naive(Image image)
{
	constexpr int KERNEL_SIZE = 7;

	const float blur[KERNEL_SIZE][KERNEL_SIZE] = { { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0} };

	convolution_gpu_naive(
		image.get_raw_data(),
		image.get_width(),
		image.get_height(),
		blur[0],
		KERNEL_SIZE,
		image.get_bpp(),
		image.get_raw_data()
	);
}

void conv_gpu_cufft(Image image)
{
	constexpr int KERNEL_SIZE = 7;

	const float blur[KERNEL_SIZE][KERNEL_SIZE] = { { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0},
					 { 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0, 1.0 / 49.0} };

	convolution_gpu_cufft(
		image.get_raw_data(),
		image.get_width(),
		image.get_height(),
		blur[0],
		KERNEL_SIZE,
		image.get_bpp(),
		image.get_raw_data()
	);
}

int main()
{
	std::vector<Image> images_to_test;
	
	images_to_test.push_back(Image("test/32x32.png"));
	images_to_test.push_back(Image("test/64x64.png"));
	images_to_test.push_back(Image("test/512x512.png"));
	images_to_test.push_back(Image("test/1920x1080.jpg"));
	//images_to_test.push_back(Image("test/3840x2160.jpg"));
	//images_to_test.push_back(Image("test/7680x4320.jpg"));
	
	printf("Comparison: Grayscale\n");
	perform_comparison(
		images_to_test,
		grayscale_cpu,
		grayscale_gpu
	);

	printf("\n\nComparison: Brightness\n");
	perform_comparison(
		images_to_test,
		brightness_cpu,
		brightness_gpu
	);

	printf("\n\nComparison: Color Extraction\n");
	perform_comparison(
		images_to_test,
		hsv_cpu,
		hsv_gpu
	);

	printf("\n\nComparison: Convolution, CPU vs GPU\n");
	perform_comparison(
		images_to_test,
		conv_cpu,
		conv_gpu_naive
	);


	printf("\n\nComparison: Convolution, GPU naive vs GPU cufft\n");
	perform_comparison(
		images_to_test,
		conv_gpu_naive,
		conv_gpu_cufft
	);


	Image test("test/512x512.png");

	float blur5[5][5]{
		{0.04, 0.04, 0.04, 0.04, 0.04},
		{0.04, 0.04, 0.04, 0.04, 0.04},
		{0.04, 0.04, 0.04, 0.04, 0.04},
		{0.04, 0.04, 0.04, 0.04, 0.04},
		{0.04, 0.04, 0.04, 0.04, 0.04},
	};

	float edge3[3][3]{
		{-1, -2, -1},
		{-2, 12, -2},
		{-1, -2, -1},
	};

	float edge5[5][5]{
		{0, 0, -1, 0, 0},
		{0, -1, -2, -1, 0},
		{-1, -2, 16, -2, -1},
		{0, -1, -2, -1, 0},
		{0, 0, -1, 0, 0},
	};

	convolution_gpu_cufft(test.get_raw_data(),
		test.get_width(),
		test.get_height(),
		blur5[0],
		5,
		test.get_bpp(),
		test.get_raw_data()
	);
	test.write("output_blur.png");

	test = Image("test/512x512.png");

	convolution_gpu_cufft(test.get_raw_data(),
		test.get_width(),
		test.get_height(),
		edge3[0],
		3,
		test.get_bpp(),
		test.get_raw_data()
	);

	test.write("output_edge.png");
}