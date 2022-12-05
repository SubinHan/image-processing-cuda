#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Image.h"

namespace PointWiseTransformer
{
	template <typename Func>
	Image transform(const Image image)
	{
		Image result(image.get_width(), image.get_height());
		Func transformer;
		for (int y = 0; y < image.get_height(); y++)
		{
			for (int x = 0; x < image.get_width(); x++)
			{
				result.set_pixel_at(x, y, transformer(image.get_pixel_at(x, y)));
			}
		}

		return result;
	}

	template <typename Func>
	__global__ void create_transformer(Func* transformer)
	{
		transformer = new Func();
	}

	template <typename Func>
	__global__ void perform_transform0(Func* d_transformer, uint8_t* const d_image, const int width, const int height, const int bpp, uint8_t* d_result)
	{
		int tx = blockDim.x * blockIdx.x + threadIdx.x;
		int ty = blockDim.y * blockIdx.y + threadIdx.y;
		int logical_tid = ty * width + tx;
		int image_offset = logical_tid * bpp;

		Pixel* pixel_in = new Pixel(d_image[image_offset + R], d_image[image_offset + G], d_image[image_offset + B]);
		Pixel* pixel_out = new Pixel();

		d_transformer(pixel_in, pixel_out);

		if (bpp == 3)
		{
			d_result[image_offset + R] = pixel_out->red;
			d_result[image_offset + G] = pixel_out->green;
			d_result[image_offset + B] = pixel_out->blue;
		}
		else
		{
			d_result[image_offset] = pixel_out->intensity;
		}

		delete pixel_in;
		delete pixel_out;
	}

	template <typename Func>
	Image transform_gpu(const Image image)
	{
		dim3 dg(image.get_width() / PointWiseTransformer::BLOCK_SIZE_X + 1, image.get_height() / PointWiseTransformer::BLOCK_SIZE_Y + 1, 1);
		dim3 db(PointWiseTransformer::BLOCK_SIZE_X, PointWiseTransformer::BLOCK_SIZE_Y, 1);

		uint8_t* d_image;
		uint8_t* d_transformer;
		uint8_t* d_result;

		const int bpp = image.get_bpp();
		const int width = image.get_width();
		const int height = image.get_height();

		const int image_buffer_size = width * height * bpp * sizeof(uint8_t);

		checkCudaErrors(cudaMalloc((void**)&d_image, image_buffer_size));
		checkCudaErrors(cudaMalloc((void**)&d_result, image_buffer_size));
		Func* d_transformer;
		create_transformer<Func> << <1, 1 >> > (d_transformer);

		checkCudaErrors(cudaMemcpy(d_image, image.get_raw_data(), image_buffer_size, cudaMemcpyHostToDevice));

		perform_transform0 << <dg, db >> > (d_image, d_transformer, width, height, bpp, d_result);

		//d_transform_gpu <<<dg, db >>> (d_image, width, height, bpp, d_result);

		Image result(image.get_width(), image.get_height());

		checkCudaErrors(cudaMemcpy(result.get_raw_data(), image.get_raw_data(), image_buffer_size, cudaMemcpyHostToDevice));

		cudaFree(d_image);
		cudaFree(d_transformer);
		cudaFree(d_result);

		return result;
	}	
}