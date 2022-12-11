#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>

void convert_grayscale_cpu(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp
);

void convert_grayscale_gpu(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp
);

__global__ void convert_grayscale_gpu0(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp
);
