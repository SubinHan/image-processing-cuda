#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cinttypes>

void extract_color_cpu(
	uint8_t* image,
	int width,
	int height,
	int bpp,
	uint8_t* red,
	uint8_t* green,
	uint8_t* blue
);

void extract_color_gpu(
	uint8_t* image,
	int width,
	int height,
	int bpp,
	uint8_t* red,
	uint8_t* green,
	uint8_t* blue
);

__global__ void extract_color_gpu0(
	uint8_t* image,
	int width,
	int height,
	int bpp,
	uint8_t* red,
	uint8_t* green,
	uint8_t* blue
);