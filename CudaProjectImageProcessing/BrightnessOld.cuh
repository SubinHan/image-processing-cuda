#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cinttypes>

void change_brightness_cpu(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp,
	float percent
);

void change_brightness_gpu(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp,
	float percent
);

__global__ void change_brightness_gpu0(
	uint8_t* image,
	const int width,
	const int height,
	const int bpp,
	float percent
);
