#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Image.h"

class PointWiseTransformer
{
public:

private:
	static constexpr int COLOR_SIZE = 256;
	static constexpr int BPP = 3;
	static constexpr int R = 0;
	static constexpr int G = 1;
	static constexpr int B = 2;

	static constexpr int BLOCK_SIZE_X = 16;
	static constexpr int BLOCK_SIZE_Y = 16;

	Image image;

public:
	Image transform();
	Image transform_gpu();

	virtual Pixel transform_pixel(Pixel p) = 0;
};