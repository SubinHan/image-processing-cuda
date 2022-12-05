#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Image.h"

namespace PointWiseTransformer
{
	template <typename Func>
	Image transform(const Image image);

	template <typename Func>
	Image transform_gpu(const Image image);
};