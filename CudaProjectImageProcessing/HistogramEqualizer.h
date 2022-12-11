#pragma once

#include "Image.h"
#include "PointWiseTransformer.cuh"

namespace PointWiseTransformer
{
	Image histogram_equalize_cpu(Image image);
}