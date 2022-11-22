#pragma once

#include "Image.h"
#include "PointWiseTransformer.h"

class HistogramEqualizer : PointWiseTransformer
{
public:
	Pixel transform_pixel(Pixel p);

private:

};