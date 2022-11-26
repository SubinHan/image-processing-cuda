#include <cstdint>

class ConvolutionCalculator
{
public:
	static void convolution(
		const uint8_t* image_in,
		const double* kernel,
		const int image_width,
		const int image_height,
		const int kernel_width,
		const int kernel_height,
		const int bpp,
		uint8_t* image_out
	);

};

