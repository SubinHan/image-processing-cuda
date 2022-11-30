#include <cstdint>

class ConvolutionCalculator
{
public:
	static void convolution_naive(const uint8_t* image_in,
		const float* kernel,
		const int image_width,
		const int image_height,
		const int kernel_width,
		const int kernel_height,
		const int bpp,
		uint8_t* image_out
	);

	static void convolution_cufft(
		const uint8_t* image_in,
		const float* kernel,
		const int image_width,
		const int image_height,
		const int kernel_width,
		const int kernel_height,
		const int bpp,
		uint8_t* image_out
	);

};

