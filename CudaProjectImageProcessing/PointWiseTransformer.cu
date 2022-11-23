#include "PointWiseTransformer.cuh"

Image PointWiseTransformer::transform()
{
	Image result(image.get_width(), image.get_height());
	for (int y = 0; y < image.get_height(); y++)
	{
		for (int x = 0; x < image.get_width(); x++)
		{
			result.set_pixel_at(x, y, transform_pixel(image.get_pixel_at(x, y)));
		}
	}

	return result;
}

Image PointWiseTransformer::transform_gpu()
{
	dim3 dg(image.get_width() / BLOCK_SIZE_X + 1, image.get_height() / BLOCK_SIZE_Y + 1, 1);
	dim3 db(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

	Image* dev_image;
	Image* dev_transformer;
	Image* dev_result;

	const int image_buffer_size = image.get_width() * image.get_height() * BPP * sizeof(uint8_t);
	const int transformer_buffer_size = BPP * COLOR_SIZE * sizeof(uint8_t);

	return Image(image.get_width(), image.get_height());
}