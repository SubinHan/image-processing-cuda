#include "HistogramEqualizer.h"

int find_min(int* counts)
{
	for (int i = 0; i < PointWiseTransformer::COLOR_SIZE; i++)
	{
		if (counts[i] > 0)
		{
			return i;
		}
	}
}

void count_intensity(const Image image, int* result)
{
	constexpr int R = 0;
	constexpr int G = 1;
	constexpr int B = 2;
	const int bpp = image.get_bpp();
	const int height = image.get_height();
	const int width = image.get_width();

	for (int i = 0; i < bpp; i++)
	{
		for (int j = 0; j < PointWiseTransformer::COLOR_SIZE; j++)
		{
			*(result + i * PointWiseTransformer::COLOR_SIZE + j) = 0;
		}
	}

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Pixel pixel = image.get_pixel_at(width, height);

			(*(result + R * PointWiseTransformer::COLOR_SIZE + pixel.red))++;
			(*(result + G * PointWiseTransformer::COLOR_SIZE + pixel.green))++;
			(*(result + B * PointWiseTransformer::COLOR_SIZE + pixel.blue))++;
		}
	}
}

void calculate_cumulative(int* counts, int* cumulative)
{
	cumulative[0] = counts[0];
	for (int i = 1; i < PointWiseTransformer::COLOR_SIZE; i++)
	{
		cumulative[i] = cumulative[i - 1] + counts[i];
	}
}

int get_offset(const int x, const int y, const int width)
{
	return y * width + x;
}

Image transform_cpu(Image image, uint8_t* const transformer)
{
	const int height = image.get_height();
	const int width = image.get_width();
	const int bpp = image.get_bpp();

	Image result(image.get_width(), image.get_height());

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Pixel pixel = image.get_pixel_at(j, i);
			int logical_offset = get_offset(j, i, width);
			int image_offset = logical_offset * bpp;

			pixel.red = *(transformer + PointWiseTransformer::COLOR_SIZE * 0 + pixel.red);
			pixel.green = *(transformer + PointWiseTransformer::COLOR_SIZE * 0 + pixel.green);
			pixel.blue = *(transformer + PointWiseTransformer::COLOR_SIZE * 0 + pixel.blue);

			image.set_pixel_at(j, i, pixel);

			//*(result + image_offset + 1) = *(transformer + PointWiseTransformer::COLOR_SIZE * 0 + pixel.red);
			//*(result + image_offset + 2) = *(transformer + PointWiseTransformer::COLOR_SIZE * 0 + pixel.red);
		}
	}

	return result;
}

struct Transformer
{
	uint8_t* transformer;

	Transformer(Image image)
	{
		int count[3][PointWiseTransformer::COLOR_SIZE];
		count_intensity(image, count[0]);

		uint8_t transformer[3][PointWiseTransformer::COLOR_SIZE];

		const int bpp = image.get_bpp();
		const int width = image.get_width();
		const int height = image.get_height();

		for (int i = 0; i < bpp; i++)
		{
			int min_intensity = find_min(count[i]);
			int count_min_intensity;

			int cumulative[PointWiseTransformer::COLOR_SIZE];
			calculate_cumulative(count[i], cumulative);
			count_min_intensity = cumulative[min_intensity];

			for (int j = 0; j < PointWiseTransformer::COLOR_SIZE; j++)
			{
				transformer[i][j] = round(((double)(cumulative[j] - count_min_intensity) / (double)(width * height - count_min_intensity) * (double)(COLOR_SIZE - 1)));
			}
		}
	}

	Pixel operator()(Pixel input)
	{
		input.red = *(transformer + PointWiseTransformer::COLOR_SIZE * 0 + input.red);
		input.green = *(transformer + PointWiseTransformer::COLOR_SIZE * 0 + input.green);
		input.blue = *(transformer + PointWiseTransformer::COLOR_SIZE * 0 + input.blue);
	}
};

Image PointWiseTransformer::histogram_equalize_cpu(Image image)
{
	return transform<Transformer>(image);
}