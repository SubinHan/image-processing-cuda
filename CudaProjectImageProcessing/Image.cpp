#pragma once

#include "Image.h"
#include <string>
#include "stb_image.h"
#include "stb_image_write.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION  
#endif

Image::Image(std::string path)
{
	int width, height, bpp;
	this->image = stbi_load(path.c_str(), &width, &height, &bpp, 0);
	this->width = width;
	this->height = height;
	this->bpp = bpp;
}

Image::~Image()
{
	free(image);
}

int Image::get_width()
{
	return width;
}

int Image::get_height()
{
	return height;
}

bool Image::is_color()
{
	return bpp == 3 || 4;
}

int Image::get_offset(int x, int y)
{
	int offset = (y * width + x) * bpp;
	return offset;
}

Pixel Image::get_pixel_at(int x, int y)
{
	int offset = get_offset(x, y);

	Pixel result = {};

	if (is_color())
	{
		result.red = image[offset + 0];
		result.green = image[offset + 1];
		result.blue = image[offset + 2];
		result.intensity = 0.2126 * (float)result.red
			+ 0.7152 * (float)result.green
			+ 0.0722 * (float)result.blue;
		return result;
	}

	result.intensity = image[offset];
	result.red = result.blue = result.green = result.intensity;

	return result;
}

void Image::set_pixel_at(int x, int y, Pixel pixel)
{
	int offset = get_offset(x, y);

	if (is_color())
	{
		image[offset + 0] = pixel.red;
		image[offset + 1] = pixel.green;
		image[offset + 2] = pixel.blue;
		return;
	}

	image[offset] = pixel.intensity;
}

void Image::write(std::string path)
{
	stbi_write_png(path.c_str(), width, height, bpp, image, width * bpp);
}
