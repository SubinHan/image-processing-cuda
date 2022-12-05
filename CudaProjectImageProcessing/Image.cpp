#pragma once

#include "stb_image.h"
#include "stb_image_write.h"

#include "Image.h"
#include <string>

Image::Image(std::string path)
{
	int width, height, bpp;
	this->image = stbi_load(path.c_str(), &width, &height, &bpp, 0);
	this->width = width;
	this->height = height;
	this->bpp = bpp;
}

Image::Image(int width, int height, int bpp)
{
	this->image = new uint8_t[width * height * bpp];

	std::fill(image, image + width * height * bpp, Image::DEFAULT_INTENSITY);

	this->width = width;
	this->height = height;
	this->bpp = bpp;
}

Image::~Image()
{
	free(image);
}

int Image::get_width() const
{
	return width;
}

int Image::get_height() const
{
	return height;
}

bool Image::is_color() const
{
	return bpp == 3 || bpp == 4;
}

int Image::get_offset(int x, int y) const
{
	int offset = (y * width + x) * bpp;
	return offset;
}

Pixel Image::get_pixel_at(int x, int y) const
{
	int offset = get_offset(x, y);

	Pixel result = {};

	if (is_color())
	{
		result.red = image[offset + 0];
		result.green = image[offset + 1];
		result.blue = image[offset + 2];
		result.intensity = 0.2126f * static_cast<float>(result.red)
			+ 0.7152f * static_cast<float>(result.green)
			+ 0.0722f * static_cast<float>(result.blue);
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

void Image::write(std::string path) const
{
	stbi_write_png(path.c_str(), width, height, bpp, image, width * bpp);
}

uint8_t* Image::get_raw_data() const
{
	return image;
}

int Image::get_bpp() const
{
	return bpp;
}
